"""Train a response predictor with a dedicated rare-event branch."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from risk_surrogate import ExceedanceNetwork

THRESHOLDS = np.array([0.02, 14.71, 0.01], dtype=np.float32)
NAMES = ('peak_drift', 'peak_floor_acceleration', 'residual_drift')


def batch_pairs(buildings, records):
    return np.stack(np.meshgrid(buildings, records, indexing='ij'), axis=-1).reshape(-1, 2)


def make_tensors(data, pairs):
    b, g = pairs[:, 0], pairs[:, 1]
    return (torch.from_numpy(data['cont'][b].reshape(-1, 57, 2)),
            torch.from_numpy(data['mask'][b].reshape(-1, 57)),
            torch.from_numpy(data['geom'][b]),
            torch.from_numpy(data['gm'][g]),
            torch.from_numpy(data['response_norm'][b, g]),
            torch.from_numpy((data['response'][b, g] > THRESHOLDS).astype('float32')))


def infer(model, data, pairs, batch=1024):
    means, probabilities, labels, outcomes = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            x, m, geom, gm, truth, indicator = make_tensors(data, pairs[i:i+batch])
            pred, _, logits = model(x, m, geom, gm)
            means.append(np.expm1(pred.numpy()*data['response_std'] + data['response_mu']).clip(0))
            probabilities.append(torch.sigmoid(logits).numpy())
            labels.append(indicator.numpy())
            outcomes.append(data['response'][pairs[i:i+batch, 0], pairs[i:i+batch, 1]])
    return np.concatenate(means), np.concatenate(probabilities), np.concatenate(labels), np.concatenate(outcomes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--width', type=int, default=192)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    data = dict(np.load(ROOT / 'data/processed/designsafe_processed.npz'))
    split = json.loads((ROOT / 'data/splits/split.json').read_text())
    train = batch_pairs(split['buildings']['train'], split['ground_motions']['train'])
    valid = batch_pairs(split['buildings']['val'], split['ground_motions']['val'])
    test = batch_pairs(split['buildings']['test'], split['ground_motions']['test'])
    exceed_count = np.array([(data['response'][train[:, 0], train[:, 1], j] > THRESHOLDS[j]).sum() for j in range(3)])
    imbalance = (len(train) - exceed_count)/np.maximum(exceed_count, 1)
    # Capping the ratio prevents one extremely rare response from dominating every update.
    positive_weight = torch.tensor(np.minimum(imbalance, 50), dtype=torch.float32)
    net = ExceedanceNetwork(args.width)
    optim = torch.optim.AdamW(net.parameters(), lr=.0006, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(1, args.epochs), eta_min=.00006)
    best = float('inf'); best_epoch = 0; history = []
    checkpoint = ROOT / 'checkpoints' / 'risk_aware_surrogate.pt'
    for epoch in range(args.epochs):
        net.train(); idx = np.random.permutation(len(train)); total = 0.; total_count = 0
        for i in range(0, len(idx), 512):
            batch = train[idx[i:i+512]]
            x, mask, geom, gm, y, classes = make_tensors(data, batch)
            mean, logvar, logits = net(x, mask, geom, gm)
            loss, _, _ = net.objective(mean, logvar, logits, y, classes, positive_weight)
            optim.zero_grad(); loss.backward(); clip_grad_norm_(net.parameters(), 1.); optim.step()
            total += loss.item()*len(batch); total_count += len(batch)
        net.eval(); vtot=0.; n=0
        with torch.inference_mode():
            for i in range(0, len(valid), 1024):
                batch = valid[i:i+1024]
                x, mask, geom, gm, y, classes = make_tensors(data, batch)
                mean, logvar, logits = net(x, mask, geom, gm)
                loss, _, _ = net.objective(mean, logvar, logits, y, classes, positive_weight)
                vtot += loss.item() * len(batch); n += len(batch)
        row={'epoch': epoch+1, 'training_objective': total/total_count,
             'validation_objective': vtot/n, 'learning_rate': optim.param_groups[0]['lr']}
        history.append(row)
        if vtot/n < best:
            best=vtot/n;best_epoch=epoch+1
            torch.save({'model': net.state_dict(), 'width': args.width, 'seed': args.seed,
                        'weights': positive_weight, 'epoch': best_epoch}, checkpoint)
        scheduler.step()
        if (epoch+1) % 5 == 0:
            print('epoch', epoch+1, 'train', round(row['training_objective'], 4),
                  'validation', round(row['validation_objective'], 4), flush=True)
    pd.DataFrame(history).to_csv(ROOT / 'logs' / 'risk_surrogate_training.csv', index=False)
    best_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net.load_state_dict(best_state['model'])
    estimates, prob, truth_labels, measured = infer(net, data, test)
    rows=[]
    for j, key in enumerate(NAMES):
        target=truth_labels[:, j]; score=prob[:, j]
        rows.append({'response': key, 'positives': int(target.sum()), 'test_pairs': len(target),
                     'prevalence': float(target.mean()),
                     'roc_auc': float(roc_auc_score(target, score)) if target.min() != target.max() else None,
                     'average_precision': float(average_precision_score(target, score)),
                     'brier_score': float(brier_score_loss(target, score)),
                     'recall_at_0_05': float(((score >= .05) & (target > .5)).sum() / max(target.sum(), 1)),
                     'positive_prediction_rate_at_0_05': float((score >= .05).mean()),
                     'mae': float(np.mean(np.abs(estimates[:, j] - measured[:, j]))),
                     'r2': float(1 - np.sum((estimates[:, j]-measured[:, j]) ** 2) /
                                  np.sum((measured[:, j]-measured[:, j].mean())**2))})
    scores = pd.DataFrame(rows)
    scores.to_csv(ROOT / 'results' / 'risk_surrogate_test_metrics.csv', index=False)
    np.savez_compressed(ROOT / 'results' / 'risk_surrogate_test_predictions.npz',
                        probability=prob, labels=truth_labels, predicted=estimates,
                        observed=measured, test_pairs=test)
    report={'best_epoch': best_epoch, 'best_validation_objective': best,
            'training_pairs': len(train), 'validation_pairs': len(valid), 'test_pairs': len(test),
            'train_positive_counts': exceed_count.tolist(), 'positive_class_weight': positive_weight.tolist(),
            'test_metrics': rows, 'interpretation': 'Proxy screening only; no nonlinear analysis of generated designs.'}
    (ROOT/'results'/'risk_surrogate_metrics.json').write_text(json.dumps(report, indent=2))
    fig = ROOT/'results'/'audit_figures'
    plt.figure(figsize=(7.2, 4.6))
    plt.plot([h['epoch'] for h in history], [h['training_objective'] for h in history], label='Training')
    plt.plot([h['epoch'] for h in history], [h['validation_objective'] for h in history], label='Validation')
    plt.title('Multitask response and exceedance prediction')
    plt.xlabel('Epoch'); plt.ylabel('Training objective');plt.legend();plt.grid(alpha=.2)
    plt.tight_layout();plt.savefig(fig/'risk_surrogate_training.png', dpi=200);plt.close()
    fig2,axes=plt.subplots(1,3,figsize=(12,3.8))
    for j,name in enumerate(NAMES):
        bins=np.linspace(0,1,11)
        assigned=np.digitize(prob[:,j],bins)-1
        avg=np.array([truth_labels[assigned==i,j].mean() if np.any(assigned==i) else np.nan for i in range(10)])
        axes[j].plot((bins[1:]+bins[:-1])/2,avg,marker='o')
        axes[j].plot([0,1],[0,1],ls='--',color='gray')
        axes[j].set_title(name);axes[j].set_xlim(0,1);axes[j].set_ylim(0,1)
        axes[j].set_xlabel('Predicted exceedance probability')
    axes[0].set_ylabel('Observed exceedance rate')
    fig2.tight_layout();fig2.savefig(fig/'risk_probability_calibration.png',dpi=200);plt.close(fig2)
    print(json.dumps(report, indent=2),flush=True)


if __name__ == '__main__':
    main()
