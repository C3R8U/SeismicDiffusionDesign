"""Evaluate the saved extreme-response predictor on untouched building/record pairs."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from risk_surrogate import ExceedanceNetwork
from train_risk_surrogate import batch_pairs,infer,NAMES

def main():
    torch.set_num_threads(4)
    d=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
    split=json.loads((ROOT/'data/splits/split.json').read_text())
    test=batch_pairs(split['buildings']['test'],split['ground_motions']['test'])
    state=torch.load(ROOT/'checkpoints/risk_aware_surrogate.pt',map_location='cpu',weights_only=True)
    model=ExceedanceNetwork(state['width'])
    model.load_state_dict(state['model'])
    predicted,prob,labels,observed=infer(model,d,test)
    rows=[]
    for j,name in enumerate(NAMES):
        y=labels[:,j];p=prob[:,j];actual=observed[:,j];p_response=predicted[:,j]
        rows.append({'response':name,'positives':int(y.sum()),'test_pairs':len(y),
            'prevalence':float(y.mean()),
            'roc_auc':float(roc_auc_score(y,p)) if y.min()!=y.max() else None,
            'average_precision':float(average_precision_score(y,p)),
            'brier_score':float(brier_score_loss(y,p)),
            'recall_at_0_05':float(np.sum((p>=.05)&(y>.5))/max(y.sum(),1)),
            'prediction_rate_at_0_05':float((p>=.05).mean()),
            'mae':float(np.mean(np.abs(p_response-actual))),
            'r2':float(1-np.sum((p_response-actual)**2)/np.sum((actual-actual.mean())**2))})
    pd.DataFrame(rows).to_csv(ROOT/'results/risk_surrogate_test_metrics.csv',index=False)
    np.savez_compressed(ROOT/'results/risk_surrogate_test_predictions.npz',
         probability=prob,labels=labels,predicted=predicted,observed=observed,test_pairs=test)
    report={'checkpoint_epoch':int(state['epoch']),'test_pairs':len(test),'metrics':rows,
            'validation':'Measured heldout paired response; new generated designs have not undergone nonlinear analysis.'}
    (ROOT/'results/risk_surrogate_metrics.json').write_text(json.dumps(report,indent=2))
    fig, axes=plt.subplots(1,3,figsize=(12,3.8))
    for j,name in enumerate(NAMES):
        cuts=np.quantile(prob[:,j],np.linspace(0,1,11))
        cuts=np.unique(cuts)
        x=[];y=[];n=[]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            selected=(prob[:,j]>=lo)&(prob[:,j]<=hi)
            x.append(float(prob[selected,j].mean()));y.append(float(labels[selected,j].mean()));n.append(int(selected.sum()))
        axes[j].plot(x,y,'o-',label='Observed by risk decile')
        axes[j].plot([0,max(x)*1.2],[0,max(x)*1.2],ls='--',color='gray')
        axes[j].set_title(name);axes[j].set_xlabel('Estimated exceedance probability')
        axes[j].set_ylim(bottom=0);axes[j].grid(alpha=.2)
    axes[0].set_ylabel('Observed frequency')
    fig.tight_layout();fig.savefig(ROOT/'results/audit_figures/risk_probability_calibration.png',dpi=200);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
