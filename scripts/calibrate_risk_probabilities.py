"""Calibrate rare-event probabilities using a validation-only logistic map."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,average_precision_score,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from risk_surrogate import ExceedanceNetwork
from train_risk_surrogate import batch_pairs,infer,NAMES

def main():
    torch.set_num_threads(4)
    d=dict(np.load(R/'data/processed/designsafe_processed.npz'))
    splits=json.loads((R/'data/splits/split.json').read_text())
    ck=torch.load(R/'checkpoints/risk_aware_surrogate.pt',map_location='cpu',weights_only=True)
    model=ExceedanceNetwork(ck['width']);model.load_state_dict(ck['model']);model.eval()
    validation=batch_pairs(splits['buildings']['val'],splits['ground_motions']['val'])
    test=batch_pairs(splits['buildings']['test'],splits['ground_motions']['test'])
    _,pv,yv,_=infer(model,d,validation)
    _,pt,yt,_=infer(model,d,test)
    calibrated=np.zeros_like(pt)
    report=[];coefficients=[]
    for j,name in enumerate(NAMES):
        xval=logit(np.clip(pv[:,j],1e-6,1-1e-6)).reshape(-1,1)
        x_test=logit(np.clip(pt[:,j],1e-6,1-1e-6)).reshape(-1,1)
        mapper=LogisticRegression(solver='lbfgs',C=1,max_iter=200,random_state=42)
        mapper.fit(xval,yv[:,j])
        calibrated[:,j]=mapper.predict_proba(x_test)[:,1]
        coefficients.append({'response':name,'slope':float(mapper.coef_[0,0]),
                'intercept':float(mapper.intercept_[0]),'validation_positives':int(yv[:,j].sum())})
        report.append({'response':name,'heldout_positives':int(yt[:,j].sum()),
           'brier_before':float(brier_score_loss(yt[:,j],pt[:,j])),
           'brier_after':float(brier_score_loss(yt[:,j],calibrated[:,j])),
           'logloss_before':float(log_loss(yt[:,j],pt[:,j])),
           'logloss_after':float(log_loss(yt[:,j],calibrated[:,j])),
           'average_precision':float(average_precision_score(yt[:,j],calibrated[:,j])),
           'roc_auc':float(roc_auc_score(yt[:,j],calibrated[:,j]))})
    np.savez_compressed(R/'results/calibrated_risk_predictions.npz',
            test_pairs=test,calibrated_probability=calibrated,uncalibrated_probability=pt,observed_exceedance=yt)
    pd.DataFrame(report).to_csv(R/'results/risk_calibration_metrics.csv',index=False)
    pd.DataFrame(coefficients).to_csv(R/'results/risk_calibration_coefficients.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(12.2,4.2))
    for j,name in enumerate(NAMES):
        y=yt[:,j]
        for name2,pred in [('Raw',pt[:,j]),('Calibrated',calibrated[:,j])]:
            edges=np.unique(np.quantile(pred,np.linspace(0,1,11)))
            bins=np.digitize(pred,edges[1:-1],right=True)
            x=[];rate=[]
            for b in range(len(edges)-1):
                idx=bins==b
                if idx.any():x.append(float(pred[idx].mean()));rate.append(float(y[idx].mean()))
            axs[j].plot(x,rate,marker='o',label=name2)
        axs[j].set_title(name);axs[j].set_xlabel('Average predicted probability')
        axs[j].grid(alpha=.2)
    axs[0].set_ylabel('Observed event fraction')
    axs[0].legend()
    fig.tight_layout();fig.savefig(R/'results/audit_figures/calibrated_risk_reliability.png',dpi=200);plt.close(fig)
    results={'fit_data':'validation structures and validation ground-motion records only',
             'evaluation_data':'heldout test structures and heldout test records',
             'coefficients':coefficients,'metrics':report,
             'scope':'Calibrated probability is an empirical classification score on the reference design distribution; it is not validated on generated structures.'}
    (R/'results/risk_calibration_report.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
