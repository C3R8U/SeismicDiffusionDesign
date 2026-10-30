"""Read available site-specific seismic records by explicit column identifier."""
from __future__ import annotations
import argparse,io,zipfile,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LEVELS=('SLE','DBE','MCE')
VARIABLES=(('PeakStoryDrift','PSDR',1.),('PeakFloorAcceleration','PFA',.0254),('ResidualStoryDrift','RSDR',1.))
LIMITS=np.array([.02,14.71,.01])


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',default='/mnt/data/DesignSafe.zip')
    args=p.parse_args();archive=zipfile.ZipFile(args.archive)
    prefix=archive.namelist()[0].split('/')[0]+'/data/'
    subgroup=pd.read_csv(io.BytesIO(archive.read(prefix+'SubgroupBuildingIDs.csv'))).iloc[:,0].astype(int).tolist()
    rows=[];widths=[];missing=[]
    for bid in subgroup:
        for level in LEVELS:
            observations=[]
            for folder,suffix,factor in VARIABLES:
                path=f'{prefix}StructuralResponses/EDPsUnderSiteSpecificGMs/{level}/{folder}/Building_{bid}_{suffix}.csv'
                try: frame=pd.read_csv(io.BytesIO(archive.read(path)))
                except KeyError:
                    missing.append({'building_id':bid,'level':level,'quantity':folder})
                    observations.append(None);continue
                width=len(frame.columns)
                widths.append({'building_id':bid,'level':level,'quantity':folder,'record_columns':width})
                frame=frame.apply(pd.to_numeric,errors='coerce')*factor
                max_response=frame.max(axis=0)
                observations.append(max_response)
            if any(x is None for x in observations):continue
            matched=set(observations[0].index)
            for array in observations[1:]:matched &= set(array.index)
            for gm in sorted(matched, key=lambda x:int(x.replace('GM',''))):
                values=[float(arr.loc[gm]) for arr in observations]
                if all(np.isfinite(values)):
                    rows.append({'building_id':bid,'hazard_level':level,'record':gm,
                       'peak_drift':values[0],'peak_floor_acc_mps2':values[1],
                       'residual_drift':values[2],
                       'passes_all_study_thresholds':bool(np.all(np.array(values)<=LIMITS))})
    out=ROOT/'results'
    frame=pd.DataFrame(rows);size=pd.DataFrame(widths)
    frame.to_csv(out/'site_specific_responses_aligned.csv',index=False)
    size.to_csv(out/'site_specific_record_counts.csv',index=False)
    summary=frame.groupby(['building_id','hazard_level']).agg(
        aligned_records=('record','count'),
        fraction_passing=('passes_all_study_thresholds','mean'),
        max_peak_drift=('peak_drift','max'),
        max_peak_floor_acc_mps2=('peak_floor_acc_mps2','max'),
        max_residual_drift=('residual_drift','max')).reset_index()
    summary.to_csv(out/'site_specific_structure_summary.csv',index=False)
    counts=size.groupby(['level','quantity','record_columns']).size().reset_index(name='files')
    counts.to_csv(out/'site_specific_column_widths.csv',index=False)
    report={'subgroup_buildings':len(subgroup),'files_found':len(widths),
       'files_missing':len(missing),'aligned_building_record_level_triplets':len(frame),
       'level_aligned_records':{k:int(len(frame[frame.hazard_level==k])) for k in LEVELS},
       'raw_column_widths':counts.to_dict('records'),
       'mean_reference_fraction_passing_per_level':summary.groupby('hazard_level')['fraction_passing'].mean().to_dict(),
       'interpretation':'Published reference structures, not generated structures. Study thresholds are not universal design-code limits.'}
    (out/'site_specific_data_report.json').write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(8,4.8))
    values=[summary[summary.hazard_level==level].fraction_passing.to_numpy() for level in LEVELS]
    ax.boxplot(values,tick_labels=LEVELS)
    ax.set_ylabel('Fraction of records below all study thresholds')
    ax.set_title('Published reference structures across seismic hazard levels')
    ax.grid(alpha=.2,axis='y')
    fig.tight_layout();fig.savefig(out/'audit_figures/site_specific_reference_performance.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for level in LEVELS:
        vals=frame[frame.hazard_level==level]['peak_drift']
        ax.hist(vals,bins=np.linspace(0,.1,42),alpha=.45,label=level,density=True)
    ax.set_xlabel('Maximum interstory drift ratio')
    ax.set_ylabel('Density')
    ax.set_title('Published seismic demand by ground-motion category')
    ax.legend();fig.tight_layout()
    fig.savefig(out/'audit_figures/site_specific_peak_drift.png',dpi=200);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='raw_column_widths'},indent=2),flush=True)

if __name__=='__main__':main()
