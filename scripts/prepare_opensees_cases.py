"""Prepare original published OpenSees dynamic cases with explicit ground-motion inputs."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def stage(archive,building_id,record_index,destination,execute=False):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        data_root=z.namelist()[0].split('/')[0]+'/data/'
        original=data_root+f'NonlinearStructuralModels/Building_{building_id}/DynamicAnalysis/'
        baseline=data_root+'NonlinearStructuralModels/BaselineFiles/DynamicAnalysis/'
        members=[m for m in z.namelist() if m.startswith(baseline) and m.endswith('.tcl')]
        overrides=[m for m in z.namelist() if m.startswith(original) and m.endswith('.tcl')]
        if not overrides:raise ValueError('No nonlinear model for this building ID')
        for item in members+overrides:(destination/Path(item).name).write_bytes(z.read(item))
        gm_root=data_root+'GroundMotions/MirandaGroundMotions/'
        filenames=z.read(gm_root+'GroundMotionInfo/GMFileNames.txt').decode().splitlines()
        dts=np.loadtxt(z.open(gm_root+'GroundMotionInfo/GMtimeSteps.txt'))
        if record_index<0 or record_index>=len(filenames):raise ValueError('Unknown ground-motion record')
        ground_motion=filenames[record_index]
        paths=destination/'Histories';paths.mkdir(exist_ok=True)
        acceleration=z.read(gm_root+'Histories/'+ground_motion)
        (paths/ground_motion).write_bytes(acceleration)
        duration_count=len(np.fromstring(acceleration.decode(),sep=' '))
        if duration_count==0:raise RuntimeError('Empty acceleration time history')
    tcl=f'''# Structural time-history case using the source project's nonlinear member models.
set baseDir [pwd]
set rootDir [pwd]
set dataDir Output
set dt {float(dts[record_index]):.12g}
set numPoints {duration_count}
set eqNumber 1
set eqFileName [list {{{ground_motion}}}]
set scale 100
set MCEScaleFactor 1.0
source Model.tcl
'''
    (destination/'run_dynamic_case.tcl').write_text(tcl)
    command=['OpenSees','run_dynamic_case.tcl']
    solver=shutil.which('OpenSees')
    report={'building_id':int(building_id),'record_index_zero_based':record_index,'record_file':ground_motion,
            'time_step_s':float(dts[record_index]),'points':duration_count,'source_tcl_count':len(members)+len(overrides),
            'staged_tcl_count':len(list(destination.glob('*.tcl'))),'acceleration_sha256':hashlib.sha256(acceleration).hexdigest(),
            'openSees_available':bool(solver),'analysis_executed':False,
            'member_properties_changed':False,
            'model_type':'published reference model, not a newly generated design',
            'run_command':'OpenSees run_dynamic_case.tcl'}
    if execute:
        if not solver:report['error']='OpenSees is not installed in the current runtime'
        else:
            run=subprocess.run(command,cwd=destination,capture_output=True,text=True,timeout=7200)
            (destination/'solver.stdout.log').write_text(run.stdout)
            (destination/'solver.stderr.log').write_text(run.stderr)
            report.update({'analysis_executed':True,'returncode':run.returncode,'success':run.returncode==0 and 'Analysis Completed' in run.stdout})
    (destination/'analysis_status.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=Path('/mnt/data/DesignSafe.zip'))
    parser.add_argument('--buildings',nargs='+',type=int,default=[0,756]);parser.add_argument('--record-index',type=int,default=0)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    reports=[]
    for bid in args.buildings:
        output=ROOT/'results/advanced/openSees_cases'/f'building_{bid}_record_{args.record_index}'
        reports.append(stage(args.archive,bid,args.record_index,output,execute=args.execute))
    path=ROOT/'results/advanced/openSees_cases/summary.json';path.write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
