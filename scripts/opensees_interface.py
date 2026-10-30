"""Stage existing structural models for OpenSees and report solver availability."""
import argparse,json,shutil,subprocess,zipfile
from pathlib import Path

def stage_model(archive,building_id,destination,analysis='DynamicAnalysis'):
    """Leave the published model intact and add the shared solver and material scripts."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        root=z.namelist()[0].split('/')[0]+'/data/NonlinearStructuralModels/'
        candidates=(root+f'Building_{building_id}/{analysis}/',root+f'BaselineFiles/{analysis}/')
        count=0
        for member in z.namelist():
            if member.endswith('/') or not member.lower().endswith('.tcl'):continue
            for prefix in candidates:
                if member.startswith(prefix):
                    (destination/Path(member).name).write_bytes(z.read(member));count+=1;break
    if not (destination/'Model.tcl').exists():raise FileNotFoundError('Structural model entry point is missing')
    return count

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',required=True);ap.add_argument('--building-id',type=int,default=0)
    ap.add_argument('--destination',default='results/opensees_staging/building_0')
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();n=stage_model(args.archive,args.building_id,args.destination)
    executable=shutil.which('OpenSees')
    result={'building_id':args.building_id,'scripts_staged':n,'entry_point':'Model.tcl',
            'openSees_executable_found':bool(executable),'analysis_executed':False,
            'notes':'The published model needs its earthquake input, time-step parameters, and ground-motion files. Newly generated sections need a separate validated model-building pipeline.'}
    if args.execute:
        if not executable:raise RuntimeError('OpenSees is not available on PATH')
        raise RuntimeError('Set earthquake inputs and verify the published solver variables before dynamic analysis')
    path=Path(args.destination)/'stage_report.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
