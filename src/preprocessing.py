"""Load building configurations, response matrices, and seismic records."""
from __future__ import annotations
import io, json, re, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

MAX_STORIES = 19
ROLE_NAMES = ("exterior_column", "interior_column", "beam")
LBS_PER_FT_TO_KG_PER_M = 0.45359237 / 0.3048


def read_csv(z, root, name):
    return pd.read_csv(io.BytesIO(z.read(root + name)))


def process_archive(archive, destination, seed=42):
    """Keep structural designs separate from their repeated seismic response records."""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(archive)
    top = z.namelist()[0].split('/')[0]
    root = top + '/data/'
    summary = read_csv(z, root, 'DesignSummary.csv')
    ids = summary['building ID'].astype(int).to_numpy()
    count = len(ids)
    assert count == 621 and len(set(ids)) == count
    all_section_names = set()
    designs = []
    info = []
    for row in summary.itertuples(index=False, name=None):
        bid = int(row[0])
        size = read_csv(z, root, f'BuildingDesigns/Building_{bid}/ConstructionSize.csv')
        geometry = read_csv(z, root, f'BuildingDesigns/Building_{bid}/Geometry.csv').iloc[0]
        n = int(geometry['number of story'])
        assert len(size) == n and n <= MAX_STORIES
        names = size.iloc[:, :3].astype(str).to_numpy()
        assert all(re.fullmatch(r'W\d+X\d+(?:\.\d+)?', item) for item in names.ravel())
        all_section_names.update(names.ravel().tolist())
        designs.append(names)
        info.append([n, int(geometry['number of X bay']), float(geometry['X bay width']) * .3048,
                     float(geometry['first story height']) * .3048,
                     float(geometry['typical story height']) * .3048,
                     float(row[5]), float(row[6]), float(row[7])])
    section_names = sorted(all_section_names)
    lookup = {name: i for i, name in enumerate(section_names)}
    k = len(section_names)
    section_mass = np.array([float(name.split('X')[1]) * LBS_PER_FT_TO_KG_PER_M for name in section_names], dtype=np.float32)
    section_depth = np.array([float(re.match(r'W(\d+)', name).group(1)) * .0254 for name in section_names], dtype=np.float32)
    section_area = section_mass / 7850.0
    section_props = np.column_stack([section_area, section_depth]).astype('float32')
    sec = np.zeros((count, MAX_STORIES, 3), dtype=np.int64)
    mask = np.zeros((count, MAX_STORIES, 3), dtype=bool)
    for i, names in enumerate(designs):
        n = len(names)
        sec[i, :n] = np.vectorize(lookup.__getitem__)(names)
        mask[i, :n] = True
    response = np.full((count, 240, 3), np.nan, dtype=np.float32)
    response_invalid = np.zeros((count, 240, 3), dtype=bool)
    dirs = ('PeakStoryDrift', 'PeakFloorAcceleration', 'ResidualStoryDrift')
    suffixes = ('PSDR', 'PFA', 'RSDR')
    for i, bid in enumerate(ids):
        for j, (folder, suffix) in enumerate(zip(dirs, suffixes)):
            path = f'StructuralResponses/EDPsUnder240GMs/{folder}/Building_{bid}_{suffix}.csv'
            try:
                frame = read_csv(z, root, path)
            except KeyError:
                paths = [p for p in z.namelist() if p.startswith(root+f'StructuralResponses/EDPsUnder240GMs/{folder}/Building_{bid}_')]
                if len(paths) != 1: raise RuntimeError(f'Unknown response layout for {bid}, {folder}')
                frame = pd.read_csv(io.BytesIO(z.read(paths[0])))
            values = frame.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
            assert values.shape[1] == 240, (bid, folder, values.shape)
            response[i,:,j] = np.nanmax(values, axis=0)
            response_invalid[i,:,j] = ~np.isfinite(values).all(axis=0)
    # Floor accelerations in the published response tables use inches per second squared.
    response[:,:,1] *= .0254
    gm_root = root + 'GroundMotions/MirandaGroundMotions/'
    steps = np.loadtxt(io.BytesIO(z.read(gm_root+'GroundMotionInfo/GMtimeSteps.txt')))
    filenames = z.read(gm_root+'GroundMotionInfo/GMFileNames.txt').decode().splitlines()
    assert len(steps) == len(filenames) == 240
    gm_features = []
    for j, name in enumerate(filenames):
        acc = np.loadtxt(io.BytesIO(z.read(gm_root+'Histories/'+name)), dtype=np.float32).reshape(-1)
        # Recorded Miranda accelerograms are expressed in g.
        pga = float(np.max(np.abs(acc))) * 9.80665
        rms = float(np.sqrt(np.mean(acc*acc))) * 9.80665
        significant = float((np.abs(acc) > np.max(np.abs(acc)) * .05).sum()) * float(steps[j])
        duration = float(len(acc) * steps[j])
        gm_features.append([np.log1p(pga),np.log1p(rms),np.log1p(significant),np.log1p(duration)])
    gm_features=np.asarray(gm_features, dtype=np.float32)
    # Stratify by geometry while keeping the original building the unit of splitting.
    rng = np.random.default_rng(seed)
    strata = {}
    for i, v in enumerate(info): strata.setdefault((v[0],v[1]), []).append(i)
    training, validation, testing = [], [], []
    for group in strata.values():
        group = rng.permutation(group)
        n=len(group); a=int(round(.70*n)); b=int(round(.15*n))
        training.extend(group[:a]);validation.extend(group[a:a+b]);testing.extend(group[a+b:])
    # Correct rounding without assigning any building to more than one set.
    groups = [training, validation, testing]
    targets = [435,93,93]
    while [len(x) for x in groups] != targets:
        surplus = next(i for i in range(3) if len(groups[i]) > targets[i])
        shortage = next(i for i in range(3) if len(groups[i]) < targets[i])
        groups[shortage].append(groups[surplus].pop())
    gm_ids = rng.permutation(240)
    gm_splits = {'train':gm_ids[:168].tolist(), 'val':gm_ids[168:204].tolist(), 'test':gm_ids[204:].tolist()}
    # All response-based scaling and performance labels below use training ground motions only.
    tr = groups[0]
    cont = np.log(section_props[sec].clip(min=1e-8))
    mu = cont[tr][mask[tr]].mean(axis=0)
    std = cont[tr][mask[tr]].std(axis=0).clip(min=1e-5)
    cont = (cont-mu)/std
    geom=np.asarray(info,dtype=np.float32)
    geom_mu=geom[tr].mean(axis=0);geom_std=geom[tr].std(axis=0).clip(min=1e-5)
    geom_norm=(geom-geom_mu)/geom_std
    gm_mu=gm_features[gm_splits['train']].mean(axis=0)
    gm_std=gm_features[gm_splits['train']].std(axis=0).clip(min=1e-5)
    gm_norm=(gm_features-gm_mu)/gm_std
    # All performance labels are training-event summaries; never take test-event quantiles.
    perf=np.nanquantile(response[:,gm_splits['train'],:],.9,axis=1).astype(np.float32)
    response_log=np.log1p(np.clip(response,0,None))
    response_mu=np.nanmean(response_log[np.ix_(tr,gm_splits['train'])],axis=(0,1))
    response_std=np.nanstd(response_log[np.ix_(tr,gm_splits['train'])],axis=(0,1)).clip(min=1e-5)
    response_norm=(response_log-response_mu)/response_std
    perf_log=np.log1p(np.clip(perf,0,None))
    perf_mu=perf_log[tr].mean(axis=0);perf_std=perf_log[tr].std(axis=0).clip(min=1e-5)
    perf_norm=(perf_log-perf_mu)/perf_std
    prior=np.zeros((3,k),np.float32)
    observed_roles=np.ones((3,k),dtype=bool)
    for role in range(3):
        labels=sec[tr,:,role][mask[tr,:,role]]
        for label in labels: prior[role,label]+=1
        # Any catalog W-shape can be proposed; engineering checks handle role suitability.
        prior[role]+=.001
        prior[role]/=prior[role].sum()
    np.savez_compressed(destination/'designsafe_processed.npz', ids=ids, section_idx=sec, mask=mask, cont=cont.astype('float32'), geom=geom_norm,
                        geom_raw=geom, perf=perf_norm, perf_raw=perf, response=response, response_norm=response_norm,
                        response_invalid=response_invalid, gm=gm_norm, gm_raw=gm_features,
                        section_props=section_props, section_names=np.array(section_names), prior=prior, observed_roles=observed_roles,
                        cont_mu=mu,cont_std=std, geom_mu=geom_mu,geom_std=geom_std,gm_mu=gm_mu,gm_std=gm_std,
                        perf_mu=perf_mu,perf_std=perf_std,response_mu=response_mu,response_std=response_std)
    split={'buildings':{key: np.array(value,dtype=int).tolist() for key,value in zip(('train','val','test'),groups)},
           'ground_motions':gm_splits,'seed':seed,
           'ground_motion_event_split':'record-level only; earthquake-event IDs are absent from the supplied metadata'}
    (destination.parent/'splits').mkdir(exist_ok=True)
    (destination.parent/'splits'/'split.json').write_text(json.dumps(split,indent=2))
    missing=int((~np.isfinite(response)).sum());invalid=int(response_invalid.sum())
    audit={'archive_name':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
           'buildings':int(count),'ground_motions':240,'response_pairs':int(count*240),
           'section_labels':k,'training_buildings':len(training),'validation_buildings':len(validation),
           'test_buildings':len(testing),'nonfinite_peak_values':missing,'invalid_full_response_columns':invalid,
           'response_unit_conversions':{'peak_floor_acceleration':'in/s^2 to m/s^2, factor 0.0254',
                                         'miranda_ground_acceleration':'g to m/s^2, factor 9.80665'},
           'section_property_limitation':'Catalog lacks verified I and Z; uses steel-weight-derived area and nominal depth only.',
           'ground_motion_event_ids':'unavailable in archive metadata'}
    (destination/'data_audit.json').write_text(json.dumps(audit,indent=2))
    pd.DataFrame({'building_id':ids, 'num_stories':geom[:,0].astype(int),'num_bays':geom[:,1].astype(int),'span_m':geom[:,2],
                  'p90_drift':perf[:,0],'p90_floor_acc_mps2':perf[:,1],'p90_residual_drift':perf[:,2]}).to_csv(destination/'design_summary.csv',index=False)
    return audit
