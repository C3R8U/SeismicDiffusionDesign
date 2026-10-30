"""Read engineering section properties and construct metric-unit design features."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CATALOG_URL='https://raw.githubusercontent.com/GUAN-XINGQUAN/AutoSDAPlatform/master/AllSectionDatabase.csv'


def load_catalog(csv_path):
    df=pd.read_csv(csv_path)
    required=('section size','A','d','Ix','Zx','Iy','Zy')
    missing=set(required)-set(df.columns)
    if missing:raise ValueError(f'Missing section columns: {sorted(missing)}')
    if df['section size'].duplicated().any():raise ValueError('Duplicate section identifiers')
    # AllSectionDatabase expresses areas in in2, lengths in inches, inertias in in4, and moduli in in3.
    convert={'A':.0254**2,'d':.0254,'Ix':.0254**4,'Zx':.0254**3,
             'Iy':.0254**4,'Zy':.0254**3}
    for key,factor in convert.items():df[key+'_SI']=pd.to_numeric(df[key],errors='raise')*factor
    return df


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--catalog',type=Path)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    catalog=args.catalog or ROOT/'data/AllSectionDatabase.csv'
    if args.download:
        catalog.parent.mkdir(exist_ok=True,parents=True)
        # A network connection is needed only when the source file is not already present.
        with urlopen(CATALOG_URL,timeout=30) as response:
            catalog.write_bytes(response.read())
    if not catalog.exists():
        raise FileNotFoundError('Supply --catalog or enable --download to obtain the published section database')
    df=load_catalog(catalog)
    observed=np.load(ROOT/'data/processed/designsafe_processed.npz')['section_names']
    catalog_rows=df.set_index('section size')
    unresolved=sorted(set(observed)-set(catalog_rows.index))
    if unresolved:raise ValueError(f'Sections absent from the catalog: {unresolved}')
    attributes=('A_SI','d_SI','Ix_SI','Zx_SI','Iy_SI','Zy_SI')
    values=catalog_rows.loc[observed,list(attributes)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.all(values>0):raise ValueError('Nonphysical catalog properties')
    np.savez_compressed(ROOT/'data/processed/verified_section_properties.npz',
                        section_names=observed, properties=values, field_names=np.array(attributes))
    result={'input_file':str(catalog),'sha256':hashlib.sha256(catalog.read_bytes()).hexdigest(),
            'catalog_rows':len(df),'matched_design_sections':len(observed),
            'properties':attributes,'unit_system':'SI',
            'source':CATALOG_URL}
    (ROOT/'data/processed/section_catalog_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
