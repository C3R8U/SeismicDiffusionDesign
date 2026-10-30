"""Read the original DesignSafe research archive and prepare structural design tensors."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from preprocessing import process_archive
from engineering import catalog_arrays
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    audit=process_archive(args.archive,ROOT/'data/processed',seed=args.seed)
    design=np.load(ROOT/'data/processed/designsafe_processed.npz')
    attrs=catalog_arrays(ROOT/'data/catalog/section_properties_source.txt',design['section_names'])
    np.savez_compressed(ROOT/'data/processed/verified_section_properties.npz',section_names=design['section_names'],
                        properties=attrs,fields=np.array(['A_m2','depth_m','Ix_m4','Zx_m3','Iy_m4','Zy_m3']))
    audit.update({'matched_engineering_section_properties':len(attrs),'property_dimensions':6})
    (ROOT/'data/processed/advanced_data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
