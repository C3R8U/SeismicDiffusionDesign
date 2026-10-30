"""Visualize the W-shape selections for reference, raw, and joint-screened frames."""
from __future__ import annotations
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    archive=np.load(ROOT/'data/processed/designsafe_processed.npz')
    names=archive['section_names']
    out=ROOT/'results/advanced/figures';out.mkdir(exist_ok=True,parents=True)
    for src in sorted((ROOT/'results/advanced').glob('generated_seed_*.json')):
        designs=json.loads(src.read_text())
        for d in designs:
            entry=np.where(archive['ids']==d['building_id'])[0][0]
            n=d['story_count']
            reference=archive['section_idx'][entry,:n]
            before=np.asarray(d['sections_raw']);after=np.asarray(d['sections_repaired'])
            fig,axes=plt.subplots(1,3,figsize=(13,max(4,min(12,n*.34+2))),sharey=True)
            for ax,labels,title in zip(axes,[reference,before,after],['Published reference','Diffusion candidate','Joint-screen adjustment']):
                image=np.log1p(archive['section_props'][labels,0])*1e3
                ax.imshow(image,aspect='auto',cmap='viridis',interpolation='nearest',origin='lower')
                for floor in range(n):
                    for role in range(3):
                        ax.text(role,floor,('N/A' if d['bays']==1 and role==1 else names[labels[floor,role]].replace('W','')),ha='center',va='center',fontsize=6,color='white')
                ax.set_xticks([0,1,2],['Exterior','Interior','Beam'],rotation=30)
                ax.set_title(title)
            axes[0].set_yticks(range(n),[f'Level {i+1}' for i in range(n)])
            fig.suptitle(f"W-shape selections | Seed {d['seed']} | Building {d['building_id']}")
            fig.tight_layout()
            tag=f"w_section_layout_seed_{d['seed']}_building_{d['building_id']}"
            fig.savefig(out/(tag+'.png'),dpi=230,bbox_inches='tight')
            fig.savefig(out/(tag+'.pdf'),bbox_inches='tight')
            plt.close(fig)
if __name__=='__main__':main()
