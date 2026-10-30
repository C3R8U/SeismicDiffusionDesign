"""Fast checks for graph layout, design masks, and probability outputs."""
import json, sys,unittest
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'src'))
from models import DualDiffusion,DualDiffusionProcess,make_graph
from risk import seismic_cvar

class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=dict(np.load(P/'data/processed/designsafe_processed.npz'))
        cls.s=json.loads((P/'data/splits/split.json').read_text())

    def test_unique_splits(self):
        parts=[set(self.s['buildings'][x]) for x in ('train','val','test')]
        self.assertEqual(len(set().union(*parts)),621)
        self.assertEqual(len(set().intersection(parts[0],parts[1])),0)
        self.assertEqual(len(set().intersection(parts[0],parts[2])),0)
        self.assertEqual(len(set().intersection(parts[1],parts[2])),0)

    def test_physical_values(self):
        d=self.d
        self.assertEqual(d['response'].shape,(621,240,3))
        self.assertEqual(d['section_props'].shape,(130,2))
        self.assertTrue(np.isfinite(d['response']).all())
        self.assertTrue(np.all(d['section_props']>0))

    def test_graph_connections(self):
        adjacency=make_graph()
        self.assertEqual(tuple(adjacency.shape),(57,57))
        self.assertTrue(torch.equal(adjacency,adjacency.T))
        self.assertEqual(adjacency[0,1],1)
        self.assertEqual(adjacency[0,3],1)

    def test_noisy_model_shapes(self):
        data=self.d
        network=DualDiffusion(130,dim=32,layers=1,heads=4,prior=data['prior'],allowed=data['observed_roles'])
        diffusion=DualDiffusionProcess(network,steps=50)
        diffusion.set_catalog((np.log(data['section_props'])-data['cont_mu'])/data['cont_std'])
        ix=0
        design=torch.from_numpy(data['cont'][[ix]].reshape(1,57,2))
        sections=torch.from_numpy(data['section_idx'][[ix]].reshape(1,57))
        mask=torch.from_numpy(data['mask'][[ix]].reshape(1,57))
        geom=torch.from_numpy(data['geom'][[ix]])
        perf=torch.from_numpy(data['perf'][[ix]])
        loss,_=diffusion.objective(design,sections,geom,perf,mask)
        self.assertTrue(torch.isfinite(loss).item())

    def test_cvar_loss(self):
        samples=torch.tensor([[[.01,5.,.002],[.04,9.,.004],[.06,20.,.02]]])
        risk=seismic_cvar(samples,alpha=.9)
        self.assertGreater(risk.item(),0)

if __name__=='__main__':unittest.main()
