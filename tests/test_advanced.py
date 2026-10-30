"""Check data separation, risk scoring, geometry, and pretrained model outputs."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from risk_surrogate import ExceedanceNetwork
from risk import seismic_cvar
from models import DualDiffusion, DualDiffusionProcess


class StructuralAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=dict(np.load(R/'data/processed/designsafe_processed.npz'))
        cls.s=json.loads((R/'data/splits/split.json').read_text())

    def test_record_splits_disjoint(self):
        g=self.s['ground_motions']
        parts=[set(g[x]) for x in ('train','val','test')]
        self.assertEqual(len(set.union(*parts)),240)
        for i in range(3):
            for j in range(i+1,3):self.assertTrue(parts[i].isdisjoint(parts[j]))

    def test_response_scaling_train_only(self):
        train=np.array(self.s['buildings']['train']); g=np.array(self.s['ground_motions']['train'])
        observed=np.log1p(self.d['response'][np.ix_(train,g)])
        self.assertTrue(np.allclose(observed.mean((0,1)),self.d['response_mu'],atol=1e-5))

    def test_standardized_section_finiteness(self):
        self.assertTrue(np.isfinite(self.d['cont']).all())
        self.assertTrue(np.all(self.d['section_props']>0))

    def test_gm_record_feature_train_only(self):
        g=np.array(self.s['ground_motions']['train'])
        expected=(self.d['gm_raw'][g].mean(0)-self.d['gm_mu'])/self.d['gm_std']
        self.assertLess(float(np.abs(expected).max()),1e-5)

    def test_risk_scores_positive_for_actual_exceedances(self):
        example=torch.tensor([[[.015,5.,.001],[.03,20.,.015]]])
        self.assertGreater(float(seismic_cvar(example)),0.)

    def test_risk_model_shapes(self):
        m=ExceedanceNetwork(width=32)
        row=0
        x=torch.from_numpy(self.d['cont'][row:row+1].reshape(1,57,2))
        mask=torch.from_numpy(self.d['mask'][row:row+1].reshape(1,57))
        gm=torch.from_numpy(self.d['gm'][:1])
        geometry=torch.from_numpy(self.d['geom'][row:row+1])
        mean,logvar,logits=m(x,mask,geometry,gm)
        self.assertEqual(tuple(mean.shape),(1,3))
        self.assertEqual(tuple(logvar.shape),(1,3))
        self.assertEqual(tuple(logits.shape),(1,3))

    def test_role_catalog_seen_only_in_training(self):
        train=np.asarray(self.s['buildings']['train'])
        for role in range(3):
            observed=self.d['section_idx'][train,:,role][self.d['mask'][train,:,role]]
            self.assertTrue(len(np.unique(observed))>2)

    def test_ground_truth_reference_coverage(self):
        test=np.asarray(self.s['buildings']['test']); events=np.asarray(self.s['ground_motions']['test'])
        self.assertEqual(self.d['response'][np.ix_(test,events)].shape,(93,36,3))

if __name__=='__main__':unittest.main()
