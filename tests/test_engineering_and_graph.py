"""Checks for physical topology, section dimensions, and coupled model gradients."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hierarchy import physical_topology,TopologyBank
from engineering import catalog_arrays,scwb_screen,steel_mass,repair_scwb
from advanced_models import HierarchicalDualDiT,PairedDiffusion

class EngineeringGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=dict(np.load(ROOT/'data/processed/designsafe_processed.npz'))
        cls.props=catalog_arrays(ROOT/'data/catalog/section_properties_source.txt',cls.data['section_names'])

    def test_frame_member_counts(self):
        for story,bay in [(1,1),(5,3),(19,5)]:
            t=physical_topology(story,bay)
            self.assertEqual(int(t['member_mask'].sum()),story*(2*bay+1))
            self.assertEqual(int(t['node_mask'].sum()),(story+1)*(bay+1))

    def test_all_design_shapes_in_catalog(self):
        self.assertEqual(self.props.shape,(130,6))
        self.assertTrue(np.all(np.isfinite(self.props)))
        self.assertTrue(np.all(self.props>0))
        self.assertAlmostEqual(self.props[0,2]/(.0254**4),933,places=4)
        self.assertAlmostEqual(self.props[0,3]/(.0254**3),164,places=4)

    def test_member_mass_and_joint_screen(self):
        for i in [0,120,400,620]:
            geometry=self.data['geom_raw'][i]
            sec=self.data['section_idx'][i]
            mass=steel_mass(sec,self.props,geometry)
            status=scwb_screen(sec,self.props,geometry)
            self.assertGreater(mass,0)
            self.assertTrue(np.isfinite(status['min_ratio']))
            self.assertLessEqual(status['fraction_over_1'],1)

    def test_graph_differentiates_sections(self):
        data=self.data
        bank=TopologyBank(data['geom_raw'])
        mask=torch.from_numpy(data['mask'][:2].reshape(2,57))
        top=bank.take([0,1],'cpu')
        network=HierarchicalDualDiT(130,width=64,depth=2,heads=4,graph_layers=2)
        sec=torch.from_numpy(data['section_idx'][:2].reshape(2,57))
        x=torch.randn(2,57,6)
        y,p=network(x,sec,torch.tensor([3,5]),torch.from_numpy(data['geom'][:2]),torch.from_numpy(data['perf'][:2]),mask,top)
        self.assertEqual(y.shape,(2,57,6))
        self.assertEqual(p.shape,(2,57,130))
        (y.square().mean()+p.square().mean()).backward()
        self.assertTrue(network.properties.weight.grad.abs().sum()>0)

if __name__=='__main__':unittest.main()
