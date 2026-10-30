"""Independent checks of response predictions, structural sections, and design screening.

The measurements in this file come from recorded responses or frozen trained
networks. Only the published response files support measured seismic outcomes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from models import SeismicSurrogate

LIMITS = np.array([0.02, 14.71, 0.01], dtype=np.float64)
NAMES = ('peak_drift', 'floor_acceleration', 'residual_drift')
LABELS = ('Peak drift ratio', 'Floor acceleration (m/s2)', 'Residual drift ratio')


def get_data():
    arrays = dict(np.load(PROJECT / 'data/processed/designsafe_processed.npz'))
    splits = json.loads((PROJECT / 'data/splits/split.json').read_text())
    return arrays, splits


def build_predictor():
    stored = torch.load(PROJECT / 'checkpoints/surrogate_0.pt', map_location='cpu', weights_only=True)
    predictor = SeismicSurrogate(width=int(stored['width']))
    predictor.load_state_dict(stored['model'])
    predictor.eval()
    return predictor


def predict_pairs(predictor, data, pairs, batch=2048):
    """Preserve pair order so observed responses remain aligned with predictions."""
    outputs, standard_deviations = [], []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch):
            selected = pairs[start:start + batch]
            buildings, records = selected[:, 0], selected[:, 1]
            means, logvars = predictor(
                torch.from_numpy(data['cont'][buildings].reshape(-1, 57, 2)),
                torch.from_numpy(data['mask'][buildings].reshape(-1, 57)),
                torch.from_numpy(data['geom'][buildings]),
                torch.from_numpy(data['gm'][records]),
            )
            means = means.numpy() * data['response_std'] + data['response_mu']
            sigmas = np.exp(0.5 * logvars.numpy()) * data['response_std']
            outputs.append(np.expm1(means).clip(0))
            standard_deviations.append(sigmas)
    return np.concatenate(outputs), np.concatenate(standard_deviations)


def make_pairs(buildings, records):
    return np.stack(np.meshgrid(buildings, records, indexing='ij'), -1).reshape(-1, 2).astype('int64')


def compute_metrics(truth, pred):
    result = {}
    for j, name in enumerate(NAMES):
        target, estimate = truth[:, j], pred[:, j]
        errors = estimate - target
        variance = np.sum((target - target.mean()) ** 2)
        result[name] = {
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'median_absolute_error': float(np.median(np.abs(errors))),
            'bias': float(errors.mean()),
            'r2': float(1 - np.sum(errors ** 2) / variance) if variance > 0 else None,
        }
    return result


def empirical_cvar(y, alpha=.9):
    """Average the worst earthquake-record losses for each structural design."""
    exceedance = np.maximum(y / LIMITS - 1, 0) ** 2
    loss = exceedance.mean(axis=-1)
    count = max(1, int(np.ceil(loss.shape[-1] * (1 - alpha))))
    return np.partition(loss, -count, axis=-1)[..., -count:].mean(-1)


def boxplot(data, filename, ylabel, title):
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(data, tick_labels=list(data), showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=.2, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def inspect_designs(data, splits):
    result = {}
    labels = data['section_idx']; mask = data['mask']
    for role, name in enumerate(('exterior_column', 'interior_column', 'beam')):
        train = np.asarray(splits['buildings']['train'], dtype=int)
        known = np.unique(labels[train, :, role][mask[train, :, role]])
        result[name] = {}
        for partition in ('val', 'test'):
            buildings = np.asarray(splits['buildings'][partition], dtype=int)
            observed = labels[buildings, :, role][mask[buildings, :, role]]
            result[name][partition] = {
                'catalog_categories_unseen_in_training_role': int(np.sum(~np.isin(np.unique(observed), known))),
                'fraction_of_tokens_unseen_in_training_role': float(np.mean(~np.isin(observed, known))),
            }
    return result


def bootstrap_mae(paired_true, paired_pred, num_buildings, num_gms, seed=42, draws=500):
    """Resample buildings, not repeated response pairs."""
    rng = np.random.default_rng(seed)
    error = np.abs(paired_true - paired_pred).reshape(num_buildings, num_gms, 3).mean(axis=1)
    choice = rng.integers(0, num_buildings, (draws, num_buildings))
    values = error[choice].mean(axis=1)
    low, high = np.quantile(values, [.025, .975], axis=0)
    return {key: {'lower': float(low[j]), 'upper': float(high[j])} for j, key in enumerate(NAMES)}


def response_baselines(data, split, predictor, output):
    building_sets = split['buildings']; record_sets = split['ground_motions']
    scenarios = {
        'seen_buildings_unseen_records': ('train', 'test'),
        'unseen_buildings_seen_records': ('test', 'train'),
        'unseen_buildings_unseen_records': ('test', 'test'),
    }
    summary = {}; scenario_predictions = {}
    for name, (bgroup, ggroup) in scenarios.items():
        b = np.asarray(building_sets[bgroup], dtype=int)
        gm = np.asarray(record_sets[ggroup], dtype=int)
        pairs = make_pairs(b, gm)
        predictions, sigma = predict_pairs(predictor, data, pairs)
        ground_truth = data['response'][pairs[:, 0], pairs[:, 1]]
        metrics = compute_metrics(ground_truth, predictions)
        summary[name] = {'pairs': len(pairs), 'buildings': len(b), 'ground_motions': len(gm), 'metrics': metrics}
        if name == 'unseen_buildings_unseen_records':
            summary[name]['building_bootstrap_mae_95_percent_ci'] = bootstrap_mae(
                ground_truth, predictions, len(b), len(gm))
            scenario_predictions = {'pairs': pairs, 'observed': ground_truth, 'predicted': predictions, 'sigma_log': sigma}
        print(name, [(n, round(metrics[n]['r2'], 3)) for n in NAMES], flush=True)
    np.savez_compressed(output / 'all_heldout_predictions.npz', **scenario_predictions)
    return summary, scenario_predictions


def actual_reference_risks(data, split, heldout, output):
    buildings = np.asarray(split['buildings']['test'], dtype=int)
    records = np.asarray(split['ground_motions']['test'], dtype=int)
    observed = data['response'][np.ix_(buildings, records)]
    estimates = heldout['predicted'].reshape(len(buildings), len(records), 3)
    true_risk = empirical_cvar(observed)
    predicted_risk = empirical_cvar(estimates)
    observed_pass = np.all(observed <= LIMITS, axis=-1)
    pass_fraction = observed_pass.mean(-1)
    true_positive = true_risk > 0
    predicted_positive = predicted_risk > 0
    comparison = pd.DataFrame({
        'building_id': data['ids'][buildings],
        'stories': data['geom_raw'][buildings, 0].astype(int),
        'observed_cvar': true_risk,
        'surrogate_cvar': predicted_risk,
        'observed_fraction_of_records_passing_all_thresholds': pass_fraction,
        'observed_meets_90_percent_record_rule': pass_fraction >= .90,
        'observed_any_exceedance': true_positive,
        'surrogate_any_exceedance': predicted_positive,
    })
    comparison.to_csv(output / 'reference_risk_calibration.csv', index=False)
    stats = {
        'reference_structures': len(buildings),
        'heldout_records_per_structure': len(records),
        'empirical_reference_compliance_90_percent_of_records': float(np.mean(pass_fraction >= .90)),
        'reference_with_observed_positive_cvar': int(true_positive.sum()),
        'reference_with_predicted_positive_cvar': int(predicted_positive.sum()),
        'missed_positive_risk_references': int((true_positive & ~predicted_positive).sum()),
        'reference_mean_observed_cvar': float(true_risk.mean()),
        'reference_mean_predicted_cvar': float(predicted_risk.mean()),
        'surrogate_cvar_correlation': float(np.corrcoef(true_risk, predicted_risk)[0, 1]) if np.std(true_risk) > 0 and np.std(predicted_risk) > 0 else None,
        'ground_truth_only': 'All observed compliance statistics describe published reference designs, not generated designs.',
    }
    f = output / 'audit_figures'
    plt.figure(figsize=(6, 5))
    plt.scatter(true_risk, predicted_risk, s=24, alpha=.65)
    plt.xlabel('Measured reference CVaR')
    plt.ylabel('Surrogate-predicted reference CVaR')
    plt.title('Tail-risk calibration on held-out structures and records')
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(f / 'reference_risk_calibration.png', dpi=200)
    plt.close()
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    plt.hist(pass_fraction, bins=bins, edgecolor='white')
    plt.xlabel('Share of unseen ground motions meeting every limit')
    plt.ylabel('Number of reference designs')
    plt.title('Measured performance of published reference structures')
    plt.tight_layout()
    plt.savefig(f / 'observed_reference_pass_fraction.png', dpi=200)
    plt.close()
    return stats


def candidate_comparison(data, split, predictor, output):
    generated = json.loads((PROJECT / 'results/generated_designs.json').read_text())
    baseline = pd.read_csv(PROJECT / 'results/baseline_generated_candidates.csv')
    all_records = np.asarray(split['ground_motions']['test'], dtype=int)
    gm_t = torch.from_numpy(data['gm'][all_records])
    name_to_index = {int(name): i for i, name in enumerate(data['ids'])}
    catalog = (np.log(data['section_props']) - data['cont_mu']) / data['cont_std']
    source_candidates = []
    for entry in generated:
        source_candidates.append((entry['building_id'], 'graph_diffusion', entry['candidate'], np.array(entry['sections'], dtype=int)))
    # The saved baseline table contains summary values but not its complete section arrays.
    # Candidate risk comparisons for those baselines are limited to their saved common scenarios.
    rows = []
    for building_id, method, candidate_id, labels in source_candidates:
        index = name_to_index[int(building_id)]
        n = len(labels)
        features = np.zeros((57, 2), dtype=np.float32)
        features[:n] = catalog[labels]
        mask = data['mask'][index].reshape(57)
        with torch.inference_mode():
            means, logvars = predictor(torch.from_numpy(np.repeat(features[None], len(all_records), axis=0)),
                 torch.from_numpy(np.repeat(mask[None], len(all_records), axis=0)),
                 torch.from_numpy(np.repeat(data['geom'][index:index + 1], len(all_records), axis=0)), gm_t)
        physical = np.expm1(means.numpy() * data['response_std'] + data['response_mu']).clip(min=0)
        scored_risk = float(empirical_cvar(physical[None])[0])
        origin = next(r for r in generated if int(r['building_id']) == building_id and int(r['candidate']) == candidate_id)
        reference = np.asarray(origin['reference_sections'], dtype=int)
        original = np.asarray(data['geom_raw'][index], dtype=float)
        num_stories, bays = int(original[0]), int(original[1])
        heights = np.array([original[3]] + [original[4]] * (num_stories - 1))
        lengths = np.tile(np.array([2 * heights, (bays - 1) * heights, np.ones(num_stories) * bays * original[2]]).T.reshape(-1), 1)
        section_mass = data['section_props'][:, 0] * 7850
        mass = float(np.sum(section_mass[labels] * lengths))
        reference_mass = float(np.sum(section_mass[reference] * lengths))
        rows.append({'building_id': building_id, 'candidate': candidate_id, 'method': method,
                     'surrogate_cvar_on_36_unseen_records': scored_risk,
                     'section_catalog_membership': 1.0,
                     'nominal_frame_steel_mass_kg': mass,
                     'reference_nominal_steel_mass_kg': reference_mass,
                     'mass_change_vs_reference_percent': 100 * (mass / reference_mass - 1),
                     'seismic_compliance_verified': False})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'generated_candidate_unseen_record_screening.csv', index=False)
    stats = {
        'generated_structures_evaluated': len(table),
        'distinct_buildings': int(table['building_id'].nunique()),
        'screening_records_per_candidate': len(all_records),
        'candidate_mean_mass_change_percent': float(table['mass_change_vs_reference_percent'].mean()),
        'candidate_median_mass_change_percent': float(table['mass_change_vs_reference_percent'].median()),
        'candidates_with_zero_surrogate_cvar': int((table['surrogate_cvar_on_36_unseen_records'] == 0).sum()),
        'nonlinear_structural_validation': 'not performed',
        'baseline_caution': 'Baseline outputs have no archived per-candidate section configurations, so a fully independent matched re-evaluation is unavailable.'
    }
    figure = output / 'audit_figures'
    plt.figure(figsize=(7, 4.5))
    plt.scatter(table['mass_change_vs_reference_percent'], table['surrogate_cvar_on_36_unseen_records'], s=40, alpha=.7)
    plt.xlabel('Nominal mass change against the published design (%)')
    plt.ylabel('Surrogate CVaR on unseen ground-motion records')
    plt.title('Generated designs: cost–risk screening only')
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(figure / 'generated_design_mass_risk_screening.png', dpi=200)
    plt.close()
    return stats


def sensitivity_checks(data, split, predictor, output):
    indices = np.asarray(split['buildings']['test'])[:20]
    gms = np.asarray(split['ground_motions']['test'])[:12]
    records = []
    for index in indices:
        original = data['cont'][index].reshape(57, 2).copy()
        mask = data['mask'][index].reshape(57)
        for factor in (0.8, 1., 1.2):
            features = original.copy()
            features[mask, 0] += np.log(factor) / data['cont_std'][0]
            pairs = len(gms)
            with torch.inference_mode():
                mean, _ = predictor(torch.from_numpy(np.repeat(features[None], pairs, axis=0)),
                    torch.from_numpy(np.repeat(mask[None], pairs, axis=0)),
                    torch.from_numpy(np.repeat(data['geom'][index:index+1], pairs, axis=0)),
                    torch.from_numpy(data['gm'][gms]))
            physical = np.expm1(mean.numpy() * data['response_std'] + data['response_mu']).clip(0)
            records.append({'building_id': int(data['ids'][index]), 'area_factor': factor,
                            'peak_drift_prediction': float(np.mean(physical[:, 0])),
                            'acceleration_prediction': float(np.mean(physical[:, 1])),
                            'residual_drift_prediction': float(np.mean(physical[:, 2]))})
    table = pd.DataFrame(records)
    table.to_csv(output / 'surrogate_area_sensitivity.csv', index=False)
    pivot = table.pivot(index='building_id', columns='area_factor', values='peak_drift_prediction')
    compliant = float(np.mean((pivot[.8] >= pivot[1.]) & (pivot[1.] >= pivot[1.2])))
    # This is a counterfactual predictor diagnostic, not a physically valid resize.
    fig = output / 'audit_figures'
    plt.figure(figsize=(7, 4.5))
    for building_id, group in table.groupby('building_id'):
        plt.plot(group['area_factor'], group['peak_drift_prediction'], alpha=.18)
    plt.plot(table.groupby('area_factor')['peak_drift_prediction'].mean(), color='black', marker='o', lw=2, label='Mean')
    plt.xlabel('Continuous area multiplier')
    plt.ylabel('Surrogate peak drift prediction')
    plt.title('Counterfactual area sensitivity of frozen surrogate')
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig / 'surrogate_area_sensitivity.png', dpi=200)
    plt.close()
    return {'tested_reference_structures': len(indices), 'records_per_structure': len(gms),
            'monotonically_decreasing_peak_drift_fraction': compliant,
            'caution': 'Area-only perturbations need not describe any valid catalog section or maintain the structural moment of inertia.'}


def draw_cross_scenario(scenarios, output):
    for variable in NAMES:
        values = [scenarios[key]['metrics'][variable]['r2'] for key in scenarios]
        plt.figure(figsize=(8, 4.6))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), ['Seen building / New GM', 'New building / Seen GM', 'New building / New GM'], rotation=12, ha='right')
        plt.ylabel('R-squared')
        plt.title(f'Surrogate response prediction: {variable}')
        plt.axhline(0, color='black', lw=.7)
        plt.tight_layout()
        plt.savefig(output / 'audit_figures' / f'generalization_{variable}.png', dpi=200)
        plt.close()


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    data, splits = get_data()
    output = PROJECT / 'results'
    (output / 'audit_figures').mkdir(exist_ok=True)
    predictor = build_predictor()
    scenario_metrics, heldout = response_baselines(data, splits, predictor, output)
    reference = actual_reference_risks(data, splits, heldout, output)
    generated = candidate_comparison(data, splits, predictor, output)
    sensitivity = sensitivity_checks(data, splits, predictor, output)
    draw_cross_scenario(scenario_metrics, output)
    result = {
        'source': 'DesignSafe published designs and frozen trained models',
        'cross_scenario_surrogate': scenario_metrics,
        'published_reference_seismic_risk': reference,
        'generated_candidate_surrogate_screening': generated,
        'counterfactual_sensitivity': sensitivity,
        'role_catalog_coverage': inspect_designs(data, splits),
        'claim_status': {'verified_structural_compliance_92_6_percent': False,
                         'improvement_16_8_percentage_points': False,
                         'cvar_reduction_18_4_percent': False,
                         'verified_mass_reduction_7_3_percent': False},
    }
    (output / 'independent_audit_metrics.json').write_text(json.dumps(result, indent=2))
    print('AUDIT_FINISHED', json.dumps({
        'reference': reference, 'generated': generated, 'sensitivity': sensitivity}, indent=2), flush=True)


if __name__ == '__main__':
    main()
