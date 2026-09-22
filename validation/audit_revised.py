"""Independent artifact/metric audit; does not mutate the frozen experiment.

Run after the experiment and parent analysis:
    python validation/audit_revised.py --require-complete
Without that flag, available outputs are checked and missing stages are listed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phyroute.checkpoint import load_checkpoint

OUT = ROOT / 'results/revised'
LEARNED = ['nn_margin', 'nn_spread', 'spread_norm', 'classifier_conf', 'error_ratio']
PHYSICS = ['phy_d', 'phy_min', 'nn_approval_signed_physics', 'physics_only_abs_margin',
           'physics_approval_only', 'conservative_and', 'physics_residual_margin']
REPLAY_CELLS = [(4101, 9101, 'id', 'base'), (4201, 9201, 'id', 'close34_open01'),
                (4301, 9301, 'id', 'close35_open31'), (4101, 9301, 'id', 'close36_open27'),
                (4201, 9101, 'pv_shift', 'base')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def infer(model, X):
    h = (X - model.x_mean) / model.x_scale
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        h = h @ weight + bias
        if index != len(model.weights)-1: h = np.maximum(h, 0)
    return 1 / (1 + np.exp(-h[:, 0])) if model.classifier else h * model.y_scale + model.y_mean


def predict(models, X):
    member_outputs = np.stack([infer(models[f'tiny_{i}'], X) for i in range(5)])
    member_margins = np.minimum(1.05-member_outputs[:, :, 1], member_outputs[:, :, 0]-.95)
    mean = member_outputs.mean(axis=0)
    nn = np.minimum(1.05-mean[:, 1], mean[:, 0]-.95)
    sigma = np.std(member_margins, axis=0, ddof=1)
    ratio = 10**infer(models['error_probe'], np.column_stack([X, nn, sigma]))[:, 0] / (np.abs(nn)+1e-6)
    return {'m_nn': nn, 'sigma': sigma, 'legacy_mean_margin': member_margins.mean(axis=0),
            'p_viol': infer(models['classifier'], X), 'error_ratio': ratio}


def residual_margin(model, X, lin):
    output = lin + infer(model, np.column_stack([X, lin]))
    return np.minimum(1.05-output[:, 1], output[:, 0]-.95)


def gate(name, pred, lin):
    nn, sigma = pred['m_nn'], pred['sigma']; a = nn >= 0; b = lin >= 0
    if name == 'nn_margin': return a, -np.abs(nn)
    if name == 'nn_spread': return a, sigma
    if name == 'spread_norm': return a, -np.abs(nn)/(sigma+1e-6)
    if name == 'classifier_conf': return a, -np.abs(pred['p_viol']-.5)
    if name == 'error_ratio': return a, pred['error_ratio']
    if name == 'phy_d': return a, np.where(a != b, np.abs(lin), -np.abs(lin))
    if name == 'phy_min': return a, -np.minimum(np.abs(nn), np.abs(lin))
    if name == 'nn_approval_signed_physics': return a, np.where(a, -lin, -np.inf)
    if name == 'physics_only_abs_margin': return b, -np.abs(lin)
    if name == 'physics_approval_only': return b, np.where(b, -lin, -np.inf)
    if name == 'conservative_and': return a & b, -np.abs(lin)
    if name == 'physics_residual_margin': return pred['m_residual'] >= 0, -np.abs(pred['m_residual'])
    raise ValueError(name)


def quantile_cut(score, target):
    finite = score[np.isfinite(score)]
    if not len(finite) or target == 0: return np.inf
    floor = finite.min() - max(1., np.ptp(finite))
    ranked = np.where(np.isfinite(score), score, floor)
    if target == 1: return np.nextafter(floor, -np.inf)
    return float(np.quantile(ranked, 1-target))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'validation/revised_integrity.json')
    args = parser.parse_args()
    checks, failures, pending = [], [], []
    def check(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': clean(details)})
        if not passed: failures.append(name)
    def close(a, b): return bool(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))
    cfg = yaml.safe_load((ROOT/'configs/revised.yaml').read_text())
    manifest = json.loads((OUT/'run_manifest.json').read_text())
    identity = {k: manifest[k] for k in ['config', 'topologies', 'source_sha256']}
    check('run_identity', canonical(identity) == manifest['run_hash'] and manifest['config'] == cfg)
    check('immutable_source_hashes', all(digest(ROOT/'phyroute'/k) == h for k, h in manifest['source_sha256'].items()))
    frozen = json.loads((ROOT/'protocol/FROZEN.json').read_text())
    check('frozen_protocol_hashes', all(digest(ROOT/p) == h for p, h in frozen['files'].items()))
    inventory = json.loads((ROOT/cfg['topology_inventory']).read_text())['topologies']
    check('topology_counts', Counter(t['partition'] for t in inventory) == {'heldout': 44, 'development': 15})
    topology_ids = ['base'] + [t['id'] for t in inventory]
    variants = ['current', 'stale'] + [f'impedance_{s:g}' for s in cfg['impedance_scales']] + ['branch_noise_10pct']
    all_models, residuals, calibrations = {}, {}, {}
    for seed in cfg['training_seeds']:
        base_path = OUT/'models'/f'train_{seed}.npz'
        residual_path = ROOT/'results/models'/f'topology_{seed}.npz'
        if not base_path.exists() or not residual_path.exists():
            pending.append(f'models {seed}'); continue
        base_cfg = {'run_hash': manifest['run_hash'], 'kind': 'models', 'seed': seed}
        models, meta = load_checkpoint(base_path, expected_config=base_cfg)
        all_models[seed] = models
        check(f'base_model_names_{seed}', set(models) == {*(f'tiny_{i}' for i in range(5)), 'classifier', 'error_probe'})
        check(f'base_model_provenance_{seed}', meta['source_metadata']['source_sha256'] == manifest['source_sha256']
              and not meta['source_metadata']['calibration_used_for_fitting'])
        history = meta['source_metadata']['histories']
        check(f'training_history_lengths_{seed}', all(len(h['loss_curve']) == h['iterations'] == len(h['validation_scores']) for h in history.values()))
        residual_cfg = {'study_config': cfg, 'train_seed': seed, 'model_kind': 'development_topology_physics_residual'}
        rm, rmeta = load_checkpoint(residual_path, expected_config=residual_cfg)
        residuals[seed] = rm['residual']
        provenance = rmeta['source_metadata']
        desired_topologies = {'base'} | {t['id'] for t in inventory if t['partition'] == 'development'}
        check(f'residual_training_allocation_{seed}', provenance['training_rows'] == 48000
              and {r['topology_id'] for r in provenance['topologies']} == desired_topologies
              and all(r['n'] == 3000 for r in provenance['topologies']))
        check(f'residual_source_inventory_{seed}', provenance['source_sha256'] == manifest['source_sha256']['topology_model.py']
              and provenance['inventory_sha256'] == digest(ROOT/cfg['topology_inventory']))
        data = {name: arrays(OUT/'data'/f'train_{seed}__{name}.npz')['X'] for name in ['train', 'probe', 'calibration']}
        rowsets = {name: set(map(bytes, np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize*X.shape[1]))).ravel())) for name, X in data.items()}
        check(f'no_train_probe_calibration_overlap_{seed}', all(not (rowsets[a] & rowsets[b]) for a, b in itertools.combinations(rowsets, 2)))
        cal_path = OUT/'models'/f'train_{seed}__calibration.json'
        if not cal_path.exists(): pending.append(f'calibration {seed}'); continue
        cal = json.loads(cal_path.read_text()); calibrations[seed] = cal
        caldata = arrays(OUT/'data'/f'train_{seed}__calibration.npz')
        cp = predict(models, caldata['X'])
        from phyroute.feeder import build_feeder
        from phyroute.lindistflow import LinDistFlow
        voltages = LinDistFlow(build_feeder()).voltages(caldata['p_net'], caldata['q_net'], caldata['v0'])
        extrema = np.column_stack([voltages.min(axis=1), voltages.max(axis=1)])
        lin = np.minimum(1.05-extrema[:, 1], extrema[:, 0]-.95)
        cp['m_residual'] = residual_margin(residuals[seed], caldata['X'], extrema)
        thresholds_ok = all(close(cal['thresholds'][p][str(r)], quantile_cut(gate(p, cp, lin)[1], r)) for p in LEARNED+PHYSICS for r in cfg['calibration_budgets'])
        error = np.sort(np.abs(cp['m_nn']-caldata['margin']))
        conformal_ok = all(close(cal['conformal_radii'][str(alpha)], error[math.ceil((len(error)+1)*(1-alpha))-1]) for alpha in cfg['conformal_alpha'])
        check(f'independent_calibration_thresholds_{seed}', thresholds_ok and conformal_ok)
    cache_jsons = [p for folder in ['data', 'physics', 'predictions'] for p in (OUT/folder).glob('*.json')]
    cache_hash_bad = []
    physics_info = {}
    for p in cache_jsons:
        meta = json.loads(p.read_text()); npz = p.with_suffix('.npz')
        if not npz.exists() or digest(npz) != meta['sha256'] or meta['identity']['run_hash'] != manifest['run_hash']:
            cache_hash_bad.append(str(p.relative_to(ROOT))); continue
        if p.parent.name == 'physics':
            d = arrays(npz); valid = d['valid']; n = cfg['n_test']; detail = meta['details']['solver']
            good = (valid.dtype == bool and valid.shape == (n,) and np.isfinite(d['margin'][valid]).all()
                    and np.isnan(d['margin'][~valid]).all() and detail['n_generated'] == n
                    and detail['n_resolved'] == int(valid.sum()) and detail['n_unresolved'] == int((~valid).sum())
                    and set(k.removeprefix('lin__') for k in d if k.startswith('lin__')) == set(variants)
                    and set(k.removeprefix('extrema__') for k in d if k.startswith('extrema__')) == set(variants))
            good = good and all(close(d['lin__'+v], np.minimum(1.05-d['extrema__'+v][:, 1], d['extrema__'+v][:, 0]-.95)) for v in variants)
            check('physics_cache_'+p.stem, good)
            key = (meta['identity']['seed'], meta['identity']['condition'], meta['identity']['topology_id'])
            physics_info[key] = {'n': int(valid.sum()), 'unresolved': int((~valid).sum()),
                                 'violations': int((d['margin'][valid] < 0).sum())}
    check('cache_file_hashes', not cache_hash_bad, {'checked': len(cache_jsons), 'bad': cache_hash_bad})
    expected_physics = {(te, 'id', topo) for te in cfg['test_seeds'] for topo in topology_ids} | {(te, 'pv_shift', 'base') for te in cfg['test_seeds']}
    if set(physics_info) != expected_physics: pending.append(f'physics caches: {len(physics_info)}/{len(expected_physics)}')
    expected_cells = {(tr, te, 'id', topo) for tr in cfg['training_seeds'] for te in cfg['test_seeds'] for topo in topology_ids}
    expected_cells |= {(tr, te, 'pv_shift', 'base') for tr in cfg['training_seeds'] for te in cfg['test_seeds']}
    expected_keys = set()
    for p in LEARNED+PHYSICS:
        for v in (['not_applicable'] if p in LEARNED else variants):
            expected_keys.add((p, v, 'no_routing', 0.))
            expected_keys.update((p, v, 'frozen_calibration', r) for r in cfg['calibration_budgets'])
            expected_keys.update((p, v, 'matched_diagnostic', r) for r in cfg['matched_budgets'])
    expected_keys.update(('conformal_margin', 'not_applicable', 'conformal_calibration', a) for a in cfg['conformal_alpha'])
    expected_keys.add(('always_exact', 'not_applicable', 'always', 1.))
    frames = []; seen_cells = set(); row_errors = []; replayed = []
    for path in sorted((OUT/'evaluation').glob('*.csv')):
        meta = json.loads(path.with_suffix('.json').read_text())
        frame = pd.read_csv(path)
        check('csv_hash_'+path.stem, digest(path) == meta['sha256'] and len(frame) == meta['rows'] == 549)
        cells = set(map(tuple, frame[['train_seed', 'test_seed', 'condition', 'topology_id']].values))
        if len(cells) != 1: row_errors.append((path.name, 'mixed cells')); continue
        cell = cells.pop(); seen_cells.add(cell)
        tr, te, condition, topo = cell
        expected_identity = {'run_hash': manifest['run_hash'], 'kind': 'evaluation',
                             'training_seed': int(tr), 'test_seed': int(te),
                             'topology_id': topo, 'condition': condition}
        if meta['identity'] != expected_identity: row_errors.append((path.name, 'CSV provenance identity'))
        keys = list(map(tuple, frame[['policy', 'physics_variant', 'operating_mode', 'target']].values))
        desired = expected_keys | {('known_topology_fallback', 'not_applicable', 'always', float(topo != 'base'))}
        if len(set(keys)) != len(keys) or set(keys) != desired: row_errors.append((path.name, 'missing/duplicate policy cells'))
        ref = physics_info.get((te, condition, topo))
        if ref is None: pending.append('missing physics for '+path.name); continue
        if not all((frame[k] == v).all() for k, v in ref.items()): row_errors.append((path.name, 'reference counts'))
        derived = {'unsafe_rate': frame.unsafe/frame.n,
                   'false_alarm_rate': frame.false_alarms/(frame.n-frame.violations),
                   'conditional_risk': frame.unsafe/frame.accepted,
                   'miss_rate': frame.unsafe/frame.violations,
                   'accuracy': 1-(frame.unsafe+frame.false_alarms)/frame.n,
                   'acceptance_coverage': frame.accepted/frame.n,
                   'realized_escalation': frame.escalated/frame.n,
                   'realized_escalation_all': (frame.escalated+frame.escalated_unresolved)/frame.generated}
        if not all(close(frame[k], v) for k, v in derived.items()): row_errors.append((path.name, 'derived rates'))
        if not ((frame.n+frame.unresolved == frame.generated).all() and (frame.generated == cfg['n_test']).all()
                and (frame.accepted_unresolved+frame.escalated_unresolved <= frame.unresolved).all()):
            row_errors.append((path.name, 'unresolved accounting'))
        exact = frame[frame.policy == 'always_exact'].iloc[0]
        if exact.unsafe != 0 or exact.false_alarms != 0 or exact.realized_escalation_all != 1:
            row_errors.append((path.name, 'always-exact control'))
        if tr in calibrations:
            frozen_rows = frame[frame.operating_mode == 'frozen_calibration']
            if not all(close(float(r.threshold), calibrations[tr]['thresholds'][r.policy][str(r.target)]) for r in frozen_rows.itertuples()):
                row_errors.append((path.name, 'unfrozen threshold'))
        if cell in REPLAY_CELLS and tr in calibrations:
            X = arrays(OUT/'data'/f'test_{te}__{condition}.npz')['X']
            d = arrays(OUT/'physics'/f'test_{te}__{condition}__{topo}.npz')
            pred = predict(all_models[tr], X)
            cache_path = OUT/'predictions'/f'train_{tr}__test_{te}__{condition}.npz'
            cached = arrays(cache_path)
            if not all(close(pred[k], cached[k]) for k in pred): row_errors.append((path.name, 'prediction replay'))
            valid = d['valid']; truth = valid & (d['margin'] < 0)
            residual_cache = {v: residual_margin(residuals[tr], X, d['extrema__'+v]) for v in variants}
            for r in frame.itertuples():
                if r.policy == 'always_exact': cheap, esc = ~truth, np.ones(len(X), bool)
                elif r.policy == 'known_topology_fallback': cheap, esc = pred['m_nn'] >= 0, np.full(len(X), topo != 'base')
                elif r.policy == 'conformal_margin':
                    cheap = pred['m_nn'] >= 0; esc = np.abs(pred['m_nn']) <= float(calibrations[tr]['conformal_radii'][str(r.target)])
                else:
                    lin = d['lin__current'] if r.physics_variant == 'not_applicable' else d['lin__'+r.physics_variant]
                    if r.policy == 'physics_residual_margin': pred['m_residual'] = residual_cache[r.physics_variant]
                    cheap, score = gate(r.policy, pred, lin)
                    if r.operating_mode == 'no_routing': esc = np.zeros(len(X), bool)
                    elif r.operating_mode == 'frozen_calibration': esc = score > float(calibrations[tr]['thresholds'][r.policy][str(r.target)])
                    else:
                        tau = quantile_cut(score, r.target); esc = score > tau
                        if not close(tau, float(r.threshold)): row_errors.append((path.name, 'retrospective threshold '+r.policy))
                final = cheap.copy(); final[esc] = valid[esc] & ~truth[esc]
                expected_counts = {'unsafe': int((final & truth & valid).sum()),
                                   'false_alarms': int((~final & ~truth & valid).sum()),
                                   'accepted': int((final & valid).sum()), 'escalated': int((esc & valid).sum()),
                                   'accepted_unresolved': int((final & ~valid).sum()),
                                   'escalated_unresolved': int((esc & ~valid).sum())}
                if any(getattr(r, k) != v for k, v in expected_counts.items()):
                    row_errors.append((path.name, 'count replay', r.policy, r.physics_variant, r.operating_mode, r.target))
            replayed.append(cell)
        frames.append(frame)
    check('evaluation_row_integrity', not row_errors, {'errors': row_errors[:40], 'error_count': len(row_errors), 'replayed_cells': replayed})
    if seen_cells != expected_cells: pending.append(f'evaluation cells: {len(seen_cells)}/{len(expected_cells)}')
    summary_path = OUT/'analysis/summary.json'
    if summary_path.exists() and frames:
        summary = json.loads(summary_path.read_text()); df = pd.concat(frames, ignore_index=True)
        check('analysis_csv_inventory', summary['evaluation_files'] == len(frames) == 549 and summary['rows'] == len(df) == 301401)
        nominal = df[(df.partition == 'heldout') & df.physics_variant.isin(['current', 'not_applicable']) & (df.operating_mode == 'frozen_calibration')]
        aggregate_bad = []
        for row in summary['nominal_primary']:
            group = nominal[(nominal.policy == row['policy']) & np.isclose(nominal.target, row['target'])]
            for metric in ['unsafe_rate', 'false_alarm_rate', 'conditional_risk', 'miss_rate', 'realized_escalation']:
                source = 'realized_escalation_all' if metric == 'realized_escalation' else metric
                topo_mean = group.groupby('topology_id')[source].mean()
                seed_mean = group.groupby('train_seed')[source].mean()
                for key, expected in [(metric, topo_mean.mean()), (metric+'_worst_topology', topo_mean.max()),
                                      (metric+'_seed_min', seed_mean.min()), (metric+'_seed_max', seed_mean.max())]:
                    if not close(row[key], expected): aggregate_bad.append([row['policy'], row['target'], key])
        check('nominal_aggregation_and_denominators', not aggregate_bad, aggregate_bad)
        reference_total = {k: sum(v[k] for v in physics_info.values()) for k in ['n', 'unresolved']}
        coverage = summary['reference_coverage']
        check('unique_reference_coverage', coverage['resolved'] == reference_total['n'] and coverage['unresolved'] == reference_total['unresolved']
              and coverage['generated'] == len(expected_physics)*cfg['n_test'])
        # Reconstruct every reported summary family from the original CSVs.
        masks = {
            'mismatch_primary': ((df.partition == 'heldout') & (df.operating_mode == 'frozen_calibration'), ['policy', 'physics_variant', 'target']),
            'conformal_heldout': ((df.partition == 'heldout') & (df.operating_mode == 'conformal_calibration'), ['policy', 'target']),
            'matched_diagnostic': ((df.partition == 'heldout') & df.physics_variant.isin(['current', 'not_applicable']) & (df.operating_mode == 'matched_diagnostic'), ['policy', 'target']),
            'base_conditions': ((df.partition == 'base') & df.physics_variant.isin(['current', 'not_applicable']) & df.operating_mode.isin(['frozen_calibration', 'conformal_calibration']), ['condition', 'policy', 'operating_mode', 'target']),
            'no_routing': ((df.partition == 'heldout') & df.physics_variant.isin(['current', 'not_applicable']) & (df.operating_mode == 'no_routing'), ['policy']),
        }
        for family, (mask, keys) in masks.items():
            subset = df[mask]
            groups = {key if isinstance(key, tuple) else (key,): value for key, value in subset.groupby(keys)}
            bad = []
            if len(summary[family]) != len(groups): bad.append('group count')
            for row in summary[family]:
                group = groups.get(tuple(row[k] for k in keys))
                if group is None: bad.append('missing group'); continue
                for metric in ['unsafe_rate', 'false_alarm_rate', 'conditional_risk', 'miss_rate', 'realized_escalation']:
                    source = 'realized_escalation_all' if metric == 'realized_escalation' else metric
                    topology_means = group.groupby('topology_id')[source].mean()
                    seed_means = group.groupby('train_seed')[source].mean()
                    pairs = [(metric, topology_means.mean()), (metric+'_worst_topology', topology_means.max()),
                             (metric+'_seed_min', seed_means.min()), (metric+'_seed_max', seed_means.max())]
                    if any(not close(row[k], value) for k, value in pairs): bad.append([tuple(row[k] for k in keys), metric])
            check('aggregation_'+family, not bad, bad)
    else: pending.append('analysis summary')
    latency_path = OUT/'analysis/latency.json'
    if latency_path.exists():
        latency = json.loads(latency_path.read_text()); bad = []
        expected_latency = set(itertools.product(['base', 'close35_open31'],
            ['always_exact', 'physics_only_abs_margin', 'nn_margin', 'phy_d', 'physics_residual_margin'], [1, 64, 512]))
        observed_latency = [(r['topology_id'], r['policy'], r['batch_size']) for r in latency['rows']]
        if set(observed_latency) != expected_latency or len(observed_latency) != len(expected_latency): bad.append('workload inventory')
        for row in latency['rows']:
            calls = np.asarray(row['all_call_us']); batch = row['batch_size']
            if len(calls) != row['repeats'] or not np.isfinite(calls).all() or not (calls > 0).all(): bad.append('invalid call samples')
            expected = {'median_us_per_call': np.median(calls), 'p95_us_per_call': np.quantile(calls, .95),
                        'mean_us_per_scenario': calls.mean()/batch, 'median_us_per_scenario': np.median(calls)/batch}
            if not all(close(row[k], v) for k, v in expected.items()): bad.append('incorrect timing summary')
        check('latency_raw_samples_and_summary', not bad, bad)
    else: pending.append('latency summary')
    report = {'run_hash': manifest['run_hash'], 'audit_source_sha256': digest(__file__),
              'reviewed_analysis_sources': {p: digest(ROOT/p) for p in ['scripts/analyze_revised.py', 'scripts/measure_revised_latency.py']},
              'checks': checks, 'failures': failures, 'pending': pending,
              'complete': not pending, 'passed': not failures and (not args.require_complete or not pending),
              'limits': ['Independent calibration and five complete selected evaluation-cell replays; other CSV rows checked by inventories, algebra and hashes.',
                         'No neural retraining in this audit; original run recorded fresh training histories.',
                         'Escalation denominators include all generated inputs; decision-error denominators use only solver-resolved inputs.']}
    args.out.write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ['passed', 'complete', 'failures', 'pending']}))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
