"""Predetermined solver comparison: five paired draws for every branch exchange.

Run from the repository root:
    python validation/solver_crosscheck.py
The preset seed and scenario count are fixed here before this validation runs.
Any convergence failure or error above tolerance makes the command fail after
writing all observed validation outcomes. Validation points are not selected
from outcome errors and do not enter model fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandapower

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phyroute.feeder import build_feeder, pandapower_voltages
from phyroute.powerflow import solve_bfs
from phyroute.scenarios import ID, sample

SEED = 20260922
N_SCENARIOS = 5
MAX_ABS_TOLERANCE_PU = 1e-8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'validation/solver_crosscheck.json')
    args = parser.parse_args()
    inventory_path = ROOT / 'protocol/topologies.json'
    inventory = json.loads(inventory_path.read_text())['topologies']
    assert len(inventory) == 59, 'frozen inventory must contain 59 branch exchanges'
    base = build_feeder()
    ds = sample(base, ID, N_SCENARIOS, np.random.default_rng(SEED))
    pv = ds.X[:, 64:64 + len(base.pv_bus)]
    p_load = ds.p_net.copy()
    p_load[:, base.pv_bus] += pv
    rows = []
    for item in inventory:
        feeder = build_feeder(open_lines=(item['open_line'],), close_lines=(item['close_line'],))
        row = {**item, 'scenarios': [], 'passed': True}
        try:
            vm, info = solve_bfs(feeder, ds.p_net, ds.q_net, ds.v0, return_info=True)
            row['batch_solver_diagnostics'] = info
        except Exception as exc:
            row['passed'] = False
            row['batch_error'] = f'{type(exc).__name__}: {exc}'
            vm = None
        for i in range(N_SCENARIOS):
            result = {'scenario_index': i}
            candidate = None if vm is None else vm[i]
            if candidate is None:
                # Diagnose the unchanged preset points individually after a batch
                # failure. Still keep the batch failure in the validation result.
                try:
                    candidate, info = solve_bfs(feeder, ds.p_net[i], ds.q_net[i], ds.v0[i], return_info=True)
                    candidate = candidate[0]
                    result['individual_solver_diagnostics'] = info
                except Exception as exc:
                    result['bfs_error'] = f'{type(exc).__name__}: {exc}'
            try:
                reference = pandapower_voltages(feeder, p_load[i], ds.q_net[i], pv[i], float(ds.v0[i]))
                error = None if candidate is None else float(np.max(np.abs(candidate - reference)))
                result.update(max_absolute_error_pu=error,
                              pandapower_converged=bool(feeder.net.converged),
                              all_finite=bool(np.isfinite(reference).all()),
                              passed=bool(feeder.net.converged and np.isfinite(reference).all()
                                          and error is not None and error < MAX_ABS_TOLERANCE_PU))
            except Exception as exc:
                result.update(error=f'{type(exc).__name__}: {exc}', passed=False)
            row['scenarios'].append(result)
            row['passed'] = row['passed'] and result['passed']
        rows.append(row)
    errors = [x['max_absolute_error_pu'] for row in rows for x in row['scenarios']
              if x.get('max_absolute_error_pu') is not None]
    output = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'preset': {'scenario_seed': SEED, 'scenarios_per_topology': N_SCENARIOS,
                         'scenario_generator': 'ID', 'paired_injections_across_topologies': True,
                         'tolerance_pu': MAX_ABS_TOLERANCE_PU},
              'provenance': {'python': platform.python_version(), 'platform': platform.platform(),
                             'numpy': np.__version__, 'pandapower': pandapower.__version__,
                             'topology_inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                             'solver_sha256': hashlib.sha256((ROOT / 'phyroute/powerflow.py').read_bytes()).hexdigest(),
                             'scenario_array_sha256': hashlib.sha256(ds.X.tobytes()).hexdigest()},
              'n_topologies': len(rows), 'n_reference_scenarios': sum(len(r['scenarios']) for r in rows),
              'maximum_absolute_error_pu': max(errors, default=None),
              'passed': all(row['passed'] for row in rows), 'results': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: output[key] for key in ['n_topologies', 'n_reference_scenarios',
                                                 'maximum_absolute_error_pu', 'passed']}))
    if not output['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
