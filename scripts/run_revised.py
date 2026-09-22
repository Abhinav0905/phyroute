"""Run one resumable phase of the revised frozen-protocol experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

# Set before NumPy/BLAS imports. Explicit one-thread execution is part of protocol.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyroute.revised import Experiment, atomic_json  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--config", default="configs/revised.yaml")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, help="training seed for train")
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--test-seed", type=int)
    parser.add_argument("--partition", choices=("all", "development", "heldout"), default="all")
    parser.add_argument("--topology", help="optional one-topology diagnostic")
    args = parser.parse_args()
    started = time.time()
    try:
        with threadpool_limits(limits=1):
            experiment = Experiment(args.config, args.out)
            if args.command == "train":
                if args.seed is None: parser.error("train requires --seed")
                experiment.train(args.seed)
                from phyroute.topology_model import train_topology_model
                train_topology_model(experiment.root, experiment.cfg, args.seed)
            elif args.command == "prepare":
                if args.test_seed is None: parser.error("prepare requires --test-seed")
                experiment.prepare(args.test_seed, args.partition, args.topology)
            else:
                if args.train_seed is None or args.test_seed is None:
                    parser.error("evaluate requires --train-seed and --test-seed")
                experiment.evaluate(args.train_seed, args.test_seed, args.partition, args.topology)
        print(f"[done] {args.command} {time.time()-started:.1f}s", flush=True)
    except Exception as exc:
        failure = {"command": vars(args), "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_s": time.time()-started, "traceback": traceback.format_exc()}
        for field in ("iterations", "failed_scenarios"):
            if hasattr(exc, field): failure[field] = getattr(exc, field)
        logdir = Path(args.out).resolve() if args.out else Path(args.config).resolve().parents[1] / "results" / "revised"
        atomic_json(logdir / "failures" / f"{time.time_ns()}.json", failure)
        raise


if __name__ == "__main__":
    main()
