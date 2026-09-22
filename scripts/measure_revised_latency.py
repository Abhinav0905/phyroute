"""Measure complete fixed-threshold cascades on preset varied workloads."""
from pathlib import Path
import json
import os
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from threadpoolctl import threadpool_limits
from phyroute.feeder import V_MIN, V_MAX
from phyroute.lindistflow import LinDistFlow
from phyroute.powerflow import solve_bfs, voltage_margin
from phyroute.revised import Experiment, environment, tiny_predictions
from phyroute.topology_model import train_topology_model, predict_topology_model, voltage_extrema

ROOT = Path(__file__).resolve().parents[1]


def main():
    exp = Experiment(ROOT / "configs/revised.yaml")
    seed, test_seed = 4101, 9101
    models = exp.train(seed)
    residual_model, _ = train_topology_model(ROOT, exp.cfg, seed)
    cal_path = exp.output / "models" / f"train_{seed}__calibration.json"
    thresholds = json.loads(cal_path.read_text())["thresholds"]
    ds = exp.test_data(test_seed)
    methods = ["always_exact", "physics_only_abs_margin", "nn_margin", "phy_d", "physics_residual_margin"]
    selected = [t for t in exp.topologies if t["id"] in ["base", "close35_open31"]]
    result = {"training_seed": seed, "test_seed": test_seed, "calibration_budget": .1,
              "environment": environment(), "topologies": [t["id"] for t in selected],
              "scope": "One trained seed, nominal verifier, preselected original and original RC-A topologies. Complete NumPy routes include prediction, selection, gathers and strict BFS fallback. Varied cyclic batches, two warmups; timing order rotates. No NR retries in timed workloads. p95 is across calls, not a deployment service-level guarantee.",
              "rows": []}
    from phyroute.revised import feeder_for
    with threadpool_limits(limits=1):
        for topo in selected:
            f = feeder_for(topo)
            ldf = LinDistFlow(f)
            def run(method, ix):
                X = ds["X"][ix]; p, q, v0 = ds["p_net"][ix], ds["q_net"][ix], ds["v0"][ix]
                if method == "always_exact":
                    return voltage_margin(solve_bfs(f,p,q,v0), V_MIN,V_MAX)>=0, len(ix)
                if method in ["physics_only_abs_margin", "phy_d", "physics_residual_margin"]:
                    extrema = voltage_extrema(ldf.voltages(p,q,v0))
                    lin = np.minimum(V_MAX-extrema[:,1],extrema[:,0]-V_MIN)
                if method in ["nn_margin", "phy_d"]:
                    # Both methods need the mean prediction, not ensemble spread.
                    mean = np.mean([models[f"tiny_{i}"].forward(X) for i in range(exp.cfg["n_tiny"])],axis=0)
                    nn = np.minimum(V_MAX-mean[:,1],mean[:,0]-V_MIN)
                if method == "nn_margin": cheap, score = nn>=0, -np.abs(nn)
                elif method == "physics_only_abs_margin": cheap,score = lin>=0,-np.abs(lin)
                elif method == "phy_d":
                    cheap=nn>=0; score=np.where(cheap!=(lin>=0),np.abs(lin),-np.abs(lin))
                else:
                    _, margin=predict_topology_model(residual_model,X,extrema)
                    cheap,score=margin>=0,-np.abs(margin)
                esc=score>float(thresholds[method]["0.1"])
                approve=cheap.copy()
                if esc.any(): approve[esc]=voltage_margin(solve_bfs(f,p[esc],q[esc],v0[esc]),V_MIN,V_MAX)>=0
                return approve,int(esc.sum())
            for batch, repeats in [(1,200),(64,80),(512,30)]:
                timings={p:[] for p in methods}; counts={p:[] for p in methods}
                for method in methods:
                    for k in range(2): run(method,(np.arange(batch)+k*batch)%len(ds["X"]))
                for rep in range(repeats):
                    ix=(np.arange(batch)+rep*batch)%len(ds["X"])
                    order=methods[rep%len(methods):]+methods[:rep%len(methods)]
                    for method in order:
                        start=time.perf_counter_ns(); approve,n_esc=run(method,ix)
                        elapsed=(time.perf_counter_ns()-start)/1000
                        if approve.shape!=(batch,): raise ValueError("wrong decision shape")
                        timings[method].append(elapsed);counts[method].append(n_esc)
                for method in methods:
                    t=np.asarray(timings[method]);row={"topology_id":topo["id"],"policy":method,"batch_size":batch,"repeats":repeats,
                        "median_us_per_call":float(np.median(t)),"p95_us_per_call":float(np.quantile(t,.95)),
                        "mean_us_per_scenario":float(t.mean()/batch),"median_us_per_scenario":float(np.median(t)/batch),
                        "realized_escalation":sum(counts[method])/(batch*repeats),"all_call_us":t.tolist()}
                    result["rows"].append(row)
                    print(topo["id"],batch,method,round(row["median_us_per_scenario"],3),flush=True)
    path=exp.output/"analysis/latency.json";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":main()
