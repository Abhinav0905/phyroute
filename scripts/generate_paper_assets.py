"""Generate the paper's numeric tables from completed, audited run summaries."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"paper/generated_tables"
PAPER.mkdir(parents=True,exist_ok=True)
summary=json.loads((ROOT/"results/revised/analysis/summary.json").read_text())
latency=json.loads((ROOT/"results/revised/analysis/latency.json").read_text())

LABELS={"nn_margin":"Neural margin","nn_spread":"Ensemble spread","spread_norm":"Spread-normalized margin",
        "classifier_conf":"Classifier confidence","error_ratio":"Learned error ratio","phy_d":"PhyRoute-D",
        "phy_min":"Minimum margin","nn_approval_signed_physics":"Neural approval-only",
        "physics_only_abs_margin":"Physics only","physics_approval_only":"Physics approval-only",
        "conservative_and":"Joint approval","physics_residual_margin":"Residual MLP"}
def get(policy,variant=None):
    rows=summary['nominal_primary'] if variant is None else summary['mismatch_primary']
    return next(x for x in rows if x['policy']==policy and x['target']==.1 and (variant is None or x['physics_variant']==variant))
def f(rate,places=3):
    value=rate*100
    if 0 < value < 0.5 * 10**(-places):
        return rf'$< {10**(-places):.{places}f}$'
    return f'{value:.{places}f}'
def write(name,text):(PAPER/name).write_text(text+'\n')

lines=[r'\begin{table}[tb]',r'\caption{Nominal current verifier, frozen 10\% calibration thresholds. Values are topology-macro percentages on 44 held-out exchanges, averaged over three training and three test seeds. Escalation uses all generated draws; unsafe approvals use resolved draws and false rejections use resolved feasible draws.}',r'\label{tab:nominal}',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Route & Escalation & Unsafe & False rejection \\',r'\midrule']
for policy,label in LABELS.items():
    r=get(policy); lines.append(f'{label} & {f(r["realized_escalation"],2)} & {f(r["unsafe_rate"])} & {f(r["false_alarm_rate"])} '+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('nominal.tex','\n'.join(lines))

lines=[r'\begin{table}[tb]',r'\caption{Verifier mismatch with frozen nominal 10\% calibration thresholds. Entries give unsafe approvals / false rejections in percent, with the same macro averaging and denominators as Table~\ref{tab:nominal}. The physical reference is unchanged.}',r'\label{tab:mismatch}',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Verifier & PhyRoute-D & Physics only & Residual MLP \\',r'\midrule']
for variant,label in [('current','Correct'),('branch_noise_10pct',r'Branch noise $\pm10\%$'),('impedance_0.8',r'$R,X\times0.80$'),('impedance_0.9',r'$R,X\times0.90$'),('impedance_0.95',r'$R,X\times0.95$'),('impedance_1.05',r'$R,X\times1.05$'),('impedance_1.1',r'$R,X\times1.10$'),('impedance_1.2',r'$R,X\times1.20$'),('stale','Stale topology')]:
    cells=[]
    for pol in ['phy_d','physics_only_abs_margin','physics_residual_margin']:
        r=get(pol,variant);cells.append(f'{f(r["unsafe_rate"])} / {f(r["false_alarm_rate"])}')
    lines.append(label+' & '+' & '.join(cells)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('mismatch.tex','\n'.join(lines))

lines=[r'\begin{table}[tb]',r'\caption{Measured complete-route mean latency in $\mu$s per scenario on an Apple M4 with one BLAS thread. Base is the original network; RC-A is the previously studied close35/open31 exchange. Batch size is shown in each column. Training seed 4101, test seed 9101; frozen 10\% calibration thresholds.}',r'\label{tab:latency}',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Route & Base, 1 & RC-A, 1 & Base, 512 & RC-A, 512 \\',r'\midrule']
for pol,label in [('always_exact','Always exact'),('physics_only_abs_margin','Physics only'),('nn_margin','Neural margin'),('phy_d','PhyRoute-D'),('physics_residual_margin','Residual MLP')]:
    cells=[]
    for top,batch in [('base',1),('close35_open31',1),('base',512),('close35_open31',512)]:
        r=next(x for x in latency['rows'] if x['policy']==pol and x['topology_id']==top and x['batch_size']==batch)
        cells.append(f'{r["mean_us_per_scenario"]:.2f}')
    lines.append(label+' & '+' & '.join(cells)+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('latency.tex','\n'.join(lines))

# A machine-readable ledger anchors hand-written interpretation to exact outputs.
ledger={'nominal_10pct':{p:get(p) for p in LABELS},'source_summary':'results/revised/analysis/summary.json',
        'source_latency':'results/revised/analysis/latency.json'}
(PAPER/'numeric_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
print('Generated three manuscript tables and numeric ledger')
