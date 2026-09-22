# Protocol amendments

## 1 Reference nonconvergence handling

Recorded 22 September 2026 after the predetermined solver cross-check and before the revised routing outcome evaluation. On one of the 59 branch exchanges (close34_open01), the batch containing five predetermined ID draws failed backward/forward-sweep convergence. The other 58 batches matched pandapower. The frozen ranges and all 59 exchanges are retained.

The numerical reference will use strict backward/forward sweep first. A failed batch is subdivided to isolate difficult draws, which are retried with pandapower Newton-Raphson. Accept only finite, converged reference outputs. If both methods fail, preserve the original row and label it unresolved rather than safe or physically infeasible; nonconvergence alone does not establish absence of an AC solution.

For every condition record generated, solver-resolved and unresolved counts. Voltage-error and decision-risk metrics use only solver-resolved draws, with explicit denominators; report unresolved fractions separately by topology. Primary paired comparisons use the same resolved subset for every method. These conditional metrics do not bound risk on unresolved scenarios. No topology is excluded on the basis of its results. Runtime measurements use declared resolved workloads and do not conceal the separate cost of reference-label fallback.

This amendment changes reference failure handling and the interpretation of denominators. It does not change the generator, methods, topology split, training hyperparameters or threshold-selection rules. The audit found the issue before inspecting comparative routing performance on the revised held-out study.
