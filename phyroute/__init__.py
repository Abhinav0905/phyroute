"""PhyRoute: voltage screening under feeder reconfiguration.

The package contains a radial feeder model (IEEE 33-bus), an exact vectorised
backward/forward-sweep power-flow solver, a linearised DistFlow proxy used as a
physics-only baseline, neural and physics-augmented models, and routing policies
that decide when a cheap prediction should be escalated to an exact solve.
"""
__version__ = "0.2.0"
