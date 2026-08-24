# physics_simulation

A Python framework for exploring an idealized Field-Reversed Configuration
(FRC) plasma. It provides grid and magnetic-coil models, a Lorentz-force test
particle solver, a simple rigid-rotor plasma model, and a telemetry dashboard.

## Installation

Create and activate a virtual environment, then install the runtime and test
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the examples

Each module has a small smoke-test example:

```bash
python environment.py
python physics.py
MPLBACKEND=Agg python dashboard.py
```

## Run the pulsed FRC Digital Twin

The Dash control room solves a coupled, time-dependent 0D pulse at every
control update. It includes compression work, D--He3/D--T fusion reactivity,
bremsstrahlung and synchrotron losses, beta monitoring, and inductive direct
energy conversion for D--He3 operation.

```bash
python app.py
```

Open <http://127.0.0.1:8050>. This is an educational, idealized model and is
not a reactor design or a safety analysis.

Run the automated regression tests with:

```bash
python -m pytest
```

## Array conventions

On a cylindrical `VacuumVessel`, grid field arrays are ordered as
`(theta, r, z)`. Dashboard radial profiles use the `theta=0`, `z=0` slice;
axial profiles use the `theta=0`, `r=0` slice.
