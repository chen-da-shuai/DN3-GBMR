# DN3-GBMR

Source code for the DN3-GBMR mobile-robot navigation method described in:

> A cerebral-cerebellar heterogeneous collaborative navigation method for
> mobile robots in complex environments

## Contents

- `src/DN3_GBMR.py`: DN3-GBMR training and evaluation entry point.
- `src/GraphQAgent.py`: experience-memory component used by the DN3 module.
- `src/G_Dict.py`: graph-memory data structure and graph-value update routines.
- `requirements.txt`: Python dependencies used by the released source code.

## Method overview

DN3 provides incremental local motion control from the ten-dimensional
relative-state representation and the 36-direction action space. GBMR stores
few-shot state-action experience in a graph, performs shortcut-based
experience reconstruction and value updating, and supplies global navigation
guidance through the action-fusion mechanism.

## Environment

The experiments were implemented in Python. The code uses NumPy, Pandas,
NetworkX, Annoy, SciPy, scikit-image, Matplotlib, Seaborn, and TensorFlow
compatibility APIs. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The main script expects the relative `path/` directory used by the original
experiment project. Pretrained DN3 arrays and the output directories should
be placed there before running the pretrained-model evaluation branch. The
default simulation parameters, state representation, action space, reward
settings, graph-memory settings, and evaluation protocol are defined in
`src/DN3_GBMR.py`.

## Running

Run the entry point from a project directory that contains the expected
relative `path/` folders:

```bash
python src/DN3_GBMR.py
```

The script writes experiment records and graph statistics to the configured
relative output paths. Before a large experiment, check the output paths and
the number of training repetitions in the `__main__` block.

## Reproducibility notes

The source release contains the algorithm implementation and its directly
used graph-memory modules. Reproducing the reported numerical results also
requires the environment-generation procedure, pretrained model arrays, and
the random-seed settings used in the corresponding experiment scripts.
These files are kept separate from the core source release because they can
be large and experiment-specific.
