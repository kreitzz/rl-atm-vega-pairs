"""Submission implementation of the paper's retained options sizing strategy."""

# Set these before importing NumPy, Arrow or scikit-learn. No worker pools.
import os

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS",
):
    os.environ[_name] = "1"
