"""Fixed robot dimensions and joint limits shared across model consumers."""
import numpy as np


SIDES = ("left", "right")
LOW_DEG = np.array([-178., -120., -178., -60., -178., -60., -60.])
HIGH_DEG = np.array([178., 120., 178., 145., 178., 60., 60.])
