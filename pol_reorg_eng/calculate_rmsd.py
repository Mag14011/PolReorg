import numpy as np

def calculate_rmsd(coords1, coords2):
    """Calculate the root mean square deviation (RMSD) between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
