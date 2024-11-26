import numpy as np

def apply_transformation(coords, rotation_matrix, translation_vector):
    """Apply the transformation matrix to the reference structure coordinates."""
#   transformed_coords = np.dot(coords, rotation_matrix) + translation_vector
    transformed_coords = np.dot(rotation_matrix, coords.T).T + translation_vector
    return transformed_coords
