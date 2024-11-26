import numpy as np

def angle_between_vectors(a, b):
    # Compute the dot product
    dot_product = np.dot(a, b)
    # Compute the magnitudes of the vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_a * norm_b)
    # Compute the angle in radians
    angle_radians = np.arccos(cos_theta)
    # Convert angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
