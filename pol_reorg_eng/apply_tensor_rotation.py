import numpy as np

def is_valid_rotation_matrix(matrix):
    # Check if the matrix is 3x3
    if matrix.shape != (3, 3):
        return False

    # Check if the matrix is orthogonal (R * R^T should be identity)
    identity = np.eye(3)
    if not np.allclose(np.dot(matrix, matrix.T), identity, atol=1e-6):
        return False

    # Check if the determinant is 1
    if not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        return False

    return True

def rotate_tensor(tensor, rotation_matrix):
    if not is_valid_rotation_matrix(rotation_matrix):
        raise ValueError("The provided matrix is not a valid rotation matrix.")

    rotation_matrix_T = rotation_matrix.T
    rotated_tensor = rotation_matrix @ tensor @ rotation_matrix_T
    return rotated_tensor
