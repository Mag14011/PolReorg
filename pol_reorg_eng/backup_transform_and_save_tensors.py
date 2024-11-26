import read_transformation_matrices as rtm
import read_tensors as rt
import apply_tensor_rotation as atr
import write_transformed_tensors as wtt
import numpy as np

def transform_and_save_tensors(matrices_path, tensors_path, output_path, frame_indices=None):
    """Read matrices and tensors, transform the tensors, and save the results to a file.
    
    Args:
        matrices_path (str): Path to the file containing transformation matrices.
        tensors_path (str): Path to the file containing tensors to transform.
        output_path (str): Path where transformed tensors will be saved.
        frame_indices (array-like, optional): Specific frame indices to process.
                                            If None, process all frames.
                                            
    Returns:
        np.ndarray: Array of transformed tensors with shape (n_frames, 3, 3)
    """

    # Read the transformation matrices and tensors from their respective files
    matrices = rtm.read_transformation_matrices(matrices_path)
    tensor = rt.read_tensors(tensors_path)

    transformed_tensors = []

    # Filter matrices if frame_indices is provided
    if frame_indices is not None:
        matrices = [m for m in matrices if m[0] in frame_indices]

    # Add frame number tracking
    processed_frames = set()

    # Apply the transformation to the tensor for each selected frame
    for frame, rotation_matrix, translation_vector in matrices:
        if frame in processed_frames:
            print(f"Warning: Duplicate frame number detected: {frame}")
        processed_frames.add(frame)

        transformed_tensor = atr.rotate_tensor(tensor, rotation_matrix)
        transformed_tensors.append((frame, transformed_tensor))

    # Write the transformed tensors to the output file
    wtt.write_transformed_tensors(output_path, transformed_tensors)

    # Convert list of (frame, tensor) tuples to numpy array of just tensors
    # Sort by frame number first to ensure correct order
    transformed_tensors.sort(key=lambda x: x[0])
    tensors_array = np.array([tensor for frame, tensor in transformed_tensors])

    return tensors_array
