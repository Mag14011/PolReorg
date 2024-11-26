import numpy as np
import read_transformation_matrices as rtm
import apply_tensor_rotation as atr
import write_transformed_tensors as wtt

def transform_and_save_tensors(matrices_path, tensor, output_path, frame_indices=None):
    """Read matrices, transform the tensor, and save the results to a file.
    
    Args:
        matrices_path (str): Path to the file containing transformation matrices.
        tensor (numpy.ndarray): 3x3 tensor to transform
        output_path (str): Path where transformed tensors will be saved.
        frame_indices (array-like, optional): Specific frame indices to process.
                                            If None, process all frames.
                                            
    Returns:
        numpy.ndarray: Array of transformed tensors with shape (n_frames, 3, 3)
    """

    print(f"\nDebug transform_and_save_tensors:")
    print(f"matrices_path: {matrices_path}")
    print(f"frame_indices length: {len(frame_indices) if frame_indices is not None else 'None'}")

    # Read the transformation matrices 
    matrices = rtm.read_transformation_matrices(matrices_path)
    print(f"Initial matrices length: {len(matrices)}")

    # Filter matrices if frame_indices is provided
    if frame_indices is not None:
        matrices = [m for m in matrices if m[0] in frame_indices]
    print(f"Filtered matrices length: {len(matrices)}")

    # Add frame number tracking
    processed_frames = set()

    # Apply the transformation to the tensor for each selected frame
    transformed_tensors = []  # For file output
    all_transformed = []      # For numpy array conversion
    
    for frame, rotation_matrix, translation_vector in matrices:
        if frame in processed_frames:
            print(f"Warning: Duplicate frame number detected: {frame}")
        processed_frames.add(frame)

        transformed = atr.rotate_tensor([tensor], rotation_matrix)[0]  # Get first tensor since we only passed one
        transformed_tensors.append((frame, transformed))
        all_transformed.append(transformed)

    # Write the transformed tensors to the output file
    wtt.write_transformed_tensors(output_path, transformed_tensors)
    
    # Convert list of transformed tensors to numpy array and return
    return np.array(all_transformed)
