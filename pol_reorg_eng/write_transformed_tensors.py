import numpy as np

def write_transformed_tensors(file_path, transformed_tensors):
    """Write the transformed tensors to a file."""
    with open(file_path, 'w') as file:
        for frame, tensor in transformed_tensors:
            # Squeeze the tensor to remove the singleton dimension
            tensor = np.squeeze(tensor)

            # Debug print to check tensor values and types
#           print(f"Frame: {frame}, Tensor: {tensor}, Types: {tensor.dtype}")
#           print(f"Elements: {tensor[0, 0]}, {tensor[0, 1]}, {tensor[0, 2]}, {tensor[1, 1]}, {tensor[1, 2]}, {tensor[2, 2]}")

            file.write(f"{frame} {float(tensor[0, 0]):.4f} {float(tensor[0, 1]):.4f} {float(tensor[0, 2]):.4f} {float(tensor[1, 1]):.4f} {float(tensor[1, 2]):.4f} {float(tensor[2, 2]):.4f}\n")
