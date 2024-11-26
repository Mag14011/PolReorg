import csv
import numpy as np

def read_transformation_matrices(file_path):
    """Read the transformation matrices from a CSV file and return a list of matrices."""
    matrices = []
    frame_numbers = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            frame = int(row[0])
            frame_numbers.append(frame)
            rotation_matrix = np.array(row[1:10]).reshape((3, 3)).astype(float)
            translation_vector = np.array(row[10:13]).astype(float)
            matrices.append((frame, rotation_matrix, translation_vector))

#   if frame_numbers:
#       print(f"Total number of frames: {len(frame_numbers)}")
#       print(f"First frame number: {frame_numbers[0]}")
#       print(f"Last frame number: {frame_numbers[-1]}")

    return matrices
