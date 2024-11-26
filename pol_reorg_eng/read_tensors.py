import numpy as np

def read_tensors(file_path):
    """Read the polarizability tensors from a file and return a list of tensors."""
    tensors = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.replace('|', ' ').split()
            tensor = np.array([
                [float(line[1]), float(line[2]), float(line[3])],
                [float(line[2]), float(line[4]), float(line[5])],
                [float(line[3]), float(line[5]), float(line[6])]
            ])
            tensors.append(tensor)
    return tensors
