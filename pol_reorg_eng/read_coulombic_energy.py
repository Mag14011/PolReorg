import numpy as np

def read_coulombic_energy(filename, common_indices):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()[1:]  # Skip the first line (header)
            coulombic_energy = np.array([float(line.split()[3]) for line in lines])[common_indices]
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        raise
    return coulombic_energy
