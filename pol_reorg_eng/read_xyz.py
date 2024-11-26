import numpy as np

def read_xyz(file_path):
    """ Read multi-frame XYZ file and return list of atom symbols and coordinates for each frame. """
    frames = []
    with open(file_path, 'r') as file:
        while True:
            # Read the number of atoms
            line = file.readline().strip()
            if not line:  # End of file
                break
            
            try:
                atom_count = int(line)
            except ValueError:
                break  # Exit if we encounter a non-integer line where we expect an atom count
            
            # Read the comment line
            comment = file.readline().strip()

            # Read the atomic symbols and coordinates
            symbols = []
            coords = []
            for _ in range(atom_count):
                line = file.readline().strip().split()
                if len(line) < 4:
                    break  # Exit loop if the line does not contain valid data
                symbols.append(line[0])
                coords.append([float(x) for x in line[1:]])

            if symbols and coords:
                frames.append((symbols, np.array(coords)))
    
    return frames
