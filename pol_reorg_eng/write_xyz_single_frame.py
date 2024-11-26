def write_xyz_single_frame(filename, frame):
    """Write a single frame to an XYZ file."""
    symbols, coords = frame
    num_atoms = len(symbols)

    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"Frame\n")
        for symbol, coord in zip(symbols, coords):
            f.write(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
