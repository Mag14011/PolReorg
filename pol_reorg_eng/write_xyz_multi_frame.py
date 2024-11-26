def write_xyz_multi_frame(filename, sim_frames, ref_frames):
    """Write simulation structure and transformed reference structure to a multi-frame XYZ file."""
    with open(filename, 'w') as f:
        for sim_frame, ref_frame in zip(sim_frames, ref_frames):
            sim_symbols, sim_coords = sim_frame
            ref_symbols, ref_coords = ref_frame
            num_atoms = len(sim_symbols) + len(ref_symbols)

            f.write(f"{num_atoms}\n")
            f.write(f"Frame\n")

            for symbol, coord in zip(sim_symbols, sim_coords):
                f.write(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")

            for symbol, coord in zip(ref_symbols, ref_coords):
                f.write(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
