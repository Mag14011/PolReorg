import numpy as np

def transform_and_save_fields(matrices, E_x, E_y, E_z, output_path, frame_indices=None):
    """Transform electric field vectors using provided rotation matrices.
    
    Args:
        matrices: List of tuples (frame_idx, rotation_matrix, translation) from superimpose_and_save
        E_x, E_y, E_z: Arrays of electric field components for each frame (V/Å)
        output_path: Path to save transformed field vectors
        frame_indices: Optional specific frames to process
        
    Returns:
        Tuple of (E_x_transformed, E_y_transformed, E_z_transformed) arrays
    """
    # Filter matrices if frame_indices provided
    if frame_indices is not None:
        matrices = [m for m in matrices if m[0] in frame_indices]

    # Initialize arrays for transformed fields
    num_frames = len(matrices)
    E_x_transformed = np.zeros(num_frames)
    E_y_transformed = np.zeros(num_frames)
    E_z_transformed = np.zeros(num_frames)
    
    # Process each frame
    for i, (frame_idx, R, _) in enumerate(matrices):
        # Create field vector and rotate it
        E_vec = np.array([E_x[frame_idx], E_y[frame_idx], E_z[frame_idx]])
        E_transformed = np.dot(R, E_vec)
        
        # Store transformed components
        E_x_transformed[i] = E_transformed[0]
        E_y_transformed[i] = E_transformed[1]
        E_z_transformed[i] = E_transformed[2]
        
        # Validate rotation preserved magnitude
        orig_mag = np.sqrt(np.sum(E_vec**2))
        new_mag = np.sqrt(np.sum(E_transformed**2))
        if not np.isclose(orig_mag, new_mag, rtol=1e-5):
            raise ValueError(f"Field magnitude not preserved in frame {frame_idx}: "
                           f"Original {orig_mag:.6f} ≠ New {new_mag:.6f}")
    
    # Calculate statistics
    magnitudes = np.sqrt(E_x_transformed**2 + E_y_transformed**2 + E_z_transformed**2)
    avg_mag = np.mean(magnitudes)
    avg_x = np.mean(E_x_transformed)
    avg_y = np.mean(E_y_transformed)
    avg_z = np.mean(E_z_transformed)
    
    std_mag = np.std(magnitudes)
    std_x = np.std(E_x_transformed)
    std_y = np.std(E_y_transformed)
    std_z = np.std(E_z_transformed)
    
    # Write transformed fields to file
    with open(output_path, 'w') as f:
        # Write header
        f.write('@    title "Rotated Electric Field"\n')
        f.write('@    xaxis  label "Frame"\n')
        f.write('@    yaxis  label "V/Å"\n')
        f.write('#frame    Magnitude          Efield_X            Efield_Y            Efield_Z\n')
        f.write('@type xy\n')
        
        # Write data
        for i, (frame_idx, _, _) in enumerate(matrices):
            mag = magnitudes[i]
            f.write(f'{frame_idx:<8} {mag:<18.6f} '
                   f'{E_x_transformed[i]:<18.6f} '
                   f'{E_y_transformed[i]:<18.6f} '
                   f'{E_z_transformed[i]:<18.6f}\n')
        
        # Write statistics
        f.write('#---#\n')
        f.write(f'#AVG:     {avg_mag:<18.6f} {avg_x:<18.6f} {avg_y:<18.6f} {avg_z:<18.6f}\n')
        f.write(f'#STDEV:   {std_mag:<18.6f} {std_x:<18.6f} {std_y:<18.6f} {std_z:<18.6f}\n')
    
    return E_x_transformed, E_y_transformed, E_z_transformed
