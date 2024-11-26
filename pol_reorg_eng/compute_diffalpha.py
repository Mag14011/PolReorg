import numpy as np
import compute_mean_tensor as cmt

def compute_diffalpha(initial_state_filename, final_state_filename, basis_set_scaling, output_filename, frame_indices=None):
    """
    Compute difference in polarizability tensors between initial and final states.
    
    Args:
        initial_state_filename (str): Path to file containing initial state tensors
        final_state_filename (str): Path to file containing final state tensors
        basis_set_scaling (float): Scaling factor for basis set
        output_filename (str): Path to save output
        frame_indices (array-like, optional): Specific frame indices to process.
                                            If None, process all frames.
    """
    try:
        # Read the polarizability tensor data from the files
        initial_state_data = np.loadtxt(initial_state_filename, usecols=(1, 2, 3, 4, 5, 6))
        final_state_data = np.loadtxt(final_state_filename, usecols=(1, 2, 3, 4, 5, 6))

        # Print the shape to confirm the data dimensions
#       print(f"Initial state data shape: {initial_state_data.shape}")
#       print(f"Final state data shape: {final_state_data.shape}")
#       print(f"Selected frame indices shape: {frame_indices.shape}")
#       print(f"Max frame index: {frame_indices.max()}")

        # Filter data based on frame_indices if provided
        if frame_indices is not None:
            initial_state_data = initial_state_data[frame_indices]
            final_state_data = final_state_data[frame_indices]
            indices = frame_indices
        else:
            indices = np.arange(len(initial_state_data))

        # Print the selected indices to verify they are in range
#       print(f"Processing indices: {indices}")

        # Extract tensor components
        initial_alpha_xx = initial_state_data[:, 0]
        initial_alpha_xy = initial_state_data[:, 1]
        initial_alpha_xz = initial_state_data[:, 2]
        initial_alpha_yy = initial_state_data[:, 3]
        initial_alpha_yz = initial_state_data[:, 4]
        initial_alpha_zz = initial_state_data[:, 5]
        final_alpha_xx = final_state_data[:, 0]
        final_alpha_xy = final_state_data[:, 1]
        final_alpha_xz = final_state_data[:, 2]
        final_alpha_yy = final_state_data[:, 3]
        final_alpha_yz = final_state_data[:, 4]
        final_alpha_zz = final_state_data[:, 5]

        # Compute mean tensors
        initial_state_alpha = cmt.compute_mean_tensor(
            initial_alpha_xx, initial_alpha_xy, initial_alpha_xz, 
            initial_alpha_yy, initial_alpha_yz, initial_alpha_zz
        )
        final_state_alpha = cmt.compute_mean_tensor(
            final_alpha_xx, final_alpha_xy, final_alpha_xz, 
            final_alpha_yy, final_alpha_yz, final_alpha_zz
        )

        # Compute differences
        diffalpha_xx = basis_set_scaling * (final_alpha_xx - initial_alpha_xx)
        diffalpha_xy = basis_set_scaling * (final_alpha_xy - initial_alpha_xy)
        diffalpha_xz = basis_set_scaling * (final_alpha_xz - initial_alpha_xz)
        diffalpha_yy = basis_set_scaling * (final_alpha_yy - initial_alpha_yy)
        diffalpha_yz = basis_set_scaling * (final_alpha_yz - initial_alpha_yz)
        diffalpha_zz = basis_set_scaling * (final_alpha_zz - initial_alpha_zz)
        
        mean_diff_alpha = cmt.compute_mean_tensor(
            diffalpha_xx, diffalpha_xy, diffalpha_xz, 
            diffalpha_yy, diffalpha_yz, diffalpha_zz
        )

        # Write output file
        with open(output_filename, 'w') as f:
            f.write("Index, Initial Alpha (xx xy xz yy yz zz), Final Alpha (xx xy xz yy yz zz), Diff Alpha (xx xy xz yy yz zz)\n")
            for i, idx in enumerate(indices):
                f.write(
                    f"{idx}, "
                    f"{initial_alpha_xx[i]:.6f} {initial_alpha_xy[i]:.6f} {initial_alpha_xz[i]:.6f} "
                    f"{initial_alpha_yy[i]:.6f} {initial_alpha_yz[i]:.6f} {initial_alpha_zz[i]:.6f}, "
                    f"{final_alpha_xx[i]:.6f} {final_alpha_xy[i]:.6f} {final_alpha_xz[i]:.6f} "
                    f"{final_alpha_yy[i]:.6f} {final_alpha_yz[i]:.6f} {final_alpha_zz[i]:.6f}, "
                    f"{diffalpha_xx[i]:.6f} {diffalpha_xy[i]:.6f} {diffalpha_xz[i]:.6f} "
                    f"{diffalpha_yy[i]:.6f} {diffalpha_yz[i]:.6f} {diffalpha_zz[i]:.6f}\n"
                )

    except FileNotFoundError:
        print(f"Error: File {initial_state_filename} or {final_state_filename} not found.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    return (initial_state_alpha, final_state_alpha, mean_diff_alpha,
            diffalpha_xx, diffalpha_xy, diffalpha_xz,
            diffalpha_yy, diffalpha_yz, diffalpha_zz,
            indices)
