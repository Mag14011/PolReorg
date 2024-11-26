import numpy as np
import csv
from tqdm import tqdm

def read_xyz(file_path):
    """Read an XYZ file and return the atomic symbols and coordinates for all frames."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frames = []
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        symbols = []
        coords = []
        for line in lines[i + 2:i + 2 + num_atoms]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        frames.append((symbols, np.array(coords)))
        i += num_atoms + 2

    return frames

def write_xyz(file_path, symbols, coords, append=False):
    """Write atomic symbols and coordinates to an XYZ file."""
    mode = 'a' if append else 'w'
    with open(file_path, mode) as file:
        file.write(f"{len(symbols)}\n")
        file.write("Generated by Python script\n")
        for symbol, coord in zip(symbols, coords):
            coord_str = ' '.join(f"{x:.6f}" for x in coord)
            file.write(f"{symbol} {coord_str}\n")

def measure_fit(ref_coords, mob_coords):
    """Calculate the best-fit transformation matrix (rotation and translation) that aligns the mobile coordinates to the reference coordinates."""
    ref_coords = np.array(ref_coords)
    mob_coords = np.array(mob_coords)
    centroid_ref = np.mean(ref_coords, axis=0)
    centroid_mob = np.mean(mob_coords, axis=0)
    ref_centered = ref_coords - centroid_ref
    mob_centered = mob_coords - centroid_mob
    H = np.dot(mob_centered.T, ref_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_ref - np.dot(R, centroid_mob)
    return R, t

def compute_rmsd(coords1, coords2):
    """Compute the RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def save_transformation_to_csv(file_path, frame_idx, R, t):
    """Save the transformation matrix (rotation and translation) to a CSV file."""
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row = [frame_idx] + R.flatten().tolist() + t.tolist()
        writer.writerow(row)

def superimpose_and_save(ref_file, mob_file, output_file, rmsd_file, transformation_csv, frame_indices=None):
    """Superimpose mobile coordinates on reference coordinates and save the transformation matrix to a CSV file.

    Args:
        ref_file (str): Path to the reference XYZ file (single frame).
        mob_file (str): Path to the mobile XYZ file (multiple frames).
        output_file (str): Path to save the combined XYZ output.
        rmsd_file (str): Path to save the RMSD values.
        transformation_csv (str): Path to save the transformation matrices.
        frame_indices (array-like, optional): Specific mobile frame indices to process. If None, process all frames.

    Returns:
        str: Path to the CSV file containing transformation matrices.
    """
    # Read the reference and mobile coordinates
    ref_symbols, ref_coords = read_xyz(ref_file)[0]  # Take first (and only) frame
    mob_frames = read_xyz(mob_file)

    # Determine which mobile frames to process
    if frame_indices is not None:
        frames_to_process = [(idx, mob_frames[idx]) for idx in frame_indices if idx < len(mob_frames)]
        print(f"Processing {len(frames_to_process)} selected frames out of {len(mob_frames)} total frames...")
    else:
        frames_to_process = list(enumerate(mob_frames))
        print(f"Processing all {len(mob_frames)} frames...")

    # Prepare output files
    open(output_file, 'w').close()
    open(rmsd_file, 'w').close()
    with open(transformation_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'R11', 'R12', 'R13', 'R21', 'R22', 'R23', 
                        'R31', 'R32', 'R33', 't1', 't2', 't3'])

    # Process each selected mobile frame
    for frame_idx, (mob_symbols, mob_coords) in tqdm(frames_to_process, 
                                                    desc="Superimposing structures",
                                                    unit="frame"):
        # Calculate the best-fit transformation
        R, t = measure_fit(ref_coords, mob_coords)

        # Apply the transformation to the mobile coordinates
        transformed_coords = np.dot(mob_coords, R.T) + t

        # Compute the RMSD for this alignment
        rmsd = compute_rmsd(ref_coords, transformed_coords)

        # Write the combined coordinates to the output XYZ file
        combined_symbols = ref_symbols + mob_symbols
        combined_coords = np.vstack([ref_coords, transformed_coords])
        write_xyz(output_file, combined_symbols, combined_coords, append=True)

        # Write the RMSD to the RMSD output file
        with open(rmsd_file, 'a') as rmsd_out:
            rmsd_out.write(f"Frame {frame_idx}: RMSD = {rmsd:.6f}\n")

        # Save the transformation matrix and translation vector to the CSV file
        save_transformation_to_csv(transformation_csv, frame_idx, R, t)

    print("\nProcessing complete!")
    return transformation_csv

# Example usage:
if __name__ == "__main__":
    ref_file_path = "reference.xyz"
    mob_file_path = "mobile.xyz"
    output_xyz_path = "aligned_output.xyz"
    rmsd_output_path = "rmsd_output.txt"
    transformation_csv_path = "transformation_matrices.csv"
    
    # Example with specific mobile frames
    selected_frames = [0, 2, 4]  # Process only frames 0, 2, and 4 from mobile structure
    result_csv = superimpose_and_save(
        ref_file_path, mob_file_path, output_xyz_path, 
        rmsd_output_path, transformation_csv_path,
        frame_indices=selected_frames
    )
    print(f"Transformation matrices saved to: {result_csv}")