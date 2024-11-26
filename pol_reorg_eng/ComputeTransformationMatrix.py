import csv
import read_xyz as rxyz
import write_xyz_single_frame as wsf
import get_transformation_matrix_vmd as gtmvmd
import apply_coord_transformation as act
import calculate_rmsd as crmsd
import write_xyz_multi_frame as wmf

def ComputeTransformationMatrix(refXYZ, simXYZ, ref_indices, sim_indices, folder_path, transfmdXYZ, transMat, rmsdFile):
    """Process simulation XYZ files by reading, superimposing, and writing the results."""
    frames_a = rxyz.read_xyz(simXYZ)
    frames_b = rxyz.read_xyz(refXYZ)

    print(f""" 
   Frame Lengths
    Ref. coords = {len(frames_b)} (file = {refXYZ})
    Sim. coords = {len(frames_a)} (file = {simXYZ})
""")

    if len(frames_b) != 1:
        raise ValueError(f"Reference file {refXYZ} has {len(frames_b)} frames; must contain exactly one frame.")

    ref_symbols, ref_coords = frames_b[0]
    num_frames = len(frames_a)

    transformed_frames = []
    transformation_matrices = []
    rmsd_values = []

    for i in range(num_frames):
        print(f"    Processing frame {i} ...")

        symbols_a, coords_a = frames_a[i]
        temp_sim_filename = f"temp_sim_frame_{i}.xyz"
        wsf.write_xyz_single_frame(temp_sim_filename, (symbols_a, coords_a))

        rotation_matrix, translation_vector = gtmvmd.get_transformation_matrix_vmd(refXYZ, temp_sim_filename, ref_indices, sim_indices)

#       print(rotation_matrix)
#       print(translation_vector)

        transformed_ref_coords = act.apply_coord_transformation(ref_coords, rotation_matrix, translation_vector)
        final_rmsd = crmsd.calculate_rmsd(coords_a, transformed_ref_coords)

        transformed_frames.append((ref_symbols, transformed_ref_coords))
        transformation_matrices.append((i, rotation_matrix, translation_vector))
        rmsd_values.append(final_rmsd)

    with open(f"{folder_path}/{rmsdFile}", "w") as rmsd_file:
        for i, rmsd in enumerate(rmsd_values):
            rmsd_file.write(f"Frame {i}: RMSD = {rmsd:.8f}\n")

    max_rmsd = max(rmsd_values)
    print(f"   Maximum RMSD: {max_rmsd:.8f}")

    wmf.write_xyz_multi_frame(f"{folder_path}/{transfmdXYZ}", frames_a, transformed_frames)
    
    with open(f"{folder_path}/{transMat}", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33", "Tx", "Ty", "Tz"])
        for i, (_, rotation_matrix, translation_vector) in enumerate(transformation_matrices):
            writer.writerow([
                i,
                rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2],
                rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2],
                rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2],
                translation_vector[0], translation_vector[1], translation_vector[2]
            ])
    
    return f"{folder_path}/{transMat}"
