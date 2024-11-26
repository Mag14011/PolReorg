import re
import subprocess

def get_transformation_matrix_vmd(ref_xyz, sim_xyz, ref_indices, sim_indices, tcl_script="measure_fit.tcl"):
    """Retrieve the transformation matrix using VMD for given indices of reference and simulation structures."""
    ref_indices_space = ' '.join(ref_indices.split(','))
    sim_indices_space = ' '.join(sim_indices.split(','))

    tcl_content = f"""
    mol new {ref_xyz} type xyz
    mol new {sim_xyz} type xyz
    set ref_structure [atomselect 0 "index {ref_indices_space}"]
    set sim_structure [atomselect 1 "index {sim_indices_space}"]
    set trans_mat [measure fit $ref_structure $sim_structure]
    quit
    """

    with open(tcl_script, 'w') as f:
        f.write(tcl_content)

    result = subprocess.run(['vmd', '-dispdev', 'text', '-e', tcl_script],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output = result.stdout
    lines = output.splitlines()
    matrix = []

    matrix_lines = re.findall(r'\{([^\}]*)\}', output)
    for line in matrix_lines:
        try:
            elements = list(map(float, line.split()))
            matrix.append(elements)
        except ValueError as e:
            print(f"Error parsing line '{line}': {e}")

    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Transformation matrix could not be parsed correctly.")

    rotation_matrix = [matrix[0][:3], matrix[1][:3], matrix[2][:3]]
    translation_vector = [matrix[0][3], matrix[1][3], matrix[2][3]]

    flat_matrix = [
        rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2],
        rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2],
        rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2],
        translation_vector[0], translation_vector[1], translation_vector[2]
    ]

    return rotation_matrix, translation_vector
