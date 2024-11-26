def generate_template_paths(output_file="temp_FilePaths.txt"):
    """
    Generate a template file paths configuration file.
    
    Args:
        output_file (str): Name of the output template file
    """
    template_content = """# Reference coordinates and polarizability tensors
red_ref_coords        = /path/to/reference/reduced.xyz
ox_ref_coords         = /path/to/reference/oxidized.xyz
red_ref_tensor        = /path/to/reference/reduced_tensor.dat
ox_ref_tensor         = /path/to/reference/oxidized_tensor.dat

# Simulation XYZ files
reactant_donor_xyz    = /path/to/trajectory/reactant_donor.xyz
reactant_acceptor_xyz = /path/to/trajectory/reactant_acceptor.xyz
product_donor_xyz     = /path/to/trajectory/product_donor.xyz
product_acceptor_xyz  = /path/to/trajectory/product_acceptor.xyz

# Coulombic State Energies
reactant_coulombic    = /path/to/energies/reactant_coulombic.dat
product_coulombic     = /path/to/energies/product_coulombic.dat"""

    try:
        with open(output_file, 'w') as f:
            f.write(template_content)
        print(f"\nTemplate file created: {output_file}")
        print("Edit this file with your actual file paths before running the analysis.")
        print("** You MUST rename the file FilePaths.txt.")
    except Exception as e:
        print(f"Error creating template file: {e}")
        sys.exit(1)

