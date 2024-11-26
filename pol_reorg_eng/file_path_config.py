import os
import sys

def construct_simulation_paths(file_paths, reactant_donor_id, reactant_acceptor_id, product_donor_id, product_acceptor_id):
    """
    Construct simulation-specific file paths using base paths and IDs.
    """
    base_path = file_paths['base_path']

    simulation_paths = {
        'reactant_donor_xyz': f'{base_path}/r{reactant_donor_id}/r{reactant_donor_id}.xyz',
        'reactant_donor_efield': f'{base_path}/r{reactant_donor_id}/FE-HEH{reactant_donor_id}/ElecField.dat',
        'reactant_acceptor_xyz': f'{base_path}/r{reactant_donor_id}/o{reactant_acceptor_id}.xyz',
        'reactant_acceptor_efield': f'{base_path}/r{reactant_donor_id}/FE-HEH{reactant_acceptor_id}/ElecField.dat',
        'reactant_coulombic': f'{base_path}/r{reactant_donor_id}/Reorg/VEGf.dat',

        'product_donor_xyz': f'{base_path}/r{product_donor_id}/r{product_donor_id}.xyz',
        'product_donor_efield': f'{base_path}/r{product_donor_id}/FE-HEH{product_donor_id}/ElecField.dat',
        'product_acceptor_xyz': f'{base_path}/r{product_donor_id}/o{product_acceptor_id}.xyz',
        'product_acceptor_efield': f'{base_path}/r{product_donor_id}/FE-HEH{product_acceptor_id}/ElecField.dat',
        'product_coulombic': f'{base_path}/r{reactant_donor_id}/Reorg/VEGb.dat'
    }

    return {**file_paths, **simulation_paths}

def validate_simulation_paths(paths_dict):
    """
    Validate that all paths in the dictionary exist.

    Args:
        paths_dict (dict): Dictionary of paths to validate

    Raises:
        SystemExit: If any path does not exist
    """
    missing_files = []
    for key, path in paths_dict.items():
        if not os.path.exists(path):
            missing_files.append((key, path))

    if missing_files:
        print("\nError: The following required files are missing:")
        for key, path in missing_files:
            print(f"  {key}:")
            print(f"    Path: {path}")
            print(f"    Directory exists: {os.path.exists(os.path.dirname(path))}")
        print(f"\nCurrent working directory: {os.getcwd()}")
        sys.exit(1)

def read_file_paths(reactant_donor_id=None, reactant_acceptor_id=None,
                   product_donor_id=None, product_acceptor_id=None):
    """
    Read file paths from FilePaths.txt configuration file and validate all paths.
    If simulation IDs are provided, also validates simulation-specific paths.

    Args:
        reactant_donor_id (str, optional): ID of the reactant donor
        reactant_acceptor_id (str, optional): ID of the reactant acceptor
        product_donor_id (str, optional): ID of the product donor
        product_acceptor_id (str, optional): ID of the product acceptor

    Returns:
        dict: Dictionary containing all validated file paths
    """
    file_paths = {}
    config_file = "FilePaths.txt"

    # Convert config_file to absolute path relative to current working directory
    if not os.path.isabs(config_file):
        config_file = os.path.abspath(config_file)

    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at: '{config_file}'")
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = [part.strip() for part in line.split('=', 1)]
                    except ValueError:
                        print(f"Error: Invalid format in {config_file} at line {line_number}: '{line}'")
                        sys.exit(1)

                    # Convert to absolute path if it isn't already
                    if not os.path.isabs(value):
                        value = os.path.abspath(value)

                    # Normalize path (resolve any .. or . components)
                    value = os.path.normpath(value)

                    file_paths[key] = value

        # Validate required paths
        required_paths = [
            'red_ref_coords',
            'ox_ref_coords',
            'red_ref_tensor',
            'ox_ref_tensor',
            'base_path'
        ]

        missing_paths = [path for path in required_paths if path not in file_paths]
        if missing_paths:
            print(f"Error: Missing required paths in {config_file}: {', '.join(missing_paths)}")
            sys.exit(1)

        # First validate the base configuration paths
        print("\nValidating base configuration paths...")
        validate_simulation_paths(file_paths)
        print("Base configuration paths validated successfully.")

        # If simulation IDs are provided, validate simulation-specific paths
        if all(id is not None for id in [reactant_donor_id, reactant_acceptor_id,
                                       product_donor_id, product_acceptor_id]):
            print("\nValidating simulation-specific paths...")
            simulation_paths = construct_simulation_paths(
                file_paths,
                reactant_donor_id,
                reactant_acceptor_id,
                product_donor_id,
                product_acceptor_id
            )
            validate_simulation_paths(simulation_paths)
            print("Simulation-specific paths validated successfully.")
            return simulation_paths

        return file_paths

    except PermissionError:
        print(f"Error: Permission denied reading {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {config_file}: {str(e)}")
        sys.exit(1)
