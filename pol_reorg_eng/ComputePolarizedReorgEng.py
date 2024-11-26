import os 
os.environ['NUMEXPR_MAX_THREADS'] = '64' 
import re
import sys 
import csv
import tempfile
import subprocess
import argparse
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

ProgDir = sys.path[0]
if (os.path.exists(f"{ProgDir}/Modules") == True): 
    sys.path.append(f"{ProgDir}/Modules")

import generate_template_paths as gtp
import run_analysis as ra
from run_analysis import construct_metadata
import save_output_file as sof
from plot_efields import create_field_analysis_plot
import plot_efield_dependent_reorganization_energy as pefdre
import plot_energy_decomposition as ped

import create_output_folder as cof
import format_energy_analysis_output as feao

def generate_folder_name(reactant_donor_id, reactant_acceptor_id, use_synthetic, **kwargs):
    """
    Generate a standardized folder name based on calculation parameters.

    Args:
        reactant_donor_id (str): ID of reactant donor
        reactant_acceptor_id (str): ID of reactant acceptor
        use_synthetic (bool): Whether using synthetic fields
        **kwargs: Additional parameters depending on calculation type
            For MD fields:
                filter_type (str, optional): Type of filter (threshold, variance, exact)
                filter_value (float, optional): Filter value
                filter_tolerance (float, optional): Tolerance for exact filter
                iterative_run (bool): Whether this is an iterative run
                increment (float): Increment value for iteration
            For synthetic fields:
                field_approach (str): 'direction_from_md' or 'heme_based'
                iterative_run (bool): Whether this is an iterative run
                iterate_position (str): Position to iterate over or 'all'
                increment (float): Increment value for iteration
                For each position (rd, ra, pd, pa):
                    {pos}_distribution (str): Distribution type
                    {pos}_magnitude (float): Field magnitude
                    {pos}_std_dev (float, optional): Standard deviation for gaussian
                    {pos}_range (float, optional): Range for uniform
                    {pos}_direction (str, optional): Direction for heme-based

    Returns:
        str: Standardized folder name
    """
    base = f"{reactant_donor_id}_{reactant_acceptor_id}"

    # MD-derived fields
    if not use_synthetic:
        if not kwargs.get('filter_type'):
            return f"{base}_md_nofilter"

        filter_type = kwargs['filter_type']
        filter_value = kwargs['filter_value']

        if not kwargs.get('iterative_run'):
            # Single run with filter
            if filter_type == 'exact':
                filter_tol = kwargs.get('filter_tolerance', 0.01)
                return f"{base}_md_{filter_type}_{filter_value:.2f}_tol{filter_tol:.2f}"
            return f"{base}_md_{filter_type}_{filter_value:.2f}"
        else:
            # Iterative run with filter
            increment = kwargs['increment']
            return f"{base}_md_{filter_type}_iter_0.00to{filter_value:.2f}_inc{increment:.2f}"

    # Synthetic fields
    field_approach = 'md' if kwargs['field_approach'] == 'direction_from_md' else 'heme'

    def format_position_params(pos, include_direction=False):
        """Helper function to format parameters for a position"""
        dist = kwargs[f'{pos}_distribution']
        mag = kwargs[f'{pos}_magnitude']

        if dist == 'gaussian':
            std = kwargs.get(f'{pos}_std_dev', 0)
            param = f"{mag:.2f}s{std:.2f}"
        elif dist == 'uniform':
            range_val = kwargs.get(f'{pos}_range', 0)
            param = f"{mag:.2f}r{range_val:.2f}"
        else:  # fixed
            param = f"{mag:.2f}"

        base_str = f"{pos}-{dist}{param}"
        if include_direction and field_approach == 'heme':
            direction = kwargs.get(f'{pos}_direction', 'xx')
            return f"{base_str}_{direction}"
        return base_str

    if not kwargs.get('iterative_run'):
        # Single run
        positions = ['rd', 'ra', 'pd', 'pa']
        pos_params = [format_position_params(pos, True) for pos in positions]
        return f"{base}_syn_{field_approach}_{'_'.join(pos_params)}"

    # Iterative run
    increment = kwargs['increment']
    if kwargs['iterate_position'] == 'all':
        # All positions
        distributions = [kwargs[f'{pos}_distribution'] for pos in ['rd', 'ra', 'pd', 'pa']]
        base_name = f"{base}_syn_{field_approach}_iter_all_{'_'.join(distributions)}_0.00to"

        # Get maximum magnitude across all positions
        max_mag = max(kwargs[f'{pos}_magnitude'] for pos in ['rd', 'ra', 'pd', 'pa'])

        if field_approach == 'heme':
            directions = [kwargs[f'{pos}_direction'] for pos in ['rd', 'ra', 'pd', 'pa']]
            return f"{base_name}{max_mag:.2f}_inc{increment:.2f}_{'_'.join(directions)}"
        return f"{base_name}{max_mag:.2f}_inc{increment:.2f}"

    # Single position iteration
    pos = kwargs['iterate_position']
    dist = kwargs[f'{pos}_distribution']
    max_mag = kwargs[f'{pos}_magnitude']

    if field_approach == 'heme':
        direction = kwargs[f'{pos}_direction']
        return f"{base}_syn_{field_approach}_iter_{pos}-{dist}_0.00to{max_mag:.2f}_inc{increment:.2f}_{direction}"
    return f"{base}_syn_{field_approach}_iter_{pos}-{dist}_0.00to{max_mag:.2f}_inc{increment:.2f}"

if __name__ == "__main__":
    print("""
 #######################################################################################
 # Assessment of Active Site Polarizability on Electron Transfer Reorganization Energy #
 #                                     Version 2                                       #
 #                         Written by Matthew J. Guberman-Pfeffer                      #
 #                             Last Modified: 11/09/2024                               #
 #######################################################################################
 """)

###################################################################################################
    # Input parameters
    parser = argparse.ArgumentParser(
        description='''The script accepts both command-line arguments and interactive input.

    Example Commands:
    ---------------
    1. MD-derived fields without filtering:
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields no 
            --set_efield_filter no

    2. MD-derived fields with filtering (single run):
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields no 
            --set_efield_filter yes 
            --filter_type threshold 
            --filter_value 0.5

    3. MD-derived fields with filtering (iterative):
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields no 
            --set_efield_filter yes 
            --filter_type threshold 
            --filter_value 0.5 
            --iterative_run yes 
            --increment 0.1

    4. Independent synthetic fields using heme-based coordinates (single run):
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields yes 
            --field_approach heme_based 
            --donor_n_pair1 "0 2"      
            --donor_n_pair2 "1 3"      
            --acceptor_n_pair1 "0 2"   
            --acceptor_n_pair2 "1 3"   
            --reactant_donor_distribution fixed 
            --reactant_donor_direction xx 
            --reactant_donor_magnitude 0.5
            --reactant_acceptor_distribution fixed 
            --reactant_acceptor_direction xy 
            --reactant_acceptor_magnitude 0.3
            --product_donor_distribution fixed 
            --product_donor_direction xz 
            --product_donor_magnitude 0.4
            --product_acceptor_distribution fixed 
            --product_acceptor_direction yy 
            --product_acceptor_magnitude 0.2

    5. Independent synthetic fields with MD-derived directions and fixed magnitudes (single run):
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields yes 
            --field_approach direction_from_md 
            --reactant_donor_distribution fixed 
            --reactant_donor_magnitude 0.5
            --reactant_acceptor_distribution fixed 
            --reactant_acceptor_magnitude 0.3
            --product_donor_distribution fixed 
            --product_donor_magnitude 0.4
            --product_acceptor_distribution fixed 
            --product_acceptor_magnitude 0.2

    6. Independent synthetic fields with MD-derived directions and gaussian-distributed magnitudes (single run):
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            -basis_set_scaling 1.0 
            --use_synthetic_fields yes 
            --field_approach direction_from_md 
            --reactant_donor_distribution gaussian 
            --reactant_donor_magnitude 0.5 
            --reactant_donor_std_dev 0.1 
            --reactant_acceptor_distribution gaussian 
            --reactant_acceptor_magnitude 0.3 
            --reactant_acceptor_std_dev 0.05 
            --product_donor_distribution gaussian 
            --product_donor_magnitude 0.4 
            --product_donor_std_dev 0.08 
            --product_acceptor_distribution gaussian 
            --product_acceptor_magnitude 0.2 
            --product_acceptor_std_dev 0.04

    7. Independent synthetic fields with MD-derived directions and gaussian-distributed magnitudes (single run):
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields yes\
            --field_approach direction_from_md 
            --reactant_donor_distribution uniform 
            --reactant_donor_magnitude 0.5 
            --reactant_donor_range 0.2 
            --reactant_acceptor_distribution uniform 
            --reactant_acceptor_magnitude 0.3 
            --reactant_acceptor_range 0.1 
            --product_donor_distribution uniform 
            --product_donor_magnitude 0.4 
            --product_donor_range 0.15 
            --product_acceptor_distribution uniform
            --product_acceptor_magnitude 0.2 
            --product_acceptor_range 0.08

    8. Independent synthetic fields with sequential iteration:
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields yes 
            --field_approach direction_from_md 
            --reactant_donor_distribution fixed 
            --reactant_donor_magnitude 0.5
            --reactant_acceptor_distribution fixed 
            --reactant_acceptor_magnitude 0.3
            --product_donor_distribution fixed 
            --product_donor_magnitude 0.4
            --product_acceptor_distribution fixed 
            --product_acceptor_magnitude 0.2
            --iterative_run yes 
            --iterate_position reactant_donor 
            --increment 0.1

    9. Independent synthetic fields with sequential iteration over all positions:
        python main.py 
            --reactant_donor_id 123 
            --reactant_acceptor_id 456 
            --basis_set_scaling 1.0 
            --use_synthetic_fields yes 
            --field_approach direction_from_md 
            --reactant_donor_distribution fixed 
            --reactant_donor_magnitude 0.5
            --reactant_acceptor_distribution fixed 
            --reactant_acceptor_magnitude 0.3
            --product_donor_distribution fixed 
            --product_donor_magnitude 0.4
            --product_acceptor_distribution fixed 
            --product_acceptor_magnitude 0.2
            --iterative_run yes 
            --iterate_position all 
            --increment 0.1''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Basic arguments

    parser.add_argument('--generate-template', action='store_true',
                   help='Generate a template file paths configuration file')

    parser.add_argument('--reactant_donor_id', type=str, 
                    help='Residue ID of the reactant donor')
    parser.add_argument('--reactant_acceptor_id', type=str, 
                    help='Residue ID of the reactant acceptor')
    parser.add_argument('--basis_set_scaling', type=float, 
                    help='Basis set scaling factor for the difference polarizability tensors')

    # Electric field source arguments
    parser.add_argument('--use_synthetic_fields', type=str, choices=['yes', 'no'], 
                    help='Use synthetic electric fields instead of MD fields')

    # Filter arguments (used only for MD fields)
    parser.add_argument('--set_efield_filter', type=str, choices=['yes', 'no'], 
                    help="Filter the electric field? (for MD fields only)")
    parser.add_argument('--filter_type', type=str, 
                    choices=['threshold', 'variance', 'exact'], 
                    help="Filter type for MD fields: 'threshold', 'variance', or 'exact'")
    parser.add_argument('--filter_value', type=float, 
                    help="Filter value for MD fields")
    parser.add_argument('--filter_tolerance', type=float, 
                    help="Tolerance for exact magnitude filtering of MD fields")

    # Synthetic field approach arguments (for synthetic fields)
    field_approach_group = parser.add_mutually_exclusive_group()
    field_approach_group.add_argument('--field_approach', type=str,
                    choices=['direction_from_md', 'heme_based'],
                    help='Approach for synthetic field generation')

    # Heme-based synthetic field approach specific arguments (shared across all positions)
    heme_based_group = parser.add_argument_group('heme_based approach arguments')
    heme_based_group.add_argument('--donor_n_pair1', type=str,
                    help='Space-separated indices for donor N pair 1 (e.g., "0 2")')
    heme_based_group.add_argument('--donor_n_pair2', type=str,
                    help='Space-separated indices for donor N pair 2 (e.g., "1 3")')
    heme_based_group.add_argument('--acceptor_n_pair1', type=str,
                    help='Space-separated indices for acceptor N pair 1 (e.g., "0 2")')
    heme_based_group.add_argument('--acceptor_n_pair2', type=str,
                    help='Space-separated indices for acceptor N pair 2 (e.g., "1 3")')

    # Reactant donor synthetic field parameters
    reactant_donor_group = parser.add_argument_group('reactant donor field parameters')
    reactant_donor_group.add_argument('--reactant_donor_distribution', type=str, 
                    choices=['fixed', 'gaussian', 'uniform'],
                    help='Distribution type for reactant donor synthetic fields')
    reactant_donor_group.add_argument('--reactant_donor_direction', type=str,
                    choices=['xx', 'xy', 'xz', 'yy', 'yz', 'zz', 'random'],
                    help='Direction for reactant donor synthetic fields (heme_based only)')
    reactant_donor_group.add_argument('--reactant_donor_magnitude', type=float,
                    help='Magnitude (fixed) or mean magnitude (gaussian/uniform) for reactant donor')
    reactant_donor_group.add_argument('--reactant_donor_std_dev', type=float,
                    help='Standard deviation for reactant donor gaussian distribution')
    reactant_donor_group.add_argument('--reactant_donor_range', type=float,
                    help='Range width for reactant donor uniform distribution')

    # Reactant acceptor synthetic field parameters
    reactant_acceptor_group = parser.add_argument_group('reactant acceptor field parameters')
    reactant_acceptor_group.add_argument('--reactant_acceptor_distribution', type=str, 
                    choices=['fixed', 'gaussian', 'uniform'],
                    help='Distribution type for reactant acceptor synthetic fields')
    reactant_acceptor_group.add_argument('--reactant_acceptor_direction', type=str,
                    choices=['xx', 'xy', 'xz', 'yy', 'yz', 'zz', 'random'],
                    help='Direction for reactant acceptor synthetic fields (heme_based only)')
    reactant_acceptor_group.add_argument('--reactant_acceptor_magnitude', type=float,
                    help='Magnitude (fixed) or mean magnitude (gaussian/uniform) for reactant acceptor')
    reactant_acceptor_group.add_argument('--reactant_acceptor_std_dev', type=float,
                    help='Standard deviation for reactant acceptor gaussian distribution')
    reactant_acceptor_group.add_argument('--reactant_acceptor_range', type=float,
                    help='Range width for reactant acceptor uniform distribution')

    # Product donor synthetic field parameters
    product_donor_group = parser.add_argument_group('product donor field parameters')
    product_donor_group.add_argument('--product_donor_distribution', type=str, 
                    choices=['fixed', 'gaussian', 'uniform'],
                    help='Distribution type for product donor synthetic fields')
    product_donor_group.add_argument('--product_donor_direction', type=str,
                    choices=['xx', 'xy', 'xz', 'yy', 'yz', 'zz', 'random'],
                    help='Direction for product donor synthetic fields (heme_based only)')
    product_donor_group.add_argument('--product_donor_magnitude', type=float,
                    help='Magnitude (fixed) or mean magnitude (gaussian/uniform) for product donor')
    product_donor_group.add_argument('--product_donor_std_dev', type=float,
                    help='Standard deviation for product donor gaussian distribution')
    product_donor_group.add_argument('--product_donor_range', type=float,
                    help='Range width for product donor uniform distribution')

    # Product acceptor synthetic field parameters
    product_acceptor_group = parser.add_argument_group('product acceptor field parameters')
    product_acceptor_group.add_argument('--product_acceptor_distribution', type=str, 
                    choices=['fixed', 'gaussian', 'uniform'],
                    help='Distribution type for product acceptor synthetic fields')
    product_acceptor_group.add_argument('--product_acceptor_direction', type=str,
                    choices=['xx', 'xy', 'xz', 'yy', 'yz', 'zz', 'random'],
                    help='Direction for product acceptor synthetic fields (heme_based only)')
    product_acceptor_group.add_argument('--product_acceptor_magnitude', type=float,
                    help='Magnitude (fixed) or mean magnitude (gaussian/uniform) for product acceptor')
    product_acceptor_group.add_argument('--product_acceptor_std_dev', type=float,
                    help='Standard deviation for product acceptor gaussian distribution')
    product_acceptor_group.add_argument('--product_acceptor_range', type=float,
                    help='Range width for product acceptor uniform distribution')

    # Iteration arguments 
    parser.add_argument('--iterative_run', type=str, choices=['yes', 'no'], 
                    help="Run analysis iteratively")
    parser.add_argument('--iterate_position', type=str,
                    choices=['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor', 'all'],
                    help="Position to iterate over for synthetic fields (required if iterative_run is yes)")
    parser.add_argument('--increment', type=float, 
                    help="Size of increment for iterative analysis")

    args = parser.parse_args()

    if args.generate_template:
        gtp.generate_template_paths()
        sys.exit(0)

    def get_input(prompt, cast=str, choices=None):
        while True:
            value = input(prompt).strip().lower()
            if choices and value not in choices:
                print(f"Invalid input. Please enter one of {choices}.")
            else:
                try:
                    return cast(value)
                except ValueError:
                    print(f"Invalid input. Please enter a {cast.__name__}.")

    def get_field_params(position_name, args_dict, field_approach):
        """Helper function to collect field parameters for a specific position
        
        Args:
            position_name (str): Name of the position (e.g., 'reactant_donor')
            args_dict (dict): Dictionary of command line arguments
            field_approach (str): The field generation approach being used
            
        Returns:
            dict: Field parameters for the specified position
        """
        prefix = f"{position_name}_"
    
        # Get distribution
        distribution = getattr(args_dict, f"{prefix}distribution")
        if not distribution:
            raise ValueError(f"Distribution must be specified for {position_name}")
        
        # Get magnitude
        magnitude = getattr(args_dict, f"{prefix}magnitude")
        if magnitude is None:
            raise ValueError(f"Magnitude must be specified for {position_name}")
    
        # Get direction for heme-based approach
        direction = None
        if field_approach == 'heme_based':
            direction = getattr(args_dict, f"{prefix}direction")
            if not direction:
                raise ValueError(f"Direction must be specified for {position_name} when using heme-based approach")
    
        params = {
            'approach': field_approach,  # Add the approach to each position's parameters
            'distribution': distribution,
            'magnitude': float(magnitude),
        }
    
        # Add direction if using heme-based approach
        if field_approach == 'heme_based':
            params['direction'] = direction
    
        # Add optional parameters based on distribution
        if distribution == 'gaussian':
            std_dev = getattr(args_dict, f"{prefix}std_dev")
            if std_dev is not None:
                params['std_dev'] = float(std_dev)
        elif distribution == 'uniform':
            range_width = getattr(args_dict, f"{prefix}range")
            if range_width is not None:
                params['range_width'] = float(range_width)
            
        return params

    # Get required input parameters
    reactant_donor_id = args.reactant_donor_id or input("Enter the residue ID of the reactant donor: ")
    reactant_acceptor_id = args.reactant_acceptor_id or input("Enter the residue ID of the reactant acceptor: ")
    basis_set_scaling = args.basis_set_scaling or float(input("Enter the basis set scaling factor for the difference polarizability tensors: "))

    # Set product IDs based on reactant IDs
    product_donor_id = reactant_acceptor_id
    product_acceptor_id = reactant_donor_id

    # Determine whether to use synthetic fields
    use_synthetic = args.use_synthetic_fields or get_input(
        "Do you want to use synthetic electric fields? (yes/no): ",
        choices=['yes', 'no'])

    if use_synthetic == 'yes':
        # Get field generation approach
        field_approach = args.field_approach or get_input(
            "Enter field generation approach (direction_from_md/heme_based): ",
            choices=['direction_from_md', 'heme_based'])
        
        # Get N pairs for heme-based approach
        donor_n_pair1 = None
        donor_n_pair2 = None
        acceptor_n_pair1 = None
        acceptor_n_pair2 = None
        if field_approach == 'heme_based':
            def parse_pair(pair_str):
                try:
                    if pair_str is None:
                        return None
                    idx1, idx2 = map(int, pair_str.split())
                    if idx1 < 0 or idx2 < 0:  # Only check that indices are non-negative
                        raise ValueError
                    return (idx1, idx2)
                except ValueError:
                    return None

            # Get donor N pairs
            donor_n_pair1 = parse_pair(args.donor_n_pair1)
            while donor_n_pair1 is None:
                pair_input = input("Enter indices for donor N pair 1 (two space-separated numbers 0-3, e.g., '0 2'): ")
                donor_n_pair1 = parse_pair(pair_input)
                if donor_n_pair1 is None:
                    print("Invalid input. Please enter two space-separated integers between 0 and 3.")

            donor_n_pair2 = parse_pair(args.donor_n_pair2)
            while donor_n_pair2 is None:
                pair_input = input("Enter indices for donor N pair 2 (two space-separated numbers 0-3, e.g., '1 3'): ")
                donor_n_pair2 = parse_pair(pair_input)
                if donor_n_pair2 is None:
                    print("Invalid input. Please enter two space-separated integers between 0 and 3.")

            # Get acceptor N pairs
            acceptor_n_pair1 = parse_pair(args.acceptor_n_pair1)
            while acceptor_n_pair1 is None:
                pair_input = input("Enter indices for acceptor N pair 1 (two space-separated numbers 0-3, e.g., '0 2'): ")
                acceptor_n_pair1 = parse_pair(pair_input)
                if acceptor_n_pair1 is None:
                    print("Invalid input. Please enter two space-separated integers between 0 and 3.")

            acceptor_n_pair2 = parse_pair(args.acceptor_n_pair2)
            while acceptor_n_pair2 is None:
                pair_input = input("Enter indices for acceptor N pair 2 (two space-separated numbers 0-3, e.g., '1 3'): ")
                acceptor_n_pair2 = parse_pair(pair_input)
                if acceptor_n_pair2 is None:
                    print("Invalid input. Please enter two space-separated integers between 0 and 3.")

        # Get field parameters for each position
        synthetic_params = {
            'approach': field_approach,
            'donor_n_pair1': donor_n_pair1 if field_approach == 'heme_based' else None,
            'donor_n_pair2': donor_n_pair2 if field_approach == 'heme_based' else None,
            'acceptor_n_pair1': acceptor_n_pair1 if field_approach == 'heme_based' else None,
            'acceptor_n_pair2': acceptor_n_pair2 if field_approach == 'heme_based' else None,
            'reactant_donor': get_field_params('reactant_donor', args, field_approach),
            'reactant_acceptor': get_field_params('reactant_acceptor', args, field_approach),
            'product_donor': get_field_params('product_donor', args, field_approach),
            'product_acceptor': get_field_params('product_acceptor', args, field_approach)
        }
        
        # Set filter parameters to None for synthetic fields
        set_filter = 'no'
        filter_type = None
        filter_value = None
        filter_tolerance = None
        
    else:
        # Original MD field filtering logic
        synthetic_params = None
        set_filter = args.set_efield_filter or get_input(
            "Do you want to filter the electric field? (yes/no): ",
            choices=['yes', 'no'])
        
        if set_filter == 'yes':
            filter_type = args.filter_type or get_input(
                "Enter filter type ('threshold', 'variance', or 'exact'): ",
                choices=['threshold', 'variance', 'exact'])
            filter_value = args.filter_value
            if filter_value is None:
                filter_value = float(input("Enter the filter value: "))
            
            if filter_type == 'exact':
                filter_tolerance = args.filter_tolerance
                if filter_tolerance is None:
                    filter_tolerance = float(input("Enter the tolerance for exact magnitude filtering: "))
            else:
                filter_tolerance = None
        else:
            filter_type = None
            filter_value = None
            filter_tolerance = None

    # Get iteration parameters
    iterative_run = args.iterative_run or get_input(
        "Do you want to iteratively run the analysis? (yes/no): ",
        choices=['yes', 'no'])

    if iterative_run == 'yes':
        if use_synthetic == 'yes':
            # Get position to iterate for synthetic fields
            iterate_position = args.iterate_position or get_input(
                "Enter position to iterate over (reactant_donor/reactant_acceptor/product_donor/product_acceptor/all): ",
                choices=['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor', 'all'])
        else:
            # For MD fields, position iteration doesn't apply
            iterate_position = None
            
        increment = args.increment
        if increment is None:
            if use_synthetic == 'yes':
                increment = float(input("Enter the size of the increment for field magnitude: "))
            else:
                increment = float(input("Enter the size of the increment for filter value: "))
    else:
        iterate_position = None
        increment = None
###################################################################################################
    # Review input parameters
    print("\n=== Parameter Review ===")
    print("\nCore Parameters:")
    print(f"  Reactant donor ID:    {reactant_donor_id}")
    print(f"  Reactant acceptor ID: {reactant_acceptor_id}")
    print(f"  Product donor ID:     {product_donor_id}")
    print(f"  Product acceptor ID:  {product_acceptor_id}")
    print(f"  Basis set scaling:    {basis_set_scaling}")

    def print_field_params(position_name, params):
        """Helper function to print field parameters for a position"""
        print(f"\n{position_name.replace('_', ' ').title()} Field Parameters:")
        print(f"  Distribution:     {params['distribution']}")
        if params.get('direction') is not None:
            print(f"  Direction:        {params['direction']}")
        print(f"  Magnitude:        {params['magnitude']:.3f} V/Å")
        if params.get('std_dev') is not None:
            print(f"  Std deviation:    {params['std_dev']:.3f}")
        if params.get('range_width') is not None:
            print(f"  Range width:      {params['range_width']:.3f}")

    if use_synthetic == 'yes':
        print("\nField Generation Settings:")
        print(f"  Approach:          {synthetic_params['approach']}")

        if synthetic_params['approach'] == 'heme_based':
            print("  N pairs:")
            print("  Donor:")
            print(f"    Pair 1:         {synthetic_params['donor_n_pair1']}")
            print(f"    Pair 2:         {synthetic_params['donor_n_pair2']}")
            print("  Acceptor:")
            print(f"    Pair 1:         {synthetic_params['acceptor_n_pair1']}")
            print(f"    Pair 2:         {synthetic_params['acceptor_n_pair2']}")

        # Print parameters for each position
        print_field_params("Reactant Donor", synthetic_params['reactant_donor'])
        print_field_params("Reactant Acceptor", synthetic_params['reactant_acceptor'])
        print_field_params("Product Donor", synthetic_params['product_donor'])
        print_field_params("Product Acceptor", synthetic_params['product_acceptor'])

    else:
        print("\nMD Field Settings:")
        if set_filter == 'yes':
            print(f"  Filter type:         {filter_type}")
            print(f"  Filter value:        {filter_value}")
            if filter_type == 'exact':
                print(f"  Filter tolerance:    {filter_tolerance}")
        else:
            print("  No field filtering applied")

    if iterative_run == 'yes':
        print("\nIteration Settings:")
        if use_synthetic == 'yes':
            print(f"  Position(s):        {iterate_position}")
            print(f"  Increment:          {increment:.3f}")
            
            print("\nIteration Ranges:")
            def print_iteration_range(position_name):
                """Helper function to print iteration range for a position"""
                max_magnitude = synthetic_params[position_name]['magnitude']
                steps = int(max_magnitude / increment) + 1
                print(f"\n  {position_name.replace('_', ' ').title()}:")
                print(f"    Range: 0.000 to {max_magnitude:.3f} V/Å")
                print(f"    Steps: {steps}")
                print(f"    Values: " + 
                    ", ".join([f"{i*increment:.3f}" for i in range(min(3, steps))] +
                            (["..."] if steps > 3 else []) +
                            ([f"{max_magnitude:.3f}"] if steps > 3 else [])))
            
            if iterate_position == 'all':
                for pos in ['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor']:
                    print_iteration_range(pos)
            else:
                print_iteration_range(iterate_position)
        else:
            # Original MD field iteration display
            print(f"  Filter type:         {filter_type}")
            current_value = 0
            steps = int(filter_value / increment) + 1
            print(f"  Range: {current_value:.3f} to {filter_value:.3f}")
            print(f"  Steps: {steps}")
            print(f"  Increment:          {increment:.3f}")
            print("  Values: " + 
                ", ".join([f"{i*increment:.3f}" for i in range(min(3, steps))] +
                        (["..."] if steps > 3 else []) +
                        ([f"{filter_value:.3f}"] if steps > 3 else [])))

    print("\n" + "="*50)

#   # Ask for confirmation
#   proceed = get_input("\nDo you want to proceed with these parameters? (yes/no): ", choices=['yes', 'no'])
#   if proceed == 'no':
#       print("\nExiting program at user request.")
#       sys.exit(0)
###################################################################################################
    # Setup directories and filenames

    print("\nCreating output directories...")
    
    # Generate the base output folder name based on calculation parameters
    folder_params = {
        'use_synthetic': (use_synthetic == 'yes'),
    }
    
    if use_synthetic == 'yes':
        folder_params.update({
            'field_approach': synthetic_params['approach'],
            'iterative_run': (iterative_run == 'yes'),
            'iterate_position': iterate_position if iterative_run == 'yes' else None,
            'increment': increment if iterative_run == 'yes' else None,
        })
        
        # Add parameters for each position
        for pos in ['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor']:
            short_pos = pos.split('_')[0][0] + pos.split('_')[1][0]  # Convert to rd, ra, pd, pa
            pos_params = synthetic_params[pos]
            
            folder_params.update({
                f'{short_pos}_distribution': pos_params['distribution'],
                f'{short_pos}_magnitude': pos_params['magnitude'],
                f'{short_pos}_std_dev': pos_params.get('std_dev'),
                f'{short_pos}_range': pos_params.get('range_width'),
                f'{short_pos}_direction': pos_params.get('direction')
            })
    else:
        folder_params.update({
            'filter_type': filter_type,
            'filter_value': filter_value,
            'filter_tolerance': filter_tolerance,
            'iterative_run': (iterative_run == 'yes'),
            'increment': increment if iterative_run == 'yes' else None
        })
    
    # Generate the standardized folder name
    main_output_folder = generate_folder_name(
        reactant_donor_id=reactant_donor_id,
        reactant_acceptor_id=reactant_acceptor_id,
        **folder_params
    )
    
    # Create subfolder structure (this part remains the same)
    txt_output_folder = os.path.join(main_output_folder, "txt")
    png_output_folder = os.path.join(main_output_folder, "png")
    mp4_output_folder = os.path.join(main_output_folder, "mp4")

    # Create all required directories (this part remains the same)
    cof.create_output_folder(main_output_folder)
    cof.create_output_folder(txt_output_folder)
    cof.create_output_folder(png_output_folder)
    cof.create_output_folder(mp4_output_folder)

    # Function to create output filenames
    def create_position_filename(base_name, params):
        """Create appropriate filename based on iteration type and parameters"""
        if params.get('use_synthetic'):
            position = params.get('position', '')
            magnitude = params.get('magnitude', 0)
            return f"{base_name}_field_variation_{position.replace('_', '-')}_{magnitude:.3f}.txt"
        else:
            filter_type = params.get('filter_type', '')
            filter_value = params.get('filter_value', 0)
            return f"{base_name}_EFieldDependentReorgEng_{filter_type}_{filter_value:.3f}.txt"

###################################################################################################
    def debug_results_tuple(results):
        """
        Debug utility to print the contents and dimensions of each element in the results tuple.
    
        Args:
            results: Tuple of results from run_analysis.py containing field statistics,
                    frame counts, energies, and other analysis data
        """
        # Expected indices based on run_analysis.py return tuple
        element_names = [
            # Field statistics [0-7]
            "reactant donor mean electric field magnitude",
            "reactant donor electric field standard deviation",
            "reactant acceptor mean electric field magnitude",
            "reactant acceptor electric field standard deviation", 
            "product donor mean electric field magnitude",
            "product donor electric field standard deviation",
            "product acceptor mean electric field magnitude",
            "product acceptor electric field standard deviation",
        
            # Frame counts [8-9]
            "reactant trajectory frame count",
            "product trajectory frame count",
        
            # Reactant state energies and variances [10-17]
            "mean reactant state coulombic energy",
            "variance of reactant state coulombic energy",
            "mean reactant donor polarization energy",
            "variance of reactant donor polarization energy",
            "mean reactant acceptor polarization energy", 
            "variance of reactant acceptor polarization energy",
            "average total reactant energy",
            "variance of total reactant energy",
        
            # Product state energies and variances [18-25]
            "mean product state coulombic energy",
            "variance of product state coulombic energy",
            "mean product donor polarization energy",
            "variance of product donor polarization energy",
            "mean product acceptor polarization energy",
            "variance of product acceptor polarization energy",
            "average total product energy",
            "variance of total product energy",
        
            # Reorganization energies [26-27]
            "unpolarized reorganization energy",
            "polarized reorganization energy"
        ]
    
        print("\n=== Results Tuple Debug Information ===")
        print(f"Total elements in tuple: {len(results)}\n")
    
        for i, (name, value) in enumerate(zip(element_names, results)):
            # Get type and shape/length information
            value_type = type(value).__name__
            if hasattr(value, 'shape'):
                dimension_info = f"shape={value.shape}"
            elif hasattr(value, '__len__'):
                dimension_info = f"length={len(value)}"
            else:
                dimension_info = "scalar"
            
            print(f"[{i:2d}] {name}")
            print(f"     Type: {value_type}")
            print(f"     Dimensions: {dimension_info}")
            print(f"     Value: {value if dimension_info == 'scalar' else '...'}\n")
###################################################################################################
    # Iterative analysis    
    if iterative_run == 'yes':
        if use_synthetic == 'yes':
            # Dictionary to store all results for final summary
            all_results = []
            position_results = {
                'reactant_donor': [],
                'reactant_acceptor': [],
                'product_donor': [],
                'product_acceptor': []
            }
            
            # Determine which positions to iterate over
            positions_to_iterate = (['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor'] 
                                if iterate_position == 'all' 
                                else [iterate_position])
            
            # Iterate over each selected position
            for current_position in positions_to_iterate:
                print(f"\nAnalyzing field variation for {current_position.replace('_', ' ').title()}")
                
                max_magnitude = synthetic_params[current_position]['magnitude']
                magnitude_values = np.linspace(0, max_magnitude, int(max_magnitude / increment) + 1)
                original_magnitude = synthetic_params[current_position]['magnitude']
                
                for idx, current_magnitude in enumerate(magnitude_values):
                    print(f"\nProcessing {current_position} magnitude: {current_magnitude:.3f} V/Å "
                        f"(Step {idx + 1}/{len(magnitude_values)})")
                    
                    synthetic_params[current_position]['magnitude'] = current_magnitude
                    
                    result = ra.run_analysis(
                        reactant_donor_id, reactant_acceptor_id,
                        product_donor_id, product_acceptor_id,
                        basis_set_scaling,
                        use_synthetic_fields=True,
                        synthetic_field_params=synthetic_params,
                        iterative_run=True,
                        iteration=idx,
                        output_folders={'txt': txt_output_folder,
                                'png': png_output_folder,
                                'mp4': mp4_output_folder}
                    )
                    
                    # Store result for both position-specific and overall summaries
                    result_with_metadata = {
                        'position': current_position,
                        'magnitude': current_magnitude,
                        'iteration': idx,
                        'data': result
                    }
                    position_results[current_position].append(result_with_metadata)
                    all_results.append(result_with_metadata)
                
                # Restore original magnitude
                synthetic_params[current_position]['magnitude'] = original_magnitude
            
            # After all iterations complete, generate summary outputs
            print("\nGenerating summary outputs...")
#           print("\n Debugging result tuple")
#           debug_results_tuple(results)  
            
            # 1. Create combined summary file
            summary_filename = f"{reactant_donor_id}_{reactant_acceptor_id}_field_variation_summary.txt"
            summary_output = feao.format_energy_analysis_output(
                results=all_results,
                include_header=True,
                use_synthetic=True,
                synthetic_params=synthetic_params,
                filter_type=None,
                filter_value=None,
                filter_tolerance=None
            )
            sof.save_output_file(txt_output_folder, summary_filename, summary_output)
            
            # 2. Generate comprehensive reorganization energy plot
            pefdre.plot_efield_dependent_reorganization_energy(
                all_results,
                os.path.join(png_output_folder, 'combined_field_dependence.png'),
                include_all_positions=True
            )
        
            # 3. Generate energy decomposition animation
            animation_data = [{
                'reactant_donor_field': r['data'][0],         # Mean field magnitude
                'reactant_donor_std': r['data'][1],           # Field std dev
                'reactant_acceptor_field': r['data'][2],      # Mean field magnitude
                'reactant_acceptor_std': r['data'][3],        # Field std dev
                'product_donor_field': r['data'][4],          # Mean field magnitude
                'product_donor_std': r['data'][5],            # Field std dev
                'product_acceptor_field': r['data'][6],       # Mean field magnitude
                'product_acceptor_std': r['data'][7],         # Field std dev
                'reactant_frames': r['data'][8],              # Frame count
                'product_frames': r['data'][9],               # Frame count
                'reactant_coulombic': r['data'][10],          # Mean energy
                'reactant_coulombic_var': r['data'][11],      # Energy variance
                'reactant_donor_polarization': r['data'][12], # Mean energy
                'reactant_donor_polarization_var': r['data'][13], # Energy variance
                'reactant_acceptor_polarization': r['data'][14], # Mean energy
                'reactant_acceptor_polarization_var': r['data'][15], # Energy variance
                'reactant_total': r['data'][16],              # Mean energy
                'reactant_total_var': r['data'][17],          # Energy variance
                'product_coulombic': r['data'][18],           # Mean energy
                'product_coulombic_var': r['data'][19],       # Energy variance
                'product_donor_polarization': r['data'][20],  # Mean energy
                'product_donor_polarization_var': r['data'][21], # Energy variance
                'product_acceptor_polarization': r['data'][22], # Mean energy
                'product_acceptor_polarization_var': r['data'][23], # Energy variance
                'product_total': r['data'][24],               # Mean energy
                'product_total_var': r['data'][25],           # Energy variance
                'reorg_unpolarized': r['data'][26],          # Reorganization energy
                'reorg_polarized': r['data'][27]             # Reorganization energy
            } for r in all_results]
            
            ped.generate_energy_decomposition_animation(
                results=animation_data,
                output_path=os.path.join(mp4_output_folder, 'animation')
            )

        else:
            # MD field filtering iteration logic
            results = []
            position_results = {
                'reactant_donor': [],
                'reactant_acceptor': [],
                'product_donor': [],
                'product_acceptor': []
            }
            all_results = []
            print("\nIterating over filter values...")
            
            for current_value in np.linspace(0, filter_value, int(filter_value / increment) + 1):
                print(f"\nProcessing filter value: {current_value:.3f}")
                
                result = ra.run_analysis(
                    reactant_donor_id, reactant_acceptor_id,
                    product_donor_id, product_acceptor_id,
                    basis_set_scaling,
                    use_synthetic_fields=False,
                    filter_type=filter_type,
                    filter_value=current_value,
                    filter_tolerance=filter_tolerance,
                    iterative_run=True,
                    output_folders={'txt': txt_output_folder,
                                'png': png_output_folder,
                                'mp4': mp4_output_folder}
                )
                
                if result[8] > 0:  # Check reactant frame count
                    # Store result for both position-specific and overall summaries
                    result_with_metadata = {
                        'magnitude': current_value,
                        'data': result
                    }
                    # Store in each position's results since MD fields affect all positions
                    for position in ['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor']:
                        position_results[position].append(result_with_metadata)
                    all_results.append(result_with_metadata)
            
            if not results:
                print("\nNo valid results were obtained for any filter value.")
                sys.exit(1)
            
            # After all iterations complete, generate summary outputs
            print("\nGenerating summary outputs...")
#           print("\n Debugging result tuple")
#           debug_results_tuple(results)  
            
            # 1. Create combined summary file
            summary_filename = f"{reactant_donor_id}_{reactant_acceptor_id}_field_variation_summary.txt"
            summary_output = feao.format_energy_analysis_output(
                results=all_results,
                include_header=True,
                use_synthetic=False,
                synthetic_params=None,
                filter_type=filter_type,
                filter_value=filter_value,
                filter_tolerance=filter_tolerance
            )
            sof.save_output_file(txt_output_folder, summary_filename, summary_output)
            
            # 2. Generate comprehensive reorganization energy plot
            pefdre.plot_efield_dependent_reorganization_energy(
                all_results,
                os.path.join(png_output_folder, 'combined_field_dependence.png'),
                include_all_positions=True
            )
                        
            # 3. Generate energy decomposition animation
            animation_data = [{
                'reactant_donor_field': r['data'][0],
                'reactant_donor_std': r['data'][1],
                'reactant_acceptor_field': r['data'][2],
                'reactant_acceptor_std': r['data'][3],
                'product_donor_field': r['data'][4],
                'product_donor_std': r['data'][5],
                'product_acceptor_field': r['data'][6],
                'product_acceptor_std': r['data'][7],
                'reactant_frames': r['data'][8],
                'product_frames': r['data'][9],
                'reactant_coulombic': r['data'][10],
                'reactant_coulombic_var': r['data'][11],
                'reactant_donor_polarization': r['data'][12],
                'reactant_donor_polarization_var': r['data'][13],
                'reactant_acceptor_polarization': r['data'][14],
                'reactant_acceptor_polarization_var': r['data'][15],
                'reactant_total': r['data'][16],
                'reactant_total_var': r['data'][17],
                'product_coulombic': r['data'][18],
                'product_coulombic_var': r['data'][19],
                'product_donor_polarization': r['data'][20],
                'product_donor_polarization_var': r['data'][21],
                'product_acceptor_polarization': r['data'][22],
                'product_acceptor_polarization_var': r['data'][23],
                'product_total': r['data'][24],
                'product_total_var': r['data'][25],
                'reorg_unpolarized': r['data'][26],
                'reorg_polarized': r['data'][27]
            } for r in all_results]

            ped.generate_energy_decomposition_animation(
                results=animation_data,
                output_path=os.path.join(mp4_output_folder, 'animation')
            )

    else:
        # Single run case
        print("\nProcessing single calculation...")
        result = ra.run_analysis(
            reactant_donor_id, reactant_acceptor_id,
            product_donor_id, product_acceptor_id,
            basis_set_scaling,
            use_synthetic_fields=(use_synthetic == 'yes'),
            synthetic_field_params=synthetic_params if use_synthetic == 'yes' else None,
            filter_type=filter_type if use_synthetic == 'no' else None,
            filter_value=filter_value if use_synthetic == 'no' else None,
            filter_tolerance=filter_tolerance if use_synthetic == 'no' else None,
            iterative_run=False,
            output_folders={'txt': txt_output_folder,
                        'png': png_output_folder,
                        'mp4': mp4_output_folder}
        )

#       print("\n Debugging result tuple")
#       debug_results_tuple(result) 
        
        # Process single run result
        if result[1] > 0:
            # Format and save result
            formatted_output = feao.format_energy_analysis_output(
                results=result,
                include_header=True,
                use_synthetic=(use_synthetic == 'yes'),
                synthetic_params=synthetic_params if use_synthetic == 'yes' else None,
                filter_type=filter_type if use_synthetic == 'no' else None,
                filter_value=filter_value if use_synthetic == 'no' else None,
                filter_tolerance=filter_tolerance if use_synthetic == 'no' else None
            )
            
            # Create appropriate filename for single run
            if use_synthetic == 'yes':
                filename_base = f"{reactant_donor_id}_{reactant_acceptor_id}_synthetic"
                for pos in ['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor']:
                    params = synthetic_params[pos]
                    filename_base += f"_{pos}_{params['distribution']}"
                    if params.get('direction'):
                        filename_base += f"_{params['direction']}"
                    filename_base += f"_{params['magnitude']:.3f}"
                filename = f"{filename_base}.txt"
            else:
                suffix = "no_filter" if filter_type is None else f"{filter_type}_{filter_value:.3f}"
                filename = f"{reactant_donor_id}_{reactant_acceptor_id}_EFieldDependentReorgEng_{suffix}.txt"
            
            sof.save_output_file(txt_output_folder, filename, formatted_output)
            
            # Generate plots for single run
            try:
                print("\nGenerating analysis plots...")
                
                # Energy decomposition plot
                ped.plot_energy_decomposition(
                    # Reactant state
                    reactant_coulombic=float(result[10]),           
                    reactant_coulombic_var=float(result[11]),       
                    reactant_donor_polarization=float(result[12]),   
                    reactant_donor_polarization_var=float(result[13]), 
                    reactant_acceptor_polarization=float(result[14]), 
                    reactant_acceptor_polarization_var=float(result[15]), 
                    reactant_total=float(result[16]),               
                    reactant_total_var=float(result[17]),           
                    
                    # Product state
                    product_coulombic=float(result[18]),           
                    product_coulombic_var=float(result[19]),       
                    product_donor_polarization=float(result[20]),   
                    product_donor_polarization_var=float(result[21]), 
                    product_acceptor_polarization=float(result[22]), 
                    product_acceptor_polarization_var=float(result[23]), 
                    product_total=float(result[24]),               
                    product_total_var=float(result[25]),           
                    
                    # Reorganization energies
                    reorg_unpolarized=float(result[26]),           
                    reorg_polarized=float(result[27]),             
                    
                    output_path=png_output_folder,
#                   field_magnitude=(synthetic_params['reactant_donor']['magnitude'] 
#                               if use_synthetic == 'yes' else None)
                )
                              
                # Define input filenames
                reactant_input_file = os.path.join(txt_output_folder, 
                    f"{reactant_donor_id}_{reactant_acceptor_id}_reactant_state_no_filter.txt")
                product_input_file = os.path.join(txt_output_folder, 
                    f"{reactant_donor_id}_{reactant_acceptor_id}_product_state_no_filter.txt")

                # Define input filenames based on whether using synthetic or MD fields
                if use_synthetic == 'yes':
                    # Construct synthetic field suffix
                    def get_position_suffix(pos_params):
                        suffix = f"{pos_params['magnitude']:.3f}"
                        if pos_params.get('direction'):  # Add direction for heme-based approach
                            suffix = f"{pos_params['direction']}_{suffix}"
                        if pos_params['distribution'] != 'fixed':
                            suffix = f"{pos_params['distribution']}_{suffix}"
                        return suffix

                    synthetic_suffix = (
                        f"synthetic_reactant-donor_{get_position_suffix(synthetic_params['reactant_donor'])}_"
                        f"reactant-acceptor_{get_position_suffix(synthetic_params['reactant_acceptor'])}_"
                        f"product-donor_{get_position_suffix(synthetic_params['product_donor'])}_"
                        f"product-acceptor_{get_position_suffix(synthetic_params['product_acceptor'])}"
                    )

                    reactant_input_file = os.path.join(txt_output_folder, 
                                                    f"{reactant_donor_id}_{reactant_acceptor_id}_reactant_state_{synthetic_suffix}.txt")
                    product_input_file = os.path.join(txt_output_folder, 
                                                    f"{reactant_donor_id}_{reactant_acceptor_id}_product_state_{synthetic_suffix}.txt")
                else:
                    # Original MD field filenames
                    reactant_input_file = os.path.join(txt_output_folder, 
                                                    f"{reactant_donor_id}_{reactant_acceptor_id}_reactant_state_no_filter.txt")
                    product_input_file = os.path.join(txt_output_folder, 
                                                    f"{reactant_donor_id}_{reactant_acceptor_id}_product_state_no_filter.txt")

                # Check if files exist
                if not os.path.exists(reactant_input_file):
                    raise FileNotFoundError(f"Reactant state file not found: {reactant_input_file}")
                if not os.path.exists(product_input_file):
                    raise FileNotFoundError(f"Product state file not found: {product_input_file}")
            
                print("\nGenerating electric field magnitude and direction plots...")
                # Define output filenames
                reactant_plot_file = os.path.join(png_output_folder, "field_analysis_reactant.png")
                product_plot_file = os.path.join(png_output_folder, "field_analysis_product.png")
            
                # Create plots
                create_field_analysis_plot(reactant_input_file, reactant_plot_file)
                create_field_analysis_plot(product_input_file, product_plot_file)
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"An error occurred while creating analysis plots: {e}")
        else:
            print("\nNo valid frames were found. Check your parameters.")
            sys.exit(1)

    print("\nProcess completed successfully.")
