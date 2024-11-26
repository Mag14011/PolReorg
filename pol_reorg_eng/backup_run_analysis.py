import os
import sys
import logging
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, NamedTuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Local imports
import read_file_paths as rfp
import create_output_folder as cof
import read_xyz as rxyz 
import superimpose_and_save as ss
import read_tensors as rt
import read_transformation_matrices as rtm
import transform_and_save_fields as tasf
import transform_and_save_tensors as tast
import compute_diffalpha as cda
import read_electric_field as ref
import compute_polarization_energy as cpe
import read_coulombic_energy as rce
import write_output_file as wof
import plot_coordinate_transformation_rmsd as pctrmsd
import plot_electric_field_dist as pefd
import plot_visualize_polarizability as pvp
import plot_tensor_field as ptf
import plot_energy_decomposition as ped
import process_and_plot_data as papd
import compute_and_write_reorganization_energies as cawre
#from file_path_config import read_file_paths, construct_simulation_paths

class SyntheticFieldGenerator:
    """Generate synthetic electric field vectors using either MD-derived directions
    or heme-based coordinate system."""
    
    def __init__(self, num_frames: int, approach: str = 'direction_from_md',
                 n_pair1: Optional[Tuple[int, int]] = None, 
                 n_pair2: Optional[Tuple[int, int]] = None):
        """
        Initialize generator with frame count and approach.
        
        Args:
            num_frames: Number of frames to generate
            approach: Method for generating fields ('direction_from_md' or 'heme_based')
            n_pair1: Optional indices of opposing N atoms defining x-axis (required for heme_based)
            n_pair2: Optional indices of opposing N atoms defining y-axis (required for heme_based)
        """
        self.num_frames = num_frames
        if approach not in ['direction_from_md', 'heme_based']:
            raise ValueError("Approach must be 'direction_from_md' or 'heme_based'")
        self.approach = approach
        
        # Only validate n_pairs for heme_based approach
        if self.approach == 'heme_based':
            if n_pair1 is None or n_pair2 is None:
                raise ValueError("n_pair1 and n_pair2 are required for heme_based approach")
        self.n_pair1 = n_pair1
        self.n_pair2 = n_pair2

    def _validate_rotation(self, original: np.ndarray, rotated: np.ndarray) -> None:
        """
        Validate rotation of field vectors.
        
        Args:
            original: Original field vectors (n_frames, 3)
            rotated: Rotated field vectors (n_frames, 3)
            
        Raises:
            ValueError: If rotation validation fails
        """
        # Check magnitude preservation
        orig_mag = np.linalg.norm(original, axis=1)
        rot_mag = np.linalg.norm(rotated, axis=1)
        if not np.allclose(orig_mag, rot_mag, rtol=1e-5):
            raise ValueError("Field vector magnitudes not preserved during rotation")
            
        # Check orthogonality preservation if vectors were orthogonal
        if original.shape[0] >= 2:
            orig_dot = np.abs(np.sum(original[0] * original[1]) / 
                            (np.linalg.norm(original[0]) * np.linalg.norm(original[1])))
            rot_dot = np.abs(np.sum(rotated[0] * rotated[1]) / 
                           (np.linalg.norm(rotated[0]) * np.linalg.norm(rotated[1])))
            if orig_dot < 1e-5 and not rot_dot < 1e-5:
                raise ValueError("Orthogonality not preserved during rotation")

    def write_heme_axes_xyz(self, qm_symbols: List[str],
                        qm_coords: np.ndarray,
                        x_axis: np.ndarray,
                        y_axis: np.ndarray,
                        z_axis: np.ndarray,
                        output_path: str,
                        scale: float = 2.0) -> None:
        """
        Create an XYZ file showing the heme structure with dummy atoms representing the axes.

        Args:
            qm_symbols: List of atomic symbols from QM structure
            qm_coords: Array of atomic coordinates
            x_axis: Unit vector for x-axis
            y_axis: Unit vector for y-axis
            z_axis: Unit vector for z-axis
            output_path: Path to save the visualization XYZ file
            scale: Scale factor for axis vectors (default 2.0 Å)
        """
        # Find Fe atom to use as origin
        fe_idx = [i for i, sym in enumerate(qm_symbols) if sym == 'Fe'][0]
        origin = qm_coords[fe_idx]

        # Create points for axis visualization
        # Scale the unit vectors and add to origin
        x_point = origin + x_axis * scale
        y_point = origin + y_axis * scale
        z_point = origin + z_axis * scale

        # Write XYZ file with original structure plus axis points
        with open(output_path, 'w') as f:
            # Total atoms = original structure + 3 axis points
            f.write(f"{len(qm_symbols) + 3}\n")
            f.write("Heme structure with coordinate axes\n")

            # Write original structure
            for symbol, coords in zip(qm_symbols, qm_coords):
                f.write(f"{symbol:2s} {coords[0]:10.6f} {coords[1]:10.6f} {coords[2]:10.6f}\n")

            # Add dummy atoms for axes (using He for visualization)
            f.write(f"He {x_point[0]:10.6f} {x_point[1]:10.6f} {x_point[2]:10.6f}\n")  # X-axis
            f.write(f"Ne {y_point[0]:10.6f} {y_point[1]:10.6f} {y_point[2]:10.6f}\n")  # Y-axis
            f.write(f"Ar {z_point[0]:10.6f} {z_point[1]:10.6f} {z_point[2]:10.6f}\n")  # Z-axis

    def get_heme_axes(self, qm_symbols: List[str],
                    qm_coords: np.ndarray,
                    output_dir: Optional[str] = None,
                    position_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract coordinate system from heme in QM structure.

        Args:
            qm_symbols: List of atomic symbols
            qm_coords: numpy array of atomic coordinates (n_atoms, 3)
            output_dir: Optional directory for visualization output
            position_name: Optional name of position for output files

        Returns:
            x_axis, y_axis, z_axis: Unit vectors defining heme-based coordinate system
        """
        print("\nDebug: Starting get_heme_axes")
        print(f"Input N pairs: n_pair1={self.n_pair1}, n_pair2={self.n_pair2}")

        # Find Fe atom
        fe_candidates = [i for i, sym in enumerate(qm_symbols) if sym == 'Fe']
        if len(fe_candidates) != 1:
            raise ValueError(f"Found {len(fe_candidates)} Fe atoms, expected 1")
        fe_idx = fe_candidates[0]
        fe_pos = qm_coords[fe_idx]
        print(f"Found Fe at index {fe_idx}")

        # Find all N atoms close to Fe (typical Fe-N distance ~2.0 Å)
        n_candidates = []
        all_n_atoms = []  # Track all nitrogen atoms for debugging
        for i, (sym, pos) in enumerate(zip(qm_symbols, qm_coords)):
            if sym == 'N':
                dist = np.linalg.norm(pos - fe_pos)
                all_n_atoms.append((i, dist))
                if 1.8 < dist < 2.2:  # Typical Fe-N range
                    n_candidates.append((i, pos))

        print("\nAll nitrogen atoms found:")
        for idx, dist in all_n_atoms:
            print(f"N atom at index {idx}: distance to Fe = {dist:.3f} Å")

        print("\nPyrrole nitrogen candidates (1.8 Å < d < 2.2 Å):")
        for idx, pos in n_candidates:
            print(f"Candidate N at index {idx}: position = {pos}")

        if len(n_candidates) != 4:
            raise ValueError(f"Found {len(n_candidates)} coordinating N atoms, expected 4 pyrrole nitrogens")

        # Extract positions and keep track of global indices
        n_global_indices = [idx for idx, _ in n_candidates]
        n_positions = np.array([pos for _, pos in n_candidates])
        n_centered = n_positions - fe_pos

        print("\nGlobal indices of pyrrole nitrogens:", n_global_indices)
        print("Looking for n_pair1:", self.n_pair1)
        print("Looking for n_pair2:", self.n_pair2)

        # Sort by angle for consistent ordering
        angles = np.arctan2(n_centered[:,1], n_centered[:,0])
        cyclic_order = np.argsort(angles)

        # Create mapping from global to local indices
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(n_global_indices)}
        print("\nGlobal to local index mapping:", global_to_local)

        # Find local indices corresponding to the specified pairs
        try:
            n_pair1_local = (global_to_local[self.n_pair1[0]], global_to_local[self.n_pair1[1]])
            n_pair2_local = (global_to_local[self.n_pair2[0]], global_to_local[self.n_pair2[1]])
        except KeyError as e:
            print(f"\nError: Could not find N atom with index {e} in the pyrrole nitrogens.")
            print("Available pyrrole nitrogen indices:", n_global_indices)
            print("Requested pairs:", self.n_pair1, self.n_pair2)
            raise

        # Define axes using local indices in cyclic order
        x_raw = n_positions[cyclic_order[n_pair1_local[1]]] - \
                n_positions[cyclic_order[n_pair1_local[0]]]
        x_axis = x_raw / np.linalg.norm(x_raw)

        # y-axis from specified pair, then project to ensure orthogonality
        y_raw = n_positions[cyclic_order[n_pair2_local[1]]] - \
                n_positions[cyclic_order[n_pair2_local[0]]]
        y_axis = y_raw - np.dot(y_raw, x_axis) * x_axis  # Project out x component
        y_axis = y_axis / np.linalg.norm(y_axis)

        # z-axis from cross product to ensure right-handed system
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Validate orthogonality
        if not np.allclose([np.dot(x_axis, y_axis),
                        np.dot(y_axis, z_axis),
                        np.dot(z_axis, x_axis)], 0, atol=1e-6):
            raise ValueError("Generated axes are not orthogonal")

        if output_dir is not None and position_name is not None:
            axes_xyz = os.path.join(output_dir, f"heme_axes_{position_name}.xyz")
            self.write_heme_axes_xyz(qm_symbols, qm_coords, x_axis, y_axis, z_axis, axes_xyz)
            print(f"\nWrote heme axes visualization to: {axes_xyz}")
            print("Axes represented by dummy atoms:")
            print("  He: X-axis")
            print("  Ne: Y-axis")
            print("  Ar: Z-axis")

            # Print detailed information about the coordinate system
            print("\nCoordinate system details:")
            print(f"X-axis: {x_axis}")
            print(f"Y-axis: {y_axis}")
            print(f"Z-axis: {z_axis}")
            print("\nN atom information:")
            print("Global indices of pyrrole nitrogens:", n_global_indices)
            print("Local ordering after cyclic sort:", cyclic_order)
            print(f"N pair 1 (global indices): {self.n_pair1} -> (local indices): {n_pair1_local}")
            print(f"N pair 2 (global indices): {self.n_pair2} -> (local indices): {n_pair2_local}")

        return x_axis, y_axis, z_axis


    def generate_fields(self,
                    distribution: str,
                    direction: Optional[str],
                    magnitude: float,
                    md_fields: Optional[Tuple[np.ndarray, ...]] = None,
                    qm_structure: Optional[Tuple[List[str], np.ndarray]] = None,
                    rotation_matrices: Optional[np.ndarray] = None,
                    std_dev: Optional[float] = None,
                    range_width: Optional[float] = None,
                    output_dir: Optional[str] = None,
                    position_name: Optional[str] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate electric field vectors using specified approach.
        """
        # Print input shapes for debugging
        if md_fields is not None:
            Ex, Ey, Ez = md_fields
            print(f"Input MD field shapes: Ex: {Ex.shape}, Ey: {Ey.shape}, Ez: {Ez.shape}")

        # Generate magnitudes based on distribution type
        if distribution == 'fixed':
            magnitudes = np.full(self.num_frames, magnitude)
        elif distribution == 'gaussian':
            if std_dev is None:
                raise ValueError("std_dev required for gaussian distribution")
            magnitudes = np.random.normal(magnitude, std_dev, self.num_frames)
            magnitudes = np.clip(magnitudes, 0, None)  # Ensure non-negative
        elif distribution == 'uniform':
            if range_width is None:
                raise ValueError("range_width required for uniform distribution")
            half_width = range_width / 2
            low = max(0, magnitude - half_width)  # Ensure non-negative
            high = magnitude + half_width
            magnitudes = np.random.uniform(low, high, self.num_frames)
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

        print(f"Generated magnitudes shape: {magnitudes.shape}")

        if self.approach == 'direction_from_md':
            if md_fields is None:
                raise ValueError("MD fields required for direction_from_md approach")

            Ex, Ey, Ez = md_fields

            # Ensure all inputs are 1D arrays
            Ex = np.asarray(Ex).flatten()
            Ey = np.asarray(Ey).flatten()
            Ez = np.asarray(Ez).flatten()

            print(f"Flattened MD field shapes: Ex: {Ex.shape}, Ey: {Ey.shape}, Ez: {Ez.shape}")

            # Verify all arrays have the expected length
            if not (len(Ex) == len(Ey) == len(Ez) == self.num_frames):
                raise ValueError(f"Array length mismatch. Expected {self.num_frames}, "
                            f"got Ex: {len(Ex)}, Ey: {len(Ey)}, Ez: {len(Ez)}")

            # Normalize MD field vectors to get directions
            E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
            unit_x = Ex / E_mag
            unit_y = Ey / E_mag
            unit_z = Ez / E_mag

            print(f"Unit vector shapes: x: {unit_x.shape}, y: {unit_y.shape}, z: {unit_z.shape}")

            # Apply synthetic magnitudes to MD directions
            result_x = unit_x * magnitudes
            result_y = unit_y * magnitudes
            result_z = unit_z * magnitudes

            print(f"Final result shapes: x: {result_x.shape}, y: {result_y.shape}, z: {result_z.shape}")

            return result_x, result_y, result_z

        elif self.approach == 'heme_based':
            if qm_structure is None or rotation_matrices is None:
                raise ValueError("QM structure and rotation matrices required for heme_based approach")
            if direction is None:
                raise ValueError("direction must be specified for heme_based approach")
                
            # Get heme-based coordinate system
            qm_symbols, qm_coords = qm_structure
            x_axis, y_axis, z_axis = self.get_heme_axes(
                qm_symbols, 
                qm_coords,
                output_dir=output_dir,
                position_name=position_name
            )

            if direction == 'random':
                # Generate random angles in spherical coordinates
                theta = np.random.uniform(0, np.pi, self.num_frames)
                phi = np.random.uniform(0, 2*np.pi, self.num_frames)
                
                # Convert to Cartesian coordinates in heme coordinate system
                # Start with unit vectors in standard coordinates
                v = np.zeros((self.num_frames, 3))
                v[:,0] = np.sin(theta) * np.cos(phi)  # x component
                v[:,1] = np.sin(theta) * np.sin(phi)  # y component
                v[:,2] = np.cos(theta)                # z component
                
                # Transform to heme coordinate system
                heme_basis = np.array([x_axis, y_axis, z_axis]).T
                v = np.array([np.dot(heme_basis, vi) for vi in v])

            elif direction == 'xx':
                v = np.tile(x_axis, (self.num_frames, 1))
            elif direction == 'yy':
                v = np.tile(y_axis, (self.num_frames, 1))
            elif direction == 'zz':
                v = np.tile(z_axis, (self.num_frames, 1))
            elif direction == 'xy':
                v = np.tile((x_axis + y_axis) / np.sqrt(2), (self.num_frames, 1))
            elif direction == 'xz':
                v = np.tile((x_axis + z_axis) / np.sqrt(2), (self.num_frames, 1))
            elif direction == 'yz':
                v = np.tile((y_axis + z_axis) / np.sqrt(2), (self.num_frames, 1))
            else:
                raise ValueError(f"Unknown direction: {direction}")
            
            # Apply magnitude to direction vector
            v = v * magnitudes[:, np.newaxis]
            
            # Rotate into MD frames
            rotated_fields = np.zeros((self.num_frames, 3))
            for i, R in enumerate(rotation_matrices):
                rotated_fields[i] = np.dot(R, v[i])
                
            # Validate transformations
            self._validate_rotation(v, rotated_fields)

            return (rotated_fields[:,0], 
                    rotated_fields[:,1], 
                    rotated_fields[:,2])
        else:
            raise ValueError(f"Unknown synthetic field approach: {self.approach}")

def construct_frame_tensors(components: Tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Construct 3x3 polarizability tensors for each frame from components.
    
    Args:
        components: Tuple of (xx, xy, xz, yy, yz, zz) arrays
        
    Returns:
        Array of shape (n_frames, 3, 3) containing tensor for each frame
    """
    xx, xy, xz, yy, yz, zz = components
    num_frames = len(xx)
    tensors = np.zeros((num_frames, 3, 3))
    for i in range(num_frames):
        tensors[i] = np.array([[xx[i], xy[i], xz[i]],
                              [xy[i], yy[i], yz[i]],
                              [xz[i], yz[i], zz[i]]])
    return tensors

def append_iteration_number(base_filename: str, iteration: Optional[int] = None) -> str:
    """
    Add iteration number to filename while preserving extension.
    
    Args:
        base_filename: Original filename
        iteration: Optional iteration number (None for non-iterative runs)
        
    Returns:
        Modified filename with iteration number inserted before extension
    """
    if iteration is None:
        return base_filename
        
    base, ext = os.path.splitext(base_filename)
    return f"{base}_iteration_{iteration:03d}{ext}"

def construct_output_path(output_folder: str, 
                         base_filename: str, 
                         iteration: Optional[int] = None) -> str:
    """
    Construct full output path with optional iteration number.
    
    Args:
        output_folder: Directory for output
        base_filename: Base filename
        iteration: Optional iteration number
        
    Returns:
        Full path with iteration number if specified
    """
    modified_filename = append_iteration_number(base_filename, iteration)
    return os.path.join(output_folder, modified_filename)

def construct_metadata(use_synthetic_fields: bool,
                      synthetic_field_params: Optional[Dict[str, Any]] = None,
                      filter_type: Optional[str] = None,
                      filter_value: Optional[float] = None,
                      filter_tolerance: Optional[float] = None) -> str:
    """
    Construct metadata string describing analysis parameters.
    """
    lines = ["Field Parameters:"]

    if use_synthetic_fields:
        lines.extend([
            "Field Type: Synthetic",
            f"Generation Approach: {synthetic_field_params['approach']}"
        ])

        if synthetic_field_params['approach'] == 'heme_based':
            lines.extend([
                f"Donor N pair 1: {synthetic_field_params['donor_n_pair1']}",
                f"Donor N pair 2: {synthetic_field_params['donor_n_pair2']}",
                f"Acceptor N pair 1: {synthetic_field_params['acceptor_n_pair1']}",
                f"Acceptor N pair 2: {synthetic_field_params['acceptor_n_pair2']}"
            ])

        # Function to format position parameters
        def format_position_params(position_name: str, params: Dict[str, Any]) -> List[str]:
            position_lines = [
                f"\n{position_name.replace('_', ' ').title()} Field Parameters:",
                f"Distribution: {params['distribution']}",
                f"Magnitude: {params['magnitude']:.3f} V/Å"
            ]

            if params.get('direction') is not None:
                position_lines.append(f"Direction: {params['direction']}")

            if params.get('std_dev') is not None:
                position_lines.append(f"Standard Deviation: {params['std_dev']:.3f}")

            if params.get('range_width') is not None:
                position_lines.append(f"Range Width: {params['range_width']:.3f}")

            return position_lines

        # Add parameters for each position
        for position in ['reactant_donor', 'reactant_acceptor',
                        'product_donor', 'product_acceptor']:
            lines.extend(format_position_params(
                position, synthetic_field_params[position]
            ))

    else:
        lines.extend([
            "Field Type: MD-derived",
            f"Filtering Applied: {'Yes' if filter_type else 'No'}"
        ])

        if filter_type:
            lines.extend([
                f"Filter Type: {filter_type}",
                f"Filter Value: {filter_value:.3f}"
            ])

            if filter_type == 'exact' and filter_tolerance is not None:
                lines.append(f"Filter Tolerance: {filter_tolerance:.3f}")

    return "\n".join(lines)

def get_frame_count(xyz_file_path: str) -> int:
    """Count frames in XYZ trajectory file."""
    with open(xyz_file_path, 'r') as f:
        first_line = f.readline()
        atoms_per_frame = int(first_line)
        f.seek(0)
        total_lines = sum(1 for _ in f)
        return (total_lines // (atoms_per_frame + 2)) 

def write_electric_field(filename: str, Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray) -> None:
    """
    Write electric field components to a file in the MD-derived format.

    Input fields are in V/Å and are converted to MV/cm for writing.
    Conversion factor: 1 V/Å = 100 MV/cm

    Args:
        filename (str): Path to output file
        Ex (np.ndarray): x-component of electric field (in V/Å)
        Ey (np.ndarray): y-component of electric field (in V/Å)
        Ez (np.ndarray): z-component of electric field (in V/Å)
    """
    # Print input array shapes
    print(f"Input array shapes: Ex: {Ex.shape}, Ey: {Ey.shape}, Ez: {Ez.shape}")

    # Ensure inputs are 1D numpy arrays
    Ex = np.asarray(Ex).flatten()
    Ey = np.asarray(Ey).flatten()
    Ez = np.asarray(Ez).flatten()

    # Print flattened array shapes
    print(f"Flattened array shapes: Ex: {Ex.shape}, Ey: {Ey.shape}, Ez: {Ez.shape}")

    # Verify all arrays have the same length
    if not (len(Ex) == len(Ey) == len(Ez)):
        raise ValueError(f"Arrays must have same length. Got lengths: Ex: {len(Ex)}, Ey: {len(Ey)}, Ez: {len(Ez)}")

    # Convert from V/Å to MV/cm (multiply by 100)
    Ex_mvcm = Ex * 100
    Ey_mvcm = Ey * 100
    Ez_mvcm = Ez * 100

    # Calculate magnitude (in MV/cm)
    magnitudes = np.sqrt(Ex_mvcm**2 + Ey_mvcm**2 + Ez_mvcm**2)
    num_frames = len(Ex)

    print(f"Number of frames to process: {num_frames}")

    # Calculate statistics (in MV/cm)
    avg_mag = np.mean(magnitudes).item()
    std_mag = np.std(magnitudes).item()
    avg_x = np.mean(Ex_mvcm).item()
    avg_y = np.mean(Ey_mvcm).item()
    avg_z = np.mean(Ez_mvcm).item()
    std_x = np.std(Ex_mvcm).item()
    std_y = np.std(Ey_mvcm).item()
    std_z = np.std(Ez_mvcm).item()

    print(f"Writing to file: {filename}")
    with open(filename, 'w') as f:
        # Write header
        f.write('@    title "Electric Field"\n')
        f.write('@    xaxis  label "Time (ps)"\n')
        f.write('@    yaxis  label "MV/cm"\n')
        f.write('#time      Magnitude          Efield_X            Efield_Y            Efield_Z\n')
        f.write('@type xy\n')

        # Write data - assume 20 ps timesteps as in the example
        for i in range(num_frames):
            if i % 1000 == 0:  # Print progress every 1000 frames
                print(f"Processing frame {i}/{num_frames}")

            time = (i + 1) * 20  # Start at 20, increment by 20
            mag = magnitudes[i].item()
            ex = Ex_mvcm[i].item()
            ey = Ey_mvcm[i].item()
            ez = Ez_mvcm[i].item()
            f.write(f'{time:<10} {mag:<18.6f} {ex:<18.6f} {ey:<18.6f} {ez:<18.6f}\n')

        # Write footer
        f.write('#---#\n')
        f.write(f'#AVG:     {avg_mag:<18.6f} {avg_x:<18.6f} {avg_y:<18.6f} {avg_z:<18.6f}\n')
        f.write(f'#STDEV:   {std_mag:<18.6f} {std_x:<18.6f} {std_y:<18.6f} {std_z:<18.6f}\n')

    print(f"Finished writing {num_frames} frames to {filename}")

def filter_electric_fields(donor_efield_path: str,
                         acceptor_efield_path: str,
                         filter_type: Optional[str] = None,
                         filter_value: Optional[float] = None,
                         filter_tolerance: Optional[float] = None) -> Tuple:
    """
    Filter electric field vectors and return common indices between donor and acceptor.
    
    Args:
        donor_efield_path: Path to donor electric field file
        acceptor_efield_path: Path to acceptor electric field file
        filter_type: Type of filter to apply ('threshold', 'variance', 'exact', or None)
        filter_value: Value for filtering
        filter_tolerance: Tolerance for exact filtering
        
    Returns:
        Tuple of (E_x_donor, E_y_donor, E_z_donor,
                 E_x_acceptor, E_y_acceptor, E_z_acceptor,
                 common_indices)
    """
    # Read and filter donor fields
    E_x_donor, E_y_donor, E_z_donor, efield_donor_indices = \
        ref.read_electric_field(donor_efield_path, filter_type, filter_value, filter_tolerance)
    
    # Read and filter acceptor fields
    E_x_acceptor, E_y_acceptor, E_z_acceptor, efield_acceptor_indices = \
        ref.read_electric_field(acceptor_efield_path, filter_type, filter_value, filter_tolerance)

    # Find common indices between donor and acceptor
    common_indices = np.intersect1d(efield_donor_indices, efield_acceptor_indices)
#   print(f"****** efield_donor_indices = {efield_donor_indices}; efield_donor_indices = {efield_acceptor_indices}")
#   print(f"****** efield_donor_indices = {len(efield_donor_indices)}; efield_donor_indices = {len(efield_acceptor_indices)}")
#   print(f"****** common_indices =  {common_indices}")
#   print(f"****** length: {len(common_indices)}")

    # Filter fields to common indices
    E_x_donor = E_x_donor[np.isin(efield_donor_indices, common_indices)]
    E_y_donor = E_y_donor[np.isin(efield_donor_indices, common_indices)]
    E_z_donor = E_z_donor[np.isin(efield_donor_indices, common_indices)]
    E_x_acceptor = E_x_acceptor[np.isin(efield_acceptor_indices, common_indices)]
    E_y_acceptor = E_y_acceptor[np.isin(efield_acceptor_indices, common_indices)]
    E_z_acceptor = E_z_acceptor[np.isin(efield_acceptor_indices, common_indices)]
    
    return (E_x_donor, E_y_donor, E_z_donor,
            E_x_acceptor, E_y_acceptor, E_z_acceptor,
            common_indices)

def inject_synthetic_fields(xyz_file_path: str,
                          output_dir: str,
                          field_params: Dict[str, Any],
                          md_fields: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                          qm_structure: Optional[Tuple[List[str], np.ndarray]] = None,
                          rotation_matrices: Optional[np.ndarray] = None,
                          position_name: str = '',
                          iteration: Optional[int] = None) -> str:
    """
    Generate synthetic fields based on specified approach and parameters.
    
    Args:
        xyz_file_path: Path to XYZ trajectory file
        output_dir: Directory for output files
        field_params: Parameters for this specific position's field generation
            {
                'distribution': str ('fixed', 'gaussian', 'uniform'),
                'direction': str (for heme_based: 'xx', 'xy', etc.),
                'magnitude': float,
                'std_dev': float (optional),
                'range_width': float (optional)
            }
        md_fields: Optional tuple of (Ex, Ey, Ez) for direction_from_md approach
        qm_structure: Optional tuple of (symbols, coords) for heme_based approach
        rotation_matrices: Optional array of rotation matrices for heme_based approach
        position_name: Name of position (e.g., 'reactant_donor')
        iteration: Optional iteration number
    
    Returns:
        Path to generated field file
    """
    # Validate inputs based on approach
    if field_params['approach'] == 'direction_from_md':
        if md_fields is None:
            raise ValueError("MD fields required for direction_from_md approach")
    elif field_params['approach'] == 'heme_based':
        if qm_structure is None or rotation_matrices is None:
            raise ValueError("QM structure and rotation matrices required for heme_based approach")
        if 'direction' not in field_params:
            raise ValueError("Direction must be specified for heme_based approach")
        # Check for n_pairs only when using heme_based approach
        if 'n_pair1' not in field_params or 'n_pair2' not in field_params:
            raise ValueError("n_pair1 and n_pair2 required for heme_based approach")
    else:
        raise ValueError(f"Unknown approach: {field_params['approach']}")

    # Debugging for heme-based approach
    if field_params['approach'] == 'heme_based':
        print("\nN pair configuration:")
        is_donor = 'donor' in position_name.lower()
        print("  Position type:", "Donor" if is_donor else "Acceptor")
        print("  Position name:", position_name)
        if is_donor:
            print("  Using donor pairs:")
            print(f"    N pair 1: {field_params.get('n_pair1', 'Not set')}")
            print(f"    N pair 2: {field_params.get('n_pair2', 'Not set')}")
        else:
            print("  Using acceptor pairs:")
            print(f"    N pair 1: {field_params.get('n_pair1', 'Not set')}")
            print(f"    N pair 2: {field_params.get('n_pair2', 'Not set')}")
        
        print(f"\n  Current configuration:")
        print(f"    N pair 1: {field_params.get('n_pair1', 'Not set')}")
        print(f"    N pair 2: {field_params.get('n_pair2', 'Not set')}")
        print(f"    Direction: {field_params.get('direction', 'Not set')}")
        print(f"    Distribution: {field_params.get('distribution', 'Not set')}")
        print(f"    Magnitude: {field_params.get('magnitude', 'Not set')}\n")

    # Get frame count and initialize generator
    num_frames = get_frame_count(xyz_file_path)
    print(f"\nProcessing {xyz_file_path}")
    print(f"  Position: {position_name}")
    print(f"  Number of frames: {num_frames}")

    # Determine if this is a donor or acceptor position
    is_donor = 'donor' in position_name.lower()

    if field_params['approach'] == 'direction_from_md':
        print("  Using MD-derived directions:")
        # Calculate some statistics about the MD field directions
        Ex, Ey, Ez = md_fields
        magnitudes = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        print(f"  MD field statistics:")
        print(f"    Average magnitude: {np.mean(magnitudes):.3f} V/Å")
        print(f"    Std deviation: {np.std(magnitudes):.3f} V/Å")
        print(f"    Min magnitude: {np.min(magnitudes):.3f} V/Å")
        print(f"    Max magnitude: {np.max(magnitudes):.3f} V/Å")

    # Create generator with appropriate parameters based on approach
    if field_params['approach'] == 'heme_based':
        generator = SyntheticFieldGenerator(
            num_frames=num_frames,
            approach=field_params['approach'],
            n_pair1=field_params['n_pair1'],
            n_pair2=field_params['n_pair2']
        )
    else:
        generator = SyntheticFieldGenerator(
            num_frames=num_frames,
            approach=field_params['approach']
        )

    # Generate fields using specified approach
    print(f"\nGenerating synthetic fields:")
    print(f"  Target magnitude: {field_params['magnitude']} V/Å")
    print(f"  Distribution: {field_params['distribution']}")
    if field_params.get('std_dev'):
        print(f"  Standard deviation: {field_params['std_dev']}")
    if field_params.get('range_width'):
        print(f"  Range width: {field_params['range_width']}")

    # Generate fields using specified approach
    Ex, Ey, Ez = generator.generate_fields(
        distribution=field_params['distribution'],
        direction=field_params.get('direction'),  # Only used for heme_based
        magnitude=field_params['magnitude'],
        std_dev=field_params.get('std_dev'),
        range_width=field_params.get('range_width'),
        md_fields=md_fields,
        qm_structure=qm_structure,
        rotation_matrices=rotation_matrices,
        output_dir=output_dir if output_dir else None,
        position_name=position_name if position_name else None
    )

    # Verify field shapes before writing
    expected_frames = num_frames
    if not (Ex.shape == Ey.shape == Ez.shape == (expected_frames,)):
        raise ValueError(f"Generated field shapes incorrect. Expected ({expected_frames},), "
                       f"got Ex: {Ex.shape}, Ey: {Ey.shape}, Ez: {Ez.shape}")

    # Modified filename construction
    approach = field_params['approach']
    direction = field_params.get('direction', 'md_derived')
    distribution = field_params['distribution']
    magnitude = field_params['magnitude']
    
    base_filename = f"synthetic_efield_{approach}_{direction}_{position_name}_{distribution}_{magnitude:.3f}.txt"
    if iteration is not None:
        base_filename = base_filename.replace('.txt', f'_iteration_{iteration:03d}.txt')
    
    field_path = os.path.join(output_dir, base_filename)
    
    print(f"\nSaving synthetic fields to: {field_path}")
    write_electric_field(field_path, Ex, Ey, Ez)

    # Print statistics about generated fields
    magnitudes = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    print(f"Generated field statistics:")
    print(f"  Average magnitude: {np.mean(magnitudes):.3f} V/Å")
    print(f"  Std deviation: {np.std(magnitudes):.3f} V/Å")
    print(f"  Min magnitude: {np.min(magnitudes):.3f} V/Å")
    print(f"  Max magnitude: {np.max(magnitudes):.3f} V/Å")

    return field_path

class FieldStatistics(NamedTuple):
    """Container for electric field statistics"""
    magnitude: np.ndarray
    average: float
    std_dev: float

class EnergyStatistics(NamedTuple):
    """Container for energy statistics"""
    average: float
    variance: float

class ProcessStateResult(NamedTuple):
    """Container for process_state results"""
    donor_tensors: Tuple[np.ndarray, np.ndarray, np.ndarray]  # red, ox, diffalpha
    acceptor_tensors: Tuple[np.ndarray, np.ndarray, np.ndarray]  # ox, red, diffalpha
    donor_components: Tuple[np.ndarray, ...]  # 6 components
    acceptor_components: Tuple[np.ndarray, ...]  # 6 components
    donor_fields: Tuple[np.ndarray, np.ndarray, np.ndarray]  # Ex, Ey, Ez transformed
    donor_polarization: np.ndarray
    acceptor_fields: Tuple[np.ndarray, np.ndarray, np.ndarray]  # Ex, Ey, Ez transformed
    acceptor_polarization: np.ndarray
    total_polarization: np.ndarray
    coulombic_energy: np.ndarray
    total_energy: np.ndarray
    coulombic_stats: EnergyStatistics
    total_stats: EnergyStatistics
    donor_field_stats: FieldStatistics
    acceptor_field_stats: FieldStatistics

def process_state_md_to_qm(state_name: str,
                 paths: Dict[str, str],
                 common_indices: np.ndarray,
                 E_fields: Tuple[np.ndarray, ...],
                 txt_output_folder: str,
                 basis_set_scaling: float,
                 is_synthetic: bool = False,
                 iteration: Optional[int] = None) -> ProcessStateResult:
    """
    Process a single state with MD to QM coordinate transformations of electric fields.

    Args:
        state_name: Name of state being processed ('reactant' or 'product')
        paths: Dictionary of file paths for input files
        common_indices: Array of frame indices to process
        E_fields: Tuple of electric field components in order:
                 (donor_Ex[frames], donor_Ey[frames], donor_Ez[frames],
                  acceptor_Ex[frames], acceptor_Ey[frames], acceptor_Ez[frames])
        txt_output_folder: Output directory for text files
        basis_set_scaling: Scaling factor for basis set
        is_synthetic: Whether using synthetic fields
        iteration: Optional iteration number for iterative runs

    Return tuple structure (38 elements total):
        [0-2]   Donor average tensors (reduced, oxidized, diffalpha)
        [3-5]   Acceptor average tensors (oxidized, reduced, diffalpha)
        [6-11]  Donor frame-by-frame diffalpha components (xx[frames],xy[frames],xz[frames],
                                                         yy[frames],yz[frames],zz[frames])
        [12-17] Acceptor frame-by-frame diffalpha components (xx[frames],xy[frames],xz[frames],
                                                            yy[frames],yz[frames],zz[frames])
        [18-20] Donor frame-by-frame electric field components (Ex[frames],Ey[frames],Ez[frames])
        [21]    Donor frame-by-frame polarization energies [frames]
        [22-24] Acceptor frame-by-frame electric field components (Ex[frames],Ey[frames],Ez[frames])
        [25]    Acceptor frame-by-frame polarization energies [frames]
        [26]    Frame-by-frame total polarization energies [frames]
        [27]    Frame-by-frame Coulombic energy [frames]
        [28]    Frame-by-frame total energy [frames]
        [29-32] Statistics (avg coulombic, var coulombic, avg total, var total)
        [33-38] Field statistics (donor_mag[frames], acceptor_mag[frames],
                                donor_avg, donor_std, acceptor_avg, acceptor_std)

    Raises:
        ValueError: If input arrays have inconsistent shapes or invalid values
    """
    logger.info(f"\n=== Processing {state_name.capitalize()} State ===")

    # Input validation
    logger.info("\nValidating inputs...")
    if not os.path.isdir(txt_output_folder):
        raise ValueError(f"Output directory does not exist: {txt_output_folder}")
        
    required_paths = [
        'red_ref_tensor', 'ox_ref_tensor', f'{state_name}_donor_xyz',
        'red_ref_coords', f'{state_name}_acceptor_xyz', 'ox_ref_coords',
        f'{state_name}_coulombic'
    ]
    missing_paths = [p for p in required_paths if p not in paths]
    if missing_paths:
        raise ValueError(f"Missing required paths: {', '.join(missing_paths)}")
        
    if not isinstance(common_indices, np.ndarray):
        raise ValueError("common_indices must be a numpy array")
    if len(common_indices) == 0:
        raise ValueError("common_indices cannot be empty")
    if len(E_fields) != 6:
        raise ValueError(f"Expected 6 field components, got {len(E_fields)}")
    if basis_set_scaling <= 0:
        raise ValueError(f"basis_set_scaling must be positive, got {basis_set_scaling}")

    try:
        # Step 1: Coordinate transformations
        logger.info("\nStep 1: Performing coordinate transformations from MD to QM coordinate system...")
        
        # Process donor transformation
        donor_aligned_coords = append_iteration_number(f"{state_name}_donor_aligned_coords.xyz", iteration)
        donor_rmsd = append_iteration_number(f"{state_name}_donor_transdRMSD.txt", iteration)
        donor_transforms = append_iteration_number(f"{state_name}_donor_transforms.csv", iteration)
        
        donor_transmat = ss.superimpose_and_save(
            paths['red_ref_coords'], #= QM = reference coordinates
            paths[f'{state_name}_donor_xyz'], #= MD = mobile coordinates 
            construct_output_path(txt_output_folder, donor_aligned_coords),
            construct_output_path(txt_output_folder, donor_rmsd),
            construct_output_path(txt_output_folder, donor_transforms),
            mode="md_to_qm",
            frame_indices=common_indices
        )

        # Process acceptor transformation
        acceptor_aligned_coords = append_iteration_number(f"{state_name}_acceptor_aligned_coords.xyz", iteration)
        acceptor_rmsd = append_iteration_number(f"{state_name}_acceptor_transdRMSD.txt", iteration)
        acceptor_transforms = append_iteration_number(f"{state_name}_acceptor_transforms.csv", iteration)

        acceptor_transmat = ss.superimpose_and_save(
            paths['ox_ref_coords'], #= QM = reference coordinates
            paths[f'{state_name}_acceptor_xyz'], #= MD = mobile coordinates 
            construct_output_path(txt_output_folder, acceptor_aligned_coords),
            construct_output_path(txt_output_folder, acceptor_rmsd),
            construct_output_path(txt_output_folder, acceptor_transforms),
            mode="md_to_qm",
            frame_indices=common_indices
        )

        # Step 2: Process and transform electric fields
        logger.info("\nStep 2: Processing and transforming electric fields...")
        E_x_donor, E_y_donor, E_z_donor, E_x_acceptor, E_y_acceptor, E_z_acceptor = E_fields
        
        # Validate field shapes
        field_shapes = {
            'donor_Ex': len(E_x_donor),
            'donor_Ey': len(E_y_donor),
            'donor_Ez': len(E_z_donor),
            'acceptor_Ex': len(E_x_acceptor),
            'acceptor_Ey': len(E_y_acceptor),
            'acceptor_Ez': len(E_z_acceptor)
        }
        
        if len(set(field_shapes.values())) != 1:
            raise ValueError("Inconsistent number of frames in field components: " + 
                           ", ".join(f"{k}: {v}" for k, v in field_shapes.items()))

        # Transform donor fields
        donor_matrices = rtm.read_transformation_matrices(donor_transmat)
        transformed_donor_filename = append_iteration_number(f"{state_name}_donor_transformed_efield.txt", iteration)
        
        E_x_donor_trans, E_y_donor_trans, E_z_donor_trans = tasf.transform_and_save_fields(
            donor_matrices,
            E_x_donor, E_y_donor, E_z_donor,
            construct_output_path(txt_output_folder, transformed_donor_filename),
            common_indices
        )

        # Transform acceptor fields
        acceptor_matrices = rtm.read_transformation_matrices(acceptor_transmat)
        transformed_acceptor_filename = append_iteration_number(f"{state_name}_acceptor_transformed_efield.txt", iteration)
        
        E_x_acceptor_trans, E_y_acceptor_trans, E_z_acceptor_trans = tasf.transform_and_save_fields(
            acceptor_matrices,
            E_x_acceptor, E_y_acceptor, E_z_acceptor,
            construct_output_path(txt_output_folder, transformed_acceptor_filename),
            common_indices
        )

        # Step 3: Read and process reference tensors
        logger.info("\nStep 3: Reading reference tensors...")
        red_tensor = rt.read_tensors(paths['red_ref_tensor'])  # List containing one 3x3 tensor
        ox_tensor = rt.read_tensors(paths['ox_ref_tensor'])    # List containing one 3x3 tensor

        # For single tensor case, take the first (and only) tensor from each list
        red_tensor = red_tensor[0]  # Now a 3x3 numpy array
        ox_tensor = ox_tensor[0]    # Now a 3x3 numpy array

        # Step 4: Compute difference tensors
        logger.info("\nStep 4: Computing difference polarizability tensors...")
        # Single tensor differences - now working with 3x3 arrays
        donor_diffalpha = (ox_tensor - red_tensor) * basis_set_scaling
        acceptor_diffalpha = (red_tensor - ox_tensor) * basis_set_scaling

        # Extract tensor components (single tensor)
        donor_components = [donor_diffalpha[i,j] for i in range(3) for j in range(i,3)]
        acceptor_components = [acceptor_diffalpha[i,j] for i in range(3) for j in range(i,3)]

        # Create arrays of repeated tensor components to match number of frames
        num_frames = len(E_x_donor_trans)
        donor_repeated_components = [np.full_like(E_x_donor_trans, comp) for comp in donor_components]
        acceptor_repeated_components = [np.full_like(E_x_acceptor_trans, comp) for comp in acceptor_components]

        # Step 5: Compute polarization energies using transformed fields
        logger.info("\nStep 5: Computing polarization energies...")
    
        polarization_energies_donor = cpe.compute_polarization_energy(
            *donor_repeated_components,  # Single tensor components repeated for all frames
            E_x_donor_trans, E_y_donor_trans, E_z_donor_trans  # Transformed field components
        )
    
        polarization_energies_acceptor = cpe.compute_polarization_energy(
            *acceptor_repeated_components,  # Single tensor components repeated for all frames
            E_x_acceptor_trans, E_y_acceptor_trans, E_z_acceptor_trans  # Transformed field components
        )

        total_polarization_energies = np.array(polarization_energies_donor) + \
                                np.array(polarization_energies_acceptor)
        # Step 6: Process Coulombic energies
        logger.info("\nStep 6: Processing Coulombic energies...")
        coulombic_energy = rce.read_coulombic_energy(
            paths[f'{state_name}_coulombic'],
            common_indices
        )
    
        total_energy = total_polarization_energies + coulombic_energy

        # Step 7: Compute statistics
        logger.info("\nStep 7: Computing statistics...")

        # Donor Polarization Energy Statistics
        logger.info("\nDonor Polarization Energy Statistics:")
        donor_pol_avg = np.mean(polarization_energies_donor)
        donor_pol_var = np.var(polarization_energies_donor, ddof=1)
        donor_pol_std = np.std(polarization_energies_donor, ddof=1)
        logger.info(f"- Average:           {donor_pol_avg:.4f} eV")
        logger.info(f"- Variance:          {donor_pol_var:.4f} eV²")
        logger.info(f"- Standard Dev:      {donor_pol_std:.4f} eV")
        logger.info(f"- Range:             {np.min(polarization_energies_donor):.4f} to {np.max(polarization_energies_donor):.4f} eV")

        # Acceptor Polarization Energy Statistics
        logger.info("\nAcceptor Polarization Energy Statistics:")
        acceptor_pol_avg = np.mean(polarization_energies_acceptor)
        acceptor_pol_var = np.var(polarization_energies_acceptor, ddof=1)
        acceptor_pol_std = np.std(polarization_energies_acceptor, ddof=1)
        logger.info(f"- Average:           {acceptor_pol_avg:.4f} eV")
        logger.info(f"- Variance:          {acceptor_pol_var:.4f} eV²")
        logger.info(f"- Standard Dev:      {acceptor_pol_std:.4f} eV")
        logger.info(f"- Range:             {np.min(polarization_energies_acceptor):.4f} to {np.max(polarization_energies_acceptor):.4f} eV")

        # Total Polarization Energy Statistics
        logger.info("\nTotal Polarization Energy Statistics:")
        total_pol_avg = np.mean(total_polarization_energies)
        total_pol_var = np.var(total_polarization_energies, ddof=1)
        total_pol_std = np.std(total_polarization_energies, ddof=1)
        logger.info(f"- Average:           {total_pol_avg:.4f} eV")
        logger.info(f"- Variance:          {total_pol_var:.4f} eV²")
        logger.info(f"- Standard Dev:      {total_pol_std:.4f} eV")
        logger.info(f"- Range:             {np.min(total_polarization_energies):.4f} to {np.max(total_polarization_energies):.4f} eV")

        # Coulombic Energy Statistics
        logger.info("\nCoulombic Energy Statistics:")
        coulombic_stats = EnergyStatistics(
            average=np.mean(coulombic_energy),
            variance=np.var(coulombic_energy, ddof=1)
        )
        coulombic_std = np.std(coulombic_energy, ddof=1)
        logger.info(f"- Average:           {coulombic_stats.average:.4f} eV")
        logger.info(f"- Variance:          {coulombic_stats.variance:.4f} eV²")
        logger.info(f"- Standard Dev:      {coulombic_std:.4f} eV")
        logger.info(f"- Range:             {np.min(coulombic_energy):.4f} to {np.max(coulombic_energy):.4f} eV")

        # Total Energy Statistics
        logger.info("\nTotal Energy Statistics:")
        total_stats = EnergyStatistics(
            average=np.mean(total_energy),
            variance=np.var(total_energy, ddof=1)
        )

        total_std = np.std(total_energy, ddof=1)
        logger.info(f"- Average:           {total_stats.average:.4f} eV")
        logger.info(f"- Variance:          {total_stats.variance:.4f} eV²")
        logger.info(f"- Standard Dev:      {total_std:.4f} eV")
        logger.info(f"- Range:             {np.min(total_energy):.4f} to {np.max(total_energy):.4f} eV")

        # Field Statistics (using transformed fields)
        logger.info("\nDonor Field Statistics:")
        donor_E_mag_trans = np.sqrt(E_x_donor_trans**2 + E_y_donor_trans**2 + E_z_donor_trans**2)
        donor_field_stats = FieldStatistics(
            magnitude=donor_E_mag_trans,
            average=np.mean(donor_E_mag_trans),
            std_dev=np.std(donor_E_mag_trans, ddof=1)
        )
        logger.info(f"- Average Magnitude: {donor_field_stats.average:.4f} V/Å")
        logger.info(f"- Standard Dev:      {donor_field_stats.std_dev:.4f} V/Å")
        logger.info(f"- Range:             {np.min(donor_E_mag_trans):.4f} to {np.max(donor_E_mag_trans):.4f} V/Å")
        logger.info("\nDonor Field Components:")
        logger.info(f"- Ex: avg = {np.mean(E_x_donor_trans):.4f} ± {np.std(E_x_donor_trans, ddof=1):.4f} V/Å")
        logger.info(f"- Ey: avg = {np.mean(E_y_donor_trans):.4f} ± {np.std(E_y_donor_trans, ddof=1):.4f} V/Å")
        logger.info(f"- Ez: avg = {np.mean(E_z_donor_trans):.4f} ± {np.std(E_z_donor_trans, ddof=1):.4f} V/Å")

        logger.info("\nAcceptor Field Statistics:")
        acceptor_E_mag_trans = np.sqrt(E_x_acceptor_trans**2 + E_y_acceptor_trans**2 + E_z_acceptor_trans**2)
        acceptor_field_stats = FieldStatistics(
            magnitude=acceptor_E_mag_trans,
            average=np.mean(acceptor_E_mag_trans),
            std_dev=np.std(acceptor_E_mag_trans, ddof=1)
        )
        logger.info(f"- Average Magnitude: {acceptor_field_stats.average:.4f} V/Å")
        logger.info(f"- Standard Dev:      {acceptor_field_stats.std_dev:.4f} V/Å")
        logger.info(f"- Range:             {np.min(acceptor_E_mag_trans):.4f} to {np.max(acceptor_E_mag_trans):.4f} V/Å")
        logger.info("\nAcceptor Field Components:")
        logger.info(f"- Ex: avg = {np.mean(E_x_acceptor_trans):.4f} ± {np.std(E_x_acceptor_trans, ddof=1):.4f} V/Å")
        logger.info(f"- Ey: avg = {np.mean(E_y_acceptor_trans):.4f} ± {np.std(E_y_acceptor_trans, ddof=1):.4f} V/Å")
        logger.info(f"- Ez: avg = {np.mean(E_z_acceptor_trans):.4f} ± {np.std(E_z_acceptor_trans, ddof=1):.4f} V/Å")

        logger.info("\nSummary of Key Quantities:")
        logger.info(f"- Total Energy:            {total_stats.average:.4f} ± {total_std:.4f} eV")
        logger.info(f"- Total Polarization:      {total_pol_avg:.4f} ± {total_pol_std:.4f} eV")
        logger.info(f"- Coulombic Energy:        {coulombic_stats.average:.4f} ± {coulombic_std:.4f} eV")
        logger.info(f"- Donor Field Magnitude:   {donor_field_stats.average:.4f} ± {donor_field_stats.std_dev:.4f} V/Å")
        logger.info(f"- Acceptor Field Magnitude: {acceptor_field_stats.average:.4f} ± {acceptor_field_stats.std_dev:.4f} V/Å")        

        logger.info(f"\n=== {state_name.capitalize()} State Processing Complete ===")

        return ProcessStateResult(
            donor_tensors=(red_tensor, ox_tensor, donor_diffalpha),
            acceptor_tensors=(ox_tensor, red_tensor, acceptor_diffalpha),
            donor_components=tuple(donor_diffalpha[i,j] for i in range(3) for j in range(i,3)),
            acceptor_components=tuple(acceptor_diffalpha[i,j] for i in range(3) for j in range(i,3)),
            donor_fields=(E_x_donor_trans, E_y_donor_trans, E_z_donor_trans),
            donor_polarization=polarization_energies_donor,
            acceptor_fields=(E_x_acceptor_trans, E_y_acceptor_trans, E_z_acceptor_trans),
            acceptor_polarization=polarization_energies_acceptor,
            total_polarization=total_polarization_energies,
            coulombic_energy=coulombic_energy,
            total_energy=total_energy,
            coulombic_stats=coulombic_stats,
            total_stats=total_stats,
            donor_field_stats=donor_field_stats,
            acceptor_field_stats=acceptor_field_stats
        )

    except Exception as e:
        logger.error(f"\nError processing {state_name} state: {str(e)}")
        raise RuntimeError(f"Failed to process {state_name} state") from e

def process_state_qm_to_md(state_name: str,
                 paths: Dict[str, str],
                 common_indices: np.ndarray,
                 E_fields: Tuple[np.ndarray, ...],
                 txt_output_folder: str,
                 basis_set_scaling: float,
                 is_synthetic: bool = False,
                 iteration: Optional[int] = None) -> ProcessStateResult:
    """
    Process a single state with QM to MD coordinate transformations of difference polarizability tensors.

    Args:
        state_name: Name of state being processed ('reactant' or 'product')
        paths: Dictionary of file paths for input files
        common_indices: Array of frame indices to process
        E_fields: Tuple of electric field components in order:
                 (donor_Ex[frames], donor_Ey[frames], donor_Ez[frames],
                  acceptor_Ex[frames], acceptor_Ey[frames], acceptor_Ez[frames])
        txt_output_folder: Output directory for text files
        basis_set_scaling: Scaling factor for basis set
        is_synthetic: Whether using synthetic fields
        iteration: Optional iteration number for iterative runs

    Return tuple structure (38 elements total):
        [0-2]   Donor average tensors (reduced, oxidized, diffalpha)
        [3-5]   Acceptor average tensors (oxidized, reduced, diffalpha)
        [6-11]  Donor frame-by-frame diffalpha components (xx[frames],xy[frames],xz[frames],
                                                         yy[frames],yz[frames],zz[frames])
        [12-17] Acceptor frame-by-frame diffalpha components (xx[frames],xy[frames],xz[frames],
                                                            yy[frames],yz[frames],zz[frames])
        [18-20] Donor frame-by-frame electric field components (Ex[frames],Ey[frames],Ez[frames])
        [21]    Donor frame-by-frame polarization energies [frames]
        [22-24] Acceptor frame-by-frame electric field components (Ex[frames],Ey[frames],Ez[frames])
        [25]    Acceptor frame-by-frame polarization energies [frames]
        [26]    Frame-by-frame total polarization energies [frames]
        [27]    Frame-by-frame Coulombic energy [frames]
        [28]    Frame-by-frame total energy [frames]
        [29-32] Statistics (avg coulombic, var coulombic, avg total, var total)
        [33-38] Field statistics (donor_mag[frames], acceptor_mag[frames],
                                donor_avg, donor_std, acceptor_avg, acceptor_std)

    Raises:
        ValueError: If input arrays have inconsistent shapes or invalid values
    """
    logger.info(f"\n=== Processing {state_name.capitalize()} State ===")

    # Input validation
    logger.info("\nValidating inputs...")
    
    # Validate paths and directory
    if not os.path.isdir(txt_output_folder):
        raise ValueError(f"Output directory does not exist: {txt_output_folder}")
    
    required_paths = [
        'red_ref_tensor', 'ox_ref_tensor', f'{state_name}_donor_xyz',
        'red_ref_coords', f'{state_name}_acceptor_xyz', 'ox_ref_coords',
        f'{state_name}_coulombic'
    ]
    missing_paths = [p for p in required_paths if p not in paths]
    if missing_paths:
        raise ValueError(f"Missing required paths: {', '.join(missing_paths)}")
        
    for path_key, path_value in paths.items():
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"File not found for {path_key}: {path_value}")

    # Validate numerical inputs
    if not isinstance(common_indices, np.ndarray):
        raise ValueError("common_indices must be a numpy array")
    if len(common_indices) == 0:
        raise ValueError("common_indices cannot be empty")
    if basis_set_scaling <= 0:
        raise ValueError(f"basis_set_scaling must be positive, got {basis_set_scaling}")
    if len(E_fields) != 6:
        raise ValueError(f"Expected 6 field components, got {len(E_fields)}")
    
    try:
        # Step 1: Read reference tensors 
        logger.info("\nStep 1: Reading reference tensors...")
        logger.info(f"- Reading reduced state tensor from: {paths['red_ref_tensor']}")
        red_tensor = rt.read_tensors(paths['red_ref_tensor'])[0]  # Single reference tensor
        logger.info(f"- Reading oxidized state tensor from: {paths['ox_ref_tensor']}")
        ox_tensor = rt.read_tensors(paths['ox_ref_tensor'])[0]    # Single reference tensor

        # Step 2: Computing difference polarizability tensors
        logger.info("\nStep 2: Computing difference polarizability tensors...")
    
        logger.info("\na) Processing donor difference tensor (ox - red):")
        logger.info(f"- Applying basis set scaling factor: {basis_set_scaling}")
        donor_diffalpha = (ox_tensor - red_tensor) * basis_set_scaling  # Single reference difference tensor

        logger.info("\nb) Processing acceptor difference tensor (red - ox):")
        logger.info(f"- Applying basis set scaling factor: {basis_set_scaling}")
        acceptor_diffalpha = (red_tensor - ox_tensor) * basis_set_scaling  # Single reference difference tensor

 
        # Step 3: Coordinate transformations
        logger.info("\nStep 3: Performing coordinate transformations...")
        
        # Donor transformations
        donor_aligned_coords = append_iteration_number(f"{state_name}_donor_aligned_coords.xyz", iteration)
        donor_rmsd = append_iteration_number(f"{state_name}_donor_transdRMSD.txt", iteration)
        donor_transforms = append_iteration_number(f"{state_name}_donor_transforms.csv", iteration)
        
        donor_transmat = ss.superimpose_and_save(
            paths[f'{state_name}_donor_xyz'], #= MD = reference coordinates 
            paths['red_ref_coords'],          #= QM = mobile coordinates
            construct_output_path(txt_output_folder, donor_aligned_coords),
            construct_output_path(txt_output_folder, donor_rmsd),
            construct_output_path(txt_output_folder, donor_transforms),
            mode="qm_to_md",
            frame_indices=common_indices
        )

        # Acceptor transformations
        acceptor_aligned_coords = append_iteration_number(f"{state_name}_acceptor_aligned_coords.xyz", iteration)
        acceptor_rmsd = append_iteration_number(f"{state_name}_acceptor_transdRMSD.txt", iteration)
        acceptor_transforms = append_iteration_number(f"{state_name}_acceptor_transforms.csv", iteration)

        acceptor_transmat = ss.superimpose_and_save(
            paths[f'{state_name}_acceptor_xyz'], #= MD = reference coordinates 
            paths['ox_ref_coords'],              #= QM = mobile coordinates
            construct_output_path(txt_output_folder, acceptor_aligned_coords),
            construct_output_path(txt_output_folder, acceptor_rmsd),
            construct_output_path(txt_output_folder, acceptor_transforms),
            mode="qm_to_md",
            frame_indices=common_indices
        )

        # Step 4: Transform difference tensors
        logger.info("\nStep 4: Applying coordinate transformations to difference tensors...")
    
        donor_diffalpha_trans_file = append_iteration_number(f"{state_name}_donor_diffalpha_trans.txt", iteration)
        donor_diffalpha_trans = tast.transform_and_save_tensors(  # Returns array of shape (n_frames, 3, 3)
            donor_transmat,
            donor_diffalpha,  # Single tensor that gets transformed for each frame
            construct_output_path(txt_output_folder, donor_diffalpha_trans_file),
            common_indices
        )

        acceptor_diffalpha_trans_file = append_iteration_number(f"{state_name}_acceptor_difalpha_trans.txt", iteration)
        acceptor_diffalpha_trans = tast.transform_and_save_tensors(  # Returns array of shape (n_frames, 3, 3)
            acceptor_transmat,
            acceptor_diffalpha,  # Single tensor that gets transformed for each frame
            construct_output_path(txt_output_folder, acceptor_diffalpha_trans_file),
            common_indices
        )

        # Step 5: Process electric fields
        logger.info("\nStep 5: Processing electric fields...")
        E_x_donor, E_y_donor, E_z_donor, E_x_acceptor, E_y_acceptor, E_z_acceptor = E_fields

        # Step 6: Compute polarization energies
        logger.info("\nStep 6: Computing polarization energies...")
    
        # Extract components - get array of shape (n_frames,) for each component
        donor_components = [donor_diffalpha_trans[:, i, j] for i in range(3) for j in range(i, 3)]
        acceptor_components = [acceptor_diffalpha_trans[:, i, j] for i in range(3) for j in range(i, 3)]
    
        # Calculate polarization energies for all frames
        polarization_energies_donor = cpe.compute_polarization_energy(
            *donor_components,  # Frame-by-frame tensor components
            E_x_donor, E_y_donor, E_z_donor  # Original MD field components
        )
    
        polarization_energies_acceptor = cpe.compute_polarization_energy(
            *acceptor_components,  # Frame-by-frame tensor components
            E_x_acceptor, E_y_acceptor, E_z_acceptor  # Original MD field components
        )

        total_polarization_energies = polarization_energies_donor + polarization_energies_acceptor

        # Step 7: Process Coulombic energies
        logger.info("\nStep 7: Processing Coulombic energies...")
        coulombic_energy = rce.read_coulombic_energy(
            paths[f'{state_name}_coulombic'],
            common_indices
        )

        # Calculate total energy
        total_energy = total_polarization_energies + coulombic_energy

        # Step 8: Compute statistics
        logger.info("\nStep 8: Computing statistics...")
                
        # Donor Polarization Energy Statistics
        logger.info("\nDonor Polarization Energy Statistics:")
        donor_pol_avg = np.mean(polarization_energies_donor)
        donor_pol_var = np.var(polarization_energies_donor, ddof=1)
        donor_pol_std = np.std(polarization_energies_donor, ddof=1)
        logger.info(f"- Average:           {donor_pol_avg:.4f} eV")
        logger.info(f"- Variance:          {donor_pol_var:.4f} eV²")
        logger.info(f"- Standard Dev:      {donor_pol_std:.4f} eV")
        logger.info(f"- Range:             {np.min(polarization_energies_donor):.4f} to {np.max(polarization_energies_donor):.4f} eV")

        # Acceptor Polarization Energy Statistics
        logger.info("\nAcceptor Polarization Energy Statistics:")
        acceptor_pol_avg = np.mean(polarization_energies_acceptor)
        acceptor_pol_var = np.var(polarization_energies_acceptor, ddof=1)
        acceptor_pol_std = np.std(polarization_energies_acceptor, ddof=1)
        logger.info(f"- Average:           {acceptor_pol_avg:.4f} eV")
        logger.info(f"- Variance:          {acceptor_pol_var:.4f} eV²")
        logger.info(f"- Standard Dev:      {acceptor_pol_std:.4f} eV")
        logger.info(f"- Range:             {np.min(polarization_energies_acceptor):.4f} to {np.max(polarization_energies_acceptor):.4f} eV")

        # Total Polarization Energy Statistics
        logger.info("\nTotal Polarization Energy Statistics:")
        total_pol_avg = np.mean(total_polarization_energies)
        total_pol_var = np.var(total_polarization_energies, ddof=1)
        total_pol_std = np.std(total_polarization_energies, ddof=1)
        logger.info(f"- Average:           {total_pol_avg:.4f} eV")
        logger.info(f"- Variance:          {total_pol_var:.4f} eV²")
        logger.info(f"- Standard Dev:      {total_pol_std:.4f} eV")
        logger.info(f"- Range:             {np.min(total_polarization_energies):.4f} to {np.max(total_polarization_energies):.4f} eV")

        # Coulombic Energy Statistics
        logger.info("\nCoulombic Energy Statistics:")
        coulombic_stats = EnergyStatistics(
            average=np.mean(coulombic_energy),
            variance=np.var(coulombic_energy, ddof=1)
        )
        coulombic_std = np.std(coulombic_energy, ddof=1)
        logger.info(f"- Average:           {coulombic_stats.average:.4f} eV")
        logger.info(f"- Variance:          {coulombic_stats.variance:.4f} eV²")
        logger.info(f"- Standard Dev:      {coulombic_std:.4f} eV")
        logger.info(f"- Range:             {np.min(coulombic_energy):.4f} to {np.max(coulombic_energy):.4f} eV")

        # Total Energy Statistics
        logger.info("\nTotal Energy Statistics:")
        total_stats = EnergyStatistics(
            average=np.mean(total_energy),
            variance=np.var(total_energy, ddof=1)
        )
        total_std = np.std(total_energy, ddof=1)
        logger.info(f"- Average:           {total_stats.average:.4f} eV")
        logger.info(f"- Variance:          {total_stats.variance:.4f} eV²")
        logger.info(f"- Standard Dev:      {total_std:.4f} eV")
        logger.info(f"- Range:             {np.min(total_energy):.4f} to {np.max(total_energy):.4f} eV")

        # Field Statistics (using transformed fields)
        logger.info("\nDonor Field Statistics:")
        donor_E_mag_trans = np.sqrt(E_x_donor**2 + E_y_donor**2 + E_z_donor**2)
        donor_field_stats = FieldStatistics(
            magnitude=donor_E_mag_trans,
            average=np.mean(donor_E_mag_trans),
            std_dev=np.std(donor_E_mag_trans, ddof=1)
        )
        logger.info(f"- Average Magnitude: {donor_field_stats.average:.4f} V/Å")
        logger.info(f"- Standard Dev:      {donor_field_stats.std_dev:.4f} V/Å")
        logger.info(f"- Range:             {np.min(donor_E_mag_trans):.4f} to {np.max(donor_E_mag_trans):.4f} V/Å")
        logger.info("\nDonor Field Components:")
        logger.info(f"- Ex: avg = {np.mean(E_x_donor):.4f} ± {np.std(E_x_donor, ddof=1):.4f} V/Å")
        logger.info(f"- Ey: avg = {np.mean(E_y_donor):.4f} ± {np.std(E_y_donor, ddof=1):.4f} V/Å")
        logger.info(f"- Ez: avg = {np.mean(E_z_donor):.4f} ± {np.std(E_z_donor, ddof=1):.4f} V/Å")

        logger.info("\nAcceptor Field Statistics:")
        acceptor_E_mag_trans = np.sqrt(E_x_acceptor**2 + E_y_acceptor**2 + E_z_acceptor**2)
        acceptor_field_stats = FieldStatistics(
            magnitude=acceptor_E_mag_trans,
            average=np.mean(acceptor_E_mag_trans),
            std_dev=np.std(acceptor_E_mag_trans, ddof=1)
        )
        logger.info(f"- Average Magnitude: {acceptor_field_stats.average:.4f} V/Å")
        logger.info(f"- Standard Dev:      {acceptor_field_stats.std_dev:.4f} V/Å")
        logger.info(f"- Range:             {np.min(acceptor_E_mag_trans):.4f} to {np.max(acceptor_E_mag_trans):.4f} V/Å")
        logger.info("\nAcceptor Field Components:")
        logger.info(f"- Ex: avg = {np.mean(E_x_acceptor):.4f} ± {np.std(E_x_acceptor, ddof=1):.4f} V/Å")
        logger.info(f"- Ey: avg = {np.mean(E_y_acceptor):.4f} ± {np.std(E_y_acceptor, ddof=1):.4f} V/Å")
        logger.info(f"- Ez: avg = {np.mean(E_z_acceptor):.4f} ± {np.std(E_z_acceptor, ddof=1):.4f} V/Å")

        logger.info("\nSummary of Key Quantities:")
        logger.info(f"- Total Energy:            {total_stats.average:.4f} ± {total_std:.4f} eV")
        logger.info(f"- Total Polarization:      {total_pol_avg:.4f} ± {total_pol_std:.4f} eV")
        logger.info(f"- Coulombic Energy:        {coulombic_stats.average:.4f} ± {coulombic_std:.4f} eV")
        logger.info(f"- Donor Field Magnitude:   {donor_field_stats.average:.4f} ± {donor_field_stats.std_dev:.4f} V/Å")
        logger.info(f"- Acceptor Field Magnitude: {acceptor_field_stats.average:.4f} ± {acceptor_field_stats.std_dev:.4f} V/Å")        

        logger.info(f"\n=== {state_name.capitalize()} State Processing Complete ===")

        return ProcessStateResult(
            donor_tensors=(red_tensor, ox_tensor, donor_diffalpha),
            acceptor_tensors=(ox_tensor, red_tensor, acceptor_diffalpha),
            donor_components=tuple(donor_diffalpha[i,j] for i in range(3) for j in range(i,3)),
            acceptor_components=tuple(acceptor_diffalpha[i,j] for i in range(3) for j in range(i,3)),
            donor_fields=(E_x_donor, E_y_donor, E_z_donor),
            donor_polarization=polarization_energies_donor,
            acceptor_fields=(E_x_acceptor, E_y_acceptor, E_z_acceptor),
            acceptor_polarization=polarization_energies_acceptor,
            total_polarization=total_polarization_energies,
            coulombic_energy=coulombic_energy,
            total_energy=total_energy,
            coulombic_stats=coulombic_stats,
            total_stats=total_stats,
            donor_field_stats=donor_field_stats,
            acceptor_field_stats=acceptor_field_stats
        )
        
    except Exception as e:
        logger.error(f"\nError processing {state_name} state: {str(e)}")
        raise RuntimeError(f"Failed to process {state_name} state") from e

def run_analysis(reactant_donor_id: str,
                reactant_acceptor_id: str,
                product_donor_id: str,
                product_acceptor_id: str,
                basis_set_scaling: float,
                use_synthetic_fields: bool = False,
                synthetic_field_params: Optional[Dict[str, Any]] = None,
                filter_type: Optional[str] = None,
                filter_value: Optional[float] = None,
                filter_tolerance: Optional[float] = None,
                iterative_run: bool = False,
                iteration: Optional[int] = None,
                output_folders: Dict[str, str] = None) -> Tuple:

    print("\n" + "="*80)
    print("=== Starting Analysis ===")
    print("="*80)
    
    # Validate input parameters
    if use_synthetic_fields and synthetic_field_params is None:
        raise ValueError("synthetic_field_params required when use_synthetic_fields is True")
    
    if use_synthetic_fields:
        required_positions = ['reactant_donor', 'reactant_acceptor', 
                            'product_donor', 'product_acceptor']
        for pos in required_positions:
            if pos not in synthetic_field_params:
                raise ValueError(f"Missing field parameters for position: {pos}")
    
    # Set output paths
    print("\nSetting up output directories:")
    print(f"- Text output:      {output_folders['txt']}")
    print(f"- PNG output:       {output_folders['png']}")
    print(f"- Animation output: {output_folders['mp4']}")

    txt_output_folder = output_folders['txt']
    png_output_folder = output_folders['png']
    mp4_output_folder = output_folders['mp4']
    
    # Read and validate file paths
    print("\nReading file paths from configuration...")
#   file_paths = read_file_paths()
#   paths = construct_simulation_paths(file_paths, reactant_donor_id, reactant_acceptor_id,
#                                   product_donor_id, product_acceptor_id)

    paths = rfp.read_file_paths("FilePaths.txt")

    print("\nInput files being used:")
    print("\nReference files:")
    print(f"- Reduced reference coordinates:  {paths['red_ref_coords']}")
    print(f"- Oxidized reference coordinates: {paths['ox_ref_coords']}")
    print(f"- Reduced reference tensor:       {paths['red_ref_tensor']}")
    print(f"- Oxidized reference tensor:      {paths['ox_ref_tensor']}")
    
    print("\nTrajectory files:")
    print(f"- Reactant donor:    {paths['reactant_donor_xyz']}")
    print(f"- Reactant acceptor: {paths['reactant_acceptor_xyz']}")
    print(f"- Product donor:     {paths['product_donor_xyz']}")
    print(f"- Product acceptor:  {paths['product_acceptor_xyz']}")
    
    print("\nEnergy files:")
    print(f"- Reactant coulombic: {paths['reactant_coulombic']}")
    print(f"- Product coulombic:  {paths['product_coulombic']}")
    
    if not use_synthetic_fields or \
    (use_synthetic_fields and synthetic_field_params['approach'] == 'direction_from_md'):
        print("\nElectric field files:")
        print(f"- Reactant donor:    {paths['reactant_donor_efield']}")
        print(f"- Reactant acceptor: {paths['reactant_acceptor_efield']}")
        print(f"- Product donor:     {paths['product_donor_efield']}")
        print(f"- Product acceptor:  {paths['product_acceptor_efield']}")
    
    # Validate required files exist
    print("\nValidating required files...")
    required_files = [
        paths['red_ref_coords'], paths['ox_ref_coords'],
        paths['red_ref_tensor'], paths['ox_ref_tensor'],
        paths['reactant_donor_xyz'], paths['product_donor_xyz'],
        paths['reactant_acceptor_xyz'], paths['product_acceptor_xyz'],
        paths['reactant_coulombic'], paths['product_coulombic']
    ]
    
    if not use_synthetic_fields or \
    (use_synthetic_fields and synthetic_field_params['approach'] == 'direction_from_md'):
        required_files.extend([
            paths['reactant_donor_efield'], paths['reactant_acceptor_efield'],
            paths['product_donor_efield'], paths['product_acceptor_efield']
        ])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\nMissing required files:")
        for f in missing_files:
            print(f"- {f}")
        sys.exit(1)
    print("All required files exist!")

    # Generate output file suffix based on analysis type
    print("\nConstructing output filenames...")
    suffix = "synthetic" if use_synthetic_fields else \
            "no_filter" if filter_type is None else \
            f"{filter_type}_{filter_value:.2f}"

    if use_synthetic_fields:
        # Add parameters for each position to suffix
        for pos in ['reactant_donor', 'reactant_acceptor', 
                   'product_donor', 'product_acceptor']:
            params = synthetic_field_params[pos]
            suffix += f"_{pos.replace('_', '-')}"
            if 'direction' in params:
                suffix += f"_{params['direction']}"
            if params['distribution'] != 'fixed':
                suffix += f"_{params['distribution']}"
            suffix += f"_{params['magnitude']:.3f}"

    # Construct output filenames
    reactant_base = f"{reactant_donor_id}_{reactant_acceptor_id}_reactant_state_{suffix}"
    product_base = f"{reactant_donor_id}_{reactant_acceptor_id}_product_state_{suffix}"
    reorg_base = f"{reactant_donor_id}_{reactant_acceptor_id}_reorganization_energies_{suffix}"

    if iteration is not None:
        print(f"- Adding iteration number: {iteration:03d}")
        reactant_base = f"{reactant_base}_iteration_{iteration:03d}"
        product_base = f"{product_base}_iteration_{iteration:03d}"
        reorg_base = f"{reorg_base}_iteration_{iteration:03d}"

    print("Output files:")
    print(f"- Reactant state:          {reactant_base}.txt")
    print(f"- Product state:           {product_base}.txt")
    print(f"- Reorganization energies: {reorg_base}.txt")

    if use_synthetic_fields:
        print("\n=== Generating Synthetic Electric Fields ===")
        print(f"Using approach: {synthetic_field_params['approach']}")

        # Generate synthetic fields for each position independently
        # Fields are generated using directions from corresponding MD field 
        # or heme-based coordinates
        if synthetic_field_params['approach'] == 'direction_from_md':
            # Read all MD fields without filtering to get directions
            print("\nStep 1: Reading MD fields for directions...")
            
            # Reactant State Fields
            print("\nReactant State MD Field Sources:")
            print(f"Donor directions from:    {paths['reactant_donor_efield']}")
            print(f"Acceptor directions from: {paths['reactant_acceptor_efield']}")
            
            reactant_donor_md = ref.read_electric_field(paths['reactant_donor_efield'])[:3]
            reactant_acceptor_md = ref.read_electric_field(paths['reactant_acceptor_efield'])[:3]
            
            # Product State Fields
            print("\nProduct State MD Field Sources:")
            print(f"Donor directions from:    {paths['product_donor_efield']}")
            print(f"Acceptor directions from: {paths['product_acceptor_efield']}")
            
            product_donor_md = ref.read_electric_field(paths['product_donor_efield'])[:3]
            product_acceptor_md = ref.read_electric_field(paths['product_acceptor_efield'])[:3]

            # Generate synthetic fields using MD directions for each position
            print("\nStep 2: Generating synthetic fields with MD directions...")
            
            reactant_donor_efield = inject_synthetic_fields(
                paths['reactant_donor_xyz'],
                txt_output_folder,
                synthetic_field_params['reactant_donor'],
                md_fields=reactant_donor_md,
                position_name='reactant_donor',
                iteration=iteration
            )
            
            reactant_acceptor_efield = inject_synthetic_fields(
                paths['reactant_acceptor_xyz'],
                txt_output_folder,
                synthetic_field_params['reactant_acceptor'],
                md_fields=reactant_acceptor_md,
                position_name='reactant_acceptor',
                iteration=iteration
            )
            
            product_donor_efield = inject_synthetic_fields(
                paths['product_donor_xyz'],
                txt_output_folder,
                synthetic_field_params['product_donor'],
                md_fields=product_donor_md,
                position_name='product_donor',
                iteration=iteration
            )
            
            product_acceptor_efield = inject_synthetic_fields(
                paths['product_acceptor_xyz'],
                txt_output_folder,
                synthetic_field_params['product_acceptor'],
                md_fields=product_acceptor_md,
                position_name='product_acceptor',
                iteration=iteration
            )

        elif synthetic_field_params['approach'] == 'heme_based':
            # === Approach 2: Heme-based coordinate system ===
            print("\nStep 1: Reading QM reference structures...")

            print("- Read reduced reference structure")
            qm_ref_red = rxyz.read_xyz(paths['red_ref_coords'])[0]  # First frame only
            print("- Read oxidized reference structure")
            qm_ref_ox = rxyz.read_xyz(paths['ox_ref_coords'])[0]
            
            # Calculate transformation matrices for each structure
            print("\nStep 2: Calculating transformation matrices...")
            
            # Reactant donor
            print("Processing reactant donor...")
            matrix_file = ss.superimpose_and_save(
                paths['reactant_donor_xyz'], 
                paths['red_ref_coords' if 'donor' in reactant_donor_id.lower() else 'ox_ref_coords'],
                os.path.join(txt_output_folder, "temp_reactant_donor.xyz"),
                os.path.join(txt_output_folder, "temp_reactant_donor.txt"),
                os.path.join(txt_output_folder, "temp_reactant_donor.csv")
            )
            reactant_donor_matrices = [matrix for _, matrix, _ in rtm.read_transformation_matrices(matrix_file)]
            
            # Reactant acceptor
            print("Processing reactant acceptor...")
            matrix_file = ss.superimpose_and_save(
                paths['reactant_acceptor_xyz'], 
                paths['ox_ref_coords' if 'acceptor' in reactant_acceptor_id.lower() else 'red_ref_coords'],
                os.path.join(txt_output_folder, "temp_reactant_acceptor.xyz"),
                os.path.join(txt_output_folder, "temp_reactant_acceptor.txt"),
                os.path.join(txt_output_folder, "temp_reactant_acceptor.csv")
            )
            reactant_acceptor_matrices = [matrix for _, matrix, _ in rtm.read_transformation_matrices(matrix_file)]
            
            # Product donor
            print("Processing product donor...")
            matrix_file = ss.superimpose_and_save(
                paths['product_donor_xyz'], 
                paths['red_ref_coords' if 'donor' in product_donor_id.lower() else 'ox_ref_coords'],
                os.path.join(txt_output_folder, "temp_product_donor.xyz"),
                os.path.join(txt_output_folder, "temp_product_donor.txt"),
                os.path.join(txt_output_folder, "temp_product_donor.csv")
            )
            product_donor_matrices = [matrix for _, matrix, _ in rtm.read_transformation_matrices(matrix_file)]
            
            # Product acceptor
            print("Processing product acceptor...")
            matrix_file = ss.superimpose_and_save(
                paths['product_acceptor_xyz'], 
                paths['ox_ref_coords' if 'acceptor' in product_acceptor_id.lower() else 'red_ref_coords'],
                os.path.join(txt_output_folder, "temp_product_acceptor.xyz"),
                os.path.join(txt_output_folder, "temp_product_acceptor.txt"),
                os.path.join(txt_output_folder, "temp_product_acceptor.csv")
            )
            product_acceptor_matrices = [matrix for _, matrix, _ in rtm.read_transformation_matrices(matrix_file)]

            # Generate synthetic fields using heme-based coordinates
            print("\nStep 3: Generating synthetic fields...")
            
            # Reactant Donor
            print("\nGenerating Reactant Donor Fields:")
            qm_ref = qm_ref_red if 'donor' in reactant_donor_id.lower() else qm_ref_ox

            # Reactant Donor
            reactant_donor_efield = inject_synthetic_fields(
                paths['reactant_donor_xyz'],
                txt_output_folder,
                {**synthetic_field_params['reactant_donor'],
                 'approach': synthetic_field_params['approach'],
                 'n_pair1': synthetic_field_params['donor_n_pair1'],
                 'n_pair2': synthetic_field_params['donor_n_pair2']},
                qm_structure=qm_ref,
                rotation_matrices=reactant_donor_matrices,
                position_name='reactant_donor',
                iteration=iteration
            )

            # Reactant Acceptor
            reactant_acceptor_efield = inject_synthetic_fields(
                paths['reactant_acceptor_xyz'],
                txt_output_folder,
                {**synthetic_field_params['reactant_acceptor'],
                 'approach': synthetic_field_params['approach'],
                 'n_pair1': synthetic_field_params['acceptor_n_pair1'],
                 'n_pair2': synthetic_field_params['acceptor_n_pair2']},
                qm_structure=qm_ref,
                rotation_matrices=reactant_acceptor_matrices,
                position_name='reactant_acceptor',
                iteration=iteration
            )

            # Product Donor
            product_donor_efield = inject_synthetic_fields(
                paths['product_donor_xyz'],
                txt_output_folder,
                {**synthetic_field_params['product_donor'],
                 'approach': synthetic_field_params['approach'],
                 'n_pair1': synthetic_field_params['donor_n_pair1'],
                 'n_pair2': synthetic_field_params['donor_n_pair2']},
                qm_structure=qm_ref,
                rotation_matrices=product_donor_matrices,
                position_name='product_donor',
                iteration=iteration
            )

            # Product Acceptor
            product_acceptor_efield = inject_synthetic_fields(
                paths['product_acceptor_xyz'],
                txt_output_folder,
                {**synthetic_field_params['product_acceptor'],
                 'approach': synthetic_field_params['approach'],
                 'n_pair1': synthetic_field_params['acceptor_n_pair1'],
                 'n_pair2': synthetic_field_params['acceptor_n_pair2']},
                qm_structure=qm_ref,
                rotation_matrices=product_acceptor_matrices,
                position_name='product_acceptor',
                iteration=iteration
            )

            # Clean up temporary files
            print("\nCleaning up temporary files...")
            temp_files = [
                "temp_reactant_donor.xyz", "temp_reactant_donor.txt", "temp_reactant_donor.csv",
                "temp_reactant_acceptor.xyz", "temp_reactant_acceptor.txt", "temp_reactant_acceptor.csv",
                "temp_product_donor.xyz", "temp_product_donor.txt", "temp_product_donor.csv",
                "temp_product_acceptor.xyz", "temp_product_acceptor.txt", "temp_product_acceptor.csv"
            ]
            for temp_file in temp_files:
                temp_path = os.path.join(txt_output_folder, temp_file)
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        else:
            raise ValueError(f"Unknown synthetic field approach: {synthetic_field_params['approach']}")

        # Update paths dictionary with new field paths
        paths['reactant_donor_efield'] = reactant_donor_efield
        paths['reactant_acceptor_efield'] = reactant_acceptor_efield
        paths['product_donor_efield'] = product_donor_efield
        paths['product_acceptor_efield'] = product_acceptor_efield

        # Read generated fields and compute statistics
        print("\n=== Reading Generated Synthetic Fields ===")
        print("\nReading reactant donor fields...")
        reactant_donor_Ex, reactant_donor_Ey, reactant_donor_Ez = \
            ref.read_electric_field(reactant_donor_efield)[:3]
        print(f"- Found {len(reactant_donor_Ex)} frames")
        
        print("\nReading reactant acceptor fields...")
        reactant_acceptor_Ex, reactant_acceptor_Ey, reactant_acceptor_Ez = \
            ref.read_electric_field(reactant_acceptor_efield)[:3]
        print(f"- Found {len(reactant_acceptor_Ex)} frames")
        
        print("\nReading product donor fields...")
        product_donor_Ex, product_donor_Ey, product_donor_Ez = \
            ref.read_electric_field(product_donor_efield)[:3]
        print(f"- Found {len(product_donor_Ex)} frames")
        
        print("\nReading product acceptor fields...")
        product_acceptor_Ex, product_acceptor_Ey, product_acceptor_Ez = \
            ref.read_electric_field(product_acceptor_efield)[:3]
        print(f"- Found {len(product_acceptor_Ex)} frames")

        print("\n=== Computing Field Statistics ===")
        print("\nCalculating field magnitudes...")
        # Compute field magnitudes and statistics
        reactant_donor_mag = np.sqrt(reactant_donor_Ex**2 + reactant_donor_Ey**2 + reactant_donor_Ez**2)
        print("- Reactant donor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(reactant_donor_mag), np.max(reactant_donor_mag)))
        
        reactant_acceptor_mag = np.sqrt(reactant_acceptor_Ex**2 + reactant_acceptor_Ey**2 + reactant_acceptor_Ez**2)
        print("- Reactant acceptor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(reactant_acceptor_mag), np.max(reactant_acceptor_mag)))
        
        product_donor_mag = np.sqrt(product_donor_Ex**2 + product_donor_Ey**2 + product_donor_Ez**2)
        print("- Product donor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(product_donor_mag), np.max(product_donor_mag)))
        
        product_acceptor_mag = np.sqrt(product_acceptor_Ex**2 + product_acceptor_Ey**2 + product_acceptor_Ez**2)
        print("- Product acceptor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(product_acceptor_mag), np.max(product_acceptor_mag)))

        print("\nComputing statistics for each position...")
        # Compute statistics for each position
        field_stats = {
            'reactant_donor': (np.mean(reactant_donor_mag), np.std(reactant_donor_mag)),
            'reactant_acceptor': (np.mean(reactant_acceptor_mag), np.std(reactant_acceptor_mag)),
            'product_donor': (np.mean(product_donor_mag), np.std(product_donor_mag)),
            'product_acceptor': (np.mean(product_acceptor_mag), np.std(product_acceptor_mag))
        }

        # Print statistics
        for position, (mean, std) in field_stats.items():
            print(f"- {position.replace('_', ' ').title()}:")
            print(f"  * Mean magnitude: {mean:.3f} V/Å")
            print(f"  * Std deviation: {std:.3f} V/Å")

        print("\n=== Packaging Fields for State Processing ===")
        # For synthetic fields, all frames are valid so use range as common indices
        reactant_common_indices = np.arange(len(reactant_donor_mag))
        product_common_indices = np.arange(len(product_donor_mag))
        
        print("\nPreparing reactant state fields...")
        # Package fields for state processing
        reactant_fields = (reactant_donor_Ex, reactant_donor_Ey, reactant_donor_Ez,
                          reactant_acceptor_Ex, reactant_acceptor_Ey, reactant_acceptor_Ez,
                          reactant_common_indices)
        print(f"- Included {len(reactant_common_indices)} frames")
        
        print("\nPreparing product state fields...")
        product_fields = (product_donor_Ex, product_donor_Ey, product_donor_Ez,
                         product_acceptor_Ex, product_acceptor_Ey, product_acceptor_Ez,
                         product_common_indices)
        print(f"- Included {len(product_common_indices)} frames")
                         
        # Get frame counts
        reactant_frame_count = len(reactant_common_indices)
        product_frame_count = len(product_common_indices)
        
        print(f"\nTotal frames to process:")
        print(f"- Reactant state: {reactant_frame_count}")
        print(f"- Product state:  {product_frame_count}")

    else:
        # Use MD-derived fields with optional filtering
        print("\n=== Processing MD Electric Fields ===")
        print(f"Filter settings:")
        print(f"- Type: {filter_type}")
        print(f"- Value: {filter_value}")
        if filter_tolerance:
            print(f"- Tolerance: {filter_tolerance}")

        # Filter electric fields for reactant and product states separately
        print("\n=== Processing Reactant State Fields ===")
        print(f"Reading and filtering fields from:")
        print(f"- Donor:    {paths['reactant_donor_efield']}")
        print(f"- Acceptor: {paths['reactant_acceptor_efield']}")
        
        (reactant_donor_Ex, reactant_donor_Ey, reactant_donor_Ez,
         reactant_acceptor_Ex, reactant_acceptor_Ey, reactant_acceptor_Ez,
         reactant_common_indices) = filter_electric_fields(
            paths['reactant_donor_efield'],
            paths['reactant_acceptor_efield'],
            filter_type, 
            filter_value, 
            filter_tolerance
        )
        print(f"Found {len(reactant_common_indices)} frames meeting filter criteria")

        print("\n=== Processing Product State Fields ===")
        print(f"Reading and filtering fields from:")
        print(f"- Donor:    {paths['product_donor_efield']}")
        print(f"- Acceptor: {paths['product_acceptor_efield']}")
        
        (product_donor_Ex, product_donor_Ey, product_donor_Ez,
         product_acceptor_Ex, product_acceptor_Ey, product_acceptor_Ez,
         product_common_indices) = filter_electric_fields(
            paths['product_donor_efield'],
            paths['product_acceptor_efield'],
            filter_type, 
            filter_value, 
            filter_tolerance
        )
        print(f"Found {len(product_common_indices)} frames meeting filter criteria")

        print("\n=== Computing Field Statistics ===")
        print("\nCalculating field magnitudes...")
        # Compute field magnitudes and statistics
        reactant_donor_mag = np.sqrt(reactant_donor_Ex**2 + 
                                   reactant_donor_Ey**2 + 
                                   reactant_donor_Ez**2)
        print("- Reactant donor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(reactant_donor_mag), np.max(reactant_donor_mag)))
        
        reactant_acceptor_mag = np.sqrt(reactant_acceptor_Ex**2 + 
                                      reactant_acceptor_Ey**2 + 
                                      reactant_acceptor_Ez**2)
        print("- Reactant acceptor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(reactant_acceptor_mag), np.max(reactant_acceptor_mag)))
        
        product_donor_mag = np.sqrt(product_donor_Ex**2 + 
                                  product_donor_Ey**2 + 
                                  product_donor_Ez**2)
        print("- Product donor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(product_donor_mag), np.max(product_donor_mag)))
        
        product_acceptor_mag = np.sqrt(product_acceptor_Ex**2 + 
                                     product_acceptor_Ey**2 + # Fixed Ey instead of Ex
                                     product_acceptor_Ez**2)
        print("- Product acceptor field magnitude range: {:.3f} to {:.3f} V/Å".format(
            np.min(product_acceptor_mag), np.max(product_acceptor_mag)))

        print("\nComputing statistics for each position...")
        field_stats = {
            'reactant_donor': (np.mean(reactant_donor_mag), np.std(reactant_donor_mag)),
            'reactant_acceptor': (np.mean(reactant_acceptor_mag), np.std(reactant_acceptor_mag)),
            'product_donor': (np.mean(product_donor_mag), np.std(product_donor_mag)),
            'product_acceptor': (np.mean(product_acceptor_mag), np.std(product_acceptor_mag))
        }

        # Print statistics
        for position, (mean, std) in field_stats.items():
            print(f"- {position.replace('_', ' ').title()}:")
            print(f"  * Mean magnitude: {mean:.3f} V/Å")
            print(f"  * Std deviation: {std:.3f} V/Å")

        print("\n=== Packaging Fields for State Processing ===")
        print("\nPreparing reactant state fields...")
        # Package fields for state processing
        reactant_fields = (reactant_donor_Ex, reactant_donor_Ey, reactant_donor_Ez,
                          reactant_acceptor_Ex, reactant_acceptor_Ey, reactant_acceptor_Ez,
                          reactant_common_indices)
        print(f"- Included {len(reactant_common_indices)} frames")
        
        print("\nPreparing product state fields...")
        product_fields = (product_donor_Ex, product_donor_Ey, product_donor_Ez,
                         product_acceptor_Ex, product_acceptor_Ey, product_acceptor_Ez,
                         product_common_indices)
        print(f"- Included {len(product_common_indices)} frames")
        
        # Get frame counts
        reactant_frame_count = len(reactant_common_indices)
        product_frame_count = len(product_common_indices)
        
        print(f"\nTotal frames to process:")
        print(f"- Reactant state: {reactant_frame_count}")
        print(f"- Product state:  {product_frame_count}")

    print("\n=== Processing States ===")
    print("\nStep 1: Processing reactant state...")
    print(f"- Processing {len(reactant_fields[-1])} frames")
    print("- Passing field components to process_state function")
    reactant_results = process_state_md_to_qm('reactant', paths, reactant_fields[-1], 
                                   reactant_fields[:-1], txt_output_folder, 
                                   basis_set_scaling, use_synthetic_fields,
                                   iteration=iteration)
    print("- Reactant state processing complete")

    print("\nStep 2: Processing product state...")
    print(f"- Processing {len(product_fields[-1])} frames")
    print("- Passing field components to process_state function")
    product_results = process_state_md_to_qm('product', paths, product_fields[-1], 
                                  product_fields[:-1], txt_output_folder, 
                                  basis_set_scaling, use_synthetic_fields,
                                  iteration=iteration)
    print("- Product state processing complete")

    print("\n=== Writing Output Files ===")
    print("\nStep 1: Constructing output filenames...")
    # Generate output file suffix based on analysis type
    suffix = "synthetic" if use_synthetic_fields else \
            "no_filter" if filter_type is None else \
            f"{filter_type}_{filter_value:.2f}"
   
    if use_synthetic_fields:
        # Include all positions in suffix
        positions = ['reactant_donor', 'reactant_acceptor', 'product_donor', 'product_acceptor']
        for pos in positions:
            params = synthetic_field_params[pos]
            suffix += f"_{pos.replace('_', '-')}"
            suffix += f"_{params['direction']}" if 'direction' in params else ""
            suffix += f"_{params['distribution']}" if params['distribution'] != 'fixed' else ""
            suffix += f"_{params['magnitude']:.3f}"
    
    print(f"- Base suffix: {suffix}")

    # Construct base filenames
    reactant_base = f"{reactant_donor_id}_{reactant_acceptor_id}_reactant_state_{suffix}"
    product_base = f"{reactant_donor_id}_{reactant_acceptor_id}_product_state_{suffix}"
    reorg_base = f"{reactant_donor_id}_{reactant_acceptor_id}_reorganization_energies_{suffix}"

    # Add iteration number if specified
    if iteration is not None:
        print(f"- Adding iteration number: {iteration:03d}")
        reactant_base = f"{reactant_base}_iteration_{iteration:03d}"
        product_base = f"{product_base}_iteration_{iteration:03d}"
        reorg_base = f"{reorg_base}_iteration_{iteration:03d}"

    # Add file extension
    print("- Adding file extensions")
    reactant_output = f"{reactant_base}.txt"
    product_output = f"{product_base}.txt"
    reorganization_energies_filename = f"{reorg_base}.txt"

    print("\nStep 2: Constructing metadata...")
    # Generate base metadata
    metadata = construct_metadata(
        use_synthetic_fields,
        synthetic_field_params if use_synthetic_fields else None,
        filter_type, 
        filter_value,
        filter_tolerance
    )

    # Add electric field file paths to metadata
    metadata_with_paths = metadata + "\n\nElectric Field Files:"
    metadata_with_paths += f"\nReactant Donor:    {paths['reactant_donor_efield']}"
    metadata_with_paths += f"\nReactant Acceptor: {paths['reactant_acceptor_efield']}"
    metadata_with_paths += f"\nProduct Donor:     {paths['product_donor_efield']}"
    metadata_with_paths += f"\nProduct Acceptor:  {paths['product_acceptor_efield']}\n"

    print("\nStep 3: Writing output files")
    print(f"\n1. Reactant state:")
    print(f"   - File: {reactant_output}")
    print(f"   - Path: {os.path.join(txt_output_folder, reactant_output)}")
    print(f"   - Number of frames: {len(reactant_fields[-1])}")

    # Write reactant state
    wof.write_output_file(
        txt_output_folder,
        reactant_output,
        'Reactant',
        metadata_with_paths,
        donor_tensors={
            'reduced': reactant_results.donor_tensors[0],
            'oxidized': reactant_results.donor_tensors[1],
            'diffalpha': reactant_results.donor_tensors[2]
        },
        acceptor_tensors={
            'reduced': reactant_results.acceptor_tensors[0],
            'oxidized': reactant_results.acceptor_tensors[1],
            'diffalpha': reactant_results.acceptor_tensors[2]
        },
        field_components={
            'donor': {
                'Ex': reactant_results.donor_fields[0],
                'Ey': reactant_results.donor_fields[1],
                'Ez': reactant_results.donor_fields[2],
                'polarization': reactant_results.donor_polarization
            },
            'acceptor': {
                'Ex': reactant_results.acceptor_fields[0],
                'Ey': reactant_results.acceptor_fields[1],
                'Ez': reactant_results.acceptor_fields[2],
                'polarization': reactant_results.acceptor_polarization
            }
        },
        energies={
            'total_polarization': reactant_results.total_polarization,
            'coulombic': reactant_results.coulombic_energy,
            'total': reactant_results.total_energy
        },
        num_frames=len(reactant_fields[-1]),
        statistics={
            'avg_coulombic': reactant_results.coulombic_stats.average,
            'var_coulombic': reactant_results.coulombic_stats.variance,
            'avg_total': reactant_results.total_stats.average,
            'var_total': reactant_results.total_stats.variance
        }
    )
    print("   - File written successfully")

    print(f"\n2. Product state:")
    print(f"   - File: {product_output}")
    print(f"   - Path: {os.path.join(txt_output_folder, product_output)}")
    print(f"   - Number of frames: {len(product_fields[-1])}")

    # Write product state
    wof.write_output_file(
        txt_output_folder,
        product_output,
        'Product',
        metadata_with_paths,
        donor_tensors={
            'reduced': product_results.donor_tensors[0],
            'oxidized': product_results.donor_tensors[1],
            'diffalpha': product_results.donor_tensors[2]
        },
        acceptor_tensors={
            'reduced': product_results.acceptor_tensors[0],
            'oxidized': product_results.acceptor_tensors[1],
            'diffalpha': product_results.acceptor_tensors[2]
        },
        field_components={
            'donor': {
                'Ex': product_results.donor_fields[0],
                'Ey': product_results.donor_fields[1],
                'Ez': product_results.donor_fields[2],
                'polarization': product_results.donor_polarization
            },
            'acceptor': {
                'Ex': product_results.acceptor_fields[0],
                'Ey': product_results.acceptor_fields[1],
                'Ez': product_results.acceptor_fields[2],
                'polarization': product_results.acceptor_polarization
            }
        },
        energies={
            'total_polarization': product_results.total_polarization,
            'coulombic': product_results.coulombic_energy,
            'total': product_results.total_energy
        },
        num_frames=len(product_fields[-1]),
        statistics={
            'avg_coulombic': product_results.coulombic_stats.average,
            'var_coulombic': product_results.coulombic_stats.variance,
            'avg_total': product_results.total_stats.average,
            'var_total': product_results.total_stats.variance
        }
    )
    print("   - File written successfully")

    print("\n=== Output Files Written Successfully ===")

    print("\n=== Computing Reorganization Energies ===")
    print(f"Output file: {os.path.join(txt_output_folder, reorganization_energies_filename)}")
    print("\nProcessing energy components:")
    print("- Reactant state:")
    print(f"  * Average coulombic energy: {reactant_results.coulombic_stats.average:.3f}")
    print(f"  * Average total energy: {reactant_results.total_stats.average:.3f}")
    print(f"  * Coulombic variance: {reactant_results.coulombic_stats.variance:.3f}")
    print(f"  * Total variance: {reactant_results.total_stats.variance:.3f}")
    print("- Product state:")
    print(f"  * Average coulombic energy: {product_results.coulombic_stats.average:.3f}")
    print(f"  * Average total energy: {product_results.total_stats.average:.3f}")
    print(f"  * Coulombic variance: {product_results.coulombic_stats.variance:.3f}")
    print(f"  * Total variance: {product_results.total_stats.variance:.3f}")
    
    unpolarized_energy, polarized_energy = cawre.compute_and_write_reorganization_energies(
        txt_output_folder,
        reorganization_energies_filename,
        reactant_results.coulombic_stats.average, 
        product_results.coulombic_stats.average,  # avg_coulombic_energies
        reactant_results.total_stats.average, 
        product_results.total_stats.average,  # avg_total_energies
        reactant_results.coulombic_stats.variance, 
        reactant_results.total_stats.variance, # variances reactant
        product_results.coulombic_stats.variance, 
        product_results.total_stats.variance    # variances product
    )
    
    print("\nReorganization energies computed:")
    print(f"- Unpolarized: {unpolarized_energy:.3f}")
    print(f"- Polarized:   {polarized_energy:.3f}")

    # Return values directly from results and statistics
    return (
        # Field statistics for each position [0-7]
        field_stats['reactant_donor'][0],      # reactant donor mean electric field magnitude
        field_stats['reactant_donor'][1],      # reactant donor electric field standard deviation
        field_stats['reactant_acceptor'][0],   # reactant acceptor mean electric field magnitude
        field_stats['reactant_acceptor'][1],   # reactant acceptor electric field standard deviation
        field_stats['product_donor'][0],       # product donor mean electric field magnitude
        field_stats['product_donor'][1],       # product donor electric field standard deviation
        field_stats['product_acceptor'][0],    # product acceptor mean electric field magnitude
        field_stats['product_acceptor'][1],    # product acceptor electric field standard deviation
        
        # Frame counts [8-9]
        reactant_frame_count,                  # reactant trajectory frame count
        product_frame_count,                   # product trajectory frame count
        
        # Reactant state energies and variances [10-17]
        reactant_results.coulombic_stats.average,                  # mean reactant state coulombic energy
        reactant_results.coulombic_stats.variance,                 # variance of reactant state coulombic energy  
        np.mean(reactant_results.donor_polarization),         # mean reactant donor polarization energy
        np.var(reactant_results.donor_polarization, ddof=1),  # variance of reactant donor polarization energy
        np.mean(reactant_results.acceptor_polarization),         # mean reactant acceptor polarization energy
        np.var(reactant_results.acceptor_polarization, ddof=1),  # variance of reactant acceptor polarization energy
        reactant_results.total_stats.average,                  # average total reactant energy
        reactant_results.total_stats.variance,                 # variance of total reactant energy 
        
        # Product state energies and variances [18-25]
        product_results.coulombic_stats.average,                   # mean product state coulombic energy   
        product_results.coulombic_stats.variance,                  # variance of product state coulombic energy  
        np.mean(product_results.donor_polarization),          # mean product donor polarization energy
        np.var(product_results.donor_polarization, ddof=1),   # variance of product donor polarization energy    
        np.mean(product_results.acceptor_polarization),          # mean product acceptor polarization energy
        np.var(product_results.acceptor_polarization, ddof=1),   # variance of product acceptor polarization energy   
        product_results.total_stats.average,                   # average total product energy
        product_results.total_stats.variance,                  # variance of total product energy 
        
        # Reorganization energies [26-27]
        unpolarized_energy,                    # unpolarized reorganization energy
        polarized_energy                       # polarized reorganization energy
    )
