import os
import numpy as np
import create_output_folder as cof
import save_output_file as sof
from typing import Dict, Any, Optional, List, Tuple

def format_tensor(tensor: np.ndarray, decimals: int = 6) -> str:
    """
    Format a 3x3 tensor as a string with proper alignment.
    
    Args:
        tensor: 3x3 numpy array
        decimals: Number of decimal places to show
    
    Returns:
        Formatted string representation of tensor
    """
    format_str = f"{{:>{decimals + 6}.{decimals}f}}"
    lines = []
    for row in tensor:
        lines.append("  " + " ".join(format_str.format(x) for x in row))
    return "\n".join(lines)

def format_header_section(title: str, content: List[Tuple[str, Any]], include_units: bool = True) -> str:
    """
    Format a section of header information with consistent alignment.
    
    Args:
        title: Section title
        content: List of (label, value) tuples
        include_units: Whether to preserve units in the value
    
    Returns:
        Formatted section string
    """
    lines = [f"=== {title} ===\n"]
    max_label_length = max(len(label) for label, _ in content) + 2
    
    for label, value in content:
        if isinstance(value, (float, np.float32, np.float64)):
            formatted_value = f"{value:12.6f}"
        else:
            formatted_value = str(value)
        lines.append(f"{label + ':':<{max_label_length}} {formatted_value}")
    
    return "\n".join(lines) + "\n"

def format_frame_table(field_components: Dict, energies: Dict, num_frames: int) -> str:
    """
    Format the frame-by-frame data table with proper alignment.
    
    Args:
        field_components: Dictionary containing field and polarization data
        energies: Dictionary containing energy components
        num_frames: Number of frames to process
    
    Returns:
        Formatted table string
    """
    # Define headers with units
    headers = [
        "Frame",
        "E_x_d (V/Å)", "E_y_d (V/Å)", "E_z_d (V/Å)", "U_d (eV)",
        "E_x_a (V/Å)", "E_y_a (V/Å)", "E_z_a (V/Å)", "U_a (eV)",
        "U_pol (eV)", "U_Coul (eV)", "U_total (eV)"
    ]
    
    # Create header with fixed column widths
    header_format = "{:>6} " + " ".join(["{:>12}"] * (len(headers) - 1))
    lines = [header_format.format(*headers)]
    
    # Add separator
    lines.append("-" * (6 + 12 * (len(headers) - 1)))
    
    # Format data rows
    row_format = "{:>6d} " + " ".join(["{:12.6f}"] * (len(headers) - 1))
    for i in range(num_frames):
        lines.append(row_format.format(
            i + 1,
            field_components['donor']['Ex'][i],
            field_components['donor']['Ey'][i],
            field_components['donor']['Ez'][i],
            field_components['donor']['polarization'][i],
            field_components['acceptor']['Ex'][i],
            field_components['acceptor']['Ey'][i],
            field_components['acceptor']['Ez'][i],
            field_components['acceptor']['polarization'][i],
            energies['total_polarization'][i],
            energies['coulombic'][i],
            energies['total'][i]
        ))
    
    return "\n".join(lines)

def write_output_file(folder_path: str, 
                     filename: str, 
                     state: str, 
                     metadata: Optional[str] = None,
                     donor_tensors: Dict[str, np.ndarray] = None,
                     acceptor_tensors: Dict[str, np.ndarray] = None,
                     field_components: Dict[str, Dict[str, np.ndarray]] = None,
                     energies: Dict[str, np.ndarray] = None,
                     num_frames: int = 0,
                     statistics: Dict[str, float] = None) -> None:
    """
    Write analysis results to an output file with improved formatting.
    
    Args:
        folder_path: Directory to write the output file
        filename: Name of the output file
        state: State label ('Reactant' or 'Product')
        metadata: Optional metadata string to include at the top of the file
        donor_tensors: Dictionary containing donor tensor results:
            {'reduced': array, 'oxidized': array, 'diffalpha': array}
        acceptor_tensors: Dictionary containing acceptor tensor results:
            {'reduced': array, 'oxidized': array, 'diffalpha': array}
        field_components: Dictionary containing field and polarization data:
            {'donor': {'Ex': array, 'Ey': array, 'Ez': array, 'polarization': array},
             'acceptor': {'Ex': array, 'Ey': array, 'Ez': array, 'polarization': array}}
        energies: Dictionary containing energy components:
            {'total_polarization': array, 'coulombic': array, 'total': array}
        num_frames: Number of frames analyzed
        statistics: Dictionary containing statistical results:
            {'avg_coulombic': float, 'var_coulombic': float,
             'avg_total': float, 'var_total': float}
    """
    # Ensure output folder exists
    cof.create_output_folder(folder_path)
    
    # Build content sections
    content = []
    
    # Header section
    content.append(f"=== {state} State Analysis Results ===\n")
    if metadata:
        content.append(metadata + "\n")
    content.append(f"Number of frames analyzed: {num_frames}\n")
    
    # Tensor sections
    content.append("=== Donor Tensors ===\n")
    content.append("Reduced State Average Tensor:")
    content.append(format_tensor(donor_tensors['reduced']))
    content.append("\nOxidized State Average Tensor:")
    content.append(format_tensor(donor_tensors['oxidized']))
    content.append("\nDifference Polarizability Tensor:")
    content.append(format_tensor(donor_tensors['diffalpha']))
    
    content.append("\n=== Acceptor Tensors ===")
    content.append("Reduced State Average Tensor:")
    content.append(format_tensor(acceptor_tensors['reduced']))
    content.append("\nOxidized State Average Tensor:")
    content.append(format_tensor(acceptor_tensors['oxidized']))
    content.append("\nDifference Polarizability Tensor:")
    content.append(format_tensor(acceptor_tensors['diffalpha']))
    
    # Average field components section
    field_content = [
        ("Average Donor Ex", np.mean(field_components['donor']['Ex']), "V/Å"),
        ("Average Donor Ey", np.mean(field_components['donor']['Ey']), "V/Å"),
        ("Average Donor Ez", np.mean(field_components['donor']['Ez']), "V/Å"),
        ("Average Donor Polarization Energy", np.mean(field_components['donor']['polarization']), "eV"),
        ("Average Acceptor Ex", np.mean(field_components['acceptor']['Ex']), "V/Å"),
        ("Average Acceptor Ey", np.mean(field_components['acceptor']['Ey']), "V/Å"),
        ("Average Acceptor Ez", np.mean(field_components['acceptor']['Ez']), "V/Å"),
        ("Average Acceptor Polarization Energy", np.mean(field_components['acceptor']['polarization']), "eV")
    ]
    content.append("\n=== Average Electric Field Components ===")
    for label, value, unit in field_content:
        content.append(f"{label + ':':<35} {value:12.6f} {unit}")
    
    # Statistics section
    stats_content = [
        ("Average Coulombic Energy", f"{statistics['avg_coulombic']:12.6f} eV"),
        ("Variance in Coulombic Energy", f"{statistics['var_coulombic']:12.6f} eV²"),
        ("Average Total Energy", f"{statistics['avg_total']:12.6f} eV"),
        ("Variance in Total Energy", f"{statistics['var_total']:12.6f} eV²")
    ]
    content.append("\n" + format_header_section("Energy Statistics", stats_content, include_units=False))
    
    # Frame-by-frame results
    content.append("=== Frame-by-Frame Results ===")
    content.append(format_frame_table(field_components, energies, num_frames))
    
    # Write all content to file
    with open(os.path.join(folder_path, filename), 'w') as f:
        f.write("\n".join(content))


