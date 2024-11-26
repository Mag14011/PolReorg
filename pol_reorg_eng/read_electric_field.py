import numpy as np

def read_electric_field(filename, filter_type=None, filter_value=None, filter_tolerance=None):
    """
    Read and filter electric field data from a file.
    
    Parameters:
        filename (str): Path to the electric field data file
        filter_type (str): Type of filter to apply ('threshold', 'variance', or 'exact')
        filter_value (float): Value to filter by
        filter_tolerance (float): Tolerance band for exact magnitude filtering
    
    Returns:
        tuple: (E_x, E_y, E_z, indices) - filtered electric field components and their indices
    """
    try:
        # Count the total number of lines in the file
        with open(filename, 'r') as file:
            total_lines = sum(1 for line in file)

        # Calculate the number of rows to read, skipping first 5 and last 3 rows
        rows_to_read = total_lines - 8

        # Load data from the file, skipping first 5 rows and using columns 2, 3, and 4
        data = np.loadtxt(filename, skiprows=5, max_rows=rows_to_read, usecols=(2, 3, 4))

        # Extract and scale the x, y, z components of the electric field
        E_x = data[:, 0] * 0.01
        E_y = data[:, 1] * 0.01
        E_z = data[:, 2] * 0.01

        # Create an array of original indices
        indices = np.arange(len(E_x))

        if filter_type is not None and filter_value is not None:
            # Calculate the magnitude of the electric field vectors
            magnitudes = np.sqrt(E_x**2 + E_y**2 + E_z**2)
            
            if filter_type == 'threshold':
                # Create a mask for values below or equal to filter_value
                mask = magnitudes <= filter_value
                
            elif filter_type == 'variance':
                # Initialize a boolean mask
                mask = np.ones(len(magnitudes), dtype=bool)
                # Iterate through all points
                for i in range(len(magnitudes)):
                    # Create a window of 21 points (or less at the edges)
                    window = magnitudes[max(0, i-10):min(len(magnitudes), i+10)]
                    # If the variance in this window exceeds filter_value, mark for removal
                    if np.var(window) > filter_value:
                        mask[i] = False

            elif filter_type == 'exact':
                if filter_tolerance is None:
                    raise ValueError("filter_tolerance must be specified when using 'exact' filter_type")
                # Create a mask for values within tolerance of filter_value
                mask = np.abs(magnitudes - filter_value) <= filter_tolerance

            else:
                raise ValueError(f"Unknown filter_type: {filter_type}")

            # Apply the mask to indices and electric field components
            indices = indices[mask]
            E_x = E_x[mask]
            E_y = E_y[mask]
            E_z = E_z[mask]

            if len(indices) == 0:
                print(f"\nWarning: No electric field vectors matched the {filter_type} filter criteria:")
                print(f"  Filter value: {filter_value}")
                if filter_type == 'exact':
                    print(f"  Tolerance: ±{filter_tolerance}")
                print("  Consider adjusting your filter parameters.")

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        raise
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        raise

    return E_x, E_y, E_z, indices
