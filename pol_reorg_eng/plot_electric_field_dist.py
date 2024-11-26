import os
import numpy as np
import matplotlib.pyplot as plt
import create_output_folder as cof

def plot_electric_field_dist(folder_path, filename, reactant_donor_E_mag, reactant_acceptor_E_mag, 
                           product_donor_E_mag, product_acceptor_E_mag, bin_method='sturges'):
    """
    Plots the electric field distribution and saves the figure to the specified folder.
    Uses dynamic bin calculation for consistent binning across all histograms.

    Args:
    folder_path (str): The path to the folder where the plot will be saved.
    filename (str): The name of the file to save.
    reactant_donor_E_mag (list or array): Reactant donor electric field magnitudes.
    reactant_acceptor_E_mag (list or array): Reactant acceptor electric field magnitudes.
    product_donor_E_mag (list or array): Product donor electric field magnitudes.
    product_acceptor_E_mag (list or array): Product acceptor electric field magnitudes.
    bin_method (str): Method to calculate number of bins ('sturges', 'fd', or 'scott'). Default is 'sturges'.
    """
    # Convert all inputs to numpy arrays
    data_arrays = [
        np.array(reactant_donor_E_mag),
        np.array(reactant_acceptor_E_mag),
        np.array(product_donor_E_mag),
        np.array(product_acceptor_E_mag)
    ]
    
    # Combine all data to find global min and max
    all_data = np.concatenate(data_arrays)
    
    # Calculate optimal number of bins based on all data
    if bin_method == 'sturges':
        # Sturges' rule: n = log2(N) + 1
        n_bins = int(np.ceil(np.log2(len(all_data)) + 1))
    elif bin_method == 'fd':
        # Freedman-Diaconis rule
        iqr = np.percentile(all_data, 75) - np.percentile(all_data, 25)
        bin_width = 2 * iqr / (len(all_data) ** (1/3))
        n_bins = int(np.ceil((np.max(all_data) - np.min(all_data)) / bin_width))
    elif bin_method == 'scott':
        # Scott's rule
        bin_width = 3.5 * np.std(all_data) / (len(all_data) ** (1/3))
        n_bins = int(np.ceil((np.max(all_data) - np.min(all_data)) / bin_width))
    else:
        n_bins = 50  # Default fallback
    
    # Create the plot
    fig3, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.0, 3.5), sharey=True)
    
    # Define common histogram parameters
    hist_params = dict(
        bins=n_bins,
        alpha=0.5,
        edgecolor='black',
        density=True  # Normalize the histograms for better comparison
    )
    
    # Plot reactant distributions
    axes[0].hist(reactant_donor_E_mag, color='red', label='Donor', **hist_params)
    axes[0].hist(reactant_acceptor_E_mag, color='blue', label='Acceptor', **hist_params)
    axes[0].set_xlabel('Threshold Electric Field Magnitude (V/Å)')
    axes[0].set_ylabel('Normalized Frequency')
    axes[0].legend()
    axes[0].tick_params(axis='both', direction='in')
    
    # Plot product distributions
    axes[1].hist(product_donor_E_mag, color='red', label='Donor', **hist_params)
    axes[1].hist(product_acceptor_E_mag, color='blue', label='Acceptor', **hist_params)
    axes[1].set_xlabel('Threshold Electric Field Magnitude (V/Å)')
    axes[1].legend()
    axes[1].tick_params(axis='both', direction='in')
    
    plt.tight_layout()
    
    # Ensure the output folder exists
    cof.create_output_folder(folder_path)
    
    # Define the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Save the figure
    plt.savefig(file_path, dpi=600)
    
    print(f'Figure saved to: {file_path}')
    print(f'Number of bins used: {n_bins}')


