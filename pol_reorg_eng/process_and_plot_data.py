import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def process_and_plot_data(folder_path, reactant_state_file, product_state_file, skip_header=25, usecols=(4, 8, 9, 10, 11)):
    """
    Process data from reactant and product state files and generate plots.

    Parameters:
        reactant_state_file (str): File path for reactant state data.
        product_state_file (str): File path for product state data.
        skip_header (int, optional): Number of lines to skip at the beginning of the file. Defaults to 24.
        usecols (tuple, optional): Columns to read from the file. Defaults to (3, 7, 8, 9, 10).
    """

    def read_data(filename, skip_header=24, usecols=(3, 7, 8, 9, 10)):
        """Read specific columns from the data file."""
        return np.loadtxt(filename, skiprows=skip_header, usecols=usecols)

    # Read the data
    reactant_data = read_data(f"{folder_path}/txt/{reactant_state_file}", skip_header, usecols)
    product_data = read_data(f"{folder_path}/txt/{product_state_file}", skip_header, usecols)

    # Extract columns for reactant state
    U_donor_react, U_acceptor_react, U_totp_react, U_Coulombic_react, U_total_react = reactant_data.T
#   print(U_donor_react, U_acceptor_react, U_totp_react, U_Coulombic_react, U_total_react)
    # Extract columns for product state
    U_donor_prod, U_acceptor_prod, U_totp_prod, U_Coulombic_prod, U_total_prod = product_data.T

    rfrms = np.arange(len(U_donor_react))
    x_rfrms = (rfrms * 20) / 1000
    pfrms = np.arange(len(U_donor_prod))
    x_pfrms = (pfrms * 20) / 1000

    # Determine common y-axis limits
    y_min = min(U_donor_react.min(), U_acceptor_react.min(), U_totp_react.min(), U_Coulombic_react.min(), U_total_react.min(),
                U_donor_prod.min(), U_acceptor_prod.min(), U_totp_prod.min(), U_Coulombic_prod.min(), U_total_prod.min())
    y_max = max(U_donor_react.max(), U_acceptor_react.max(), U_totp_react.max(), U_Coulombic_react.max(), U_total_react.max(),
                U_donor_prod.max(), U_acceptor_prod.max(), U_totp_prod.max(), U_Coulombic_prod.max(), U_total_prod.max())

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0))

    # Plot 1: Polarization energy of donor, acceptor, and total (reactant state)
    axs[0, 0].plot(x_rfrms, U_donor_react, color='red', alpha=0.5, label='U_donor')
    axs[0, 0].plot(x_rfrms, U_acceptor_react, color='blue', alpha=0.5, label='U_acceptor')
#   axs[0, 0].plot(x_rfrms, U_totp_react, alpha=0.5, label='U_total (Polarization)')
#   axs[0, 0].title('Reactant State: Polarization Energies')
    axs[0, 0].set_xlabel('Time (ns)')
    axs[0, 0].set_ylabel('Energy (eV)')
    axs[0, 0].tick_params(direction='in')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(y_min, y_max)
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#   axs[0, 0].savefig(f'{folder_path}/png/Reactant_State_Polarization_Energies.png')

    # Plot 2: Polarization energy of donor, acceptor, and total (product state)
    axs[0, 1].plot(x_pfrms, U_donor_prod, color='red', alpha=0.5, label='U_donor')
    axs[0, 1].plot(x_pfrms, U_acceptor_prod, color='blue', alpha=0.5, label='U_acceptor')
#   axs[0, 1].plot(x_pfrms, U_totp_prod, alpha=0.5, label='U_total (Polarization)')
#   axs[0, 1].title('Product State: Polarization Energies')
    axs[0, 1].set_xlabel('Time (ns)')
    axs[0, 1].set_ylabel('Energy (eV)')
    axs[0, 1].tick_params(direction='in')
    axs[0, 1].legend()
    axs[0, 1].set_ylim(y_min, y_max)
    axs[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#   axs[0, 1].savefig(f'{folder_path}/png/Product_State_Polarization_Energies.png')

    # Plot 3: Coulombic energy and total energy (reactant state)
    axs[1, 0].plot(x_rfrms, U_Coulombic_react, color='orange', alpha=0.5, label='U_Coulombic')
    axs[1, 0].plot(x_rfrms, U_total_react, color='green', alpha=0.5, label='U_total')
#   axs[1, 0].title('Reactant State: Coulombic and Total Energies')
    axs[1, 0].set_xlabel('Time (ns)')
    axs[1, 0].set_ylabel('Energy (eV)')
    axs[1, 0].tick_params(direction='in')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(y_min, y_max)
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#   axs[1, 0].savefig(f'{folder_path}/png/Reactant_State_Coulombic_and_Total_Energies.png')

    # Plot 4: Coulombic energy and total energy (product state)
    axs[1, 1].plot(x_pfrms, U_Coulombic_prod, color='orange', alpha=0.5, label='U_Coulombic')
    axs[1, 1].plot(x_pfrms, U_total_prod, color='green', alpha=0.5, label='U_total')
#   axs[1, 1].title('Product State: Coulombic and Total Energies')
    axs[1, 1].set_xlabel('Time (ns)')
    axs[1, 1].set_ylabel('Energy (eV)')
    axs[1, 1].tick_params(direction='in')
    axs[1, 1].legend()
    axs[1, 1].set_ylim(y_min, y_max)
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#   axs[1, 1].savefig(f'{folder_path}/png/Product_State_Coulombic_and_Total_Energies.png')

    plt.tight_layout()
    file_path=f"{folder_path}/png/Visualize_Energies.png"
    plt.savefig(file_path, dpi=600)
    print(f' **Figure saved to: {file_path}')

    # Plot 5: Distribution of Coulombic and Total energies for both states
    plt.figure(figsize=(3.3, 3.3))
    plt.hist(U_total_react, bins=30, alpha=0.5, label='U_total Reactant')
    plt.hist(U_total_prod, bins=30, alpha=0.5, label='U_total Product')
    plt.hist(U_Coulombic_react, bins=30, alpha=0.5, label='U_Coulombic Reactant')
    plt.hist(U_Coulombic_prod, bins=30, alpha=0.5, label='U_Coulombic Product')
#   plt.title('Distribution of Coulombic and Total Energies')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    file_path=f"{folder_path}/png/Distribution_of_Coulombic_and_Total_Energies.png"
    plt.savefig(file_path, dpi=600)
    print(f' **Figure saved to: {file_path}')
