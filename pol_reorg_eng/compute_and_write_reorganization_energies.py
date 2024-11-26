import save_output_file as sof

kb  = 8.617333262145e-5  # Boltzmann constant in eV/K
T   = 300                # Temperature in K

def compute_and_write_reorganization_energies(folder_path, filename, reactant_avg_coulombic_energy, product_avg_coulombic_energy, reactant_avg_total_energy, product_avg_total_energy, reactant_var_coulombic_energy, reactant_var_total_energy, product_var_coulombic_energy, product_var_total_energy):
    
    lambdaSTunpol = 0.5 * (reactant_avg_coulombic_energy - product_avg_coulombic_energy)
    lambdaSTpol = 0.5 * (reactant_avg_total_energy - product_avg_total_energy)

    sigmaUnpolR = reactant_var_coulombic_energy / (2 * kb * T)
    sigmaPolR = reactant_var_total_energy / (2 * kb * T)
    sigmaUnpolP = product_var_coulombic_energy / (2 * kb * T)
    sigmaPolP = product_var_total_energy / (2 * kb * T)
    lambdaUnpolRXN = (lambdaSTunpol ** 2) / ((sigmaUnpolR + sigmaUnpolP) / 2)
    lambdaPolRXN = (lambdaSTpol ** 2) / ((sigmaPolR + sigmaPolP) / 2)

    unpolarized_ergodicity_factor = (sigmaUnpolR + sigmaUnpolP) / (2 * lambdaSTunpol)
    polarized_ergodicity_factor = (sigmaPolR + sigmaPolP) / (2 * lambdaSTpol)

    # Prepare the content to write to the file
    content = "Reorganization Energies:\n\n"
    content += f"Unpolarized Stokes Reorganization Energy:                      {lambdaSTunpol:>8.3f}\n"
    content += f"Polarized Stokes Reorganization Energy:                        {lambdaSTpol:>8.3f}\n"
    content += f"Unpolarized Variation Reorganization Energy in Reactant State: {sigmaUnpolR:>8.3f}\n"
    content += f"Polarized Variation Reorganization Energy in Reactant State:   {sigmaPolR:>8.3f}\n"
    content += f"Unpolarized Variation Reorganization Energy in Product State:  {sigmaUnpolP:>8.3f}\n"
    content += f"Polarized Variation Reorganization Energy in Product State:    {sigmaPolP:>8.3f}\n"
    content += f"Unpolarized Reaction Reorganization Energy:                    {lambdaUnpolRXN:>8.3f}\n"
    content += f"Polarized Reaction Reorganization Energy:                      {lambdaPolRXN:>8.3f}\n"
    content += f"Unpolarized Ergodicity Factor:                                 {unpolarized_ergodicity_factor:>8.3f}\n"
    content += f"Polarized Ergodicity Factor:                                   {polarized_ergodicity_factor:>8.3f}\n"

    # Save the content to the file
    sof.save_output_file(folder_path, filename, content)

    print(f"\n Unpolarized -> Polarized Reaction Reorganization Energy: {lambdaUnpolRXN:>8.3f} -> {lambdaPolRXN:>8.3f}\n")
    return lambdaUnpolRXN, lambdaPolRXN
