def compute_polarization_energy(Da_xx, Da_xy, Da_xz, Da_yy, Da_yz, Da_zz, E_x, E_y, E_z):
    """
    Compute polarization energy with detailed term breakdown.
    
    Parameters:
    Da_xx, Da_xy, Da_xz, Da_yy, Da_yz, Da_zz: Components of the dipole polarizability tensor
    E_x, E_y, E_z: Components of the electric field vector
    
    Returns:
    float: Polarization energy value
    """
    # Calculate each term separately
    term1 = Da_xx * (E_x)**2
    term2 = 2 * Da_xy * E_x * E_y
    term3 = 2 * Da_xz * E_x * E_z
    term4 = Da_yy * (E_y)**2
    term5 = 2 * Da_yz * E_y * E_z
    term6 = Da_zz * (E_z)**2
    
    # Print each term with its components
    print(f"\nTerm breakdown:")
    print(f"1. Da_xx * E_x² = {Da_xx} * ({E_x})² = {term1:.6f}")
    print(f"2. 2 * Da_xy * E_x * E_y = 2 * {Da_xy} * {E_x} * ({E_y}) = {term2:.6f}")
    print(f"3. 2 * Da_xz * E_x * E_z = 2 * {Da_xz} * {E_x} * ({E_z}) = {term3:.6f}")
    print(f"4. Da_yy * E_y² = {Da_yy} * ({E_y})² = {term4:.6f}")
    print(f"5. 2 * Da_yz * E_y * E_z = 2 * {Da_yz} * ({E_y}) * ({E_z}) = {term5:.6f}")
    print(f"6. Da_zz * E_z² = {Da_zz} * ({E_z})² = {term6:.6f}")
    
    # Sum all terms
    sum_terms = term1 + term2 + term3 + term4 + term5 + term6
    print(f"\nSum of all terms: {sum_terms:.6f}")
    
    # Final result with -0.5 factor
    U = -0.5 * sum_terms
    print(f"Final result (-0.5 * sum): {U:.6f}")
    
    return U

# Example values
ex=0.335164    
ey=-0.190883    
ez=-0.020248

daxx=1.154706
daxy=1.054833
daxz=-4.280670
dayy=9.330970
dayz=3.449422
dazz=5.184324

# Calculate the polarization energy
result = compute_polarization_energy(daxx, daxy, daxz, dayy, dayz, dazz, ex, ey, ez)
print(f"Polarization energy: {result:.6f}")
