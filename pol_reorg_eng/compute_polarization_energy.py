def compute_polarization_energy(Da_xx, Da_xy, Da_xz, Da_yy, Da_yz, Da_zz, E_x, E_y, E_z):
    polarization_energies = []
    
    for idx, (daxx, daxy, daxz, dayy, dayz, dazz, ex, ey, ez) in enumerate(zip(Da_xx, Da_xy, Da_xz, Da_yy, Da_yz, Da_zz, E_x, E_y, E_z)):
        # Calculate each term separately
        term1 = daxx * (ex)**2
        term2 = 2 * daxy * ex * ey
        term3 = 2 * daxz * ex * ez
        term4 = dayy * (ey)**2
        term5 = 2 * dayz * ey * ez
        term6 = dazz * (ez)**2
        
        # Calculate final energy
        U = -0.5 * (term1 + term2 + term3 + term4 + term5 + term6)
        
        # Print details for first and last iterations
        if idx == 0 or idx == len(Da_xx) - 1:
            print(f"\nIteration {idx + 1}:")
            print(f"Term 1: Da_xx * (E_x)²")
            print(f"      = {daxx:.6f} * ({ex:.6f})²")
            print(f"      = {daxx:.6f} * {ex**2:.6f}")
            print(f"      = {term1:.6f}")
            
            print(f"\nTerm 2: 2 * Da_xy * E_x * E_y")
            print(f"      = 2 * {daxy:.6f} * {ex:.6f} * {ey:.6f}")
            print(f"      = {term2:.6f}")
            
            print(f"\nTerm 3: 2 * Da_xz * E_x * E_z")
            print(f"      = 2 * {daxz:.6f} * {ex:.6f} * {ez:.6f}")
            print(f"      = {term3:.6f}")
            
            print(f"\nTerm 4: Da_yy * (E_y)²")
            print(f"      = {dayy:.6f} * ({ey:.6f})²")
            print(f"      = {dayy:.6f} * {ey**2:.6f}")
            print(f"      = {term4:.6f}")
            
            print(f"\nTerm 5: 2 * Da_yz * E_y * E_z")
            print(f"      = 2 * {dayz:.6f} * {ey:.6f} * {ez:.6f}")
            print(f"      = {term5:.6f}")
            
            print(f"\nTerm 6: Da_zz * (E_z)²")
            print(f"      = {dazz:.6f} * ({ez:.6f})²")
            print(f"      = {dazz:.6f} * {ez**2:.6f}")
            print(f"      = {term6:.6f}")
            
            print(f"\nSum of all terms: {term1:.6f} + {term2:.6f} + {term3:.6f} + {term4:.6f} + {term5:.6f} + {term6:.6f}")
            print(f"                 = {term1 + term2 + term3 + term4 + term5 + term6:.6f}")
            
            print(f"\nFinal energy U = -0.5 * (sum)")
            print(f"               = -0.5 * {term1 + term2 + term3 + term4 + term5 + term6:.6f}")
            print(f"               = {U:.6f}")
            print("-" * 80)

        polarization_energies.append(U)
    
    return polarization_energies

