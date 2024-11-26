from run_analysis import construct_metadata

def format_energy_analysis_output(results, include_header=True, use_synthetic=False, synthetic_params=None,
                                filter_type=None, filter_value=None, filter_tolerance=None):
    """
    Format energy analysis results for output to file.

    Args:
        results: List of result tuples (or single tuple) from run_analysis containing:
            [0] reactant donor mean electric field magnitude
            [1] reactant donor electric field standard deviation
            [2] reactant acceptor mean electric field magnitude
            [3] reactant acceptor electric field standard deviation
            [4] product donor mean electric field magnitude
            [5] product donor electric field standard deviation
            [6] product acceptor mean electric field magnitude
            [7] product acceptor electric field standard deviation
            [8] reactant trajectory frame count
            [9] product trajectory frame count
            [10] mean reactant state coulombic energy
            [11] variance of reactant state coulombic energy
            ... (and so on through index 27)
        include_header: Boolean, whether to include header and metadata
        use_synthetic: Boolean, whether synthetic fields were used
        synthetic_params: Dict of synthetic field parameters if used
        filter_type: String, type of filter used for MD fields
        filter_value: Float, filter value used for MD fields
        filter_tolerance: Float, tolerance used for exact filter type
    """
    COLUMN_WIDTH = 7  # Define consistent column width
    content = ""

    # Add metadata if requested
    if include_header:
        if use_synthetic or filter_type:
            metadata = construct_metadata(
                use_synthetic_fields=use_synthetic,
                synthetic_field_params=synthetic_params,
                filter_type=filter_type,
                filter_value=filter_value,
                filter_tolerance=filter_tolerance
            )
            content += metadata + "\n\n"

        # Add data legend
        content += "# Data Legend (all energies in eV; all fields in V/Å):\n"
        content += " 1. Reactant Donor Field Magnitude\n"
        content += " 2. Reactant Donor Field Std Dev\n"
        content += " 3. Reactant Acceptor Field Magnitude\n"
        content += " 4. Reactant Acceptor Field Std Dev\n"
        content += " 5. Product Donor Field Magnitude\n"
        content += " 6. Product Donor Field Std Dev\n"
        content += " 7. Product Acceptor Field Magnitude\n"
        content += " 8. Product Acceptor Field Std Dev\n"
        content += " 9. Reactant Frames Count\n"
        content += "10. Product Frames Count\n"
        content += "11. Reactant Coulombic Energy\n"
        content += "12. Reactant Coulombic Variance\n"
        content += "13. Reactant Donor Polarization Energy\n"
        content += "14. Reactant Donor Polarization Variance\n"
        content += "15. Reactant Acceptor Polarization Energy\n"
        content += "16. Reactant Acceptor Polarization Variance\n"
        content += "17. Reactant Total Energy\n"
        content += "18. Reactant Total Variance\n"
        content += "19. Product Coulombic Energy\n"
        content += "20. Product Coulombic Variance\n"
        content += "21. Product Donor Polarization Energy\n"
        content += "22. Product Donor Polarization Variance\n"
        content += "23. Product Acceptor Polarization Energy\n"
        content += "24. Product Acceptor Polarization Variance\n"
        content += "25. Product Total Energy\n"
        content += "26. Product Total Variance\n"
        content += "27. Unpolarized Reorganization Energy\n"
        content += "28. Polarized Reorganization Energy\n\n"

        # Add column numbers with fixed width formatting
        content += "# Data:\n"
        header = ""
        for i in range(1, 29):  # Changed to 29 to match 28 columns
            header += f"{i:>{COLUMN_WIDTH}d}"  # Right-align numbers with consistent width
        content += header + "\n"

    # Ensure results is always a list
    if not isinstance(results, list):
        results = [results]

    # Format each result
    for result in results:
        line = ""
        # Handle the data based on the type of value
        for i, value in enumerate(result):
            if i in [8, 9]:  # Frame counts
                line += f"{int(value):>{COLUMN_WIDTH}d}"
            else:  # All other values are floats
                line += f"{float(value):>{COLUMN_WIDTH}.3f}"

        content += line + "\n"

    return content
