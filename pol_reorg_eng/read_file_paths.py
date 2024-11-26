import os
import sys

def read_file_paths(config_file="FilePaths.txt"):
    """
    Read file paths from configuration file and validate their existence.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Dictionary of validated file paths
        
    Raises:
        SystemExit: If config file is missing or any specified paths don't exist
    """
    if not os.path.isabs(config_file):
        config_file = os.path.abspath(config_file)
        
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at: '{config_file}'")
        sys.exit(1)
        
    paths = {}
    try:
        with open(config_file, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                try:
                    key, value = [part.strip() for part in line.split('=', 1)]
                except ValueError:
                    print(f"Error: Invalid format in {config_file} at line {line_number}: '{line}'")
                    sys.exit(1)
                    
                if not os.path.isabs(value):
                    value = os.path.abspath(value)
                value = os.path.normpath(value)
                paths[key] = value
                
        # Validate all paths exist
        missing_files = []
        for key, path in paths.items():
            if not os.path.exists(path):
                missing_files.append((key, path))
                
        if missing_files:
            print("\nError: The following required files are missing:")
            for key, path in missing_files:
                print(f"  {key}:")
                print(f"    Path: {path}")
                print(f"    Directory exists: {os.path.exists(os.path.dirname(path))}")
            print(f"\nCurrent working directory: {os.getcwd()}")
            sys.exit(1)
            
        return paths
        
    except PermissionError:
        print(f"Error: Permission denied reading {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {config_file}: {str(e)}")
        sys.exit(1)
