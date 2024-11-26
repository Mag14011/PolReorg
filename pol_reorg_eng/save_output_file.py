import os
import create_output_folder as cot

def save_output_file(folder_path, filename, content):
    # Ensure the output folder exists
    cot.create_output_folder(folder_path)

    # Define the full file path
    file_path = os.path.join(folder_path, filename)

    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(content)

    print(f" **File saved to: {file_path}")
