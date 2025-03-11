
"""
Functions to manage input data files.
"""



import os
import argparse

def get_new_filename(filename):
    """
    If the given input filename already exists, the
    funuction will append a numerical value to the end
    of the filename before the .filetype (e.g. .png). Each time 
    the associated script is run, the file produced will have
    an increasing (by 1) numerical index to avoid overwriting the file.

    Input: filename.png (str)
    Output: filename_1.png (str)
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{base}_{counter}{ext}"
    while os.path.isfile(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{ext}"
    return new_filename

def check_and_get_filename(filename):
    """
    A function to check if a certain file exists, and asks the user
    if they wish to overwrite the existing file or create a new file
    under a new filename with an appended numerical index.

    Input: filename.png (str)
    Output: either filename.png or filename_1.png (str)
    """
    if os.path.isfile(filename):
        print(f"The file '{filename}' already exists.")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if response == 'yes':
            print(f"Overwriting the file '{filename}'.")
            return filename
        else:
            new_filename = get_new_filename(filename)
            print(f"Saving the file as '{new_filename}'.")
            return new_filename
    else:
        print(f"The file '{filename}' does not exist. Saving as '{filename}'.")
        return filename

def main():
    parser = argparse.ArgumentParser(description="Check if a filename exists before saving a new file.")
    parser.add_argument('filename', type=str, help='The filename to check and potentially save')
    args = parser.parse_args()
    filename = check_and_get_filename(args.filename)
    # Add your file saving logic here
    with open(filename, 'w') as f:
        f.write("Example data")  # Example file saving logic

if __name__ == "__main__":
    main()


def list_parquet_files(directory):
    """List all Parquet files in the specified directory."""
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
        if files:
            print("Parquet files in the directory:")
            for file in files:
                print(file)
        else:
            print("No Parquet files found in the directory.")
    except Exception as e:
        print(f"Error accessing the directory: {directory}")
        print(e)
