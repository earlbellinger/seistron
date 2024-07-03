import os
import argparse

def get_new_filename(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{base}_{counter}{ext}"
    while os.path.isfile(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{ext}"
    return new_filename

def check_and_get_filename(filename):
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

