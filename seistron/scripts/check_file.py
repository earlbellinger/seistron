import os
import argparse

def file_exists(filename):
    return os.path.isfile(filename)

def main():
    parser = argparse.ArgumentParser(description="Check if a file exists.")
    parser.add_argument('filename', type=str, help='The file to check')
    args = parser.parse_args()

    if file_exists(args.filename):
        print(f"The file '{args.filename}' exists.")
    else:
        print(f"The file '{args.filename}' does not exist.")

if __name__ == "__main__":
    main()

