import os
import sys

def convert_size(size_bytes):
    """Convert bytes to human-readable format"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units)-1:
        size_bytes /= 1024.0
        unit_index += 1
    return f"{size_bytes:.2f} {units[unit_index]}"


def get_directory_size(directory):
    """Calculate total size of all files in directory and subdirectories"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError as e:
                print(f"Error accessing {filepath}: {e}", file=sys.stderr)
    return total_size

if __name__ == "__main__":
    # Get directory from command line or use current directory
    # directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    directory = os.getcwd()
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    try:
        total_bytes = get_directory_size(directory)
        human_readable = convert_size(total_bytes)
        print(f"Total size of {directory}: {human_readable}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)