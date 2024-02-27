import os

def find_latest_file(directory, search_string):
    """
    Find the latest file in a directory where all fiels start with the
    specified string and return the one with the largest suffix.
    This function might be useful to find the latest binary/trajectory
    file in a directory.

    Args:
    - directory: The directory to search in
    - search_string: The string to search for
    Returns:
    - latest_file: The file with the largest value after the search string
    """
    # Get all files in the directory
    all_files = os.listdir(directory)

    # Filter files starting with "Conf_"
    search_files = [f for f in all_files if f.startswith(search_string)]

    # If no such files are found, return None
    if not search_files:
        print(f"No files found in {directory} starting with {search_string}")
        return None

    # Extract values from filenames and get the file with the largest value
    latest_file = max(search_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    return latest_file