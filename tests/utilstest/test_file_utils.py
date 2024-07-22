import os

from beadspring.utils.file_utils import find_latest_file


def test_find_latest_file():
    # Create a temporary directory for testing
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)

    # Create some test files
    test_files = [
        "Conf_1.txt",
        "Conf_20.txt",
        "Conf_300.txt",
        "Conf_4000.txt",
    ]
    for file in test_files:
        open(os.path.join(test_dir, file), "w").close()

    # Test when there are files starting with "Conf_"
    latest_file = find_latest_file(test_dir, "Conf_")
    assert latest_file == "Conf_4000.txt"

    # Test when there are no files starting with "Conf_"
    latest_file = find_latest_file(test_dir, "Invalid_")
    assert latest_file is None

    # Remove the temporary directory and files after testing
    for file in test_files:
        os.remove(os.path.join(test_dir, file))
    os.rmdir(test_dir)
