import os

from beadspring.utils.file_utils import find_latest_file


def test_find_latest_file(tmp_path):
    filenames = ["Conf_1.txt", "Conf_20.txt", "Conf_300.txt", "Conf_4000.txt"]
    for name in filenames:
        (tmp_path / name).touch()

    latest_file = find_latest_file(tmp_path, "Conf_")
    assert latest_file == "Conf_4000.txt"

    latest_file = find_latest_file(tmp_path, "Invalid_")
    assert latest_file is None
