"""Tests for :mod:`beadspring.utils.file_utils`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beadspring.utils.file_utils import find_latest_file, generate_lin_log_timesteps


def test_find_latest_file(tmp_path: Path):
    files = ["Conf_1.txt", "Conf_20.txt", "Conf_300.txt", "Conf_4000.txt"]
    for name in files:
        (tmp_path / name).write_text("", encoding="utf-8")

    latest = find_latest_file(tmp_path, "Conf_")
    assert latest == "Conf_4000.txt"


def test_find_latest_file_returns_none_when_no_match(tmp_path: Path):
    assert find_latest_file(tmp_path, "Conf_") is None


def test_find_latest_file_missing_directory(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        find_latest_file(missing, "Conf_")


def test_generate_lin_log_timesteps_produces_expected_sequences(tmp_path: Path):
    log_part, linlog_part = generate_lin_log_timesteps(3, 10, save_file=True, output_file=tmp_path / "steps.txt")
    assert np.all(np.diff(log_part) >= 0)
    assert linlog_part[-1] == 11  # final_step + 1
    saved = np.loadtxt(tmp_path / "steps.txt")
    npt = np.testing
    npt.assert_allclose(saved, linlog_part)
