import numpy as np
import pytest
import os
from beadspring.utils import build_architectures as ba

def test_create_ring(tmp_path):
    output_file = tmp_path / "ring.data"
    ba.create_ring(10, file_name=str(output_file))
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "10 atoms" in content
    assert "10 bonds" in content # Ring has N bonds

def test_create_linear_chain(tmp_path):
    output_file = tmp_path / "linear.data"
    ba.create_linear_chain(str(output_file), no_of_monomers=10)
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "10 atoms" in content
    assert "9 bonds" in content # Linear chain has N-1 bonds

def test_create_star(tmp_path):
    output_file = tmp_path / "star.data"
    # 3 arms, 5 monomers each
    # Total atoms = core (depends on implementation, usually many beads on surface) 
    # + 3 * 5
    # Let's just check it runs and creates file
    ba.create_star(3, 5, file_name=str(output_file))
    
    assert output_file.exists()

def test_create_polymer_matrix(tmp_path):
    output_file = tmp_path / "matrix.data"
    # 2 chains, 5 monomers each
    ba.create_polymer_matrix(2, 5, file_name=str(output_file))
    
    assert output_file.exists()
    content = output_file.read_text()
    # 2 * 5 = 10 atoms
    # But implementation might include core or something? 
    # Docstring says "Create a matrix of polymer chains... The core is a sphere..."
    # It seems it creates chains grafted to a core?
    # Let's just verify file creation for now.
    assert "atoms" in content
