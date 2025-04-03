from smiles_preprocessor import SMILESPreprocessor

import pytest


@pytest.fixture
def empty_preprocessor():
    return SMILESPreprocessor()


def test_encode(empty_preprocessor):
    smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
    enc = empty_preprocessor.encode(smi)
    assert enc == [1, 5, 5, 20, 18, 7, 21, 7, 5, 3, 18, 5, 5, 18, 5, 5, 18, 5, 3, 5, 20, 18, 7, 21, 7, 3, 2]