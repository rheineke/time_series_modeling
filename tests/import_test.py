"""Smoke test: every module imports on this Python"""

import importlib

import pytest

MODULES = [
    "data",
    "ipsh",
    "run_eda",
    "run_graphs",
    "run_model_fit",
    "utils",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
