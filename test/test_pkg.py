import importlib

import pytest

import fairseq


def test_version():
    assert isinstance(fairseq.__version__, str)
    assert fairseq.__version__


@pytest.mark.parametrize(
    "module",
    [
        "fairseq.data",
        "fairseq.models",
        "fairseq.modules",
        "fairseq.tasks",
        "fairseq.checkpoint_utils",
        "fairseq.models.hubert",
        "fairseq.models.wav2vec",
    ],
)
def test_import(module):
    importlib.import_module(module)
