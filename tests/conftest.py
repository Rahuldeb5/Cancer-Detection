import pytest

from tumorlib import io


def pytest_collection_modifyitems(config, items):
    if io.MASK_ROOT.is_dir() and io.CT_ROOT.is_dir():
        return
    skip = pytest.mark.skip(reason=f"PanTS data not found under {io.MASK_ROOT} / {io.CT_ROOT}")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
