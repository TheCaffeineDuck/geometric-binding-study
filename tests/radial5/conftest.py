import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long check (minutes); deselect with -m 'not slow'")
