import pytest


@pytest.fixture(scope="session")
def design():
    """The closed M1 design, built once per test session (about 20 s)."""
    from design import spec as dspec, closure
    return closure.close(dspec.load())
