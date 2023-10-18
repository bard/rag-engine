import pytest


@pytest.fixture
def example_fixture() -> str:
    return "example"


@pytest.fixture(scope="session")
def example_session_fixture() -> str:
    return "example"
