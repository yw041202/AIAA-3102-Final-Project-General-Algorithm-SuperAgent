import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from partd.demo_data import create_demo_inputs


@pytest.fixture
def demo(tmp_path_factory):
    # Short names also work with Windows installations lacking long-path support.
    return create_demo_inputs(tmp_path_factory.mktemp("d") / "demo")
