"""Load spec/vehicle.yaml (the single source of truth)."""
from pathlib import Path
import yaml

SPEC_PATH = Path(__file__).resolve().parents[1] / "spec" / "vehicle.yaml"
G0 = 9.80665


def load(path=SPEC_PATH):
    with open(path) as f:
        return yaml.safe_load(f)["vehicle"]
