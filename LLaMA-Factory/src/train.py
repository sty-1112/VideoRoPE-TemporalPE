import importlib.util
import sys
from pathlib import Path

videorope_model_path = (
    Path(__file__).resolve().parents[2]
    / "videorope-transformer"
    / "modeling_videorope.py"
)

target_name = "transformers.models.qwen2_vl.modeling_qwen2_vl"

spec = importlib.util.spec_from_file_location(target_name, videorope_model_path)
module = importlib.util.module_from_spec(spec)
sys.modules[target_name] = module
spec.loader.exec_module(module)

from llamafactory.train.tuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    run_exp()


if __name__ == "__main__":
    main()