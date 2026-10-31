import copy

import pytest
import torch

from test_methods import ToyEnv


@pytest.mark.parametrize("module", ["methods.mobihet", "methods.mobihet_rel"])
def test_next_round_resume(tmp_path, module):
    Method = __import__(module, fromlist=["Method"]).Method
    env = ToyEnv()
    method = Method(env)
    method.round(1, 1e-3)
    checkpoint = tmp_path / "state.pt"
    torch.save({"method": method.state(), "gen": env.gen.get_state(), "rng": torch.get_rng_state()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    method.round(2, 1e-3)
    expected = copy.deepcopy(method.state())
    resumed_env = ToyEnv()
    resumed = Method(resumed_env)
    resumed.load_state(saved["method"])
    resumed_env.gen.set_state(saved["gen"])
    torch.set_rng_state(saved["rng"])
    resumed.round(2, 1e-3)
    for key, value in resumed.state().items():
        if torch.is_tensor(value):
            assert torch.equal(value, expected[key]), key
        else:
            assert value == expected[key], key
