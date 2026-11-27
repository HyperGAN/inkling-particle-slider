from types import SimpleNamespace

import torch
from inkling_vibe import InklingVibeModel


def test_evicted_prefix_is_frozen_but_final_block_still_receives_gradients():
    weight = torch.tensor(2.0, requires_grad=True)
    cache = {(i,): 1.0 for i in range(24)}

    def last_hidden(key):
        if key not in cache:
            assert not torch.is_grad_enabled()
            cache[key] = float(key[0])
        return weight * cache[key]

    host = SimpleNamespace(
        runtime=SimpleNamespace(prefix_cache=cache, last_hidden=last_hidden)
    )
    ids = torch.tensor([[25]])
    InklingVibeModel.response(host, ids).backward()
    assert weight.grad.item() == 25.0
    assert len(cache) == 24 and (0,) not in cache
    with torch.no_grad():
        assert InklingVibeModel.response(host, ids).item() == 50.0
