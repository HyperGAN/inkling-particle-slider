"""Inkling's packed three-pass model binding for the shared text vibe workflow."""

import torch
from lumen_studio.particles import Branch
from particle_sliders_vibe.text_model import TextModel
from torch import nn


def main(argv=None):
    from particle_sliders_vibe.text_cli import main as text_main

    return text_main(argv, model="inkling-2-mini")


def load_model(snapshot):
    from inkling2_native.native import Runtime

    # Match the native renderer's CPU budget for staging packed experts.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    runtime = Runtime(snapshot, "cuda:0", storage="mmap")
    return InklingVibeModel(runtime)


class InklingVibeModel(TextModel):
    def __init__(self, runtime):
        self.runtime = runtime
        identity = {
            "model": "inkling-2-mini",
            "native": runtime.identity,
            "protocol": "last-block-attention-v1",
        }
        super().__init__(
            runtime.model, runtime.tokenizer, runtime.model.model, identity
        )

    def encode(self, prompt, cue=""):
        content = f"{cue}\n\n{prompt}" if cue else prompt
        return self.runtime.tensor(self.runtime.encode(content))

    def response(self, ids):
        # Only the last block has trainable hooks. Earlier fixed-loop states can
        # be reused for both conditions without caching an adapted activation.
        key = tuple(ids[0].tolist())
        cache = self.runtime.prefix_cache
        if key not in cache and len(cache) >= 24:
            del cache[next(iter(cache))]
        return self.runtime.last_hidden(key).float()

    def new_adapter(self, key):
        adapter = InklingAttentionAdapter(self.tower).to(self.device)
        self.mixer.add(key, adapter)
        return adapter

    def reply(self, prompt, **kwargs):
        result = super().reply(prompt, **kwargs)
        special = self.tokenizer.decode(result["tokens"], skip_special_tokens=False)
        if "<|content_thinking|>" in special:
            result["text"] = ""
        return result


class InklingAttentionAdapter(nn.Module):
    def __init__(self, tower):
        super().__init__()
        last = len(tower.layers) - 1
        self.names = [
            f"layers.{last}.self_attn.{name}.linear"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.branches = nn.ModuleList()
        for name in self.names:
            projection = tower.get_submodule(name)
            self.branches.append(
                Branch(projection.in_features, projection.out_features)
            )


if __name__ == "__main__":
    main()
