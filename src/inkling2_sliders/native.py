"""Pinned Transformers looped text tower with original packed CUDA experts."""
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopForCausalLM, Inkling2LoopTextConfig,
)

from .packed import PackedExperts

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
REVISION = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
TRANSFORMERS_REVISION = "6173a2f679ff4ed393afc499aefc569da7e68bc1"
SHARD = "model-00001-of-00001.safetensors"
EXPERT_PREFIX = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.")
PROTOCOL = "inkling2-mini-text-packed-bf16-three-pass-v1"


class Inkling2PackedText(Inkling2LoopForCausalLM):
    @classmethod
    def is_custom_code(cls):
        # Apply the pinned upstream text key conversion once, at the text-model root.
        return False

    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            layer.mlp.experts = PackedExperts(layer.mlp.experts)



def file_sha(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def source_sha():
    digest = sha256()
    for name in ("native.py", "packed.py", "slider.py", "model-lock.json"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def verify_snapshot(path):
    lock = json.loads(Path(__file__).with_name("model-lock.json").read_text())
    for name, expected in lock.items():
        if file_sha(path / name) != expected:
            raise ValueError(f"pinned Inkling 2 Mini checksum differs for {name}")
    return {
        "model": MODEL_ID, "revision": REVISION, "files": lock,
        "transformers": TRANSFORMERS_REVISION, "implementation_sha256": source_sha(),
        "protocol": PROTOCOL,
    }


MAPPED_PROFILES = {
    "mmap": frozenset((6, 7, 9, 11, 12, 14, 17, 23)),
    "mmap-loop": frozenset(range(6, 24)),
}
STORAGE_PROFILES = ("resident", *MAPPED_PROFILES)


class Runtime:
    def __init__(self, model_dir, device, storage="resident"):
        if storage not in STORAGE_PROFILES:
            raise ValueError("select resident or mmap checkpoint storage")
        self.storage = storage
        self.path = Path(model_dir).resolve(strict=True)
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("an explicit CUDA device is required")
        self.identity = verify_snapshot(self.path)
        self.identity["storage"] = storage
        if storage in MAPPED_PROFILES:
            self.identity["resident_expert_layers"] = sorted(MAPPED_PROFILES[storage])
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        release = json.loads((self.path / "config.json").read_text())
        text = Inkling2LoopTextConfig.from_dict(release["text_config"])
        if (text.num_hidden_layers, text.hidden_size, text.loop_num_passes) != (24, 2304, 3):
            raise ValueError("unexpected fixed-loop Inkling 2 Mini text architecture")
        generation = json.loads((self.path / "generation_config.json").read_text())
        text.pad_token_id = generation["pad_token_id"]
        text.eos_token_id = generation["eos_token_id"]
        with safe_open(self.path / SHARD, framework="pt", device="cpu") as shard:
            dense = {
                name: shard.get_tensor(name) for name in shard.keys()
                if name.startswith("model.llm.") and not EXPERT_PREFIX.match(name)
            }
            self.model, report = Inkling2PackedText.from_pretrained(
                None, config=text, state_dict=dense, dtype=torch.bfloat16,
                device_map={"": str(self.device)}, attn_implementation="eager",
                output_loading_info=True,
            )
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                if report.get(key):
                    raise ValueError(f"Inkling 2 Mini failed strict dense load ({key}): {report[key]}")
            del dense
            for index, layer in enumerate(self.model.model.layers):
                layer.mlp.experts.load_packed(
                    shard, f"model.llm.layers.{index}.mlp.experts", self.device,
                    resident=storage == "resident" or index in MAPPED_PROFILES[storage],
                )
        for tensor in (*self.model.parameters(), *self.model.buffers()):
            if tensor.device != self.device:
                raise ValueError(f"model tensor resident on {tensor.device}, expected {self.device}")
        self.model.eval().requires_grad_(False)
        self.prefix_cache = {}

    def last_hidden(self, ids):
        model = self.model.model
        layer = model.layers[-1]
        if layer.is_block_boundary:
            raise ValueError("this native prefix cache requires the fixed final non-boundary layer")
        if ids in self.prefix_cache:
            args, kwargs = self.prefix_cache[ids]
            hidden = layer(*args, **kwargs)
            return model.norm(model.final_input(args[1], hidden)[0]).to(args[0].dtype)[:, -1, :]

        def record_prefix(module, args, kwargs):
            if kwargs.get("cache") is not None or args[0].requires_grad:
                raise ValueError("only frozen nonincremental Inkling prefixes may be cached")
            self.prefix_cache[ids] = (args, kwargs)

        with layer.register_forward_pre_hook(record_prefix, with_kwargs=True):
            return model(input_ids=self.tensor(ids), use_cache=False).last_hidden_state[:, -1, :]

    def encode(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("a nonempty text prompt is required")
        result = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], reasoning_effort="none",
            add_generation_prompt=True, tokenize=False,
        )
        # Prefill the released public-text channel so a short output budget measures the answer.
        rendered = result + "<|message_model|><|content_text|>"
        ids = tuple(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if not 1 <= len(ids) <= 256:
            raise ValueError("this text backend accepts up to 256 prompt tokens")
        return ids

    def tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def render(self, prompt, max_new_tokens=64):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be an integer in [1, 128]")
        inputs = self.tensor(self.encode(prompt))
        result = self.model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            eos_token_id=self.model.config.eos_token_id, pad_token_id=self.model.config.pad_token_id,
        )
        tokens = result[0, inputs.shape[1]:].tolist()
        special = self.tokenizer.decode(tokens, skip_special_tokens=False)
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if "<|content_thinking|>" in special:
            answer = None
        return {
            "text": answer, "raw_text": special, "token_ids": tokens,
            "stop": "eos" if tokens and tokens[-1] == self.model.config.eos_token_id else "length",
        }
