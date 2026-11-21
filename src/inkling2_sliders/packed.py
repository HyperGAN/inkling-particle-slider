"""Decode released expert tensors on CUDA only for experts used by the router."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import triton
import triton.language as tl


@triton.jit
def _decode(weights, scales, factors, table, experts, output,
            size: tl.constexpr, group: tl.constexpr, block: tl.constexpr):
    slot = tl.program_id(1)
    expert = tl.load(experts + slot).to(tl.int64)
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < size
    scale = tl.load(scales + expert * (size // group) + index // group, mask, other=0)
    if group == 16:
        byte = tl.load(weights + expert * (size // 2) + index // 2, mask, other=0)
        code = (byte >> ((index % 2) * 4)) & 15
        magnitude = code & 7
        value = tl.where(magnitude < 5, magnitude * 0.5,
                         tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)))
        value = tl.where(code < 8, value, -value)
        value *= tl.load(table + scale)
        value *= tl.load(factors + expert)
    else:
        code = tl.load(weights + expert * size + index, mask, other=0)
        value = tl.load(table + code) * tl.exp2(scale.to(tl.float32) - 127)
    tl.store(output + slot.to(tl.int64) * size + index, value, mask)


class PackedProjection(nn.Module):
    def __init__(self, weight, scale, amax=None, denominator=None, *, device=None):
        super().__init__()
        self.device = torch.device(device or weight.device)
        if self.device.type != "cuda" or scale.device != weight.device or weight.ndim != 3:
            raise ValueError("expert weights and scales require one CUDA execution device")
        nvfp4 = amax is not None
        expected = (torch.uint8, torch.float8_e4m3fn) if nvfp4 else (torch.float8_e4m3fn, torch.uint8)
        if (weight.dtype, scale.dtype) != expected:
            raise ValueError("expert weight or scale format differs from Inkling 2 Mini")
        self.rows = weight.shape[1]
        self.columns = weight.shape[2] * (2 if nvfp4 else 1)
        self.group_size = 16 if nvfp4 else 32
        if self.columns % self.group_size or scale.shape != (
            weight.shape[0], self.rows, self.columns // self.group_size
        ):
            raise ValueError("packed expert tensor shapes differ")
        self.num_experts = weight.shape[0]
        weight_codes, scale_codes = weight.view(torch.uint8), scale.view(torch.uint8)
        if weight.device == self.device:
            self.register_buffer("weight_codes", weight_codes, persistent=False)
            self.register_buffer("scale_codes", scale_codes, persistent=False)
        else:
            self.weight_codes, self.scale_codes = weight_codes, scale_codes
        if nvfp4:
            if amax.shape != (weight.shape[0],) or denominator.numel() != 1:
                raise ValueError("invalid NVFP4 per-expert outer scale")
            factors = amax.to(self.device).float() / denominator.to(self.device).float()
        else:
            factors = None
        self.register_buffer("factors", factors, persistent=False)

    def _stage_rows(self, source, indices):
        storage = torch.empty((len(indices), *source.shape[1:]), dtype=source.dtype, pin_memory=True)
        torch.index_select(source, 0, indices, out=storage)
        return storage.to(self.device, non_blocking=True)

    def staged(self, selected, indices):
        weights = self._stage_rows(self.weight_codes, indices)
        scales = self._stage_rows(self.scale_codes, indices)
        factors = self.factors.index_select(0, selected) if self.factors is not None else None
        return weights, scales, factors, torch.arange(selected.numel(), device=self.device)

    def decode(self, selected, table, dtype, *, staged=None):
        weights, scales, factors, indices = staged or (
            self.weight_codes, self.scale_codes, self.factors, selected,
        )
        output = torch.empty((selected.numel(), self.rows, self.columns), dtype=dtype, device=self.device)
        size = self.rows * self.columns
        _decode[(triton.cdiv(size, 512), selected.numel())](
            weights, scales, factors, table, indices, output, size, self.group_size, 512,
        )
        return output


class PackedExperts(nn.Module):
    """The upstream eager dispatch with per-chunk BF16 weight decoding."""

    def __init__(self, original):
        super().__init__()
        self.num_experts = original.num_experts
        self.activation = original.act_fn

    def load_packed(self, shard, prefix, device, *, resident=True):
        for name, source in (("gate_up", "w13"), ("down", "w2")):
            weight = shard.get_tensor(f"{prefix}.{source}_weight")
            scale = shard.get_tensor(f"{prefix}.{source}_weight_scale")
            if resident:
                weight, scale = weight.to(device), scale.to(device)
            outer = []
            if weight.dtype == torch.uint8:
                outer = [
                    shard.get_tensor(f"{prefix}.{source}_weight_{suffix}").to(device)
                    for suffix in ("amax", "outer_scale_denominator")
                ]
            setattr(self, name, PackedProjection(weight, scale, *outer, device=device))
        self.resident = resident
        table = torch.arange(256, device=device, dtype=torch.int16).to(torch.uint8)
        self.register_buffer("fp8_values", table.view(torch.float8_e4m3fn).float(), persistent=False)

    def _project_chunk(self, hidden, expert_weights, selected, cpu_selection, routes):
        gate_staged = None if self.resident else self.gate_up.staged(selected, cpu_selection)
        down_staged = None if self.resident else self.down.staged(selected, cpu_selection)
        gate_up = self.gate_up.decode(selected, self.fp8_values, hidden.dtype, staged=gate_staged)
        down = self.down.decode(selected, self.fp8_values, hidden.dtype, staged=down_staged)
        routed = []
        for slot, (route, tokens) in enumerate(routes):
            gate_up_state = F.linear(hidden[tokens], gate_up[slot])
            expert_state = self.activation(gate_up_state[:, 0::2], gate_up_state[:, 1::2])
            routed.append(F.linear(expert_state, down[slot]) * expert_weights[tokens, route, None])
        return tuple(routed)

    def _single_token(self, hidden, expert_indices, expert_weights):
        selected, permutation = expert_indices[0].sort()
        cpu_selection = None if self.resident else selected.cpu()
        gate_staged = None if self.resident else self.gate_up.staged(selected, cpu_selection)
        down_staged = None if self.resident else self.down.staged(selected, cpu_selection)
        gate_up = self.gate_up.decode(selected, self.fp8_values, hidden.dtype, staged=gate_staged)
        down = self.down.decode(selected, self.fp8_values, hidden.dtype, staged=down_staged)
        single = hidden.expand(selected.numel(), -1).unsqueeze(-1)
        gate_up_state = torch.bmm(gate_up, single).squeeze(-1)
        state = self.activation(gate_up_state[:, 0::2], gate_up_state[:, 1::2])
        routed = torch.bmm(down, state.unsqueeze(-1)).squeeze(-1)
        contributions = (routed * expert_weights[0, permutation, None]).float()
        result = torch.zeros_like(hidden, dtype=torch.float32)
        for row in contributions.unbind():
            result = result + row
        return result.to(hidden.dtype)

    def forward(self, hidden, expert_indices, expert_weights):
        if hidden.shape[0] == 1 and not torch.is_grad_enabled():
            return self._single_token(hidden, expert_indices, expert_weights)
        result = torch.zeros_like(hidden, dtype=torch.float32)
        with torch.no_grad():
            mask = F.one_hot(expert_indices, num_classes=self.num_experts).permute(2, 1, 0)
            positions = mask.nonzero()
            counts = torch.bincount(positions[:, 0], minlength=self.num_experts).tolist()
            active = [(index, count) for index, count in enumerate(counts) if count]
        offset = 0
        for start in range(0, len(active), 8):
            group = active[start:start + 8]
            cpu_selection = torch.tensor([index for index, _ in group], dtype=torch.long)
            selected = cpu_selection.to(hidden.device)
            routes = []
            for _, count in group:
                routes.append(tuple(positions[offset:offset + count, 1:].unbind(-1)))
                offset += count
            args = hidden, expert_weights, selected, cpu_selection, tuple(routes)
            if torch.is_grad_enabled() and (hidden.requires_grad or expert_weights.requires_grad):
                routed = checkpoint(self._project_chunk, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                routed = self._project_chunk(*args)
            for (_, tokens), contribution in zip(routes, routed):
                result.index_add_(0, tokens, contribution.float())
        return result.to(hidden.dtype)
