from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particle_sliders.pg09_backends import AdapterParameters
from transformers.core_model_loading import Inkling2LoopDequantizeExperts

from inkling2_sliders.cli import rate_options
from inkling2_sliders.measurement import intensity, integrity, qualify_integrity
from inkling2_sliders.packed import PackedProjection
from inkling2_sliders.slider import Slider


def test_rate_options_vary_one_optimizer_role():
    recipe = get_recipe("empf")
    rates = list(rate_options(recipe))
    assert len(asdict(recipe)) == 179
    assert len(rates) == 7
    center = rates[0][1]
    for _, absolute, options in rates[1:]:
        assert sum(x != y for x, y in zip(center, absolute)) == 1
        assert (options["lr"], options["lr"] * options["d_lr_mult"],
                options["lr"] * options["prior_lr_mult"]) == pytest.approx(absolute)


def test_instrument_and_integrity_are_separate():
    assert len(qualify_integrity()) == 9
    assert intensity("- Apples\n- Nuts\n- Crackers") == 1
    assert intensity("- Same\n- Same\n- Same") == 1 / 3
    assert intensity("Apples, nuts and crackers.") == 0
    assert integrity({"text": "Apples, nuts and crackers.", "stop": "eos"}) == "pass"
    assert integrity(None) == "unmeasured"


def test_routed_attachment_restores_base_and_starts_at_zero():
    q, o = (nn.Linear(8, 8, bias=False).requires_grad_(False) for _ in range(2))
    attention = SimpleNamespace(q_proj=SimpleNamespace(linear=q), o_proj=SimpleNamespace(linear=o))
    model = nn.Module()
    model.layers = [SimpleNamespace(self_attn=attention)]
    # Model registration is required so the core attachment transaction owns the hosts.
    model.add_module("q", q)
    model.add_module("o", o)
    inputs = torch.arange(8.).view(1, 8)
    reference = o(q(inputs)).detach()
    original = (q.forward, o.forward)
    slider = Slider(model)
    try:
        network = AdapterParameters(slider.named_parameters(), slider.scaled)
        assert network.particles is slider.particles
        with network.scaled(1.):
            assert torch.equal(o(q(inputs)), reference)
        with torch.no_grad():
            slider.projections["o_proj"].lora_up.weight.fill_(1.)
        with network.scaled(1.):
            assert not torch.equal(o(q(inputs)), reference)
            with network.scaled(0.):
                assert torch.equal(o(q(inputs)), reference)
            assert not torch.equal(o(q(inputs)), reference)
        with pytest.raises(ValueError):
            with slider.scaled(1.1):
                pass
    finally:
        slider.close()
    assert (q.forward, o.forward) == original


@pytest.mark.parametrize("nvfp4", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed expert parity needs CUDA")
def test_packed_decode_matches_pinned_upstream(nvfp4):
    gpu = torch.device("cuda:0")
    columns = 64
    group = 16 if nvfp4 else 32
    if nvfp4:
        weight = torch.randint(0, 256, (4, 8, columns // 2), dtype=torch.uint8, device=gpu)
        scale = torch.full((4, 8, columns // group), 0.75, device=gpu).to(torch.float8_e4m3fn)
        amax = torch.tensor([3., 5., 7., 11.], device=gpu)
        denominator = torch.tensor([15.], device=gpu)
    else:
        weight = torch.linspace(-4, 4, 4 * 8 * columns, device=gpu).view(4, 8, columns).to(torch.float8_e4m3fn)
        scale = torch.randint(123, 132, (4, 8, columns // group), dtype=torch.uint8, device=gpu)
        amax = denominator = None
    packed = PackedProjection(weight, scale, amax, denominator)
    table = torch.arange(256, device=gpu, dtype=torch.int16).to(torch.uint8).view(torch.float8_e4m3fn).float()
    indices = torch.tensor([3, 0, 2], device=gpu)
    names = ["weight", "scale", "amax", "denominator"]
    inp = {name: tensor for name, tensor in zip(names, (weight, scale, amax, denominator)) if tensor is not None}
    reference = Inkling2LoopDequantizeExperts().convert(inp, names, ["output"])["output"]
    assert torch.equal(packed.decode(indices, table, torch.bfloat16), reference[indices])


def test_original_text_weight_conversion_is_applied_once():
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping, get_model_conversion_mapping,
    )
    from inkling2_sliders.native import Inkling2LoopTextConfig, Inkling2PackedText

    with torch.device("meta"):
        model = Inkling2PackedText(Inkling2LoopTextConfig())
    source = get_checkpoint_conversion_mapping("inkling2_loop_text")
    active = get_model_conversion_mapping(model, add_legacy=False)
    assert len(active) == len(source)
    assert [(item.source_patterns, item.target_patterns) for item in active] == [
        (item.source_patterns, item.target_patterns) for item in source
    ]


def test_scoped_initialization_preserves_zero_up_and_owns_bank():
    from inkling2_sliders.slider import _initialize

    q, o = (nn.Linear(8, 8, bias=False).requires_grad_(False) for _ in range(2))
    attention = SimpleNamespace(q_proj=SimpleNamespace(linear=q), o_proj=SimpleNamespace(linear=o))
    model = nn.Module()
    model.layers = [SimpleNamespace(self_attn=attention)]
    model.add_module("q", q)
    model.add_module("o", o)
    slider = Slider(model)
    try:
        initial = [projection.lora_down.weight.detach().clone() for projection in slider.projections.values()]
        critic = nn.Sequential(nn.Linear(20, 16), nn.SiLU(), nn.Linear(16, 1))
        native = SimpleNamespace(slider=slider, particles=slider.particles)
        receipt = _initialize(native, critic, 17361, "test")
        assert set(receipt["components"]) == {"adapter", "bank", "critic.0", "critic.2"}
        assert receipt["completed_steps"] == 0
        assert receipt["restored_from"] is None
        for original, projection in zip(initial, slider.projections.values()):
            assert torch.equal(original, projection.lora_down.weight)
            assert torch.count_nonzero(projection.lora_up.weight) == 0
        assert torch.isfinite(slider.particles).all()
    finally:
        slider.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert dispatch parity needs CUDA")
def test_grouped_decode_replay_and_multitoken_gradient():
    from inkling2_sliders.packed import PackedExperts
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopSoftcappedSwiGLU, Inkling2LoopTextConfig,
    )

    device = torch.device("cuda:0")
    original = SimpleNamespace(num_experts=4, act_fn=Inkling2LoopSoftcappedSwiGLU(Inkling2LoopTextConfig()))
    experts = PackedExperts(original)
    experts.resident = True

    def make(rows):
        base = torch.randint(-4, 5, (4, rows, 32), device=device).float().mul_(0.25)
        packed = base.to(torch.float8_e4m3fn)
        scales = torch.full((4, rows, 1), 127, dtype=torch.uint8, device=device)
        return PackedProjection(packed, scales), base.to(torch.bfloat16)

    experts.gate_up, gate = make(64)
    experts.down, down = make(32)
    codes = torch.arange(256, device=device, dtype=torch.int16).to(torch.uint8)
    experts.fp8_values = codes.view(torch.float8_e4m3fn).float()
    hidden = torch.randn(1, 32, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 0, 3]], device=device)
    weights = torch.tensor([[0.5, 0.25, 0.25]], device=device, dtype=torch.bfloat16)
    reference = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in (0, 2, 3):
        position = (ids == expert).nonzero()[0, 1]
        state = torch.nn.functional.linear(hidden, gate[expert])
        active = experts.activation(state[:, 0::2], state[:, 1::2])
        projected = torch.nn.functional.linear(active, down[expert])
        reference += (projected * weights[0, position]).float()
    with torch.no_grad():
        actual = experts(hidden, ids, weights)
    torch.testing.assert_close(actual, reference.to(torch.bfloat16), rtol=0.0157, atol=0.0078125)
    training = hidden.expand(2, -1).clone().requires_grad_()
    result = experts(training, ids.expand(2, -1), weights.expand(2, -1))
    gradient, = torch.autograd.grad(result.float().sum(), training)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_bullet_margin_matches_native_head_and_off_is_exact():
    from inkling2_sliders.bullet import MarginRow, NativeMargin, softcap_logits

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "-"
            return {"input_ids": [12]}

    head = nn.Linear(16, 32, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    runtime = SimpleNamespace(
        model=SimpleNamespace(lm_head=head, logit_softcap=lambda x: torch.tanh(x / 30) * 30),
        tokenizer=Tokenizer(), identity={"fixture": True},
    )
    hidden = torch.linspace(-1, 1, 16, dtype=torch.bfloat16)
    outputs = head(hidden)
    dash, rival, margin = softcap_logits(runtime, hidden, 12)
    original = head.forward
    adapter = NativeMargin(runtime)
    try:
        row = MarginRow(hidden, dash, rival, margin.view(1, 1), margin.view(1, 1), 0, True, (12, 1))
        assert torch.equal(adapter.sample(row), row.neutral)
        with torch.no_grad():
            adapter.slider.projections["bullet"].lora_up.weight.fill_(2.)
        with adapter.network.scaled(1.):
            updated = head(hidden)
            assert not torch.equal(outputs[12], updated[12])
            assert torch.equal(torch.cat((outputs[:12], outputs[13:])), torch.cat((updated[:12], updated[13:])))
            assert torch.equal(softcap_logits(runtime, hidden, 12)[2].view(1, 1), adapter.sample(row))
        assert torch.equal(head(hidden), outputs)
        gradient = torch.autograd.grad(adapter.generated(row).sum(), tuple(adapter.network.parameters()),
                                       allow_unused=False)
        assert all(torch.isfinite(g).all() for g in gradient)
    finally:
        adapter.close()
    assert head.forward == original
