"""Routed next-token formatting control using native greedy decision margins."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from particle_sliders.adapter_activation import push_activation, restore_activation
from particle_sliders.audio import AudioResponseCritic
from particle_sliders.backends import AdapterParameters
from particle_sliders.engine import ConditionalParticleGAN, PG_SOURCE_PROFILE, PG_VERSION
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, _check_packet, read_packet
from particle_sliders.response import ResponseBackend, ResponseCapabilities, TRAINING as FULL_SCHEMA
from particle_sliders.routed_projection import TinyParticleProjection

from .native import Runtime, file_sha
from .slider import PRECISION, TEXT_CAPABILITIES, _initialize, _validate

BACKEND = "inkling2_mini_bullet"
PROTOCOL = "inkling2-mini-routed-bullet-margin-v1"
RESPONSE = ResponseCapabilities(
    "routed_particle_gan", "text", "native_greedy_bullet_logit_margin", "continuous_vector",
    (1, 1), PG_VERSION, PG_SOURCE_PROFILE, penalty_input_domain="continuous",
    base_precisions=(PRECISION,), qualification="Pinned Inkling 2 Mini greedy token margin",
)
LORA_RANK = 8
LORA_ALPHA = 128.


@dataclass(frozen=True)
class MarginRow:
    hidden: torch.Tensor
    dash_logit: torch.Tensor
    rival: torch.Tensor
    target: torch.Tensor
    neutral: torch.Tensor
    position: int
    edit: bool
    prefix: tuple[int, ...]


def bullet_id(runtime):
    tokens = runtime.tokenizer("-", add_special_tokens=False)["input_ids"]
    if tokens != [12]:
        raise ValueError("pinned native dash token differs")
    return tokens[0]


def softcap_logits(runtime, hidden, token):
    raw = runtime.model.lm_head(hidden)
    softcap = runtime.model.logit_softcap(raw.float())
    top = softcap.topk(2, dim=-1)
    rival = torch.where(top.indices[..., 0] == token, top.values[..., 1], top.values[..., 0])
    return raw[..., token], rival, softcap[..., token] - rival


class BulletSlider(nn.Module):
    """One rank-eight routed delta for the native Markdown dash logit."""

    def __init__(self, runtime):
        super().__init__()
        head = runtime.model.lm_head
        if getattr(head, "_bullet_slider", None) is not None:
            raise ValueError("Inkling output head is already bound")
        self.token = bullet_id(runtime)
        self.particles = nn.Parameter(torch.randn(128, 4, device=head.weight.device))
        reference = nn.Linear(head.in_features, 1, bias=False, device=head.weight.device)
        with torch.no_grad():
            reference.weight.zero_()
        reference.requires_grad_(False)
        project = TinyParticleProjection("bullet_output", reference, self.particles, LORA_RANK, LORA_ALPHA)
        self.projections = nn.ModuleDict({"bullet": project.to(device=head.weight.device, dtype=torch.float32)})
        object.__setattr__(self, "_head", head)
        self._forward = head.forward
        object.__setattr__(head, "_bullet_slider", self)
        head.forward = self.logits

    def shifted(self, hidden, base):
        return base + self.projections["bullet"](hidden.float()).squeeze(-1).to(base.dtype)

    def logits(self, hidden):
        scores = self._forward(hidden)
        if self.projections["bullet"].multiplier:
            scores = scores.clone()
            scores[..., self.token] = self.shifted(hidden, scores[..., self.token].clone())
        return scores

    @contextmanager
    def scaled(self, strength):
        if not 0 <= strength <= 1 or self._head is None:
            raise ValueError("bullet adapter requires an attached head and strength in [0,1]")
        push_activation(self, self.projections.values(), ("multiplier",))
        try:
            self.projections["bullet"].multiplier = float(strength)
            yield self
        finally:
            restore_activation(self)

    def close(self):
        if self._head is None:
            return
        if getattr(self, "_activation_stack", None) or self._head.forward != self.logits:
            raise RuntimeError("cannot detach active or overwritten Inkling bullet adapter")
        self._head.forward = self._forward
        del self._head._bullet_slider
        object.__setattr__(self, "_head", None)
        self._forward = None


class NativeMargin:
    def __init__(self, runtime):
        self.runtime = runtime
        self.model = runtime.model
        self.model_identity = runtime.identity
        self.slider = BulletSlider(runtime)
        self.network = AdapterParameters(self.slider.named_parameters(), self.slider.scaled)
        self.initialization = None

    @property
    def particles(self):
        return self.network.particles

    def generated(self, prepared):
        with self.network.scaled(1.):
            raw = self.slider.shifted(prepared.hidden, prepared.dash_logit)
        logit = self.model.logit_softcap(raw.float())
        return (logit - prepared.rival).view(1, 1)

    @torch.no_grad()
    def sample(self, prepared, *, scale=1.):
        with self.network.scaled(scale):
            raw = self.slider.shifted(prepared.hidden, prepared.dash_logit)
        return (self.model.logit_softcap(raw.float()) - prepared.rival).view(1, 1)

    @torch.no_grad()
    def response(self, *, prompt, strength=1., seed=0):
        ids = self.runtime.encode(prompt)
        hidden = self.runtime.last_hidden(ids)
        with self.network.scaled(strength):
            _, _, margin = softcap_logits(self.runtime, hidden, self.slider.token)
        return margin.view(1, 1)

    def render(self, *, prompt, max_new_tokens=64, strength=1., seed=0, images=None, output_type=None):
        with self.network.scaled(strength):
            return self.runtime.render(prompt, max_new_tokens)

    def metadata(self):
        return {
            "protocol": PROTOCOL, "model_identity": self.model_identity, "coordinates": "dash versus best rival",
            "fresh_initialization": self.initialization, "bullet_token": self.slider.token,
            "rank": LORA_RANK, "alpha": LORA_ALPHA,
        }

    def close(self):
        self.slider.close()


def _tokens(runtime, text):
    return tuple(runtime.tokenizer(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def prepare_rows(runtime, prompt_data):
    token = bullet_id(runtime)
    separator = _tokens(runtime, "\n")
    rows = []
    for row in prompt_data["training"]:
        neutral = runtime.encode(row["prompt"])
        positive = runtime.encode(prompt_data["positive_prefix"] + row["prompt"])
        shared = ()
        sites = []
        for i, item in enumerate(row["items"]):
            sites.append((len(shared), True))
            shared += (token,)
            sites.append((len(shared), False))
            shared += _tokens(runtime, " " + item)
            sites.append((len(shared), False))
            if i < 2:
                shared += separator
        def captured(prefix, positions):
            states = runtime.model.model(input_ids=runtime.tensor(prefix + shared), use_cache=False).last_hidden_state
            return states[0, [len(prefix) + i - 1 for i in positions]]
        features = captured(neutral, [i for i, _ in sites])
        positive_features = captured(positive, [i for i, edit in sites if edit])
        dash_logits, rivals, base_margins = softcap_logits(runtime, features, token)
        _, _, teacher_margins = softcap_logits(runtime, positive_features, token)
        teacher = iter(teacher_margins)
        for k, (i, edit) in enumerate(sites):
            baseline = base_margins[k].view(1, 1)
            target = next(teacher).view(1, 1) if edit else baseline
            rows.append(MarginRow(features[k].contiguous().clone(), dash_logits[k].clone(), rivals[k].clone(),
                                  target.clone(), baseline.clone(), i, edit, neutral + shared[:i]))
    return rows


def _prepare(options, *, cohort, receipt, resident, prepared):
    runtime = resident or Runtime(options["model_dir"], options["device"], options["storage"])
    if (runtime.path != Path(options["model_dir"]).resolve()
            or runtime.device != torch.device(options["device"]) or runtime.storage != options["storage"]):
        raise ValueError("resident runtime differs from the frozen training profile")
    with torch.random.fork_rng(devices=[runtime.device.index]):
        seed_initialization_rng(options["seed"], [runtime.device.index])
        rows = prepared
        if rows is None:
            rows = prepare_rows(runtime, json.loads(Path(options["training_file"]).read_text()))
        native = NativeMargin(runtime)
        try:
            conditions = torch.tensor(
                [[float(row.edit), float(row.neutral.item()) / 16., row.position / 16.] for row in rows],
                device=runtime.device)
            critic = AudioResponseCritic(1, conditions.shape[1]).to(runtime.device)
            native.initialization = (
                _initialize(native, critic, options["seed"], cohort) if receipt is None else receipt
            )
            game = ConditionalParticleGAN(
                native, rows, critic, conditions=conditions, particles=native.particles,
                preset="empf", optimizer_dense_profile="recipe",
                response_domain="continuous_vector", penalty_input_domain="continuous",
                response_scale=options["response_scale"], batch_size=options["batch"],
                seed=options["seed"], total_steps=options["steps"], recipe_overrides=options["rates"],
            )
            specification = INFERENCE.specification(
                native, model_identity=native.model_identity, protocol=PROTOCOL,
                configuration={"fresh_initialization": native.initialization, "bullet_token": native.slider.token},
            )
            return Binding(game, native.close, inference=specification, response=native.response)
        except BaseException:
            native.close()
            raise


def _validate_options(options):
    scale = options.get("response_scale")
    if type(scale) not in (int, float) or not 0 < scale <= 100:
        raise ValueError("finite measured native response scale required")
    _validate({key: value for key, value in options.items() if key != "response_scale"})


def _validate_inference(options, spec):
    if set(options) != {"model_dir", "device"} or spec.identity["protocol"] != PROTOCOL:
        raise ValueError("RAW bullet inference requires model_dir and explicit CUDA device")
    if set(spec.configuration) != {"fresh_initialization", "bullet_token"} or spec.configuration["bullet_token"] != 12:
        raise ValueError("RAW bullet configuration differs from the native token contract")


def _prepare_inference(options, spec):
    runtime = Runtime(options["model_dir"], options["device"], spec.identity["model"]["storage"])
    native = NativeMargin(runtime)
    native.initialization = spec.configuration["fresh_initialization"]
    return InferenceBinding(
        native.network, native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL),
        native.close, response=native.response, render=native.render,
    )


TRAINING = ResponseBackend(BACKEND, RESPONSE, _prepare, _validate_options)
INFERENCE = InferenceBackend(BACKEND, TEXT_CAPABILITIES, _prepare_inference, _validate_inference)


def open_training(config, *, resume=None, resident=None, prepared=None):
    packet = read_packet(resume) if resume is not None else None
    receipt = None
    if packet is not None:
        for path in Path(resume).parent.glob("quarter-*.json"):
            if json.loads(path.read_text())["integrity"] != "pass":
                raise ValueError("recorded native integrity gate prevents candidate resume")
        _check_packet(packet)
        if packet["schema"] != FULL_SCHEMA or packet["manifest"]["config"] != config.as_dict():
            raise ValueError("bullet resume requires the exact FULL training packet")
        previous = packet["payload"]["game"]["config"]["backend"]
        if previous["protocol"] != PROTOCOL:
            raise ValueError("saved checkpoint uses a different response")
        receipt = deepcopy(previous["fresh_initialization"])
    session = TRAINING.open(
        config, operation="resume" if packet else "train", cohort=config.cohort,
        receipt=receipt, resident=resident, prepared=prepared,
    )
    if packet is not None:
        try:
            session.restore(packet)
        except BaseException:
            session.close()
            raise
    return session
