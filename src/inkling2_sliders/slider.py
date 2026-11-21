"""Product binding for shared EMPF training, strict FULL resume and RAW inference."""
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
import json
from pathlib import Path

import torch
from torch import nn
from particlegan.init import R2Normal
from particle_sliders.adapter_activation import push_activation, restore_activation
from particle_sliders.adapter_attachment import atomic_attachment
from particle_sliders.audio import ARLastHiddenResponseAdapter, AudioResponseCritic, AudioRow, audio_conditions
from particle_sliders.engine import ConditionalParticleGAN, PG_SOURCE_PROFILE, PG_VERSION
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.initialization import ComponentInitialization, initialize_fresh_components
from particle_sliders.lifecycle import Binding, Capabilities, _check_packet, read_packet
from particle_sliders.response import ResponseBackend, ResponseCapabilities, TRAINING as FULL_SCHEMA
from particle_sliders.routed_projection import TinyParticleProjection
from particle_sliders.weight_policy import require_no_weight_ema

from .native import PROTOCOL, STORAGE_PROFILES, Runtime, file_sha, source_sha

BACKEND = "inkling2_mini"
WIDTH = 2304
POLICY = "seeded-keep-adapter-qr-critic-r2-bank-v1"
PRECISION = "bf16-packed-nvfp4-mxfp8"
RESPONSE_CAPABILITIES = ResponseCapabilities(
    "routed_particle_gan", "text", "native_final_AR_token_hidden", "continuous_vector",
    (1, WIDTH), PG_VERSION, PG_SOURCE_PROFILE, penalty_input_domain="continuous",
    base_precisions=(PRECISION,), qualification="Pinned Inkling 2 Mini text tower",
)
TEXT_CAPABILITIES = Capabilities(
    "routed_particle_gan", render=True, render_outputs=("text",),
    operations=("response", "render"), base_precisions=(PRECISION,),
    qualification="Pinned native greedy text generation",
)


class Slider(nn.Module):
    @atomic_attachment("model")
    def __init__(self, model):
        super().__init__()
        attention = model.layers[-1].self_attn
        targets = (("q_proj", attention.q_proj.linear), ("o_proj", attention.o_proj.linear))
        if any(type(host) is not nn.Linear or getattr(host, "_inkling_slider", None) is not None
               for _, host in targets):
            raise ValueError("expected two unbound final Inkling attention projections")
        self.particles = nn.Parameter(torch.randn(128, 4, device=targets[0][1].weight.device))
        self.projections = nn.ModuleDict()
        self._attached = []
        for name, host in targets:
            projection = TinyParticleProjection(name, host, self.particles, 8, 8.0)
            self.projections[name] = projection.to(device=host.weight.device, dtype=torch.float32)
            object.__setattr__(host, "_inkling_slider", self)
            self._attached.append((host, projection))

    @contextmanager
    def scaled(self, strength):
        if type(strength) not in (int, float) or not isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Inkling slider strength must be in [0, 1]")
        if not self._attached:
            raise RuntimeError("slider is detached")
        push_activation(self, self.projections.values(), ("multiplier",))
        try:
            for projection in self.projections.values():
                projection.multiplier = float(strength)
            yield
        finally:
            restore_activation(self)

    def close(self):
        if getattr(self, "_activation_stack", None):
            raise RuntimeError("cannot detach active Inkling slider")
        for host, projection in self._attached:
            if host._inkling_slider is not self or host.forward != projection.forward:
                raise RuntimeError("Inkling adapter no longer owns its host projection")
        for host, projection in reversed(self._attached):
            host.forward = projection.org_forward
            projection.org_forward = None
            del host._inkling_slider
        self._attached.clear()


class NativeResponse(ARLastHiddenResponseAdapter):
    family = "inkling2-mini-text"

    def __init__(self, runtime):
        self.runtime = runtime
        self.initialization = None
        slider = Slider(runtime.model.model)
        try:
            super().__init__(
                runtime.model.model, slider, model_identity=runtime.identity,
                penalty_input_domain="continuous",
            )
        except BaseException:
            slider.close()
            raise

    def _hidden(self, ids):
        return self.runtime.last_hidden(ids)

    def metadata(self):
        return {
            **super().metadata(), "protocol": PROTOCOL,
            "fresh_initialization": self.initialization,
        }

    def response(self, *, prompt, strength=1., seed=0):
        return self._response(self.runtime.encode(prompt), scale=strength)

    def render(self, *, prompt, max_new_tokens=64, strength=1., seed=0, images=None, output_type=None):
        with self.network.scaled(strength):
            return self.runtime.render(prompt, max_new_tokens)

    def close(self):
        self.slider.close()


def _seeds(module, seed, label):
    return {
        name: int.from_bytes(sha256(f"{POLICY}:{seed}:{label}:{name}".encode()).digest()[:8], "big") % 2**63
        for name, parameter in module.named_parameters() if parameter.requires_grad and parameter.numel()
    }


def _initialize(native, critic, seed, cohort):
    bank = nn.Module()
    bank.register_parameter("particles", native.particles)
    projections = native.slider.projections
    components = {
        "adapter": ComponentInitialization(projections, "keep"),
        "bank": ComponentInitialization(
            bank, "deterministic_orthogonal", _seeds(bank, seed, "bank"),
            distributions={"particles": R2Normal(0., 1.)},
        ),
    }
    for name, leaf in critic.named_modules():
        if type(leaf) is nn.Linear:
            components[f"critic.{name}"] = ComponentInitialization(
                leaf, "deterministic_orthogonal", _seeds(leaf, seed, f"critic.{name}"),
            )
    return initialize_fresh_components(
        components, cohort_id=cohort,
        source_identity={"implementation": source_sha(), "policy": POLICY},
        completed_steps=0, restored_from=None, optimizers=(),
    ).as_dict()


def _validate(options):
    expected = {
        "model_dir", "device", "training_file", "training_sha256", "steps", "seed", "batch", "rates",
        "ema_decay", "storage",
    }
    if set(options) != expected:
        raise ValueError("Inkling training requires exact model, device, data, budget, rates and EMA fields")
    require_no_weight_ema(options["ema_decay"])
    if version("particle-sliders-core") != "0.15.2":
        raise ValueError("install the reviewed Particle Sliders core 0.15.2")
    device = torch.device(options["device"])
    if device.type != "cuda" or device.index is None:
        raise ValueError("production training requires explicit CUDA placement")
    if options["storage"] not in STORAGE_PROFILES:
        raise ValueError("unknown expert storage profile")
    rates = options["rates"]
    if set(rates) != {"lr", "d_lr_mult", "prior_lr_mult"} or any(
        type(rate) not in (int, float) or not isfinite(rate) or rate <= 0 for rate in rates.values()
    ):
        raise ValueError("all three exact EMPF rate options must be finite positive numbers")
    if file_sha(options["training_file"]) != options["training_sha256"]:
        raise ValueError("training prompt file differs from the frozen cohort")


def _preflight(options, request, staged):
    if request["native"] or set(staged) != {"cohort", "receipt", "resident", "prepared"}:
        raise ValueError("use open_training for fresh training or exact resume")


def prepare_rows(runtime, rows):
    native = NativeResponse(runtime)
    try:
        return [native.prepare(AudioRow(runtime.encode(row["neutral"]), runtime.encode(row["positive"])))
                for row in rows]
    finally:
        native.close()


def _prepare(options, *, cohort, receipt, resident, prepared):
    runtime = resident or Runtime(options["model_dir"], options["device"], options["storage"])
    if (runtime.path != Path(options["model_dir"]).resolve()
            or runtime.device != torch.device(options["device"]) or runtime.storage != options["storage"]):
        raise ValueError("resident runtime differs from the frozen training path or device")
    device = runtime.device
    with torch.random.fork_rng(devices=[device.index]):
        seed_initialization_rng(options["seed"], [device.index])
        native = NativeResponse(runtime)
        try:
            rows = prepared
            if rows is None:
                with open(options["training_file"], encoding="utf-8") as source:
                    pairs = json.load(source)
                rows = [native.prepare(AudioRow(runtime.encode(row["neutral"]), runtime.encode(row["positive"])))
                        for row in pairs]
            conditions = audio_conditions(rows)
            critic = AudioResponseCritic(WIDTH, conditions.shape[1]).to(device)
            native.initialization = (
                _initialize(native, critic, options["seed"], cohort) if receipt is None else receipt
            )
            game = ConditionalParticleGAN(
                native, rows, critic, conditions=conditions, particles=native.particles,
                preset="empf", optimizer_dense_profile="recipe",
                response_domain="continuous_vector", penalty_input_domain="continuous",
                total_steps=options["steps"], seed=options["seed"], batch_size=options["batch"],
                recipe_overrides=options["rates"],
            )
            spec = INFERENCE.specification(
                native, model_identity=runtime.identity, protocol=PROTOCOL,
                configuration={"fresh_initialization": native.initialization},
            )
            return Binding(game, native.close, inference=spec, response=native.response)
        except BaseException:
            native.close()
            raise


def _validate_inference(options, spec):
    if set(options) != {"model_dir", "device"} or spec.identity["protocol"] != PROTOCOL:
        raise ValueError("RAW Inkling inference requires model_dir and an explicit CUDA device")
    if set(spec.configuration) != {"fresh_initialization"}:
        raise ValueError("RAW Inkling initialization provenance is required")


def _prepare_inference(options, spec):
    runtime = Runtime(options["model_dir"], options["device"], spec.identity["model"]["storage"])
    native = NativeResponse(runtime)
    native.initialization = spec.configuration["fresh_initialization"]
    return InferenceBinding(
        native.network, native_identity(native, model_identity=runtime.identity, protocol=PROTOCOL),
        native.close, response=native.response, render=native.render,
    )


TRAINING = ResponseBackend(BACKEND, RESPONSE_CAPABILITIES, _prepare, _validate, preflight=_preflight)
INFERENCE = InferenceBackend(BACKEND, TEXT_CAPABILITIES, _prepare_inference, _validate_inference)


def open_training(config, *, resume=None, resident=None, prepared=None):
    packet = read_packet(resume) if resume is not None else None
    receipt = None
    if packet is not None:
        for path in Path(resume).parent.glob("quarter-*.json"):
            recorded = json.loads(path.read_text())
            if recorded["integrity"] != "pass":
                raise ValueError("candidate has a recorded native integrity stop")
        _check_packet(packet)
        if packet["schema"] != FULL_SCHEMA or packet["manifest"]["config"] != config.as_dict():
            raise ValueError("resume requires an exact FULL training packet")
        receipt = packet["payload"]["game"]["config"]["backend"]["fresh_initialization"]
        if (receipt["cohort_id"] != config.cohort or receipt["completed_steps"] != 0
                or receipt["source_identity"] != {"implementation": source_sha(), "policy": POLICY}):
            raise ValueError("FULL initialization identity differs")
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
