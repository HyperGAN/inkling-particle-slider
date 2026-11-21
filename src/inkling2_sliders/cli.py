"""Run the fixed native panel or render with an independent RAW reload."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean

import torch
from particlegan import get_recipe
from particle_sliders.inference import open_inference, read_inference
from particle_sliders.lifecycle import SessionConfig, atomic_json, atomic_save, read_packet
from particle_sliders.pg09_audio import AudioRow, PreparedAudioResponse

from .measurement import intensity, integrity, qualify_integrity
from .native import STORAGE_PROFILES, Runtime, file_sha, source_sha
from .slider import BACKEND, INFERENCE, open_training, prepare_rows

PANEL = Path(__file__).with_name("panel.json")


def save(path, value):
    atomic_json(path, value)


def rate_options(recipe):
    center = (recipe.lr, recipe.lr * recipe.d_lr_mult, recipe.lr * recipe.prior_lr_mult)
    yield "stock", center, dict(lr=recipe.lr, d_lr_mult=recipe.d_lr_mult, prior_lr_mult=recipe.prior_lr_mult)
    for axis, role in enumerate(("generator", "critic", "bank")):
        for label, factor in (("half", 0.5), ("double", 2.0)):
            absolute = list(center)
            absolute[axis] *= factor
            generator, critic, bank = absolute
            yield f"{role}-{label}", tuple(absolute), dict(
                lr=generator, d_lr_mult=critic / generator, prior_lr_mult=bank / generator,
            )


def marker(message):
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), flush=True)


def capture(runtime, prompt, limit):
    output = runtime.render(prompt, limit)
    return {**output, "intensity": intensity(output["text"]) if output["text"] is not None else None, "integrity": integrity(output)}


def load_json(path):
    return json.loads(path.read_text())


def run_search(model_dir, device, out, *, storage, resume=False):
    out.mkdir(parents=True, exist_ok=resume)
    panel = load_json(PANEL)
    steps, seed, limit = (panel[key] for key in ("steps", "seed", "max_new_tokens"))
    prefix = panel["positive_prefix"]
    prompts = panel["heldout"]
    training = [dict(neutral=prompt, positive=prefix + prompt) for prompt in panel["training"]]
    training_file = out / "training.json"
    if not resume:
        save(training_file, training)
    elif load_json(training_file) != training:
        raise ValueError("previous training cases differ")
    recipe = get_recipe("empf")
    rates = list(rate_options(recipe))
    plan = {
        "panel": panel, "panel_sha256": file_sha(PANEL),
        "implementation_sha256": source_sha(),
        "runner_sha256": file_sha(__file__),
        "model_dir": str(model_dir.resolve()), "device": device, "storage": storage,
        "readout_sha256": file_sha(Path(__file__).with_name("measurement.py")),
        "training_sha256": file_sha(training_file),
        "rates": rates, "stock_recipe": asdict(recipe),
        "torch": str(torch.__version__), "transformers": version("transformers"),
        "particlegan": version("particlegan"), "particle_sliders_core": version("particle-sliders-core"),
        "weight_policy": {"ema_decay": None, "selection": "raw"},
        "scoring": "Equal-case mean distinct-bullet intensity ON minus OFF. Integrity is separate.",
    }
    if not resume:
        save(out / "plan.json", plan)
    elif load_json(out / "plan.json") != json.loads(json.dumps(plan)):
        raise ValueError("previous search plan differs")
    if (out / "result.json").exists():
        return
    qualification = qualify_integrity()
    runtime = Runtime(model_dir, device, storage)
    marker({"native": "loaded", "cuda": torch.cuda.get_device_name(device),
            "allocated_bytes": torch.cuda.memory_allocated(device)})
    controls = []
    for index, prompt in enumerate(prompts):
        control = {"prompt": prompt}
        for branch, conditioned in (("off", prompt), ("explicit_positive", prefix + prompt)):
            path = out / f"control-{index}-{branch}.json"
            if not path.exists():
                save(path, capture(runtime, conditioned, limit))
            control[branch] = load_json(path)
        controls.append(control)
    save(out / "controls.json", {"integrity_fixtures": qualification, "matched": controls})
    if any(control[branch]["integrity"] != "pass" for control in controls for branch in ("off", "explicit_positive")):
        raise ValueError("native controls did not qualify the bounded integrity instrument")
    if any(control["explicit_positive"]["intensity"] <= control["off"]["intensity"] for control in controls):
        raise ValueError("native explicit-control bracket did not qualify all four bullet cases")
    marker({"native": "panel qualified", "controls": len(controls)})
    rows_path = out / "teachers.pt"
    if not rows_path.exists():
        rows = prepare_rows(runtime, training)
        atomic_save(rows_path, {
            "rows": [{**asdict(row), "target": row.target.cpu(), "neutral": row.neutral.cpu()} for row in rows],
            "prefixes": runtime.prefix_cache,
        })
    captured = torch.load(rows_path, weights_only=True, map_location=device)
    runtime.prefix_cache = captured["prefixes"]
    rows = [PreparedAudioResponse(AudioRow(**row["row"]), row["target"], row["neutral"])
            for row in captured["rows"]]
    results = []
    for label, absolute, options in rates:
        arm = out / label
        arm.mkdir(exist_ok=True)
        settlement = arm / "settlement.json"
        if not settlement.exists():
            config = SessionConfig(
                BACKEND, f"three-bullets-empf8-{label}", model_dir=str(runtime.path), device=device,
                training_file=str(training_file.resolve()), training_sha256=file_sha(training_file),
                steps=steps, seed=seed, batch=2, rates=options, ema_decay=None, storage=storage,
            )
            save(arm / "config.json", config.as_dict())
            result = run_arm(arm, config, runtime, rows, controls, steps, limit, label, absolute)
            save(settlement, result)
        results.append(load_json(settlement))
        save(out / "settlements.json", results)
    eligible = [arm for arm in results if arm["status"] == "complete"]
    highest = max((arm["quarters"][-1]["score"] for arm in eligible), default=None)
    selected = [arm["arm"] for arm in eligible if arm["quarters"][-1]["score"] == highest]
    if len({arm["initial_network_sha256"] for arm in results}) != 1:
        raise RuntimeError("rate candidates did not share a matched adapter initialization")
    report = {"cohort": runtime.identity, "updates_per_arm": steps, "best_score": highest,
              "best_tested": selected, "arms": results,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    save(out / "result.json", report)
    marker({"best_score": highest, "best_tested": selected, "finished": len(results)})


def run_arm(arm, config, runtime, rows, controls, steps, limit, label, absolute):
    fulls = sorted(arm.glob("FULL[0-9][0-9][0-9].pt"))
    latest = fulls[-1] if fulls else None
    start = int(latest.stem[4:]) if latest else 0
    dispatch = arm / "last-dispatch.json"
    if dispatch.exists() and load_json(dispatch)["update"] > start:
        raise RuntimeError("last dispatched update has no committed FULL; reconcile it before restarting this arm")
    result = {"arm": label, "absolute_rates": absolute, "rate_options": config.options["rates"],
              "quarters": [], "status": "complete"}
    prior = sorted(arm.glob("quarter-[0-9][0-9][0-9].json"))
    for path in prior:
        result["quarters"].append(load_json(path))
    initial = arm / "RAW000.pt"
    failed = next((quarter for quarter in result["quarters"] if quarter["integrity"] != "pass"), None)
    if failed is not None:
        result["status"] = "stopped" if failed["integrity"] == "fail" else "unmeasured"
        result["initial_network_sha256"] = hash_network(read_inference(initial)["network"])
        return result
    with ExitStack() as owners:
        session = owners.enter_context(open_training(config, resume=latest, resident=runtime, prepared=rows))
        native = session.game.backend
        if start == 0:
            example = rows[0]
            if not torch.equal(native.sample(example, scale=1.), example.neutral):
                raise ValueError("initial Inkling slider changed the frozen native response")
            if latest is None:
                session.checkpoint(arm / "FULL000.pt")
            if not initial.exists():
                session.export_inference(initial)
        result["initial_network_sha256"] = hash_network(read_inference(initial)["network"])
        recorded = {quarter["updates"] for quarter in result["quarters"]}
        for update in range(max(1, start), steps + 1):
            full = arm / f"FULL{update:03d}.pt"
            if update != start:
                save(dispatch, {"update": update})
                session.step()
                session.checkpoint(full)
            if update % (steps // 4) or update in recorded:
                continue
            raw_path = arm / f"RAW{update:03d}.pt"
            raw = read_inference(raw_path) if raw_path.exists() else session.export_inference(raw_path)
            native = session.game.backend
            if hash_network(raw["network"]) != hash_network(native.network.state_dict()):
                raise RuntimeError("RAW tensor mismatch")
            cases = []
            for index, case in enumerate(controls if update == steps else controls[:1]):
                path = arm / f"native-{update:03d}-{index}.json"
                if not path.exists():
                    output = native.render(prompt=case["prompt"], max_new_tokens=limit, strength=1.)
                    save(path, {**output, "intensity": intensity(output["text"]) if output["text"] is not None else None, "integrity": integrity(output)})
                branch = load_json(path)
                cases.append({"prompt": case["prompt"], "off": case["off"], "on": branch,
                              "gain": (branch["intensity"] - case["off"]["intensity"]
                                       if branch["intensity"] is not None else None)})
            verdicts = {case["on"]["integrity"] for case in cases}
            status = "fail" if "fail" in verdicts else "unmeasured" if "unmeasured" in verdicts else "pass"
            quarter = {"updates": update, "cases": cases, "integrity": status,
                       "score": (mean(case["gain"] for case in cases)
                                 if update == steps and all(case["gain"] is not None for case in cases) else None)}
            result["quarters"].append(quarter)
            save(arm / f"quarter-{update:03d}.json", quarter)
            marker({"arm": label, "quarter": update, "integrity": status, "score": quarter["score"]})
            if status != "pass":
                result["status"] = "stopped" if status == "fail" else "unmeasured"
                break
            if update == steps // 2:
                # Restore the exact committed state without repeating its update or initializer.
                session.close()
                session = owners.enter_context(open_training(config, resume=full, resident=runtime, prepared=rows))
                if session.checkpoint()["sha256"] != read_packet(full)["sha256"]:
                    raise RuntimeError("restored FULL packet did not reproduce exactly")
                save(arm / "full-reload.json", {"updates": update, "sha256": read_packet(full)["sha256"]})
    if (arm / "full-reload.json").exists():
        result["exact_full_reloaded_at"] = load_json(arm / "full-reload.json")["updates"]
    return result


def hash_network(state):
    digest = sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="inkling2-sliders")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allocator-gib", type=float, default=26.)
    parser.add_argument("--storage", choices=STORAGE_PROFILES, default="resident")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("--out", required=True, type=Path)
    search.add_argument("--resume", action="store_true")
    render = commands.add_parser("render")
    render.add_argument("--raw", type=Path)
    render.add_argument("--prompt", required=True)
    render.add_argument("--strength", type=float, default=1.)
    render.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    budget = args.allocator_gib * 2**30
    total = torch.cuda.get_device_properties(device).total_memory
    if not 0 < budget < total:
        parser.error("allocator budget must be positive and smaller than total device memory")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    if args.command == "search":
        run_search(args.model_dir, args.device, args.out, storage=args.storage, resume=args.resume)
    elif args.raw is None:
        marker(Runtime(args.model_dir, args.device, args.storage).render(args.prompt, args.max_new_tokens))
    else:
        with open_inference(args.raw, backend=INFERENCE, options={
            "model_dir": str(args.model_dir.resolve()), "device": args.device,
        }) as session:
            marker(session.render(prompt=args.prompt, strength=args.strength, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
