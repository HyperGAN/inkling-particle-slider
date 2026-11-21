"""Fixed-profile routed bullet-margin LR screen with RAW finished-text selection."""
import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import median

import torch
from particlegan import get_recipe
from particle_sliders.inference import read_inference
from particle_sliders.lifecycle import SessionConfig, atomic_json, atomic_save, read_packet

from inkling2_sliders.bullet import BACKEND, MarginRow, open_training, prepare_rows
from inkling2_sliders.cli import hash_network, rate_options
from inkling2_sliders.measurement import intensity, integrity, qualify_integrity
from inkling2_sliders.native import Runtime, file_sha

SEED = 17361
STEPS = 64
BATCH = 6


def read(path):
    return json.loads(path.read_text())


def observed(native, prompt, strength, limit):
    output = native.render(prompt=prompt, strength=strength, max_new_tokens=limit)
    verdict = integrity(output)
    text = output["text"]
    if text and output["stop"] == "length" and len(text.strip()) >= 8 and set(text.split()[0]) == {"-"}:
        if set("".join(text.split())) == {"-"}:
            verdict = "fail"
    return {**output, "integrity": verdict, "intensity": intensity(text) if text is not None else None}


def freeze_panel(inputs):
    original = read(inputs / "baseline-panel.json")
    examples = (
        ("Carrot", "Spinach", "Peas"), ("Cow", "Horse", "Duck"),
        ("Shirt", "Pants", "Socks"), ("Chess", "Checkers", "Cards"),
        ("Rake", "Shovel", "Hoe"), ("Seeds", "Worms", "Berries"),
        ("Wood", "Steel", "Plastic"), ("Hammer", "Wrench", "Pliers"),
    )
    return {
        "training": [dict(prompt=p, items=list(words)) for p, words in zip(original["training"], examples, strict=True)],
        "positive_prefix": original["positive_prefix"],
        "matched_cases": [dict(index=i, strength=s) for i, s in
                          ((0, 1.), (1, 1.), (2, 1.), (3, 1.), (0, .5), (1, .5))],
        "max_new_tokens": original["max_new_tokens"], "steps": STEPS, "batch": BATCH, "seed": SEED,
        "metric": "Equal-case mean distinct unnumbered bullet lines ON minus OFF, each intensity capped at 3/3",
        "initial_rate_neighborhood": "stock plus half and double each role independently",
    }


def run_arm(runtime, rows, coeff, run, panel, control, label, absolute, options):
    arm = run / label
    arm.mkdir(exist_ok=True)
    training_file = run / "panel.json"
    cfg = SessionConfig(
        BACKEND, "inkling2-bullet-margin64-" + label, model_dir=str(runtime.path), device=str(runtime.device),
        training_file=str(training_file), training_sha256=file_sha(training_file),
        steps=STEPS, seed=SEED, batch=BATCH, rates=options, ema_decay=None,
        storage=runtime.storage, response_scale=coeff,
    )
    atomic_json(arm / "config.json", cfg.as_dict())
    fulls = sorted(arm.glob("FULL???.pt"))
    latest = fulls[-1] if fulls else None
    start = int(latest.stem[4:]) if latest else 0
    dispatch = arm / "last-dispatch.json"
    if dispatch.exists() and read(dispatch)["update"] > start:
        raise RuntimeError("last update has no durable FULL")
    quarters = {q["updates"]: q for q in (read(path) for path in sorted(arm.glob("quarter-*.json")))}
    failed = next((q for q in quarters.values() if q["integrity"] != "pass"), None)
    initial = arm / "RAW000.pt"
    state = "complete"
    if failed is not None:
        state = "stopped" if failed["integrity"] == "fail" else "unmeasured"
    else:
        session = open_training(cfg, resume=latest, resident=runtime, prepared=rows)
        try:
            if start == 0:
                assert torch.equal(session.game.backend.sample(rows[0]), rows[0].neutral)
                if latest is None:
                    session.checkpoint(arm / "FULL000.pt")
                if not initial.exists():
                    session.export_inference(initial)
            for step in range(max(1, start), STEPS + 1):
                full = arm / f"FULL{step:03d}.pt"
                if step > start:
                    atomic_json(dispatch, {"update": step})
                    session.step()
                    session.checkpoint(full)
                if step % (STEPS // 4) or step in quarters:
                    continue
                raw_path = arm / f"RAW{step:03d}.pt"
                raw = read_inference(raw_path) if raw_path.exists() else session.export_inference(raw_path)
                native = session.game.backend
                assert hash_network(raw["network"]) == hash_network(native.network.state_dict())
                cases = panel["matched_cases"] if step == STEPS else panel["matched_cases"][:1]
                results = []
                for index, position in enumerate(cases):
                    c = control[position["index"]]
                    path = arm / f"native-{step:03d}-{index}.json"
                    if not path.exists():
                        atomic_json(path, observed(native, c["prompt"], position["strength"], panel["max_new_tokens"]))
                    on = read(path)
                    off = c["off"]
                    gain = (Fraction(round(on["intensity"] * 3) - round(off["intensity"] * 3), 3)
                            if on["intensity"] is not None else None)
                    results.append({**position, "prompt": c["prompt"], "off": off, "on": on,
                                    "gain": str(gain) if gain is not None else None})
                    if on["integrity"] != "pass":
                        break
                integrity_result = next(
                    (x["on"]["integrity"] for x in results if x["on"]["integrity"] != "pass"), "pass"
                )
                score = (sum(Fraction(case["gain"]) for case in results) / len(results)
                         if integrity_result == "pass" and step == STEPS else None)
                quarter = {"updates": step, "integrity": integrity_result, "cases": results,
                           "score": str(score) if score is not None else None}
                quarters[step] = quarter
                atomic_json(arm / f"quarter-{step:03d}.json", quarter)
                print(json.dumps({"arm": label, "updates": step, "integrity": integrity_result,
                                  "score": quarter["score"]}), flush=True)
                if integrity_result != "pass":
                    state = "stopped" if integrity_result == "fail" else "unmeasured"
                    break
                if step == STEPS // 2:
                    session.close()
                    session = open_training(cfg, resume=full, resident=runtime, prepared=rows)
                    assert session.checkpoint()["sha256"] == read_packet(full)["sha256"]
                    atomic_json(arm / "full-replay.json", {"updates": step, "sha256": read_packet(full)["sha256"]})
        finally:
            session.close()
    result = {"arm": label, "status": state, "absolute_rates": absolute,
              "initial": hash_network(read_inference(initial)["network"]), "quarters": list(quarters.values())}
    atomic_json(arm / "settlement.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    model_dir, run = args.model_dir.resolve(), args.run.resolve()
    baseline = Path(__file__).resolve().parent
    run.mkdir(exist_ok=args.resume)
    panel = freeze_panel(baseline)
    if not args.resume:
        atomic_json(run / "panel.json", panel)
    elif panel != read(run / "panel.json"):
        raise ValueError("frozen panel differs")
    torch.set_num_threads(4)
    dev = "cuda:0"
    torch.cuda.set_per_process_memory_fraction(24 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    runtime = Runtime(model_dir, dev, "resident")
    print("Resident pinned Inkling text model loaded", flush=True)
    controls = read(baseline / "baseline-controls.json")["matched"]
    assert all(r["off"]["integrity"] == r["explicit_positive"]["integrity"] == "pass" for r in controls)
    assert len(qualify_integrity()) == 9
    targets = run / "targets.pt"
    if not targets.exists():
        rows = prepare_rows(runtime, panel)
        atomic_save(targets, [asdict(row) for row in rows])
    rows = [MarginRow(**entry) for entry in torch.load(targets, weights_only=True, map_location=dev)]
    edits = [row for row in rows if row.edit]
    first = [row for row in edits if row.position == 0]
    if len(first) != 8 or sum(row.target.item() > 0 for row in first) < 6:
        raise ValueError("explicit prompt did not qualify the first-bullet teacher margin")
    coeff = max(1., median(abs((row.target - row.neutral).item()) for row in edits))
    rates = list(rate_options(get_recipe("empf")))
    plan = {
        "model": runtime.identity, "panel_sha256": file_sha(run / "panel.json"),
        "script_sha256": file_sha(__file__), "targets_sha256": file_sha(targets),
        "response_scale": coeff, "controls_source": str(baseline / "baseline-controls.json"),
        "controls_sha256": file_sha(baseline / "baseline-controls.json"),
        "positive_first_token_margins": [row.target.item() for row in first],
        "neutral_first_token_margins": [row.neutral.item() for row in first],
        "edits": len(edits), "preservation_rows": len(rows) - len(edits),
        "rate_options": rates, "stock_recipe": asdict(get_recipe("empf")),
    }
    if (run / "plan.json").exists() and json.loads(json.dumps(plan)) != read(run / "plan.json"):
        raise ValueError("frozen plan differs")
    atomic_json(run / "plan.json", plan)
    print(json.dumps({"matched_targets": len(rows), "response_scale": coeff}), flush=True)
    results = []
    for label, absolute, options in rates:
        settlement = run / label / "settlement.json"
        result = (read(settlement) if settlement.exists()
                  else run_arm(runtime, rows, coeff, run, panel, controls, label, absolute, options))
        results.append(result)
    assert len({a["initial"] for a in results}) == 1
    scored = [(a["arm"], Fraction(a["quarters"][-1]["score"])) for a in results if a["status"] == "complete"]
    top = max((score for _, score in scored), default=None)
    winners = [name for name, score in scored if score == top]
    atomic_json(run / "result.json", {"arms": results, "best_score": str(top) if top is not None else None,
                                    "best_tested": winners, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
    print(json.dumps({"best_tested": winners, "best_score": str(top)}), flush=True)


if __name__ == "__main__":
    main()
