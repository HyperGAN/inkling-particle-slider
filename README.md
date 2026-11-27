# Inkling 2 Mini Particle Slider

A strength-controlled Markdown bullet slider for the fixed three-pass
[Inkling 2 Mini text model](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/6e7c8248f75265bdd4fbab5273304243b5d8c2cb).
The adapter adds a learned rank-eight, particle-routed increment to the native
dash logit. The pretrained text tower runs on CUDA with the released packed
NVFP4 and MXFP8 expert weights. Strength zero bypasses the adapter.

**Prompt** · Name three kinds of fabric. Use one word per example.

**OFF** · Cotton, silk, wool.

**ON**

- Cotton
- Silk
- Wool

The released checkpoint produced three bullets on all three new prompts
from a fresh wheel install. [More examples and measurements](docs/RESULTS.md)

## Run the verified checkpoint

Use Python 3.12 on Linux with a CUDA GPU. The fully resident A6000
qualification peaked at 17.78 GiB of CUDA allocations. The render command
defaults to a 24 GiB allocator cap. [Memory profiles](MEMORY.md)

The [v0.2.0 release](https://github.com/HyperGAN/inkling-particle-slider/releases/tag/v0.2.0)
includes the package, a Particle Sliders Core 0.14.4 companion wheel, and
`RAW064.pt`. Use the release wheels below to satisfy the checkpoint's
runtime source and model identity checks. The companion uses the byte-identical
reviewed runtime code with local paths removed from one provenance record.
The package pins the required
[Transformers revision](https://github.com/huggingface/transformers/commit/6173a2f679ff4ed393afc499aefc569da7e68bc1).

```bash
python -m pip install \
  https://github.com/HyperGAN/inkling-particle-slider/releases/download/v0.2.0/particle_sliders_core-0.14.4-py3-none-any.whl \
  https://github.com/HyperGAN/inkling-particle-slider/releases/download/v0.2.0/inkling2_mini_particle_sliders-0.2.0-py3-none-any.whl

hf download thinkingmachines/Inkling-2-Mini \
  --revision 6e7c8248f75265bdd4fbab5273304243b5d8c2cb \
  --local-dir ./model --exclude mtp.safetensors

curl -fL -o RAW064.pt \
  https://github.com/HyperGAN/inkling-particle-slider/releases/download/v0.2.0/RAW064.pt

inkling2-bullet-render --model-dir ./model --raw ./RAW064.pt \
  --prompt 'Name three kinds of fabric. Use one word per example.' \
  --strength 1 --max-new-tokens 32
```

Use `--strength 0` for the matched baseline. The command prints JSON with
the generated text, token IDs, and stopping status.

## Train a slider from a phrase

The separate `vibe` package runs the shared text version of Anima’s `--vibe` workflow on Inkling’s frozen three-pass CUDA model. Start in a separate Python 3.12 environment, install a CUDA build of Torch 2.13, and install the product from this checkout.

```sh
python -m pip install ./vibe
export CUDA_VISIBLE_DEVICES=0
inkling-vibe --snapshot ./model --vibe "short rhyming couplets" --out ./runs/couplets
```

Training uses the NanoGPT Jev judge. Supply `NANOGPT_API_KEY` to the training process. The judge receives the phrase, teaching prompts and generated replies. The collector tries 24 contexts, scores matched plain and cued replies, and retains successful pairs. The shared trainer freezes same-token response targets and fits RAW routed adapters to the final attention block. It compares quarter checkpoints on the same prompts at strength one, then runs Anima’s adaptive constant-strength search on the winning RAW. The selected strength must improve intact held-out replies and the reloaded export must replay the saved tokens. Use a new output directory for each run.

The first candidate uses 64 updates. Inspect `result.json` for the held-out gain of the style you trained. Generate a matched comparison with the exported `slider.safetensors`:

```sh
inkling-vibe --snapshot ./model --prompt "Welcome a neighbor to the garden." --adapter ./runs/couplets/slider.safetensors --strength 0
inkling-vibe --snapshot ./model --prompt "Welcome a neighbor to the garden." --adapter ./runs/couplets/slider.safetensors
```

## Train and evaluate

For a fresh run, install this checkout with `python -m pip install .` in a
separate environment. It uses ParticleGAN 0.11.1 and Particle Sliders Core
0.15.2 with the stock EMPF recipe and explicit continuous penalties. Download
the pinned model as above, then run the
[bullet-margin experiment](experiments/bullet-margin/run_search.py) from
the repository root. It captures native teacher responses and retains
matched outputs, RAW exports, and resumable FULL checkpoints.

```bash
python experiments/bullet-margin/run_search.py ./model ./run-bullet-margin
```

Pass `--resume` with the original paths and original software environment
to resume the same run. FULL checkpoints from v0.2.0 require its release
environment. The original prompt-end hidden-state experiment remains available
through `inkling2-sliders`. Its eight-update rates all scored zero formatting
gain. [Diagnosis](docs/RESULTS.md#why-the-first-slider-missed)

[MIT license](LICENSE)
