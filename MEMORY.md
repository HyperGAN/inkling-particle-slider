# Inkling 2 Mini text memory

The pinned source stores 18,889,358,496 bytes (17.59 GiB) of text weights
including packed experts. The resident CUDA profile needs that storage plus
the CUDA context, BF16 decode workspace, attention cache and training tensors.
The historical training launcher's 26 GiB allocator default and the
released renderer's 24 GiB allocator default are caps. Allow free memory
for other processes on a shared GPU.

The `mmap` profile retains packed expert layers 6, 7, 9, 11, 12, 14, 17 and 23
on CUDA. `mmap-loop` retains loops and downstream layers 6 through 23. The
other layers' original packed codes are mapped from the local checkpoint and
only the routed experts are transferred for CUDA decoding. Dense text weights
and all floating-point training and inference remain on GPU. The shared-host `mmap-loop` attempt uses a 17 GiB allocator cap and checks
physical GPU and host floors.
The original weights occupy about 19.53 GB on disk.

This integration bounds prompt length to 256 tokens and generation to 128 new
tokens. The LR panel uses at most 16 new tokens. The fully resident A6000 search measured 17.80 GiB peak CUDA allocations,
17.89 GiB peak reserved by CUDA, and 18.24 GiB sampled peak process usage.

The routed bullet-margin screen used the fully resident profile and peaked at
17.78 GiB of CUDA allocations across seven arms. [Working slider and measurements](docs/RESULTS.md).
