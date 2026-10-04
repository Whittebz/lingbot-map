# Experiment 1: Image-token reconstruction probe

This directory contains an additive implementation of Experiment 1.
It does not modify files in `lingbot_map/` or `demo.py`.

## Research question

Can image tokens from one local window reconstruct the preprocessed input image?

The experiment uses a frozen LingBot-Map model and a separately trained RGB decoder.
The RGB decoder is not used for inference timing.

## Conditions

- `local`: capture the output of one `frame_blocks[layer]` before global attention.
- `global_0`, `global_1`, `global_2`, `global_3`: use the four token outputs returned by the official aggregator.
- `zero`: replace patch tokens with zeros. This is a negative control.
- `shuffle`: shuffle patch positions within each frame. This is a spatial control.
- `frame_shuffle`: exchange complete frame token sets. This is a temporal control.

Start with `local` and `global_3`. Add `zero` and `shuffle` after the pipeline passes.

## Remote workflow

Run commands in `/home/dengbaichuan/research/lingbot_map` on `kt`.

1. Extract token files with `stage=extract`.
2. Train the RGB decoder with `stage=train`.
3. Evaluate PSNR, SSIM, and temporal consistency with `stage=eval`.
4. Save the command, configuration, GPU index, and commit in the run directory.

Example smoke test on the idle GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python experiments/exp1_token_reconstruction.py extract \
  --model_path models/lingbot-map.pt \
  --image_folder example/loop \
  --output_dir experiments/runs/loop_local_smoke \
  --source local --layer 23 --first_k 8 --window_size 8 \
  --num_scale_frames 4 --use_sdpa --device cuda:0
```

`CUDA_VISIBLE_DEVICES=3` maps the physical GPU 3 to `cuda:0` inside the process.

## Baseline rule

The input preprocessing, model checkpoint, window size, frame list, decoder, and optimizer must stay fixed across conditions.

The script writes a `manifest.json` file for each extraction run.
