# Pure GCA hidden-token analysis

## Run record

- Code commit: `37800aa` and `d6aa826`
- Device: physical GPU 2, selected while utilization was 0 percent
- Model: `models/lingbot-map.pt`
- Scenes: `loop`, `courthouse`, `university`
- Frames per scene: first 64 frames
- Window: 8 frames
- Scale frames: 4
- GCA layers: 4, 11, 17, 23
- Decoder training: 10 epochs, batch size 1

## What was extracted

The experiment registers a forward hook on `aggregator.global_blocks[layer]`.
The hook reads the output after the GCA block.
The experiment then removes camera, register, and scale tokens.
The saved tensor has shape `[frames, 777, 1024]`.

This is a pure GCA hidden representation.
It is different from the official aggregator output used by the original depth head.
The official output concatenates frame and global features and has 2048 channels.

## Layer results

Values are final validation values from each scene run.

| Scene | GCA layer | L1 | PSNR | Global SSIM |
| --- | ---: | ---: | ---: | ---: |
| loop | 4 | 0.06501 | 21.001 | 0.895 |
| loop | 11 | 0.07222 | 20.260 | 0.875 |
| loop | 17 | 0.09776 | 18.127 | 0.774 |
| loop | 23 | 0.10341 | 17.720 | 0.741 |
| courthouse | 4 | 0.08322 | 18.163 | 0.924 |
| courthouse | 11 | 0.09864 | 17.172 | 0.907 |
| courthouse | 17 | 0.10525 | 16.365 | 0.883 |
| courthouse | 23 | 0.10042 | 16.490 | 0.884 |
| university | 4 | 0.12177 | 15.853 | 0.699 |
| university | 11 | 0.12989 | 15.363 | 0.652 |
| university | 17 | 0.14850 | 14.460 | 0.571 |
| university | 23 | 0.15122 | 14.376 | 0.547 |

## Mean over scenes

| GCA layer | L1 | PSNR | Global SSIM |
| ---: | ---: | ---: | ---: |
| 4 | 0.09000 | 18.339 | 0.840 |
| 11 | 0.10025 | 17.598 | 0.811 |
| 17 | 0.11717 | 16.317 | 0.743 |
| 23 | 0.11835 | 16.195 | 0.724 |

## Layer 4 controls

| Scene | Condition | L1 | PSNR | Global SSIM |
| --- | --- | ---: | ---: | ---: |
| loop | GCA layer 4 | 0.06501 | 21.001 | 0.895 |
| loop | zero | 0.17873 | 13.887 | 0.057 |
| loop | shuffle | 0.17852 | 13.894 | 0.058 |
| courthouse | GCA layer 4 | 0.08322 | 18.163 | 0.924 |
| courthouse | zero | 0.27129 | 9.481 | 0.010 |
| courthouse | shuffle | 0.27154 | 9.538 | 0.010 |
| university | GCA layer 4 | 0.12177 | 15.853 | 0.699 |
| university | zero | 0.20984 | 12.271 | 0.009 |
| university | shuffle | 0.20870 | 12.274 | 0.009 |

## Interpretation

The GCA layer 4 representation has the best RGB reconstruction in all three scenes.
The reconstruction score decreases from layer 4 to the later layers.
The layer 4 real-token condition is much better than the zero and shuffle controls.
This supports that the pure GCA hidden tokens contain spatial visual information.

The result does not mean that layer 4 is the best layer for depth or pose.
The result does not prove that layer 4 K/V tokens can be compressed.
The next step is to test task outputs and then extract the actual K/V representation.
