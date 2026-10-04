# Experiment 1 formal results

## Run record

- Code commits: `e597bb6`, `445adb0`, `7386914`, `bb698b1`, `82d8f54`, `24bdadb`
- Device: physical GPU 3
- Model: `models/lingbot-map.pt`
- Scenes: `loop`, `courthouse`, `university`
- Frames per scene: first 64 frames
- Window: 8 frames
- Scale frames: 4
- Decoder training: 10 epochs, batch size 1
- Token dtype on disk: float16
- Model weights: frozen

## Conditions

- `local`: frame-block tokens from layer 23.
- `local_zero`: local tokens replaced with zeros.
- `local_shuffle`: patch positions shuffled within each frame.
- `local_frame_shuffle`: complete frame token sets exchanged across frames.
- `global0`: first official aggregator output.
- `global3`: last official aggregator output.

## Validation results

The values below are the final validation values from each scene run.
`ssim_global` is a lightweight global SSIM estimate. It is not a replacement for a standard windowed SSIM implementation.

| Scene | Condition | L1 | PSNR | Global SSIM |
| --- | --- | ---: | ---: | ---: |
| loop | local | 0.1015 | 17.890 | 0.762 |
| loop | local_zero | 0.1787 | 13.887 | 0.057 |
| loop | local_shuffle | 0.1775 | 13.921 | 0.066 |
| loop | local_frame_shuffle | 0.1306 | 15.754 | 0.573 |
| loop | global0 | 0.0657 | 21.025 | 0.898 |
| loop | global3 | 0.1018 | 17.844 | 0.782 |
| courthouse | local | 0.0999 | 16.602 | 0.886 |
| courthouse | local_zero | 0.2713 | 9.481 | 0.010 |
| courthouse | local_shuffle | 0.2716 | 9.536 | 0.010 |
| courthouse | local_frame_shuffle | 0.1655 | 12.773 | 0.673 |
| courthouse | global0 | 0.0772 | 18.539 | 0.930 |
| courthouse | global3 | 0.1087 | 16.205 | 0.877 |
| university | local | 0.1641 | 13.836 | 0.545 |
| university | local_zero | 0.2098 | 12.271 | 0.009 |
| university | local_shuffle | 0.2142 | 12.124 | 0.009 |
| university | local_frame_shuffle | 0.1985 | 12.373 | 0.188 |
| university | global0 | 0.1184 | 16.038 | 0.719 |
| university | global3 | 0.1516 | 14.334 | 0.532 |

## Mean over the three scenes

| Condition | L1 | PSNR | Global SSIM |
| --- | ---: | ---: | ---: |
| local | 0.1218 | 16.109 | 0.731 |
| local_zero | 0.2200 | 11.880 | 0.025 |
| local_shuffle | 0.2211 | 11.861 | 0.028 |
| local_frame_shuffle | 0.1649 | 13.633 | 0.478 |
| global0 | 0.0871 | 18.534 | 0.849 |
| global3 | 0.1207 | 16.128 | 0.730 |

## Interpretation

The `local` condition is better than the zero control on all three scenes.
The `local` condition is better than the spatial shuffle control on all three scenes.
These results support the claim that local image tokens contain spatial visual information.

The frame shuffle control remains above the zero control.
This result suggests that adjacent frames contain redundant visual information.

The `global0` condition is the strongest RGB reconstruction condition in this run.
The `local` and `global3` conditions have similar mean PSNR.

This run does not prove that local tokens preserve depth or pose information.
This run does not prove inference acceleration.
The next test must measure depth and pose outputs with the original heads.
