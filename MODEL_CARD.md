# Strawberry runner segmentation weights

[Download the trained checkpoint](https://github.com/Bidhan-ghimire/plant-images-to-breeding-records/releases/download/v1.0.0/strawberry-runner-yolo11x.pt) from the [v1.0.0 release](https://github.com/Bidhan-ghimire/plant-images-to-breeding-records/releases/tag/v1.0.0). Save it as `models/best.pt` beside the notebook, or set `STRAWBERRY_RUNNER_CHECKPOINT` to its location. The notebook verifies the file before loading it.

| Property | Value |
| --- | --- |
| Architecture | Ultralytics YOLO11x segmentation |
| Class | Runner (class 0) |
| Training data | Combined ground imagery (GI), aerial imagery at 5 m (AI5), and 10 m (AI10) |
| Initialization | Pretrained yolo11x-seg.pt |
| Training seed | 42 |
| File size | 124,741,857 bytes |
| SHA-256 | 69b679d12f76ccfc99e692e8481815eb2ca2c2e798126257152f76f9908152fc |
| Inference used in this demonstration | Image size 320, confidence 0.25, IoU 0.7, maximum 300 detections |
| Verified inference environment | Python 3.10.19, Ultralytics 8.4.9, PyTorch 2.10.0+cu128, CUDA 12.8, NVIDIA GPU |

This is the unmodified checkpoint from the [training reproduction project](https://github.com/Bidhan-ghimire/strawberry-runner-repro), used for the illustrated demonstration. It is not described as the authors' original checkpoint. The [training arguments](model/training_args.yaml) retain training settings with a machine-specific output path replaced by a relative path.

The demonstration's 40 photographs were already part of model validation. The displayed counts demonstrate workflow execution, not accuracy on unseen plants. Detection counts describe runner instances visible in an image; the model has not been validated here for flowers, fruit, diseases, or all acquisition conditions.

Photographs must be obtained from [Dryad](https://doi.org/10.5061/dryad.bzkh189nw). This repository includes their filename/hash list, not the dataset or saved prediction run.

**If you use these trained weights, please cite Zhou et al. (2025), “Deep learning for strawberry runner detection integrating ground and aerial imaging,” [doi:10.1016/j.atech.2025.101290](https://doi.org/10.1016/j.atech.2025.101290), and the [Dryad dataset](https://doi.org/10.5061/dryad.bzkh189nw).**

Original demonstration code is MIT licensed. The weights and Ultralytics retain their [upstream AGPL-3.0 terms](THIRD_PARTY_NOTICES.md).
