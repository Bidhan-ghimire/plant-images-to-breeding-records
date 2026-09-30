# Software, model, and data licenses

Original demonstration code by Bidhan Ghimire is offered under the [MIT License](LICENSE). You may reuse that code under its terms. The citation request below is scholarly attribution, not an additional restriction on the MIT license.

The trained YOLO11x segmentation checkpoint uses Ultralytics. Ultralytics 8.4.9 is licensed under AGPL-3.0, and [Ultralytics states that trained models use AGPL-3.0 by default](https://www.ultralytics.com/license). The weights are distributed with those upstream terms, not as MIT-only weights. A copy is provided in [licenses/AGPL-3.0.txt](licenses/AGPL-3.0.txt). Using the complete workflow with Ultralytics does not remove its applicable source-sharing obligations; see the upstream terms for the combined application.

Corresponding Ultralytics source: [version 8.4.9](https://github.com/ultralytics/ultralytics/tree/v8.4.9). The model's training workflow and configuration are available in [strawberry-runner-repro](https://github.com/Bidhan-ghimire/strawberry-runner-repro); its checkpoint identity and training settings are described in [MODEL_CARD.md](MODEL_CARD.md).

The source image dataset is available from [Dryad](https://doi.org/10.5061/dryad.bzkh189nw) under CC0. Dataset photographs are not bundled here; illustrative figures retain their source attribution.

**Please cite Zhou et al. (2025) and the Dryad dataset when using the trained model.** Full citations appear in the README and notebook. The released weights are from the reproduction project, not a claim to be the authors' original trained model.
