# 3DG-VLN

Official implementation of **3DG-VLN**, a high-resolution dual-view UAV vision-language navigation framework with online 3D directional guidance for precise *see-and-reach* target navigation.

This repository releases the dataset, source code, and pretrained weights for our 3DG-VLN work.

## Overview

3DG-VLN focuses on **UAV vision-language navigation in target-visible scenarios**, where an aerial agent is required to reach a visible target according to a concise natural-language instruction. Different from long-range search-oriented UAV-VLN settings, our task emphasizes fine-grained visual grounding, local 3D spatial reasoning, and precise waypoint-level motion prediction.

Given high-resolution egocentric observations and a coarse 3D directional cue, 3DG-VLN predicts continuous 3D waypoints to guide the UAV toward the target. During inference, an online 3D direction updating strategy is adopted to dynamically refine the target direction from current observations, helping the UAV maintain spatial alignment with the target during closed-loop navigation.

## Highlights

* **UAV-VLN-FOV Benchmark**
  We construct a high-resolution benchmark for precise target-visible UAV navigation.

* **High-Resolution Dual-View Observations**
  The agent uses front-view and downward-view egocentric images to support fine-grained visual grounding and local navigation.

* **3D Directional Guidance**
  Coarse 3D directional cues are introduced to provide spatial priors for target-oriented waypoint prediction.

* **Online Direction Updating**
  During inference, the relative target direction is dynamically updated from current observations to reduce spatial drift.

* **Released Resources**
  We release the dataset, code, and pretrained weights to support reproducible research.

## Dataset

We release the **UAV-VLN-FOV** benchmark constructed for the 3DG-VLN task.

The benchmark contains:

* 2,717 trajectories
* Concise high-level instruction
* High-resolution dual-view images
* Continuous 3D waypoint annotations
* Evaluation splits for seen, unseen-object, and unseen-scene testing

The dataset is designed to evaluate precise *see-and-reach* navigation, where the target is visible from the initial human viewpoint and the UAV is expected to approach the target accurately.

### Download

The dataset will be released at:

```text
[Dataset Download Link]
```

After downloading, please organize the dataset as follows:

```text
3DG-VLN/
├── data/
│   └── UAV-VLN-FOV/
│       ├── train/
│       ├── test/
│       ├── test_unseen_object/
│       ├── test_unseen_scene/
│       └── meta/
```

## Pretrained Weights

We provide pretrained weights for the 3DG-VLN model.

```text
[Pretrained Weight Download Link]
```

Please place the downloaded weights under:

```text
3DG-VLN/
└── checkpoints/
    └── 3dg-vln/
```

## Repository Structure

```text
3DG-VLN/
├── airsim_plugin/              # AirSim client and simulator interaction tools
├── scripts/                    # Running scripts
├── src/                        # Main source code
│   ├── common/                 # Common configurations and arguments
│   ├── model_wrapper/          # Model wrappers and visual grounding modules
│   └── vlnce_src/              # Training and evaluation code
├── utils/                      # Utility scripts and evaluation metrics
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

We recommend using Conda to create the environment.

```bash
conda create -n 3dg-vln python=3.10 -y
conda activate 3dg-vln
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use AirSim-based evaluation, please make sure that Unreal Engine, AirSim, and the corresponding simulator environment are correctly installed.

## Evaluation

To evaluate the model on the test split, run:

```bash
bash scripts/3DG-VLN_eval.sh
```

To compute navigation metrics, run:

```bash
bash scripts/metric.sh
```

The evaluation reports the following metrics:

* **SR**: Success Rate
* **OSR**: Oracle Success Rate
* **SPL**: Success weighted by Path Length
* **NE**: Navigation Error

## License

This project is released under the **Apache License 2.0**.

Please see the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This repository builds upon and benefits from several excellent open-source projects. We sincerely thank the authors and contributors of these repositories.

The implementation is partly inspired by or adapted from the following projects:

* [TravelUAV](https://github.com/xxx/TravelUAV)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* [AerialVLN](https://github.com/AirVLN/AirVLN)

If you use any third-party components, models, or datasets, please also follow their original licenses.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{xxx2026_3dgvln,
  title={3DG-VLN: High-Resolution UAV Vision-Language Navigation with Online 3D Directional Guidance},
  author={XXX and XXX and XXX},
  journal={XXX},
  year={2026}
}
```

