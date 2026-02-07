# mini-VLA Project Context

## Project Overview

**mini-VLA** is a minimal, educational implementation of a Vision-Language-Action (VLA) model. It is designed to demonstrate how to fuse visual inputs, text instructions, and robot states to generate continuous control actions using a diffusion policy. The project prioritizes simplicity and readability (approx. 150 LOC for the core model) to serve as a learning resource and prototyping template.

### Core Architecture (`models/`)

The architecture follows a modular design:
*   **`vla_diffusion_policy.py`**: The main model class `VLADiffusionPolicy` that orchestrates the components.
*   **`encoders.py`**: Contains modality-specific encoders:
    *   `ImageEncoderTinyCNN`: Encodes images.
    *   `TextEncoderTinyGRU`: Encodes text instructions.
    *   `StateEncoderMLP`: Encodes robot state vectors.
*   **`fusion.py`**: `FusionMLP` combines the embeddings from the three encoders into a single context vector.
*   **`diffusion_head.py`**: `DiffusionPolicyHead` generates actions based on the fused context using a diffusion process.

### Environment & Simulation (`envs/`)

*   Uses **MetaWorld** (via MuJoCo) for simulation environments.
*   Specific environments are defined/wrapped in `envs/metaworld_env.py` and `envs/metaworld_mt1.py`.

## Building and Running

### Prerequisites

*   Python 3.10
*   Dependencies listed in `requirements.txt` (install via `pip install -r requirements.txt`)
*   MuJoCo (required for MetaWorld)

### Key Commands

Run these commands from the project root (`/home/hongrui/Codes/hongrui-vla`).

#### 1. Data Collection
Collect demonstration trajectories using an expert policy. This generates an `.npz` dataset.

```bash
python -m scripts.collect_data \
  --env-name push-v3 \
  --camera-name corner \
  --episodes 100 \
  --max-steps 100 \
  --output-path data/metaworld_push_bc.npz
```

#### 2. Training
Train the VLA diffusion policy on the collected dataset.

```bash
python -m scripts.train \
  --dataset-path data/metaworld_push_bc.npz \
  --epochs 50 \
  --batch-size 64 \
  --d-model 128 \
  --save-path checkpoints/model.pt \
  --device cpu  # or 'cuda' if available
```

#### 3. Testing / Evaluation
Run the trained model in the simulation environment and optionally save a video.

```bash
python -m scripts.test \
  --checkpoint checkpoints/model.pt \
  --env-name push-v3 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos
```

## Development Conventions

*   **Code Style:** The codebase emphasizes minimalism. New features should aim to maintain the "hackable" nature of the project.
*   **Data Format:** Datasets are stored as `.npz` files containing arrays for `images`, `states`, `actions`, and `text_ids`.
*   **Model Configuration:** Key hyperparameters like `d_model` and `diffusion_T` are configurable via command-line arguments in `train.py` and saved with the checkpoint.
*   **Testing:** There are no explicit unit tests. `scripts/test.py` serves as the primary functional test by running the policy in the simulation.
