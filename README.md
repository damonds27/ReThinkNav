# ReThinkNav: Zero-Shot Vision-and-Language Navigation with Open-Source LLMs via Contextual Reasoning and Loop Recovery

### Installation

```bash
conda create -n rethinknav python=3.8
conda activate rethinknav
```

#### Install Habitat and Dependencies
This project builds upon [Discrete-Continuous-VLN](https://github.com/YanyuanQiao/Open-Nav). Please follow the steps below:

1. You could follow the [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN) to install [`habitat-lab`](https://github.com/facebookresearch/habitat-lab) and [`habitat-sim`](https://github.com/facebookresearch/habitat-sim) by following the official Habitat installation guide.
2. We use Habitat [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) in our experiments, the same version used in [VLN-CE](https://github.com/jacobkrantz/VLN-CE) to ensure compatibility.
3. You may refer to **requirements.txt** or **environment.yml** in this repository for the exact package versions used.

Note: Our installation instructions are adapted from Open-Nav.

### Dataset

**OpenNav_R2R-CE_100**: [Download Here](https://drive.google.com/file/d/1SfrPWqCIiivwduCYPMe-Za1wOt4eU6G9/view?usp=sharing)


Please place the downloaded files under: 

> data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/


### Scenes: Matterport3D

We use **Matterport3D (MP3D)** scene reconstructions in this project.

You can obtain the dataset by following the instructions on the [official Matterport3D project page](https://niessner.github.io/Matterport/). The download script `download_mp.py` is required to fetch the scenes.

To download the scenes:

> Requires **Python 2.7**.

```bash
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Expected directory structure:
```
- data/
  - scene_datasets/
    - mp3d/
      - {scene_id}/
        - {scene_id}.glb
        - {scene_id}_semantic.ply
        - {scene_id}.house
        - {scene_id}.navmesh
```

### Trained Network Weights

We provide several pre-trained models to support waypoint prediction and visual encoding in the ReThinkNav framework.

#### Candidate Waypoint Predictor

Path: 
> waypoint_prediction/checkpoints/check_val_best_avg_wayscore

- [RGB-D (FoV 90) weights used in our paper](https://drive.google.com/file/d/16Vk3ummmyLvpQr16TzBL-iwZNlrELOdk/view?usp=sharing)
- [Depth-only (FoV 90, R2R-CE)](https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view?usp=sharing)

These models are used to predict candidate waypoints in the environment from visual input.


#### Visual Encoder (ResNet-50 for Depth)

Path:
> data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth

- Download link: [ResNet-50 pretrained on Gibson for DD-PPO](https://zenodo.org/record/6634113/files/gibson-2plus-resnet50.pth)

This ResNet-50 depth encoder is trained for PointGoal navigation on the Gibson dataset and used to extract visual features from depth images.

#### External VLM Models

Some external models are required for Scene Perception:

- [**SpatialBot**](https://github.com/BAAI-DCAI/SpatialBot)
- [**RAM (Recognize Anything Model)**](https://github.com/xinyu1205/recognize-anything)

Please refer to their respective repositories for model download and setup instructions. These models are used to get spatial visual information to support the reasoning process of open-source LLMs.

Clone or place them under the root directory:

Path: 
> recognize_anything/

> SpatialBot3B/


## Inference

To run inference with ReThinkNav, use the provided ollama.py:

```bash
python ollama.py
```

### Choosing the Language Model
Before running inference with ReThinkNav, make sure to deploy your LLM. We use QWEN3-32B as the main navigator.
You can specify which LLM to use via the --llm argument in the ollama.py
Open-source LLMs must be deployed separately and configured before use.


### Modifying Evaluation Episodes
To change the number of evaluation episodes, edit the following field in:
```
habitat_extensions/config/vlnce_task.yaml
```
Locate this section and modify EPISODES_TO_LOAD:

```yaml
DATASET:
  TYPE: VLN-CE-v1
  SPLIT: val_unseen
  DATA_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/OpenNav_R2R-CE_100_bertidx.json.gz
  SCENES_DIR: data/scene_datasets/
  EPISODES_TO_LOAD: 1  # Change this to run more episodes
```


## 🙏 Acknowledgements

We acknowledge that some parts of our code are adapted from existing open-source projects. Specifically, we reference the following repositories: **[Open-Nav](https://github.com/YanyuanQiao/Open-Nav)**, **[DiscussNav](https://github.com/LYX0501/DiscussNav)**, **[Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN)**, **[SpatialBot](https://github.com/BAAI-DCAI/SpatialBot)**, **[RAM](https://github.com/xinyu1205/recognize-anything)**


