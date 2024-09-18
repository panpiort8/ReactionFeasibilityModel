# ReactionFeasibilityModel

## Installation
To create the conda environment, run the following commands:

```bash
conda create --name rfm python=3.11.8 -y
conda activate rfm

# If using CUDA:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.2.1+cu118 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html

# If using CPU:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/torch-2.3/cpu/repo.html

pip install -e .

pip install pre-commit
pre-commit install
```

## Data preparation
To prepare the training datasets, run the following notebooks under `notebooks/created_dataset`:
- `create_positive.ipynb`. It removes the atom mapping from the raw USPTO dataset. We call this dataset "positive".
- `extract_forward_templates.ipynb`. It extracts the forward templates from the USPTO dataset.
- `create_negative_forward.ipynb`. It creates the negative reactions by applying the forward templates to reactants from the positive dataset.
- `create_negative_shuffle.ipynb`. It creates the negative reactions by shuffling the reactants from the positive dataset. A product from a positive dataset is assigned with a reactants coming from a similar (in terms of Tanimoto distance) reaction.
- `merge_files.ipynb`. It merges the positive and negative datasets into a single one.

## Training
The default configs logs to `experiments` directory and caches the molecules encodings in `processed_graphs`. If you want to store the data on other partition, you can create a symlink to the desired location.

To train the model, run the following command:
```bash
python -m scripts.train --cfg configs/rfm_train.gin
```

If you want to use other dataset, you should create a `configs/datasets/<your_dataset>.gin` file and  can replace "include 'configs/datasets/forward_with_shuffle.gin'" with "include 'configs/datasets/<your_dataset>.gin'" in the `configs/rfm_train.gin`.

## Checkpoints
Checkpoints can be downloaded from [here](https://ujchmura-my.sharepoint.com/:f:/g/personal/piotr_gainski_doctoral_uj_edu_pl/EhHNt1xE009Eh6YI6z8b9KUBT6-2C-lsOTX5I0EWLk4lnw?e=9cPzl5) and should be placed in the `checkpoints` directory.

## Usage
See `notebooks/example.ipynb`.
