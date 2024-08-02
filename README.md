# ReactionFeasibilityModel

## Installation
To install the environment with plenty of uneccessary packages (the output of `conda env export --no-build --name retrogfn`), run:
````
conda env create -f environment_full.yaml
````

To install the minial set of packages extracted with `pipreqs`, run:

````
conda create -n python=3.11.4 retrogfn pip
conda activate retrogfn
pip install -r requirements.txt
````

## Checkpoints
Checkpoints can be downloaded from [here](https://ujchmura-my.sharepoint.com/:f:/g/personal/piotr_gainski_doctoral_uj_edu_pl/EhHNt1xE009Eh6YI6z8b9KUBT6-2C-lsOTX5I0EWLk4lnw?e=9cPzl5) and should be placed in the `checkpoints` directory.

## Usage
See `notebooks/example.ipynb`.

## Data preparation
To prepare the training datasets, run the following notebooks under `notebooks/created_dataset`:
- `create_positive.ipynb`. It removes the atom mapping from the raw USPTO dataset. We call this dataset "positive".
- `extract_forward_templates.ipynb`. It extracts the forward templates from the USPTO dataset.
- `create_negative_forward.ipynb`. It creates the negative reactions by applying the forward templates to reactants from the positive dataset.
- `create_negative_shuffle.ipynb`. It creates the negative reactions by shuffling the reactants from the positive dataset. A product from a positive dataset is assigned with a reactants coming from a similar (in terms of Tanimoto distance) reaction.
- `merge_files.ipynb`. It merges the positive and negative datasets into a single one.
