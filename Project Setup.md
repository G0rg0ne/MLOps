**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**
Check the code examples provided here : 
https://github.com/graviraja/MLOps-Basics/tree/main/week_0_project_setup

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

Install the requirements:

```
pip install -r requirements.txt
```


### Dataset

 Datasets is a lightweight library providing **two** main features:
https://huggingface.co/datasets
- **one-line dataloaders for many public datasets**: one-liners to download and pre-process any of the major public datasets (image datasets, audio datasets, text datasets in 467 languages and dialects, etc.) provided on the [HuggingFace Datasets Hub](https://huggingface.co/datasets). With a simple command like `squad_dataset = load_dataset("squad")`, get any of these datasets ready to use in a dataloader for training/evaluating a ML model (Numpy/Pandas/PyTorch/TensorFlow/JAX),
- **efficient data pre-processing**: simple, fast and reproducible data pre-processing for the public datasets as well as your own local datasets in CSV, JSON, text, PNG, JPEG, WAV, MP3, Parquet, etc. With simple commands like `processed_dataset = dataset.map(process_example)`, efficiently prepare the dataset for inspection and ML model evaluation and training.
## Running

### Training


After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Inference

After training, update the model checkpoint path in the code and run

```
python inference.py
```

### Running notebooks
I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks.

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```


