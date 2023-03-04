## HBTBD

This repository provides a reference implementation of HBTBD as described in the paper:
> HBTBD:Heterogenous Bitcoin Transaction Behavior Dataset for Anti-Money Laundering

### Dependencies

Recent versions of the following packages for Python 3 are required:
* PyTorch 1.2.0
* DGL 0.3.1
* NetworkX 2.3
* scikit-learn 0.21.3
* NumPy 1.17.2
* SciPy 1.3.1

Dependencies for the preprocessing code are not listed here.

### Datasets

The preprocessed datasets are available at:
* HBTBD - [Kaggle]https://www.kaggle.com/datasets/songjialin/hbtbd-for-aml)

### Usage

1. Create `checkpoint/` and `data/preprocessed` directories
2. Extract the zip file downloaded from the section above to `data/preprocessed`
2. Execute one of the following three commands from the project home directory:
    * `python run_Elliptic.py`
