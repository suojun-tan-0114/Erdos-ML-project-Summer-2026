# Deep Learning for Fraud Detection

This repository contains my contributions to a larger **Erdős Institute Deep Learning Boot Camp (Summer 2026)** group project on fraud detection using the IEEE-CIS Fraud Detection dataset.

**Full group project:**  
https://github.com/satyu2004/deep-learning-for-fraud-detection

## Project Overview

Fraud detection is a highly imbalanced classification problem in which individual transaction features may not capture all of the useful signal. Transactions can also be related through shared cards, addresses, email domains, devices, and other identifiers.

The group project therefore explored both strong tabular baselines and graph-based deep learning approaches, including:

- LightGBM
- Multi-layer perceptrons
- GraphSAGE and other graph neural networks
- Hybrid LightGBM + graph models
- Residual graph/topology correction approaches

The main goal was to determine whether relational information between transactions could improve fraud detection beyond a strong tabular baseline.

## My Contributions

My work focused primarily on a **Joint LightGBM–GraphSAGE model** that combines tabular predictions with graph-based relational information.

### LightGBM Baseline

A LightGBM classifier was used as a strong tabular baseline for the IEEE-CIS transaction data. The baseline captures patterns in transaction-level features such as transaction amount, card information, addresses, and engineered variables.

### Temporal, Relation-Aware GraphSAGE

I worked on a GraphSAGE model in which transactions are represented as nodes and edges are created from shared identifiers and relationships between transactions.

Examples of graph relations explored include:

- shared card information
- card + address relationships
- card + email relationships
- address + email relationships
- device + address relationships
- engineered transaction identities

To reduce temporal leakage, graph neighborhoods were constructed using historical information, with validation transactions restricted to earlier neighbors where appropriate.

### Joint LightGBM + GraphSAGE Model

The joint model combines the outputs of the tabular LightGBM model and the GraphSAGE model:

\[
z = \alpha z_{LGBM} + (1-\alpha) z_{GNN}
\]

where:

- \(z_{LGBM}\) is the LightGBM logit
- \(z_{GNN}\) is the GraphSAGE logit
- \(\alpha\) controls the contribution of each model

The LightGBM model is trained first and then frozen, while the GraphSAGE model learns complementary information from transaction features and graph neighborhoods.

This design allows the final prediction to use both **tabular feature patterns** and **relational information between connected transactions**.

## Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle: https://www.kaggle.com/competitions/ieee-fraud-detection

The dataset contains transaction-level and identity-level information. The two main files are `train_transaction.csv` and `train_identity.csv` which are joined using `TransactionID`.

Because the dataset is too large to store directly in the repository, it should be downloaded separately from Kaggle before running the notebooks.

## Evaluation

Because fraud is rare, model performance was evaluated using metrics that are more informative than accuracy alone, especially **ROC-AUC** as well as **PR-AUC**.

The experiments compare standalone tabular and graph models with hybrid approaches to determine whether graph structure adds useful predictive information. For the complete set of group experiments and reported results, see the full team repository: https://github.com/satyu2004/deep-learning-for-fraud-detection

## Group Project

The full team repository includes additional approaches beyond the work emphasized here, including:

- LightGBM baseline
- MLP baseline
- Heterogeneous GraphSAGE
- Heterogeneous GAT
- LightGBM with structural graph features and GNN embeddings
- Joint LightGBM + relation-aware GraphSAGE
- Residual topology correction using TopER, hypergraph methods, and a sheaf neural network

### Team Members

- Sathya Rengaswami
- Wenwen Li
- Lang Song
- Suo-Jun Tan

## Technologies

The project uses tools including:

- Python
- Pandas / NumPy
- LightGBM
- PyTorch
- PyTorch Geometric
- Scikit-learn
- Jupyter Notebook

## Notes

This repository is intended as a portfolio of my contributions to the group project. For the complete codebase, all modeling approaches, presentation materials, and team results, please refer to the full group repository linked above.
