# Evaluating Geostatic Stress History from CPT Data
A Hybrid Data-Driven Framework Integrating Unsupervised and Supervised Learning
Details of the methodology and case studies are in the following reference:
Gwak, D., & Ku, T. (2026). Evaluating Geostatic Stress History from CPT Data: A Hybrid Data-Driven Framework Integrating Unsupervised and Supervised Learning. Engineering Geology (Ref: ENGEO-D-25-03495R1).

Contact email: tsku@konkuk.ac.kr


### Files and Descriptions
01_preprocessing.py: Python script for data ingestion, feature engineering (log-transformation, qt_ratio, etc.), and standardization following Section 3.2.

02_clustering_optimization.py: Code for optimizing the hybrid UMAP-GMM architecture using Optuna to identify distinct geological units as described in Appendix A.4.

03_supervised_regression.py: Script for performing Bayesian hyperparameter search and training the final Random Forest model using parameters specified in Appendix A.5.

CPT DB.xlsx: A representative sample dataset containing CPT parameters and corresponding OCR labels to ensure the entire workflow is executable. (Note: The full raw dataset is confidential).
