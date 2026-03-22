import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def prepare_geotechnical_data(features):
    """
    Performs data ingestion and feature engineering for CPT-based OCR estimation.
    This module implements the preprocessing protocols defined in Section 3.2.
    
    Args:
        features (list): List of input parameters for model training.
        
    Returns:
        X_scaled (ndarray): Standardized feature matrix.
        df (DataFrame): Preprocessed dataframe containing engineered parameters.
    """
    # Load raw dataset directly from the provided Excel file
    # Ensure 'CPT DB.xlsx' is located in the same directory as this script.
    df = pd.read_excel("CPT DB.xlsx", sheet_name="clean")
    
    # Filter core geotechnical parameters required for the hybrid framework
    core_params = ['Site', 'OCR', 'depth', 'qt', 'u0', 'u2', 'σvo', 'Qt', 'Bq']
    df = df[core_params].copy()
    
    # Exclude samples with missing target labels (OCR) to ensure training integrity
    df = df[df['OCR'].notna()].reset_index(drop=True)

    # --- Feature Engineering Stage (Section 3.2) ---
    # excess_pwp: Excess pore water pressure (u2 - u0)
    df['excess_pwp'] = df['u2'] - df['u0']
    
    # qt_ratio: Normalized cone resistance based on excess pore pressure
    df['qt_ratio'] = df['qt'] / (df['excess_pwp'] + 1e-6)
    
    # qt_per_depth: Depth-normalized cone resistance
    df['qt_per_depth'] = df['qt'] / (df['depth'] + 1e-6)
    
    # Apply logarithmic transformations to normalize skewed distributions
    df['qt_log'] = np.log1p(df['qt'])
    df['u2_log'] = np.log1p(df['u2'])
    df['Qt_log'] = np.log1p(df['Qt'])
    df['OCR_log'] = np.log1p(df['OCR'])

    # Select specified features for scaling
    X = df[features].copy()
    
    # Handle missing values via mean imputation and standardize features
    # Standardization is critical for distance-based algorithms like UMAP/GMM
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, df