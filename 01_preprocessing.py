import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_geotechnical_data(features):
    """
    Performs data ingestion and feature engineering for CPT-based OCR estimation.
    This module implements the preprocessing protocols defined in Section 4.1.
    
    Args:
        features (list): List of input parameters for model training. 
                         Must now include 'u2_minus_u0' and 'qt_over_depth'.
        
    Returns:
        X_scaled (ndarray): Standardized feature matrix.
        df (DataFrame): Preprocessed dataframe containing engineered parameters.
    """
    df = pd.read_excel("CPT DB.xlsx", sheet_name="clean")
    
    # Filter core geotechnical parameters required for the hybrid framework
    core_params = ['Site', 'OCR', 'depth', 'qt', 'u0', 'u2', 'σvo', 'Qt', 'Bq']
    df = df[core_params].copy()
    
    # Exclude samples with missing target labels (OCR) to ensure training integrity
    df = df[df['OCR'].notna()].reset_index(drop=True)

    # --- Feature Engineering Stage (Appendix A.2 & Section 4.1) ---
    # Apply specific logarithmic transformations to normalize skewed distributions
    df['u2_log'] = np.log1p(df['u2'])
    df['Qt_log'] = np.log1p(df['Qt'])

    # Create derivative features (excess pore pressure and qt/depth)
    df['u2_minus_u0'] = df['u2'] - df['u0']
    df['qt_over_depth'] = df['qt'] / df['depth']

    # Select specified features for scaling
    X = df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df