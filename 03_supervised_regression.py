import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

def execute_bayesian_hyperparameter_tuning(X_train, y_train, random_state=94):
    """
    Demonstrates the automated hyperparameter tuning process for the Random Forest regressor.
    Validates the selection of optimal parameters for geostatic stress history evaluation.
    """
    # Define search space for ensemble learning optimization
    search_space = {
        'n_estimators': Integer(100, 800),
        'max_depth': Integer(5, 30),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Real(0.2, 1.0, prior='uniform'),
        'bootstrap': Categorical([True])
    }

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Bayesian Optimization using Gaussian Processes
    opt = BayesSearchCV(
        rf, search_spaces=search_space, n_iter=300, 
        cv=cv, scoring='r2', n_jobs=1, random_state=random_state
    )
    opt.fit(X_train, y_train.values.ravel())
    return opt.best_params_

def train_final_predictive_model(X_train, y_train):
    """
    Initializes and trains the final Random Forest model with optimal parameters.
    Hyperparameter configuration as per Appendix A.5:
    - Estimators: 500, Max Depth: 10, Min Samples Leaf: 5, Max Features: 0.2
    """
    final_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features=0.2,
        bootstrap=True,
        random_state=94,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train.values.ravel())
    
    # Export the serialized model for reproducibility
    joblib.dump(final_model, "final_ocr_model.pkl")
    return final_model