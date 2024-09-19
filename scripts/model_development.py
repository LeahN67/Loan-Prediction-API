import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import logging

# Set up logging for tracking and debugging purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """
    Load the training and testing datasets from .npy files.

    Returns:
        tuple: Contains training and testing data for features (X) and targets (y_binary, y_yield, y_days).
    Raises:
        FileNotFoundError: If any of the data files are not found.
    """
    try:
        X_train = np.load('../data/X_train.npy', allow_pickle=True)
        X_test = np.load('../data/X_test.npy', allow_pickle=True)
        y_binary_train = np.load('../data/y_binary_train.npy', allow_pickle=True)
        y_binary_test = np.load('../data/y_binary_test.npy', allow_pickle=True)
        y_yield_train = np.load('../data/y_yield_train.npy', allow_pickle=True)
        y_yield_test = np.load('../data/y_yield_test.npy', allow_pickle=True)
        y_days_train = np.load('../data/y_days_train.npy', allow_pickle=True)
        y_days_test = np.load('../data/y_days_test.npy', allow_pickle=True)
        
        logger.info(f"Data loaded successfully. Shapes: X_train: {X_train.shape}, y_binary_train: {y_binary_train.shape}, y_yield_train: {y_yield_train.shape}, y_days_train: {y_days_train.shape}")
        return (X_train, X_test, y_binary_train, y_binary_test, 
                y_yield_train, y_yield_test, y_days_train, y_days_test)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise

def handle_nan_values(X, y):
    """
    Handle missing values (NaNs) in both the features and target data.

    Args:
        X (numpy.ndarray): Feature data.
        y (numpy.ndarray): Target data.

    Returns:
        tuple: Imputed feature and target data with NaNs replaced by 0.
    """
    logger.info(f"Before handling NaNs - X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"NaN count in X: {np.isnan(X).sum()}, NaN count in y: {np.isnan(y).sum()}")
    
    imputer_X = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = imputer_X.fit_transform(X)
    
    imputer_y = SimpleImputer(strategy='constant', fill_value=0)
    y_imputed = imputer_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    logger.info(f"After handling NaNs - X shape: {X_imputed.shape}, y shape: {y_imputed.shape}")
    logger.info(f"NaN count in X_imputed: {np.isnan(X_imputed).sum()}, NaN count in y_imputed: {np.isnan(y_imputed).sum()}")
    
    return X_imputed, y_imputed

def train_binary_model(X_train, y_train):
    """
    Train the binary classification model (for loan repayment prediction).

    Args:
        X_train (numpy.ndarray): Feature data for training.
        y_train (numpy.ndarray): Target data for training.

    Returns:
        RandomForestClassifier: Trained binary classification model.
    """
    X_imputed, y_imputed = handle_nan_values(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_imputed, y_imputed)
    logger.info("Binary model trained successfully")
    return model

def train_yield_model(X_train, y_train):
    """
    Train the yield prediction model (regression).

    Args:
        X_train (numpy.ndarray): Feature data for training.
        y_train (numpy.ndarray): Target data for training.

    Returns:
        RandomForestRegressor: Trained regression model for yield prediction.
    """
    X_imputed, y_imputed = handle_nan_values(X_train, y_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_imputed, y_imputed)
    logger.info("Yield model trained successfully")
    return model

def train_days_model(X_train, y_train):
    """
    Train the days-to-repayment prediction model (regression).

    Args:
        X_train (numpy.ndarray): Feature data for training.
        y_train (numpy.ndarray): Target data for training.

    Returns:
        RandomForestRegressor: Trained regression model for days-to-repayment prediction, or None if no valid data exists.
    """
    X_imputed, y_imputed = handle_nan_values(X_train, y_train)
    if len(y_imputed) == 0:
        logger.error("No valid data for days-to-repayment model after handling NaNs")
        return None
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_imputed, y_imputed)
    logger.info("Days-to-repayment model trained successfully")
    return model

def evaluate_binary_model(model, X_test, y_test):
    """
    Evaluate the binary classification model using AUC-ROC score.

    Args:
        model (RandomForestClassifier): Trained binary classification model.
        X_test (numpy.ndarray): Feature data for testing.
        y_test (numpy.ndarray): Target data for testing.
    """
    X_imputed, y_imputed = handle_nan_values(X_test, y_test)
    y_pred = model.predict_proba(X_imputed)[:, 1]
    auc_roc = roc_auc_score(y_imputed, y_pred)
    logger.info(f"Binary Model AUC-ROC: {auc_roc}")

def evaluate_regression_model(model, X_test, y_test, model_name):
    """
    Evaluate a regression model (either yield or days-to-repayment).

    Args:
        model (RandomForestRegressor): Trained regression model.
        X_test (numpy.ndarray): Feature data for testing.
        y_test (numpy.ndarray): Target data for testing.
        model_name (str): Name of the model being evaluated (for logging purposes).

    Returns:
        None
    """
    if model is None:
        logger.error(f"Cannot evaluate {model_name}: model is None")
        return
    X_imputed, y_imputed = handle_nan_values(X_test, y_test)
    y_pred = model.predict(X_imputed)
    mae = mean_absolute_error(y_imputed, y_pred)
    r2 = r2_score(y_imputed, y_pred)
    logger.info(f"{model_name} MAE: {mae}")
    logger.info(f"{model_name} R2: {r2}")

def save_models(binary_model, yield_model, days_model):
    """
    Save trained models to disk.

    Args:
        binary_model (RandomForestClassifier): Trained binary classification model.
        yield_model (RandomForestRegressor): Trained yield regression model.
        days_model (RandomForestRegressor or None): Trained days-to-repayment regression model (if available).

    Raises:
        Exception: If an error occurs during model saving.
    """
    try:
        joblib.dump(binary_model, '../models/binary_model.joblib')
        joblib.dump(yield_model, '../models/yield_model.joblib')
        if days_model is not None:
            joblib.dump(days_model, '../models/days_model.joblib')
        logger.info("Models saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise

def main():
    """
    Main function to execute the model development pipeline:
    1. Load data.
    2. Train models.
    3. Evaluate models.
    4. Save models.
    """
    try:
        (X_train, X_test, y_binary_train, y_binary_test, 
         y_yield_train, y_yield_test, y_days_train, y_days_test) = load_data()

        # Train each model
        binary_model = train_binary_model(X_train, y_binary_train)
        yield_model = train_yield_model(X_train, y_yield_train)
        days_model = train_days_model(X_train, y_days_train)

        # Evaluate each model on test data
        evaluate_binary_model(binary_model, X_test, y_binary_test)
        evaluate_regression_model(yield_model, X_test, y_yield_test, "Yield Model")
        evaluate_regression_model(days_model,X_test, y_days_test, "Days-to-Repayment Model")

        # Save the trained models
        save_models(binary_model, yield_model, days_model)
        
        logger.info("Model development completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during model development: {e}")

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()
