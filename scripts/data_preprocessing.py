import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_file: str, test_file: str, repayment_file: str) -> tuple:
    """
    Load data from CSV files.
    """
    try:
        train_loans = pd.read_csv(train_file)
        test_loans = pd.read_csv(test_file)
        repayments = pd.read_csv(repayment_file)
        logger.info(f"Data loaded successfully. Train shape: {train_loans.shape}, Test shape: {test_loans.shape}, Repayments shape: {repayments.shape}")
        return train_loans, test_loans, repayments
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

def merge_datasets(loans: pd.DataFrame, repayments: pd.DataFrame) -> pd.DataFrame:
    """
    Merge loan data with repayment summary.
    """
    try:
        repayment_summary = repayments.groupby('loan_id').agg({
            'amount': 'sum',
            'paid_at': 'max',
            'transaction_type': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() else 'Unknown'
        }).reset_index()
        
        repayment_summary.columns = ['loan_id', 'total_repaid', 'last_payment_date', 'most_common_transaction_type']
        
        merged_data = pd.merge(loans, repayment_summary, on='loan_id', how='left')
        
        merged_data['total_repaid'] = merged_data['total_repaid'].fillna(0)
        merged_data['last_payment_date'] = merged_data['last_payment_date'].fillna(pd.NaT)
        merged_data['most_common_transaction_type'] = merged_data['most_common_transaction_type'].fillna('Unknown')
        
        logger.info(f"Datasets merged successfully. Shape: {merged_data.shape}")
        return merged_data
    except Exception as e:
        logger.error(f"Error while merging datasets: {e}")
        raise

def preprocess_loans(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """
    Preprocess loan data.
    """
    try:
        combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        
        categorical_cols = ['dismissal_description', 'acquisition_channel', 'sector', 'most_common_transaction_type']
        numerical_cols = ['principal', 'total_owing_at_issue', 'employee_count', 'total_repaid']
        
        imputer = SimpleImputer(strategy='most_frequent')
        combined_data[categorical_cols] = imputer.fit_transform(combined_data[categorical_cols])

        label_encoder = LabelEncoder()
        for col in categorical_cols:
            combined_data[col] = label_encoder.fit_transform(combined_data[col])

        scaler = StandardScaler()
        combined_data[numerical_cols] = scaler.fit_transform(combined_data[numerical_cols])

        train_data = combined_data[:len(train_data)]
        test_data = combined_data[len(train_data):]

        logger.info("Loan data preprocessed successfully")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error during loan preprocessing: {e}")
        raise

def parse_date(date_str: str) -> pd.Timestamp:
    """
    Parse date string into pandas Timestamp.
    """
    date_formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    logger.warning(f"Unable to parse date: {date_str}")
    return pd.NaT

def create_target_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for models.
    """
    try:
        data['repaid_within_15_days'] = (data['total_recovered_15_dpd'] >= data['total_owing_at_issue']).astype(int)
        data['approval_status'] = data['approval_status'].apply(parse_date)
        data['days_to_repayment'] = (pd.to_datetime(data['last_payment_date']) - data['approval_status']).dt.days
        data.loc[data['days_to_repayment'] < 0, 'days_to_repayment'] = np.nan
        logger.info("Target variables created successfully")
        return data
    except Exception as e:
        logger.error(f"Error creating target variables: {e}")
        raise

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features.
    """
    try:
        data['repayment_ratio'] = data['total_repaid'] / data['total_owing_at_issue']
        data['application_month'] = pd.to_datetime(data['approval_status']).dt.month
        data['avg_past_yield'] = data.groupby('business_id')['cash_yield_15_dpd'].transform('mean')
        data['avg_past_days_to_repayment'] = data.groupby('business_id')['days_to_repayment'].transform('mean')
        logger.info("Features engineered successfully")
        return data
    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        raise

def prepare_features_and_targets(data: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target variables.
    """
    features = ['principal', 'total_owing_at_issue', 'employee_count', 'dismissal_description', 
                'acquisition_channel', 'sector', 'repayment_ratio', 'application_month', 
                'avg_past_yield', 'avg_past_days_to_repayment', 'most_common_transaction_type']

    X = data[features]
    y_binary = data['repaid_within_15_days']
    y_yield = data['cash_yield_15_dpd']
    y_days = data['days_to_repayment']

    logger.info("Features and targets prepared successfully")
    return X, y_binary, y_yield, y_days

def save_processed_data(X_train, X_test, y_binary_train, y_binary_test, y_yield_train, y_yield_test, y_days_train, y_days_test):
    """
    Save processed data to files.
    """
    try:
        np.save('../data/X_train.npy',X_train)
        np.save('../data/X_test.npy', X_test)
        np.save('../data/y_binary_train.npy', y_binary_train)
        np.save('../data/y_binary_test.npy', y_binary_test)
        np.save('../data/y_yield_train.npy', y_yield_train)
        np.save('../data/y_yield_test.npy', y_yield_test)
        np.save('../data/y_days_train.npy', y_days_train)
        np.save('../data/y_days_test.npy', y_days_test)
        logger.info("Processed data saved successfully")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def main():
    try:
        train_loans, test_loans, repayments = load_data('../data/train_loan_data.csv', '../data/test_loan_data.csv', '../data/train_payment_data.csv')
        
        train_merged = merge_datasets(train_loans, repayments)
        test_merged = merge_datasets(test_loans, repayments)
        
        train_preprocessed, test_preprocessed = preprocess_loans(train_merged, test_merged)
        
        train_data = create_target_variables(train_preprocessed)
        test_data = create_target_variables(test_preprocessed)
        
        train_data = engineer_features(train_data)
        test_data = engineer_features(test_data)
        
        X_train, y_binary_train, y_yield_train, y_days_train = prepare_features_and_targets(train_data)
        X_test, y_binary_test, y_yield_test, y_days_test = prepare_features_and_targets(test_data)

        save_processed_data(X_train, X_test, y_binary_train, y_binary_test, y_yield_train, y_yield_test, y_days_train, y_days_test)
        
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    main()