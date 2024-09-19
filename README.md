# Numida Loan Prediction API

This repository contains a machine learning-based system for predicting loan performance for Numida, a company providing working capital loans to small and medium-sized businesses in Uganda.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Project Structure](#project-structure)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development](#model-development)
7. [API Usage](#api-usage)
8. [Web Interface](#web-interface)
9. [Future Improvements](#future-improvements)

## Project Overview

This system uses machine learning to predict various aspects of loan performance:
- Likelihood of repayment within 15 days
- Expected yield
- Number of days to full repayment

It also calculates a risk score and recommends an interest rate based on these predictions.

## Getting Started

### Prerequisites
- Python 3.10.0
- pip
- Jupyter Notebook (for running the EDA notebook)

### Setting up the environment

1. Clone the repository:
   ```
   git clone https://github.com/LeahN67/Loan-Prediction.git
   cd Loan-Prediction
   ```

2. Ensure you have Python 3.10.0 installed. You can check your Python version with:
   ```
   python --version
   ```
   If you don't have Python 3.10.0, you can download it from the [official Python website](https://www.python.org/downloads/release/python-3100/).

3. Create a virtual environment:
   ```
   python -m venv <environment_name>
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     <environment_name>\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source <environment_name>/bin/activate
     ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   The `requirements.txt` file is included in the repository and contains all the necessary Python packages for this project.

## Project Structure

- `eda_notebook.ipynb`: Jupyter notebook for exploratory data analysis
- `data_preprocessing.py`: Handles data loading, cleaning, and feature engineering
- `model_development.py`: Trains and evaluates the machine learning models
- `app.py`: Flask application serving the API
- `static/index.html`: Simple web interface for interacting with the API
- `data/`: Directory containing the input CSV and numpy files
- `models/`: Directory where trained models are saved
- `requirements.txt`: List of Python packages required for the project


## Exploratory Data Analysis

Before preprocessing the data and developing the models, it's crucial to understand the dataset. We've provided a Jupyter notebook for exploratory data analysis:

1. Ensure you have Jupyter Notebook installed. It should be included in the requirements.txt, but if not, you can install it with:
   ```
   pip install jupyter
   ```

2. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

3. Open the `eda_notebook.ipynb` file in the Jupyter Notebook interface.

4. Run the cells in the notebook to explore the dataset. This notebook includes:
   - Basic statistics of the features
   - Distribution plots for key variables
   - Correlation analysis
   - Visualizations of relationships between variables
   - Initial insights that informed feature engineering and model selection

Running this notebook will give you a better understanding of the dataset and the rationale behind some of the preprocessing steps and modeling choices.

## Data Preprocessing

To preprocess the data:

1. Ensure your CSV files are in the `data/` directory:
   - `train_loan_data.csv`
   - `test_loan_data.csv`
   - `train_payment_data.csv`

2. Run the preprocessing script:
   ```
   python data_preprocessing.py
   ```

This will generate preprocessed numpy files in the `data/` directory.

## Model Development

To train the models:

1. Ensure you have run the data preprocessing step.

2. Run the model development script:
   ```
   python model_development.py
   ```

This will train the models and save them in the project directory.

## API Usage

To start the API:

1. Ensure you have trained the models.

2. Run the Flask application:
   ```
   python app.py
   ```

The API will be available at `http://localhost:5003`.

To make a prediction, send a POST request to `http://localhost:5003/predict` with a JSON payload containing the loan features.

Example using curl:
```
curl -X POST -H "Content-Type: application/json" -d '{
  "principal": 1000,
  "total_owing_at_issue": 1100,
  "employee_count": 5,
  "dismissal_description": "none",
  "acquisition_channel": "online",
  "sector": "retail",
  "repayment_ratio": 0.9,
  "application_month": 6,
  "avg_past_yield": 0.15,
  "avg_past_days_to_repayment": 25,
  "most_common_transaction_type": "loan"
}' http://localhost:5003/predict
```

## Web Interface

A simple web interface is available at `http://localhost:5003` when the Flask app is running. You can use this to input loan details and see predictions.

## Future Improvements

- Implement user authentication for the API
- Add more advanced feature engineering techniques
- Implement model monitoring and retraining pipeline
- Enhance the web interface with more user-friendly features and styling
- Add more comprehensive input validation and error handling
- Deploy the API to a cloud service like AWS, GCP or Azure
