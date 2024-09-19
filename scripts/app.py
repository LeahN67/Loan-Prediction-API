from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load models
try:
    binary_model = joblib.load('../models/binary_model.joblib')
    yield_model = joblib.load('../models/yield_model.joblib')
    days_model = joblib.load('../models/days_model.joblib')
    logger.info("Models loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Error loading models: {e}")
    raise

# Define feature names
feature_names = ['principal', 'total_owing_at_issue', 'employee_count', 'dismissal_description', 
                 'acquisition_channel', 'sector', 'repayment_ratio', 'application_month', 
                 'avg_past_yield', 'avg_past_days_to_repayment', 'most_common_transaction_type']

def preprocess_input(data):
    """
    Preprocess the input data by encoding categorical features.

    Args:
        data (dict): Dictionary of input features from the request payload.

    Returns:
        dict: Processed data with categorical features encoded into numeric values.
    """
    dismissal_encoding = {"none": 0, "minor": 1, "major": 2}
    acquisition_channel_encoding = {"online": 0, "offline": 1}
    sector_encoding = {"finance": 0, "retail": 1, "healthcare": 2}
    transaction_type_encoding = {"loan": 0, "credit": 1, "debit": 2}

    # Replace categorical features with their encoded values
    data['dismissal_description'] = dismissal_encoding.get(data['dismissal_description'], -1)  # Default to -1 if not found
    data['acquisition_channel'] = acquisition_channel_encoding.get(data['acquisition_channel'], -1)
    data['sector'] = sector_encoding.get(data['sector'], -1)
    data['most_common_transaction_type'] = transaction_type_encoding.get(data['most_common_transaction_type'], -1)

    return data

@app.route('/')
def home():
    """
    Serve the home page (index.html) from the static directory.

    Returns:
        File: The index.html file from the 'static' directory.
    """
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction request by accepting JSON input, processing the features,
    and returning predictions for loan repayment probability, yield, and days to repayment.

    The prediction also includes a risk score and recommended interest rate.

    Returns:
        JSON: A dictionary with predicted values or an error message if something fails.
    """
    try:
        data = request.json
        # Check if all required features are provided in the request
        if not all(feature in data for feature in feature_names):
            missing_features = [f for f in feature_names if f not in data]
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Preprocess the input data to handle categorical variables
        data = preprocess_input(data)

        # Extract features in the right order and convert them into an array
        features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)

        # Make predictions using the loaded models
        binary_pred = binary_model.predict_proba(features)[0, 1]
        yield_pred = yield_model.predict(features)[0]
        days_pred = days_model.predict(features)[0]

        # Calculate a simple risk score and recommended interest rate
        risk_score = (1 - binary_pred) * 100  # Higher score means higher risk
        base_rate = 0.10  # 10% base rate
        risk_premium = risk_score / 1000  # 0.1% increase for each risk point
        recommended_rate = base_rate + risk_premium

        # Build the response
        response = {
            'repayment_probability': float(binary_pred),
            'predicted_yield': float(yield_pred),
            'predicted_days_to_repayment': float(days_pred),
            'risk_score': float(risk_score),
            'recommended_interest_rate': float(recommended_rate)
        }

        # Log the prediction details and return the response
        logger.info(f"Prediction made: {response}")
        return jsonify(response)

    except Exception as e:
        # Log and return the error if any issue arises during prediction
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors (not found) by returning a JSON response with an error message.

    Args:
        error: The error object.

    Returns:
        JSON: A dictionary with an error message and 404 status code.
    """
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    """
    Run the Flask application in debug mode on port 5003.
    """
    app.run(debug=True,port=5003)
