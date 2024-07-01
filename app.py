from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the Naive Bayes model from a file
with open("nb_01.sav", 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Welcome to the Lung Cancer Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON request data
    data = request.get_json()

    # Convert the JSON data to a DataFrame
    input_features = pd.DataFrame([data])

    # Make predictions using the model
    prediction = model.predict(input_features)

    # Return the prediction result as JSON
    result = {
        'prediction': int(prediction[0])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
