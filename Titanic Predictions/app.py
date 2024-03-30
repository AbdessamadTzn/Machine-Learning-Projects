from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import joblib
import os

# Load environment variables
load_dotenv()

# Get files from the env
MODEL_PATH = os.getenv("MODEL_PATH")
STATIC_IMAGE_PATH = os.getenv("STATIC_IMAGE_PATH")

# Load the model
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', STATIC_IMAGE_PATH=STATIC_IMAGE_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    pclass = request.form['pclass']
    sex = request.form['sex']
    age = request.form['age']
    sibsp = request.form['sibsp']
    parch = request.form['parch']
    fare = request.form['fare']
    embarked = request.form['embarked']
    try:
        pclass = int(pclass)
        sex = int(sex)
        age = float(age)
        sibsp = int(sibsp)
        parch = int(parch)
        fare = float(fare)
        embarked = int(embarked)

        # Make predictions
        features = [[pclass, sex, age, sibsp, parch, fare, embarked]]
        prediction = model.predict(features)[0]

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
