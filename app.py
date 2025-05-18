from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load saved files
model = joblib.load('best_calorie_model.pkl')
scaler = joblib.load('scaler.pkl')
exercise_encoder = joblib.load('exercise_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    try:
        Age = float(request.form['Age'])
        Height = float(request.form['Height'])
        Weight = float(request.form['Weight'])
        Gender = 0 if request.form['Gender'].lower() == 'male' else 1
        Body_Temp = float(request.form['Body_Temp'])
        Duration = float(request.form['Duration'])
        Heart_Rate = float(request.form['Heart_Rate'])
        Exercise_Type = request.form['Exercise_Type']

        # Encode exercise type
        Exercise_Type_enc = exercise_encoder.transform([Exercise_Type])[0]

        # Prepare features
        features = np.array([[Age, Height, Weight, Gender, Body_Temp, Duration, Heart_Rate, Exercise_Type_enc]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        return render_template('index.html', prediction_text=f'Estimated Calories Burnt: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
