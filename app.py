from flask import Flask, request, render_template
import pickle
import numpy as np
import time

app = Flask(__name__)

# Order here must match the model's target encoding
weather_classes = [
    'clear', 'cloudy', 'drizzly', 'foggy', 'hazey',
    'misty', 'rainy', 'smokey', 'thunderstorm'
]


def load_model(model_path='model/model.pkl'):
    """Load the trained weather classification model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def classify_weather(features):
    model = load_model()
    start = time.time()
    prediction_index = model.predict(features)[0]
    latency = round((time.time() - start) * 1000, 2)
    
    # Use the model’s predicted index to look up the label
    prediction = weather_classes[int(prediction_index)]
    
    return prediction, latency


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # --- Read and convert form inputs to floats ---
            temperature = float(request.form['temperature'])
            pressure = float(request.form['pressure'])
            humidity = float(request.form['humidity'])
            wind_speed = float(request.form['wind_speed'])
            wind_deg = float(request.form['wind_deg'])

            # Optional fields: default to 0 if missing/blank
            rain_1h = float(request.form.get('rain_1h', 0) or 0)
            rain_3h = float(request.form.get('rain_3h', 0) or 0)
            snow = float(request.form.get('snow', 0) or 0)
            clouds = float(request.form.get('clouds', 0) or 0)

            # --- Build numeric feature array for the model ---
            features = np.array([
                temperature, pressure, humidity,
                wind_speed, wind_deg, rain_1h,
                rain_3h, snow, clouds
            ], dtype=float).reshape(1, -1)

            # --- Classify and measure latency ---
            prediction, latency = classify_weather(features)

            # Render result page
            return render_template(
                'result.html',
                prediction=prediction,
                latency=latency
            )

        except Exception as e:
            error_msg = f"Error processing input: {e}"
            # Re-render the form with an error message
            return render_template('form.html', error=error_msg)

    # GET request – show the input form
    return render_template('form.html')


if __name__ == '__main__':
    # Run on port 5001 as before
    app.run(host="0.0.0.0", port=5001)