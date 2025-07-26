"""
Main Flask Application 
"""
import os
import json
import logging
import sys
import warnings
from flask import Flask, render_template, send_from_directory, jsonify, request

# Suppress matplotlib warnings globally - more aggressive approach
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Using categorical units to plot a list of strings')
warnings.filterwarnings('ignore', category=UserWarning, message='.*categorical units.*')

# Also suppress matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)

# Configure logging to handle Unicode properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Ensure stdout can handle Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import modular components
from app.utils import prepare_today_mandi_data, get_mandi_status
from app.weather import (
    weather_forecast_handler, 
    get_weather_data_handler, 
    get_previous_rainfall_handler,
    weather_cards_data_handler
)
from app.market import market_price_handler, get_price_data_handler
from app.predictor import (
    get_districts_handler, 
    get_coordinates_handler, 
    predict_crop_production_handler
)

# -------------------- Flask App Init -------------------- #
app = Flask(__name__, static_folder="static")

# -------------------- BASIC ROUTES -------------------- #
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/weather")
def weather():
    return render_template("Weather.html")

@app.route("/predictor")
def predictor():
    return render_template("predictor.html")

@app.route("/projects")
def projects():
    return render_template("projects.html")

@app.route("/projects2")
def projects2():
    return render_template("projects2.html")

@app.route("/market")
def market():
    return render_template("market.html")

# -------------------- API ROUTES -------------------- #
@app.route('/api/places')
def get_places():
    file_path = os.path.join(app.static_folder, 'india_places.json')
    try:
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "india_places.json not found"}, 404

# -------------------- WEATHER ROUTES -------------------- #
@app.route("/weather/forecast", methods=['POST'])
def weather_forecast():
    return weather_forecast_handler(app)

@app.route('/api/weather-data', methods=['POST'])
def get_weather_data():
    return get_weather_data_handler(app)

@app.route('/api/previous-rainfall', methods=['POST'])
def get_previous_rainfall():
    return get_previous_rainfall_handler(app)

@app.route("/weather/cards-data", methods=['POST'])
def weather_cards_data():
    return weather_cards_data_handler(app)

# -------------------- MARKET ROUTES -------------------- #
@app.route("/market/price", methods=["POST"])
def market_price():
    return market_price_handler(app)

@app.route('/api/price-data', methods=['POST'])
def get_price_data():
    return get_price_data_handler(app)

@app.route('/api/mandi-status', methods=['GET'])
def get_mandi_status_route():
    return jsonify(get_mandi_status(app))

# -------------------- CROP PREDICTION ROUTES -------------------- #
@app.route('/api/districts', methods=['GET'])
def get_districts():
    return get_districts_handler(app)

@app.route('/api/coordinates', methods=['GET'])
def get_coordinates():
    return get_coordinates_handler(app)

@app.route('/predict-crop', methods=['POST'])
def predict_crop_production():
    return predict_crop_production_handler(app)

# -------------------- STATIC FILES -------------------- #
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    # Initialize mandi data downloader
    prepare_today_mandi_data(app)
    
    # Start the Flask app
    app.run(debug=True)
