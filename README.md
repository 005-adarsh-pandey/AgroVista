
AgroVista – An Intelligent Crop Yield Prediction system, using Machine Learning and Weather Analytics
=====================================================================================================

AgroVista is a web-based platform that helps farmers and agricultural stakeholders make informed decisions. It combines machine learning, weather forecasting, MSP, and market price analysis. Designed with the unique needs of Indian agriculture in mind, AgroVista gathers data from government sources, weather APIs, and crop production statistics to provide a powerful and user-friendly decision-support tool.

The project was built using Python, Flask, and Scikit-learn. It is modular, scalable, and easy to use, even in rural areas with limited digital access.

What Does AgroVista Do?
------------------------

AgroVista is not just a crop prediction tool. It offers an integrated dashboard that answers important questions farmers often have:

- What crop should I plant this season?
- What is the expected yield based on my land and weather conditions?
- Are current market prices better than MSP (Minimum Support Price)?
- What does the 6-month weather forecast look like?

Core Modules
------------

### 1. Crop Yield Prediction

- Uses a Random Forest regression model trained on over 240,000 agricultural records from 1997 to 2015.
- Inputs include crop, area, season, location (latitude and longitude), and rainfall.
- Provides accurate yield estimates with performance metrics (R² = 0.92).

### 2. Weather Forecasting

- Offers real-time 6-month seasonal forecasts through the CDS Copernicus API.
- Uses historical data from the NASA POWER API.
- Includes rainfall and climate conditions in the prediction process.

### 3. Market Price Analysis

- Provides live mandi prices through the data.gov.in API.
- Gets MSP data from government PDF reports.
- Suggests the most profitable crop for a region based on expected yield versus price.

### 4. Web Interface

- Built with Flask, HTML, CSS, and JavaScript.
- Mobile-friendly and suitable for devices with low resources.
- Includes prediction forms, weather dashboards, and market comparison tools.
- Uses asynchronous APIs with fallback logic and error handling.

Architecture Overview
---------------------

AgroVista follows a clean three-layer architecture:

- **Frontend:** Simple and responsive user interface.
- **Backend:** Modular Python Flask app that handles prediction logic and API calls.
- **Data Layer:** Preprocessed agricultural datasets, trained machine learning models, and real-time feeds.

Technology Stack
----------------

- **Language:** Python 3.8+
- **Framework:** Flask
- **Machine Learning:** Scikit-learn (Random Forest)
- **Data Tools:** Pandas, NumPy, Joblib
- **Visualization:** Matplotlib
- **APIs:** CDS (Copernicus), NASA POWER, data.gov.in (Mandi Prices), DESAgri (MSP)

Getting Started
---------------

### Prerequisites

- Python 3.8+
- pip
- Internet connection (for real-time weather and mandi APIs)

### Installation

```bash
// for linux
sudo apt update
sudo apt install python3-pip
//Same commands for linux and windows
git clone https://github.com/005-adarsh-pandey/AgroVista/
cd AgroVista
pip install --break-system-packages -r requirements.txt
cd Website
python app.py
```

Then, open your browser and go to:

```
http://localhost:5000
```

Sample Use Cases
----------------

- A farmer wants to find out which crop will be most profitable this season.
- A policymaker needs to evaluate yield risk by district for different climate patterns.
- An agribusiness wants to integrate yield and price data into their system through an API.

Achievements
------------

- Trained on over 240,000 agricultural records across 19 years and 30+ crops.
- Integrated real-time seasonal forecasts and price data from multiple APIs.
- Achieved high accuracy (Test R²: 0.92, RMSE: 4.2 million MT).
- Created a fully modular, scalable, and responsive platform.

Future Enhancements
-------------------

- Adding soil health and irrigation source data.
- Developing a REST API for integration with external platforms.
- Creating a mobile app with support for regional languages.
- Implementing anomaly detection for early warning systems.
- Expanding to include Agmarknet and other price sources.

License
-------

This project was developed for academic and educational purposes, aimed at promoting smart agriculture in India.

**Built with purpose—for the farmers of tomorrow.**
