from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone 
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt 
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from datetime import date

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IllRun Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f7f7f7;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 400px;
            margin: 40px auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 32px 24px 24px 24px;
        }
        h2 {
            text-align: center;
            color: #333;
        }
        label {
            display: block;
            margin-top: 16px;
            color: #555;
        }
        input[type="date"], input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 4px;
            border-radius: 4px;
            border: 1px solid #ccc;
            box-sizing: border-box;
        }
        button {
            margin-top: 24px;
            width: 100%;
            padding: 10px;
            background: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
        .result {
            margin-top: 24px;
            text-align: center;
            font-size: 18px;
            color: #222;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>IllRun Project</h2>
        <p style="text-align:center;">Predict if you will be running tomorrow!</p>
        <form method="post">
            <label for="predict_date">Date:</label>
            <input type="date" id="predict_date" name="predict_date" value="{{ today }}" required>
            <label for="rest_days">Rest days before Run:</label>
            <input type="number" id="rest_days" name="rest_days" min="0" max="30" value="0" required>
            <button name="action" value="Predict">Predict!</button>
        </form>
        <div class="result">{{ status|safe }}</div>
        <div style="margin-top: 30px; padding: 10px; background: #f0f0f0; border-radius: 5px; font-size: 12px; color: #666; text-align: center;">
            <em>Frontend developed with assistance from Google Gemini 2.5 Pro LLM</em><br>
            <a href="https://github.com/tiubak/IllRun" target="_blank" style="color: #007bff; text-decoration: none;">View on GitHub</a>
        </div>
    </div>
</body>
</html>
"""

def handle_Predict():
    #input:
    predict_date_str = request.form['predict_date']
    rest_days = int(request.form['rest_days'])    
    predict_date = pd.to_datetime(predict_date_str)
    week_of_year = predict_date.isocalendar().week
    weekday = predict_date.weekday()
    is_weekend = weekday in [5, 6]
    X_input = pd.DataFrame({
        'RestDaysBeforeRun': [rest_days],
        'week_of_year': [week_of_year],
        'is_weekend': [is_weekend]
    })

    # Load data
    df = pd.read_csv("rungap-export.csv")
    # cleaning data
    df = df[df['activity'].str.contains('Running', case=False, na=False)]
    df = df[['local time', 'distance(m)']]
    df['local time'] = pd.to_datetime(df['local time'])
    df = df[(df['local time'] >  pd.to_datetime('2024-08-24')) & (df['local time'] < pd.to_datetime('2025-08-26'))]       
    # First we need to normalize the hour
    df['local time'] = pd.to_datetime(df['local time']).dt.normalize()
    df_allDays = df.groupby('local time').agg({
        'distance(m)': 'sum' 
    }).reset_index()
    #EDA
    # add not runned days
    all_days = pd.date_range(start=df_allDays['local time'].min(), end=df_allDays['local time'].max(), freq='D')
    df_allDays = df_allDays.set_index('local time').reindex(all_days).rename_axis('local time').reset_index()
    df_allDays['distance(m)'] = df_allDays['distance(m)'].fillna(0)
    df=df_allDays
    #add other features
    df['week_of_year'] = df['local time'].dt.isocalendar().week  # Add week_of_year to the final df
    last_run_date = df['local time'].where((df['distance(m)'] != 0).astype(int) == 1)
    last_run_date = last_run_date.ffill()
    # Calculate the difference in days
    df['LastDayRun'] = (df['local time'] - last_run_date).dt.days
    #add weekday
    df['weekday'] = df['local time'].dt.weekday
    #add is_weekend
    df['is_weekend'] = df['weekday'].isin([5, 6])
    df['RestDaysBeforeRun'] = df['LastDayRun'].shift(1)
    df['RestDaysBeforeRun'] = df['RestDaysBeforeRun'].fillna(0)   

    #create model

    features = ['RestDaysBeforeRun', 'week_of_year', 'is_weekend']
    target = 'distance(m)'

    X = df[features]
    y = df[target]

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,          
        learning_rate=0.05,
        random_state=42
    )
    #use whole data
    xgb_model.fit(X,y,verbose=False)
    # Predict distance
    pred_km = xgb_model.predict(X_input)[0] / 1000  # Convert to km
    
    # Determine if will run (if predicted distance > 0.1 km)
    will_run = 1 if pred_km > 0.1 else 0
    # Visual result
    # Lottie animation URLs - real working animations
    lottie_run = "https://assets3.lottiefiles.com/packages/lf20_x1gjdldd.json"  # Running animation
    lottie_couch = "https://assets5.lottiefiles.com/packages/lf20_l5qvxwtf.json"  # Relaxing/sleep animation
    lottie_player = "https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"
    if will_run == 1:
        card = f"""
        <div style='background:#e6ffe6;border-radius:10px;padding:24px 12px;text-align:center;box-shadow:0 2px 8px #b2f2b2;'>
            <div style='font-size:64px;animation: bounce 2s infinite;'>🏃‍♂️</div>
            <span style='font-size:22px;color:#228B22;font-weight:bold;'>You will be running!</span><br>
            <span style='font-size:18px;color:#333;'>Predicted distance: <b>{pred_km:.2f} km</b></span>
        </div>
        <style>
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-10px); }}
                60% {{ transform: translateY(-5px); }}
            }}
        </style>
        """
    else:
        card = f"""
        <div style='background:#ffe6e6;border-radius:10px;padding:24px 12px;text-align:center;box-shadow:0 2px 8px #f2b2b2;'>
            <div style='font-size:64px;animation: sway 3s ease-in-out infinite;'>🛋️</div>
            <span style='font-size:22px;color:#B22222;font-weight:bold;'>You will NOT be running.</span><br>
            <span style='font-size:18px;color:#333;'>Predicted distance: <b>0.00 km</b></span>
        </div>
        <style>
            @keyframes sway {{
                0%, 100% {{ transform: rotate(-2deg); }}
                50% {{ transform: rotate(2deg); }}
            }}
        </style>
        """
    return card

@app.route("/", methods=["GET", "POST"])
def index():
    status = ""
    str_today=date.today().isoformat()
    if request.method == "POST":
        action = request.form['action']
        if action == "Predict":
            status = handle_Predict()
        else:
            status = "Unknown action."
    return render_template_string(HTML, status=status, today=str_today)

if __name__ == "__main__":
    app.run(debug=True)