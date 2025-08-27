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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import joblib
from datetime import date

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IllRun project</title>
</head>
<body>
    <h2>This is it IllRun project.</h2>
    <p>It will predict if you wil be running or not tomorow!</p>
    <form method="post">
        <button name="action" value="CreateModel">Create Model</button>
    </form>
    <form method="post">
        <label for="predict_date">Date (today):</label>
        <input type="date" id="predict_date" name="predict_date" value="{{ today }}"><br/>
        <input type="checkbox" id="run_yesterday" name="run_yesterday" value="1">
        <label for="run_yesterday">Run yesterday</label><br/>
        <input type="checkbox" id="run_today" name="run_today" value="1">
        <label for="run_today">Run today</label><br/>
        <button name="action" value="Predict">Predict!</button>
    </form>
    <div style="margin-top:20px;">
        <div>{{ status|safe }}</div>
    </div>
</body>
</html>
"""

def handle_CreateModel():
    
    df = pd.read_csv("rungap-export.csv")
    df = df[df['activity'].str.contains('Running', case=False, na=False)]
    # to make sure the date is balanced, its better to filter by date (let's consider last year)
    df['local time'] = pd.to_datetime(df['local time'])
    df = df[(df['local time'] >=  pd.to_datetime('2024-08-24')) & (df['local time'] <= pd.to_datetime('2025-08-26'))]
    df = df[['local time', 'duration(s)', 'distance(m)', 'avg heartrate']]
    df['day_of_year'] = df['local time'].dt.dayofyear


    df_filled = df.groupby('day_of_year').mean()
    df_filled = df_filled.reindex(pd.RangeIndex(1, 366), fill_value=0)
    df_filled = df_filled.reset_index()
    df_filled = df_filled.rename(columns={'index': 'day_of_year'})
    df_filled = df_filled.dropna(subset=['day_of_year'])

    mask = df_filled['distance(m)'] == 0
    # as I'm using my runnings from 2024 and 2025, let's consider the day that course starts to cut between years
    # so if day of year is > 238 (24th August), then we are in 2024, else in 2025
    df_filled.loc[mask, 'local time'] = df_filled.loc[mask, 'day_of_year'].apply(
        lambda d: pd.to_datetime('2024-01-01') + pd.to_timedelta(d - 1, unit='D') if d > 238
        else pd.to_datetime('2025-01-01') + pd.to_timedelta(d - 1, unit='D')
    )

    #lets also add a new column that will say if I runned or not
    df_filled['runned'] = df_filled['distance(m)'].apply(lambda x: 1 if x>0 else 0)

    #add columns
    df_filled['runned_tomorrow'] = df_filled['runned'].shift(-1)
    df_filled['local time'] = pd.to_datetime(df_filled['local time'])
    df_filled['weekday'] = df_filled['local time'].dt.dayofweek
    df_filled['runned_today'] = df_filled['runned']
    df_filled['runned_yesterday'] = df_filled['runned'].shift(1)
    
    #clean
    df_filled = df_filled.dropna(subset=['runned_tomorrow', 'runned_yesterday'])


    #train model
    features = ['runned_today', 'runned_yesterday', 'weekday']
    target = 'runned_tomorrow'
    X = df_filled[features]
    y = df_filled[target]
    #use the whole data to train

    #Gradient BoostingClassifier give better results    
    #model = AdaBoostClassifier(n_estimators=300, random_state=42, estimator=DecisionTreeClassifier(max_depth=5))
    #model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "IllRun_model.pkl")

    loaded_model = joblib.load("IllRun_model.pkl")
    ret_stat = (
        f"IllRun_model.pkl saved!<br/>"
        f"Training accuracy: {loaded_model.score(X, y):.2f}<br/>"
        f"Test accuracy: {loaded_model.score(X, y):.2f}<br/>"
        f"Model parameters:<pre>{model.get_params()}</pre>"
        f"Feature importances:<pre>{model.feature_importances_}</pre>"
        f"Number of estimators: {len(model.estimators_)}"
    )
    return ret_stat

def handle_Predict():
        from flask import request
        loaded_model = joblib.load("IllRun_model.pkl")
        # Get form values
        predict_date = request.form.get('predict_date')
        run_yesterday = 1 if request.form.get('run_yesterday') == '1' else 0
        run_today = 1 if request.form.get('run_today') == '1' else 0

        # Convert date to weekday
        weekday = pd.to_datetime(predict_date).dayofweek if predict_date else pd.Timestamp.today().dayofweek

        # Prepare input for model
        X_input = pd.DataFrame({
            'runned_today': [run_today],
            'runned_yesterday': [run_yesterday],
            'weekday': [weekday]
        })

        # Predict
        pred = loaded_model.predict(X_input)[0]
        emoji = '🏃' if pred == 1 else '🛋️'
        result = f"Prediction for tomorrow: {'Running' if pred == 1 else 'Not running'} {emoji}"
        return result

@app.route("/", methods=["GET", "POST"])
def index():
    status = ""
    str_today=date.today().isoformat()
    if request.method == "POST":
        action = request.form['action']
        if action == "CreateModel":
            status = handle_CreateModel()
        elif action == "Predict":
            status = handle_Predict()
        else:
            status = "Unknown action."
    return render_template_string(HTML, status=status, today=str_today)

if __name__ == "__main__":
    app.run(debug=True)