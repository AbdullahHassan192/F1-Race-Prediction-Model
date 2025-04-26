import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from config import TEAM_PERFORMANCE, DRIVER_PERFORMANCE, CIRCUIT_DATA, DRIVER_TEAMS, DRIVER_ORDER


fastf1.Cache.enable_cache('fastf1_cache')

def fetch_f1_data(year, round_number): # Get data from the FastF1 api
    
    try:
      
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
    
        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
        results = results.rename(columns={'FullName': 'Driver'})
        
        # Converting time to seconds manually
        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(
                lambda x: x.total_seconds() if pd.notnull(x) else None
            )
        
       
        print("\nQualifying Results:")
        print(results.head())
        
        return results
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Current columns available:", quali.results.columns.tolist())
        return None

# Safely convert datetime into seconds
def convert_time(td): 
    try:
        return td.total_seconds()
    except Exception as e:
        print(f"Could not convert time: {td}, Error: {e}")
        return None

# Basic clean up of times
def clean_data(df):

    print("\nBefore cleaning:")
    print(df[['Driver', 'Q1', 'Q2', 'Q3']].head())
    
    for session in ['Q1', 'Q2', 'Q3']:
        df[session + '_sec'] = df[session].apply(convert_time)
    
    print("\nAfter cleaning:")
    print(df[['Driver', 'Q1_sec', 'Q2_sec', 'Q3_sec']].head())
    
    return df.dropna()


def run_regression(df):
    X = df[['Q1_sec', 'Q2_sec']]
    y = df['Q3_sec']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X)
    
    results_df = df[['Driver', 'TeamName', 'Q1_sec', 'Q2_sec', 'Q3_sec']].copy()
    results_df['Predicted_Q3'] = predictions
    results_df['Difference'] = results_df['Predicted_Q3'] - results_df['Q3_sec']
    
    results_df = results_df.sort_values('Predicted_Q3')
    

    print("\nQualifying Prediction:")
    print("=" * 70)
    print(f"{'Position':<10}{'Driver':<15}{'Team':<20}{'Predicted Time':<15}{'Actual Time':<15}")
    print("-" * 70)
    
    for idx, row in results_df.iterrows():
        pred_time = f"{row['Predicted_Q3']:.3f}"
        actual_time = f"{row['Q3_sec']:.3f}" if not pd.isna(row['Q3_sec']) else "N/A"
        print(f"{results_df.index.get_loc(idx)+1:<10}{row['Driver']:<15}{row['TeamName']:<20}{pred_time:<15}{actual_time:<15}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f'MAE: {mae:.2f} seconds')
    print(f'R^2 Score: {r2:.2f}')

# Fetch rounds of 2025 + 2024 race on circuit we're predicting on
def fetch_recent_data():
    all_data = []
    
    current_year = 2025
    for round_num in range(1, 5):  # First 4 rounds of 2025
        print(f"Fetching data for {current_year} round {round_num}...")
        df = fetch_f1_data(current_year, round_num)
        if df is not None:
            df['Year'] = current_year
            df['Round'] = round_num
            all_data.append(df)
    
    print("Fetching 2024 Saudi Arabian GP data...")
    saudi_2024 = fetch_f1_data(2024, 4) 
    if saudi_2024 is not None:
        saudi_2024['Year'] = 2024
        saudi_2024['Round'] = 4
        all_data.append(saudi_2024)
    
    return all_data

# Apply the performance factors calculated and placed in the config
def apply_performance_factors(predictions_df, circuit_name='Saudi Arabia'):
    base_time = CIRCUIT_DATA[circuit_name]['base_quali']

    # Compute team performance multipliers
    team_factors = {
        team: 1 + (data['quali_adjust'] / base_time)
        for team, data in TEAM_PERFORMANCE.items()
    }

    # Compute driver performance multipliers
    driver_factors = {
        driver: 1 + (data['quali_adjust'] / base_time)
        for driver, data in DRIVER_PERFORMANCE.items()
    }

    for idx, row in predictions_df.iterrows():
        team = row['Team']
        driver = row['Driver']

        team_factor = team_factors.get(team, 1.0)
        driver_factor = driver_factors.get(driver, 1.0)

        base_prediction = base_time * team_factor * driver_factor

        random_variation = np.random.uniform(-0.1, 0.1)
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation

    return predictions_df

#Predict q3 times for saudi GP (replace with GP to be predicted)
def predict_saudi_gp(model, latest_data):

    driver_teams = DRIVER_TEAMS

    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    
    results_df = apply_performance_factors(results_df)

    results_df = results_df.sort_values('Predicted_Q3')

    print("\nSaudi Arabian GP 2025 Qualifying Predictions:")
    print("=" * 100)
    print(f"{'Position':<10}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
    print("-" * 100)
    
    for idx, row in results_df.iterrows():
        print(f"{results_df.index.get_loc(idx)+1:<10}"
              f"{row['Driver']:<20}"
              f"{row['Team']:<25}"
              f"{row['Predicted_Q3']:.3f}s")


def get_ordered_qualifying_df(circuit_name='Saudi Arabia'):
    driver_teams = DRIVER_TEAMS
    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    results_df = apply_performance_factors(results_df, circuit_name)
    ordered_results = []
    for driver in DRIVER_ORDER:
        match = results_df[results_df['Driver'] == driver]
        if not match.empty:
            ordered_results.append(match.iloc[0])
        else:
            ordered_results.append(pd.Series({'Driver': driver, 'Predicted_Q3': np.nan}))
    final_df = pd.DataFrame(ordered_results)[['Driver', 'Predicted_Q3']].reset_index(drop=True)
    return final_df

# Separate function that does everything to make quali predictions for race.py
def build_quali_model():
    all_data = fetch_recent_data()

    if not all_data:
        print("No data found.")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')

    imputer = SimpleImputer(strategy='median')
    X = valid_data[['Q1_sec', 'Q2_sec']]
    y = valid_data['Q3_sec']

    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    return model

def generate_latest_data():
    all_data = fetch_recent_data()
    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = clean_data(combined_df)
    return combined_df



if __name__ == "__main__":
    print("Initializing enhanced F1 prediction model...")

    all_data = fetch_recent_data()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
    
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')

        imputer = SimpleImputer(strategy='median')
        
        X = valid_data[['Q1_sec', 'Q2_sec']]
        y = valid_data['Q3_sec']
        
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())
        
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        predict_saudi_gp(model, valid_data)
        
        y_pred = model.predict(X_clean)
        mae = mean_absolute_error(y_clean, y_pred)
        r2 = r2_score(y_clean, y_pred)
            
        print("\nModel Evaluation Metrics:")
        print(f'MAE: {mae:.2f} seconds')
        print(f'R^2 Score: {r2:.2f}')
    else:
        print("Failed to fetch F1 data")