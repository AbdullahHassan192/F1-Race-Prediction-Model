import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from config import TEAM_PERFORMANCE, DRIVER_PERFORMANCE, DRIVER_MAPPING, DRIVER_TEAMS
from quali import build_quali_model, generate_latest_data, predict_saudi_gp, get_ordered_qualifying_df

fastf1.Cache.enable_cache("fastf1_cache")

# Fetch 2024 Saudi Arabia race session
race_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
race_2024.load()

# Get laps + sector times
laps_df = race_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_df.dropna(inplace=True)  # Drop incomplete laps

# Turn timedelta into seconds (useful for math later)
for time_col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_df[time_col + " (s)"] = laps_df[time_col].dt.total_seconds()

# Average sector times by driver
sector_avg = laps_df.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_avg["TotalSectorTime (s)"] = (
    sector_avg["Sector1Time (s)"] + sector_avg["Sector2Time (s)"] + sector_avg["Sector3Time (s)"]
)

# Set up driver mappings from config
driver_abbr_map = DRIVER_MAPPING

# Build the qualifying model
quali_model = build_quali_model()
latest_quali = generate_latest_data()

# Predict 2025 qualifying and order them
predict_saudi_gp(quali_model, latest_quali)
qualifying_preds = get_ordered_qualifying_df().copy()

# Tweak column names for consistency
qualifying_preds.rename(columns={"Predicted_Q3": "QualifyingTime (s)"}, inplace=True)
qualifying_preds["DriverAbbr"] = qualifying_preds["Driver"].map(driver_abbr_map)

# Add race skill from config
race_skills = {
    DRIVER_MAPPING[d]: stats['race_skill']
    for d, stats in DRIVER_PERFORMANCE.items()
    if d in DRIVER_MAPPING
}
qualifying_preds["RaceSkill"] = qualifying_preds["DriverAbbr"].map(race_skills)

# Add wet skill from config, only used if rain chance >= 0.75
wet_skills = {
    DRIVER_MAPPING[d]: stats['wet_skill']
    for d, stats in DRIVER_PERFORMANCE.items()
    if d in DRIVER_MAPPING
}
qualifying_preds["WetPerformanceFactor"] = qualifying_preds["DriverAbbr"].map(wet_skills)

# Weather API stuff
API_KEY = "" # USE YOUR OWN API KEY
weather_endpoint = f"http://api.openweathermap.org/data/2.5/forecast?q=Jeddah&appid={API_KEY}&units=metric"
weather_resp = requests.get(weather_endpoint)
weather_json = weather_resp.json()

# Forecast for race time 
target_forecast = "2025-04-20 18:00:00"
forecast_info = next((f for f in weather_json["list"] if f["dt_txt"] == target_forecast), None)

rain_chance = forecast_info["pop"] if forecast_info else 0
temp_celsius = forecast_info["main"]["temp"] if forecast_info else 20

# If heavy rain predicted, adjust qualifying times by using wet performance
if rain_chance >= 0.75:
    qualifying_preds["QualifyingTime (s)"] *= qualifying_preds["WetPerformanceFactor"]

# Driver mapping from config
driver_team_lookup = {
    DRIVER_MAPPING[d]: t
    for d, t in DRIVER_TEAMS.items()
    if d in DRIVER_MAPPING
}

# Team performance from config
team_score_lookup = {
    t: info["race_score"]
    for t, info in TEAM_PERFORMANCE.items()
}

qualifying_preds["Team"] = qualifying_preds["DriverAbbr"].map(driver_team_lookup)
qualifying_preds["TeamPerformance"] = qualifying_preds["Team"].map(team_score_lookup)

# Merge sector times 
sector_avg.rename(columns={"Driver": "DriverAbbr"}, inplace=True)
merged = qualifying_preds.merge(sector_avg[["DriverAbbr", "TotalSectorTime (s)"]], on="DriverAbbr", how="left")

merged["RainProbability"] = rain_chance
merged["Temperature"] = temp_celsius

lap_time_avg = laps_df.groupby("Driver")["LapTime (s)"].mean()
merged["LapTime (s)"] = merged["DriverAbbr"].map(lap_time_avg)

# Clean dataset (drop row if missing LapTime)
final_data = merged.dropna(subset=["LapTime (s)"])

# Features and target for modeling
X = final_data[["QualifyingTime (s)", "RainProbability", "Temperature", "TeamPerformance", "RaceSkill"]]
X = X.dropna()
final_data = final_data.loc[X.index]  
y = final_data["LapTime (s)"]

# Build the race prediction model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
race_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
race_model.fit(X_train, y_train)

# Make race predictions
final_data["PredictedRaceTime (s)"] = race_model.predict(X)

# Display prediction
race_outcome = final_data.sort_values("PredictedRaceTime (s)")
print("\n Predicted 2025 Saudi Arabian GP Results:")
print(race_outcome[["Driver", "PredictedRaceTime (s)"]])

# Error of the model
y_pred_test = race_model.predict(X_test)
print(f"\nModel MAE: {mean_absolute_error(y_test, y_pred_test):.2f} seconds")

# Team performance effect
plt.figure(figsize=(12, 8))
plt.scatter(race_outcome["TeamPerformance"],
            race_outcome["PredictedRaceTime (s)"],
            c=race_outcome["QualifyingTime (s)"], cmap='coolwarm')

for idx, row in race_outcome.iterrows():
    plt.annotate(row["Driver"], (row["TeamPerformance"], row["PredictedRaceTime (s)"]), xytext=(5, 5), textcoords="offset points")

plt.colorbar(label="Qualifying Time (s)")
plt.xlabel("Team Score")
plt.ylabel("Predicted Race Time (s)")
plt.title("Impact of Team Strength on Race Performance")
plt.tight_layout()
plt.savefig("team_performance_effect.png")
plt.show()

# Race skill effect
plt.figure(figsize=(12, 8))
plt.scatter(race_outcome["RaceSkill"],
            race_outcome["PredictedRaceTime (s)"],
            c=race_outcome["QualifyingTime (s)"], cmap='viridis')

for idx, row in race_outcome.iterrows():
    plt.annotate(row["Driver"], (row["RaceSkill"], row["PredictedRaceTime (s)"]), xytext=(5, 5), textcoords="offset points")

plt.colorbar(label="Qualifying Time (s)")
plt.xlabel("Driver Race Skill")
plt.ylabel("Predicted Race Time (s)")
plt.title("Impact of Driver Skill on Race Outcome")
plt.tight_layout()
plt.savefig("race_skill_effect.png")
plt.show()

# Feature Importance plot
feat_importances = race_model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 8))
plt.barh(features, feat_importances, color="skyblue")
plt.xlabel("Impact")
plt.title("Feature Impact")
plt.savefig("feature_impact.png")
plt.tight_layout()
plt.show()
