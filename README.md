# F1 Race Prediction Model

A machine learning project to predict Formula 1 qualifying and race results using historical data and driver/team performance metrics.

## Overview

This project predicts Formula 1 race outcomes in two stages:
1. **Qualifying Prediction**: Predicts Q3 times based on historical Q1/Q2 data and team/driver performance metrics
2. **Race Prediction**: Uses the qualifying grid as input along with driver skills and team performance to predict race results

The model is currently specifically tuned for the Saudi Arabian Grand Prix but can be adapted for other circuits.

## Features

- **Data-driven predictions** using FastF1 API and historical race data
- **Comprehensive performance metrics** for teams and drivers
- **Weather-aware modeling** that adjusts predictions based on rain probability
- **Visual analysis** of feature importance and factor relationships
- **Race simulation** based on qualifying, driver skill, and team performance

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- fastf1
- requests

## Project Structure

- **config.py**: Configuration with team and driver performance metrics
- **quali.py**: Qualifying prediction model
- **race.py**: Race prediction model that builds on qualifying results

## How It Works

### Performance Metrics (config.py)

The configuration file contains carefully calibrated performance metrics for teams and drivers:

- **Team Performance**:
  - `quali_adjust`: Time adjustment for qualifying (in seconds)
  - `race_score`: Race performance multiplier (lower is better)

- **Driver Performance**:
  - `quali_adjust`: Individual qualifying boost
  - `race_skill`: Race pace multiplier (higher is better)
  - `wet_skill`: Wet weather race pace multiplier (higher is better)

These metrics were derived from analyzing historical race data, qualifying gaps, and race pace differentials between teams and drivers. The values represent relative performance differences rather than absolute measures and have been fine-tuned based on observed performance patterns.

### Qualifying Prediction (quali.py)

The qualifying prediction model:

1. Fetches historical qualifying data (2024-2025) using FastF1 API
2. Cleans and processes qualifying session times
3. Builds a Linear Regression model based on Q1 and Q2 times to predict Q3
4. Applies team and driver performance adjustments
5. Outputs predicted Q3 times for each driver

### Race Prediction (race.py)

The race prediction model:

1. Uses the qualifying results as starting input
2. Incorporates driver race skill and team performance metrics
3. Factors in weather conditions (temperature and rain probability)
4. Uses a Gradient Boosting Regressor to predict race lap times
5. Visualizes relationships between skills, team performance, and outcomes

## Visualizations

The model generates three key visualizations:

1. **Feature Importance**: Shows which factors most influence race predictions
   
![Image](https://github.com/user-attachments/assets/6273901c-bcad-486f-9517-9ff757ef5d80)
  
2. **Driver Skill Impact**: Visualizes the relationship between driver skill and race time
3. **Team Strength Impact**: Shows how team performance affects race outcomes

## Results

For the Saudi Arabian GP 2025, the model predicts qualifying and race results with reasonable accuracy:

- Qualifying prediction accuracy: MAE of ~1.24 seconds
- Race prediction accuracy: MAE of ~1.54 seconds

## Limitations and Future Work

- The model is primarily tuned for the Saudi Arabian circuit
- Weather prediction relies on a simple API and could be enhanced
- Driver-specific circuit performance isn't fully accounted for
- More granular race strategy modeling could improve predictions
