# EPITECH PROJECT, 2026 - TARDIS

## Train Delay Analysis and Prediction

The goal of this project is to analyze train delay data, build a regression model to predict delays, and display everything through an interactive dashboard.

## Project Structure

- dataset.csv: initial dataset
- cleaned_dataset.csv: output from the tardis_eda notebook
- model.pkl: trained Random Forest model
- tardis_eda.ipynb: data exploration, cleaning and feature engineering
- tardis_model.ipynb: model training, evaluation and selection
- tardis_dashboard.py: Streamlit dashboard
- requirements.txt: project dependencies
- model_report.md: detailed model results
- README.md: project documentation, setup, and usage instructions

## Setup

Install the required dependencies with: 
python3 -m pip install -r requirements.txt

### Run the Streamlit dashboard:

To start the dashboard, run: 
python3 -m streamlit run tardis_dashboard.py

## tardis_eda.ipynb - Data Cleaning and Exploration

- loads dataset.csv and inspects its structure
- removes duplicate rows
- converts the "Date" column to datetime format
- converts numeric columns stored as strings (using ',' as a decimal separator)
- fills missing values for categorical columns and drops rows with missing key fields
- creates new features: 
    Temporal features (Year, Month, Month name, Quarter, Season)
    Route (Route: departure station -> arrival station)
    Rates (cancellation rate, Departure delay rate, Arrival delay rate)
    Category (Delay severity: Low/Medium/High, based on arrival delay rate)
- exports the cleaned data to cleaned_dataset.csv
- includes visualization: delay distribution histogram, boxplot and top 10 stations by average delay

## tardis_model.ipynb - Predictive Modeling

- loads cleaned_dataset.csv
- selects features: Service, Departure station, Arrival station, Month, Year, Number of scheduled trains
- applies One-Hot Encoding to categorical columns
- splits data: 80% training/ 20% testing
- sets a baseline model (mean prediction) to have a reference point for comparison
- trains two models: Linear Regression and Random Forest Regressor
- evaluates both using MAE and R2 score
- applies GridSearchCV
- compares standard vs. tuned Random Forest performance
- displays the top 10 most important features
- saves the model to model.pkl

## tardis_dashboard.py - Streamlit Dashboard

The Dashboard has several sections:
- sidebar filters: filter data by service type and month
- summary statistics: average arrival delay, total trips, punctuality rate
- delay distribution: histogram and boxplot of arrival delays
- station-level analysis: bar chart of top 10 departure stations by average delay
- correlation heatmap: relationship between delay-rated numeric variables
- prediction interface: fill in journey details (service, stations, month, year, number of trains) and get a predicted delay in minutes
- model explanation: top 10 feature importances from the Random Forest model

## model_report.md - Model Report

Contains a detailed breakdown of the modeling process: evaluations for each model (MAE, RMSE, R2), justification for the feature selection and an explanation of the hyperparameter tuning process.
The selected model is a tuned Random Forest Regressor, which achieved the lowest MAE across all tested models.

## Note: Code is formatted with ruff as required.
