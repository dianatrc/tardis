import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="TARDIS Dashboard", layout="wide")


@st.cache_data
def load_data():
    """
    Load the cleaned dataset from .csv file.
    Returns:
        The loaded dataset.
    """
    return pd.read_csv("cleaned_dataset.csv")


@st.cache_resource
def load_model():
    """
    Load the pre-trained Random Forest model from .pkl file.
    Returns:
        The trained regression model.
    """
    with open("model.pkl", "rb") as file:
        return pickle.load(file)


df = load_data()
model = load_model()

st.title("TARDIS Dashboard")
st.write("Train delay analysis and prediction dashboard")

# Interactive filters
st.sidebar.header("Interactive Filters")
st.sidebar.write("Use the filters to get specific data.")

# Available options
service_options = sorted(df["Service"].dropna().unique())
month_options = sorted(df["Month"].dropna().unique())

# Show filters
selected_services = st.sidebar.multiselect("Select service type", service_options, default=service_options)
selected_months = st.sidebar.multiselect("Select month", month_options, default=month_options)

# Create a new filtered dataframe
filtered_df = df[
    (df["Service"].isin(selected_services)) &
    (df["Month"].isin(selected_months))
]

# Dynamic Summary Statistics
st.header("Summary Statistics")

average_delay = filtered_df["Average delay of all trains at arrival"].mean()
total_trips = len(filtered_df)

if total_trips > 0:
    punctuality_rate = (filtered_df["Average delay of all trains at arrival"] <= 5).mean() * 100
else:
    punctuality_rate = 0

col1, col2, col3 = st.columns(3)

col1.metric("Average Arrival Delay", f"{average_delay:.2f} min")
col2.metric("Total Trips", total_trips)
col3.metric("Punctuality Rate", f"{punctuality_rate:.1f}%")

# Delay Distribution Visualization
st.header("Delay Distribution")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["Average delay of all trains at arrival"].dropna(), bins=20, color='skyblue', edgecolor='black')
ax.set_title("Distribution of Average Arrival Delay")
ax.set_xlabel("Average Arrival Delay (minutes)")
ax.set_ylabel("Frequency")

st.pyplot(fig)

# Arrival Delay Boxplot
st.subheader("Arrival Delay Boxplot")

fig2, ax2 = plt.subplots(figsize=(8, 2.5))
ax2.boxplot(df["Average delay of all trains at arrival"].dropna(), vert=False)
ax2.set_title("Boxplot of Average Arrival Delay")
ax2.set_xlabel("Average Arrival Delay (minutes)")

st.pyplot(fig2)

# Station level comparative analysis
st.header("Station-level comparative analysis")

top_stations = (
    filtered_df.groupby("Departure station")["Average delay of all trains at arrival"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig_stations, ax_stations = plt.subplots(figsize=(10, 6))
top_stations.sort_values().plot(kind="barh", ax=ax_stations, color="lightcoral", edgecolor='black')
ax_stations.set_title("Top 10 Departure Stations by Average Arrival Delay")
ax_stations.set_xlabel("Average Arrival Delay (minutes)")
ax_stations.set_ylabel("Departure Station")
st.pyplot(fig_stations)

# Correlation Heatmap
st.header("Correlation Heatmap")
st.write("Visual representation of relationships between numeric delay factors.")

numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
key_numeric_cols = [col for col in numeric_cols if "Average delay" in col or "Number" in col]

if len(key_numeric_cols) > 0:
    corr_matrix = filtered_df[key_numeric_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax_corr, fmt=".2f")
    ax_corr.set_title("Correlation Between Delay Factors")
    st.pyplot(fig_corr)
else:
    st.write("Not enough numerical data to display correlation.")

# Prediction Interface
st.header("Prediction Interface")

service_options = sorted(df["Service"].dropna().unique())
departure_options = sorted(df["Departure station"].dropna().unique())
arrival_options = sorted(df["Arrival station"].dropna().unique())
month_options = sorted(df["Month"].dropna().unique())
year_options = sorted(df["Year"].dropna().unique())

service = st.selectbox("Service", service_options)
departure_station = st.selectbox("Departure station", departure_options)
arrival_station = st.selectbox("Arrival station", arrival_options)
month = int(st.selectbox("Month", month_options))
year = int(st.selectbox("Year", year_options))
scheduled_trains = st.number_input(
    "Number of scheduled trains",
    min_value=0,
    value=10,
    step=1
)

if st.button("Predict Delay"):
    input_df = pd.DataFrame({
        "Service": [service],
        "Departure station": [departure_station],
        "Arrival station": [arrival_station],
        "Month": [month],
        "Year": [year],
        "Number of scheduled trains": [int(scheduled_trains)]
    })

    input_encoded = pd.get_dummies(
        input_df,
        columns=["Service", "Departure station", "Arrival station"],
        drop_first=True
    )

    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.success(f"Predicted arrival delay: {prediction:.2f} minutes")

# Model Explanation
st.header("Model Explanation")

feature_importance = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
).sort_values(ascending=False)

st.subheader("Top 10 Feature Importances")

fig3, ax3 = plt.subplots(figsize=(10, 6))
feature_importance.head(10).sort_values().plot(
    kind="barh",
    ax=ax3
)
ax3.set_title("Top 10 Feature Importances in Random Forest")
ax3.set_xlabel("Importance Score")
ax3.set_ylabel("Feature")

st.pyplot(fig3)

st.write(
    "This section explains which variables have the strongest influence on the "
    "Random Forest model when predicting average arrival delay. Higher feature "
    "importance means that the model relies more on that variable during prediction."
)
