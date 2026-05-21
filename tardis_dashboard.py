import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="TARDIS Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)


df = load_data()
model = load_model()

st.title("TARDIS Dashboard")
st.write("Train delay analysis and prediction dashboard")

# Summary Statistics Panel
st.header("Summary Statistics")

average_delay = df["Average delay of all trains at arrival"].mean()
total_trips = len(df)
punctuality_rate = (df["Average delay of all trains at arrival"] <= 5).mean() * 100

col1, col2, col3 = st.columns(3)

col1.metric("Average Arrival Delay", f"{average_delay:.2f} min")
col2.metric("Total Trips", total_trips)
col3.metric("Punctuality Rate", f"{punctuality_rate:.1f}%")

# Delay Distribution Visualization
st.header("Delay Distribution")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["Average delay of all trains at arrival"].dropna(), bins=20)
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

