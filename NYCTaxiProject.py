import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Apply the custom theme with corrected CSS
# streamlit run "NYCTaxiProject.py"

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f2f2f2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


@st.cache
def get_data():
    # Path to your Parquet file
    file_path = 'green_tripdata_2023-01.parquet'
    # Read the Parquet file
    data = pd.read_parquet(file_path, engine='pyarrow')
    return data

with header:

    st.image("taxi_pic01.jpg", width=250)
    st.title("NYC Taxi Project")
    st.text("In this Project of NYC Taxi...")

with dataset:
    st.header("NYC Taxi Dataset")

    taxi_data = get_data()
    st.write(taxi_data.head())

    sel_col2, = st.columns(1)  # Correctly unpack the single column object

    st.subheader("Feature Distribution Plot")
    # Add a unique key for this selectbox
    input_plot_feature = sel_col2.selectbox("Select the feature for model input", options=taxi_data.columns, key='input_plot_feature')

    pulocation_disy = pd.DataFrame(taxi_data[input_plot_feature].value_counts())  # Removed .head(50) for full distribution
    st.bar_chart(pulocation_disy)


with features:
    st.header("Features")
    st.markdown('* ** first feature:** i created asdad')
    st.markdown('* ** first feature:** i created asdad')

with model_training:
    st.header("Model Training")
    st.text("Choose Hyperparameter")

    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider("What should be the max depth of the model?", min_value=10, max_value=100, value=20,
                               step=10)

    n_estimators = sel_col.selectbox("How many trees should be?", options=[100, 200, 300, "No limit"], index=0)
    sel_col.text("List of Features Data")
    # sel_col.write(taxi_data.columns)
    input_feature = sel_col.selectbox("Select the feature for model input", options=taxi_data.columns)

    if n_estimators == "No limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(X)
    disp_col.subheader("Mean Absolute Error of the model is:")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean Squared Error of the model is:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R Squared Error of the model is:")
    disp_col.write(r2_score(y, prediction))
