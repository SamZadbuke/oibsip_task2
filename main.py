import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

categorical_columns = [
    "fueltype",
    "aspiration",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "fuelsystem",
]

# Load and preprocess the data
def load_and_preprocess_data():
    data = pd.read_csv("car_price.csv", encoding='latin1')

    data[["carCompany", "carModel"]] = data["CarName"].str.split(" ", expand=True, n=1)
    data = data.drop(["CarName"], axis=1)

    X = data.drop("price", axis=1)
    y = data["price"]

    numeric_columns = [
        "symboling",
        "doornumber",
        "wheelbase",
        "carlength",
        "carwidth",
        "carheight",
        "curbweight",
        "cylindernumber",
        "enginesize",
        "boreratio",
        "stroke",
        "compressionratio",
        "horsepower",
        "peakrpm",
        "citympg",
        "highwaympg",
    ]

    encoder = OneHotEncoder(drop="first", sparse=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_columns)
    X_numeric = X[numeric_columns]
    X_processed = pd.concat([X_numeric, X_encoded], axis=1)

    return X_processed, y, numeric_columns, data, encoder

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_regressor = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    return y_test, y_pred, rf_regressor  # Return the trained model

# Streamlit app Section
def main():
    st.title("Car Price Estimation Tool")

    st.write("Welcome to the Car Price Estimation Tool!")
    st.write(
        "This tool utilizes a machine learning model to estimate car prices based on a range of features. "
        "Please provide car details, and the tool will generate a price estimate."
    )

    X_processed, y, numeric_columns, data, encoder = load_and_preprocess_data()
    y_test, y_pred, rf_regressor = train_model(X_processed, y)  # Get the trained model

    # Graph Section
    for i in range(0, len(numeric_columns), 2):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for j, feature in enumerate(numeric_columns[i : i + 2]):
            axes[j].scatter(data["price"], data[feature], color="blue")
            axes[j].set_xlabel("Car Price ($)")
            axes[j].set_ylabel(feature)
            axes[j].set_title(f"{feature} vs Car Price")
        st.pyplot(fig)

    st.markdown(
        "<h3 style='text-align: center;'>R-squared Score: {:.3f}</h2>".format(
            r2_score(y_test, y_pred)
        ),
        unsafe_allow_html=True,
    )

    # Prediction Section
    st.markdown(
        "<h3 style='text-align: center;'>Enter Car Details:</h3>", unsafe_allow_html=True
    )
    input_features = {}
    col1, col2 = st.columns(2)
    for feature in numeric_columns:
        input_features[feature] = col1.number_input(f"Enter {feature}", value=0)

    for feature in data.select_dtypes(include='object').columns:
        input_features[feature] = col2.selectbox(
            f"Select {feature}", data[feature].unique()
        )

    input_df = pd.DataFrame(input_features, index=[0])

    st.write("User Input:")
    st.write(input_df)

    input_encoded = pd.DataFrame(encoder.transform(input_df[categorical_columns]))
    input_encoded.columns = encoder.get_feature_names_out(categorical_columns)
    input_numeric = input_df[numeric_columns]
    input_processed = pd.concat([input_numeric, input_encoded], axis=1)

    predicted_price = rf_regressor.predict(input_processed)

    st.markdown(
        "<h3 style='text-align: center;'>Predicted Car Price:</h3>", unsafe_allow_html=True
    )
    st.write(
        f"<h1 style='text-align: center; font-size: 36px; color: red;'>${predicted_price[0]:,.2f}</h1>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
