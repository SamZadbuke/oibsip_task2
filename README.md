# Car Price Estimation Tool

This is a Streamlit web application that estimates car prices based on a range of features using a Random Forest Regressor model. Users can input various car details, and the tool will generate a price estimate.

## Table of Contents

- [Getting Started](#getting-started)
- [Features](#features)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Getting Started

To run this application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/car-price-estimation.git
  ```

2. Navigate to the project directory:

   ```bash
   cd car-price-estimation
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open your web browser and go to `http://localhost:8501` to use the application.

## Features

- User-friendly web interface for car price estimation.
- Visualization of the relationship between car features and prices.
- Estimates car prices based on user-provided information.
- Provides the R-squared score as a measure of model performance.

## Usage

1. Upon launching the application, you will see a web interface with an explanation of how the tool works.

2. Enter the car details in the input fields provided. Input includes both numeric and categorical features.

3. Click the "Calculate Price" button to generate a price estimate based on the provided information.

4. The estimated car price will be displayed on the screen.

## Dependencies

- Python 3.7+
- Pandas
- Streamlit
- Matplotlib
- Scikit-learn

You can install the required dependencies using the provided `requirements.txt` file.
