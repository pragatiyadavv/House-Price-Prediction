import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Load Data
file_path = r'House Price.csv'  # Update the file path as necessary
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Handle missing values
df.ffill(inplace=True)  # Forward fill to handle missing values

# Encode Categorical Variables
label_encoders = {}
for column in ['facing', 'furnishDetails', 'property_type']:
    if column in df.columns:
        le = LabelEncoder()
        le.fit(df[column].dropna())  # Fit on non-null values to avoid issues
        df[column] = le.transform(df[column].fillna('Unknown'))  # Encode and handle NaN
        label_encoders[column] = le  # Store the label encoders for future use

# Step 3: Define Features and Target Variable
X = df.drop('price', axis=1)  # Features
y = df['price']  # Target variable

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric data to NaN

# Drop rows with NaN values
X.dropna(inplace=True)  # Drop rows with NaN values
y = y[X.index]  # Align y with the remaining X

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)  # Train the model

# Step 6: Define Prediction Function
def predict_price(bedRoom, bathroom, balcony, floorNum, facing, furnishDetails, property_type, area):
    input_data = pd.DataFrame({
        'bedRoom': [bedRoom],
        'bathroom': [bathroom],
        'balcony': [balcony],
        'floorNum': [floorNum],
        'facing': [facing],
        'furnishDetails': [furnishDetails],
        'property_type': [property_type],
        'area': [area]
    })

    # Encode categorical variables for input
    for column in ['facing', 'furnishDetails', 'property_type']:
        if input_data[column].values[0] in label_encoders[column].classes_:
            input_data[column] = label_encoders[column].transform(input_data[column])
        else:
            input_data[column] = -1  # Assign a default value (or handle as you see fit)

    # Ensure input_data is numeric
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    # Predict price
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Streamlit User Interface
st.title("House Price Prediction")
st.write("Enter the details of the house to get the predicted price.")

# Input fields for the user
bedRoom = st.number_input("Number of Bedrooms", min_value=0)
bathroom = st.number_input("Number of Bathrooms", min_value=0)
balcony = st.number_input("Number of Balconies", min_value=0)
floorNum = st.number_input("Floor Number", min_value=0)
facing = st.selectbox("Facing", ['North', 'South', 'East', 'West'])
furnishDetails = st.selectbox("Furnish Details", ['Furnished', 'Semi-Furnished', 'Unfurnished'])
property_type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Villa', 'Bungalow'])
area = st.number_input("Area (in sq. ft.)", min_value=0)

# Button to make prediction
if st.button("Predict Price"):
    predicted_price = predict_price(bedRoom, bathroom, balcony, floorNum, facing, furnishDetails, property_type, area)
    st.success(f'Predicted Price: ₹{predicted_price:.2f}')
