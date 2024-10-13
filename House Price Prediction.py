import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Load Data
file_path = r'\House Price.csv'  # Update the file path as necessary
df = pd.read_csv(file_path)

# Debugging: Check the loaded data
print(df.head())
print(df.info())
print("Missing values before handling:")
print(df.isnull().sum())

# Step 2: Data Preprocessing
# Handle missing values
df.ffill(inplace=True)  # Forward fill to handle missing values
print("Missing values after handling:")
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Step 3: Encode Categorical Variables
label_encoders = {}
for column in ['facing', 'furnishDetails', 'property_type']:
    if column in df.columns:
        le = LabelEncoder()
        le.fit(df[column].dropna())  # Fit on non-null values to avoid issues
        df[column] = le.transform(df[column].fillna('Unknown'))  # Encode and handle NaN
        label_encoders[column] = le  # Store the label encoders for future use
        print(f"Unique values in '{column}': {df[column].unique()}")
    else:
        print(f"Column '{column}' does not exist in the DataFrame.")

# Step 4: Define Features and Target Variable
X = df.drop('price', axis=1)  # Features
y = df['price']  # Target variable

# Step 5: Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric data to NaN

# Step 6: Drop rows with NaN values (if any remain)
print(f'Shape of X before NaN removal: {X.shape}')
print(f'Shape of y before NaN removal: {y.shape}')

X.dropna(inplace=True)  # Drop rows with NaN values
y = y[X.index]  # Align y with the remaining X

print(f'Shape of X after NaN removal: {X.shape}')
print(f'Shape of y after NaN removal: {y.shape}')

# Check if either is empty
if X.empty or y.empty:
    print("Warning: One of the DataFrames is empty after preprocessing.")
else:
    # Step 7: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 8: Train the Model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)  # Train the model

    # Step 9: Define Prediction Function
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
                print(f"Warning: '{input_data[column].values[0]}' is an unseen label in '{column}'. Using default encoding.")
                input_data[column] = -1  # Assign a default value (or handle as you see fit)

        # Ensure input_data is numeric
        input_data = input_data.apply(pd.to_numeric, errors='coerce')

        # Predict price
        predicted_price = model.predict(input_data)
        return predicted_price[0]

    # Example Usage of the Prediction Function
    predicted_price = predict_price(2, 2, 1, 1, 'North', 'Furnished', 'Apartment', 1000)
    print(f'Predicted Price: {predicted_price}')
