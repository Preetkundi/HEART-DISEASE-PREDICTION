import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load and Preprocess the Dataset

# Sample dataset path (you can replace this with your own dataset)
data = 'heart_disease_data.csv'
# Load the dataset into pandas DataFrame

df = pd.read_csv(data)

# Step 2: Handle Missing Data (if any)
# Check for missing values
df.isnull().sum()

# No missing values, but you could drop rows with missing values like:
# df.dropna(inplace=True)

# Step 3: Encode Categorical Variables
# Label encode columns where needed (e.g., 'sex', 'cp', 'restecg', etc.)
encoder = LabelEncoder()

df['sex'] = encoder.fit_transform(df['sex'])
df['cp'] = encoder.fit_transform(df['cp'])
df['fbs'] = encoder.fit_transform(df['fbs'])
df['restecg'] = encoder.fit_transform(df['restecg'])
df['exang'] = encoder.fit_transform(df['exang'])
df['slope'] = encoder.fit_transform(df['slope'])
df['ca'] = encoder.fit_transform(df['ca'])
df['thal'] = encoder.fit_transform(df['thal'])

# Step 4: Feature Scaling
scaler = StandardScaler()

# Features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Scale the features
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Streamlit App for Prediction

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Streamlit app
st.title("Heart Disease Prediction")

st.write("""
This app predicts whether a person has heart disease based on the following input features:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise induced angina
- ST depression induced by exercise
- Slope of peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia type

Please fill in the details below to get the prediction.
""")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")

# Convert input values into a DataFrame
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
                                   'oldpeak', 'slope', 'ca', 'thal'])

# Scale the input data using the same scaler
input_data_scaled = scaler.transform(input_data)

# Button to make prediction
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction == 1:
        st.write("The model predicts: **Heart Disease Positive**")
    else:
        st.write("The model predicts: **Heart Disease Negative**")

# Display the raw input data (optional, for transparency)
st.write("Your Input Data:")
st.write(input_data)
