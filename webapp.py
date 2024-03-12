## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

## Load the trained model (assuming the model is saved as 'crop_recommendation_model.joblib')
#model = joblib.load('crop_recommendation_model.joblib')

# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df= pd.read_csv('Crop_recommendation.csv')



X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']


# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)





import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RF.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


#model = pickle.load(open('RF.pkl', 'rb'))

RF_Model_pkl=pickle.load(open('RF.pkl','rb'))

## Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # # Making predictions using the model
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

## Streamlit code for the web app interface
def main():
    # # Setting the title of the web app
    st.title("Crop Recommendation Web App")
    
    # # Input fields for the user to enter the environmental factors
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=0.0, step=0.1)
    inputs=[[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    # # Button to make predictions
    if st.sidebar.button("Predict"):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"The recommended crop is: {prediction[0]}")

## Running the main function
if __name__ == '__main__':
    main()
