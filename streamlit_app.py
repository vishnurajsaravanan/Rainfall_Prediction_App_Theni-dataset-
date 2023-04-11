import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("Rainfall Prediction App")

dat = st.selectbox("Choose area",options=['bodi','cumbum','gandamanur','gudalore','manjalardam','mayiladumparai','periakulam','utamapalayam','vaigaidam','veerapandi'])

if dat == 'bodi':
    df = pd.read_csv("Sample Data/bodi.csv") 
elif dat == 'cumbum':
        df= pd.read_csv("Sample Data/cumbum.csv")
elif dat == 'gandamanur':
        df= pd.read_csv("Sample Data/gandamanur.csv")
elif dat == 'manjalardam':
        df= pd.read_csv("Sample Data/manjalardam.csv")
elif dat == 'gudalore':
        df= pd.read_csv("Sample Data/gudalore.csv")
elif dat == 'mayiladumparai':
        df= pd.read_csv("Sample Data/mayiladumparai.csv")
elif dat == 'periakulam':
        df= pd.read_csv("Sample Data/periakulam.csv")
elif dat == 'utamapalayam':
        df= pd.read_csv("Sample Data/utamapalayam.csv")
elif dat == 'vaigaidam':
        df= pd.read_csv("Sample Data/vaigaidam.csv")
elif dat == 'veerapandi':
        df = pd.read_csv("Sample Data/veerapandi.csv")

st.header('We are using **RandomForestRegressor**')

X_train = df.iloc[:, 0:1].values
y_train = df.iloc[:, 1].values

data = st.slider('Enter year', 2018, 2030, 2018)

X_test = np.array(data).reshape(-1,1)

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, color='blue', label='Historical')
ax.plot(X_test, y_pred, color='red', label='Predicted')
ax.set_title('Rainfall Prediction')
ax.set_xlabel('Year')
ax.set_ylabel('Rainfall (mm)')
ax.legend()
st.pyplot(fig)

st.write('# The predicted value for the year',data,'is', y_pred)