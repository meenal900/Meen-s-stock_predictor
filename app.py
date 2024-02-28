import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import yfinance as yf
import streamlit as st

# Specify the ticker symbol and the timeframe

import streamlit as st
import time

def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

# Example usage
text = "Hey! Interested in stock's trend? Let's explore and ride the wave together!"
speed = 10
typewriter(text=text, speed=speed)

start_date = "2010-01-01"
end_date = "2024-02-26"
st.title("MEEN's STOCK TREND PREDICTOR")

user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start_date, end=end_date)

st.subheader("Data from 2010 to 2024")
st.write(df.describe())

#VISUALIZATIONS

st.subheader('Closing time VS Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing time VS Time chart with 100MA')
fig=plt.figure(figsize=(12,6))
ma100=df.Close.rolling(100).mean()
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)

st.subheader('Closing time VS Time chart with 100MA & 200MA')
fig=plt.figure(figsize=(12,6))
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)

#splitting data into training and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)


#load model
model=load_model('keras_model.h5')

#testing part

past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range (100 , input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test=np.array(x_test), np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_
scale_factor= 1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor



st.subheader('Predicted VS Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)