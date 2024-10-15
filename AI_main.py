# Importing modules 
# | --------------------------------------------------------------------------|
import pandas as pd
import TL_datamanagement as TL_D
import AI_model as AI_M
import tensorflow as tf
import matplotlib.pyplot as plt
# | --------------------------------------------------------------------------|


# Data management 
# | --------------------------------------------------------------------------|
tickers = list(pd.read_excel("C:/Sauvegarde/Trading_house/Ref_data.xlsx")["TICKER"])
fields =  ["PX_LAST", "PX_OPEN","LOW", 'HIGH']

df = TL_D.get_historical_data(tickers, fields, "20140101", "20240901") 

# For the sake of avoiding multiple BBG request
#df.to_excel("C:/Sauvegarde/Trading_house/AI_for_trading/df_price.xlsx")

#### Getting data
df = pd.read_excel("C:/Sauvegarde/Trading_house/AI_for_trading/df_price.xlsx")
df_data = df['CAC Index'].iloc[2:].bfill()

# | --------------------------------------------------------------------------|

# Creating Deep neural network model 
# | --------------------------------------------------------------------------|

#### model parameters
time_horizon = 10
time_target = 1
offset = 0

#### Creating Neural Network model 

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (time_horizon, 1)))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))
    
# Checking NN Model
model.summary()

# Creating model
NN_model = AI_M.neural_network(model, time_horizon, time_target, offset)

# Adding data
NN_model.add_data(df_data)
   
# Compile and fit 
history = NN_model.compile_and_fit()
   
# Model prediction
pred = NN_model.make_prediction()    
# | --------------------------------------------------------------------------|

# Checking predicted results
# | --------------------------------------------------------------------------|
plt.plot(pred[:100], color = 'blue', marker = '.', markersize=2, linewidth=1)
plt.plot(NN_model.Y_test[:100], color = 'red', marker = '.', markersize=2, linewidth=1)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('CAC Index Forecasting with CNN')
plt.show()
# | --------------------------------------------------------------------------|