# AI_model
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class neural_network():
    
    def __init__(self, model, time_horizon, time_target, offset):
        self.model = model
        self.data = None
        self.X = []
        self.Y = []
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.time_horizon = time_horizon
        self.time_target = time_target
        self.offset = offset
        
        
    def add_data(self, df_data):
        self.data = df_data
    
    def get_sliding_window(self):
        assert len(self.data) > 0

        for i in range(len(self.data) - self.time_horizon - self.offset):
            tmp_X = list(self.data.iloc[i:i + self.time_horizon].values)
            tmp_Y = list(self.data.iloc[i + self.time_horizon + self.offset : i + self.time_horizon + self.offset + self.time_target].values)

            self.X.append(tmp_X)
            self.Y.append(tmp_Y[0])
               
        return self.X, self.Y
    
    def prepare_data(self, test_size = 0.2, train_size = 0.8):
        self.X, self.Y = self.get_sliding_window()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
                                                                                self.Y,
                                                                                shuffle = False,
                                                                                test_size = 0.2,
                                                                                train_size = 0.8)
    
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.Y_train = np.array(self.Y_train)
        self.Y_test = np.array(self.Y_test)
        
        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def compile_and_fit(self):
      
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.prepare_data()
        
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])
       
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=5,
                                                          mode='min')
        
        history = self.model.fit(self.X_train, 
                                 self.Y_train,
                                 batch_size = 20,
                                 epochs = 200,
                                 callbacks=[early_stopping])

        return history

    def make_prediction(self, X = None):
        if X : 
            return self.model.predict(X)
        else : 
            return self.model.predict(self.X_test)


if __name__ == "__main__" :
    
    #### Getting data
    df = pd.read_excel("C:/Sauvegarde/Trading_house/AI_for_trading/df_price.xlsx")
    df_data = df['CAC Index'].iloc[2:].bfill()
    time_horizon = 10
    time_target = 1
    offset = 0
    
    #### Creating Neural Network model 
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape = (time_horizon,)))
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1))
        
    # Checking NN Model
    model.summary()
    
    # Creating model
    NN_model = neural_network(model, time_horizon, time_target, offset)
    
    # Adding data
    NN_model.add_data(df_data)
   
    # Compile and fit 
    history = NN_model.compile_and_fit()
   
    # Model prediction
    pred = NN_model.make_prediction()    
    
    # Checking predicted results
    import matplotlib.pyplot as plt
    plt.plot(pred[:100], color = 'blue', marker = '.', markersize=2, linewidth=0)
    plt.plot(NN_model.Y_test[:100], color = 'red', marker = '.', markersize=2, linewidth=0)
    plt.show()
    
    
    # Using LSTM : 
    
        
    
    
    
    
    time_horizon = 10
    nb_batches = 100
    
    X, Y = get_LSTM_window(df_data, nb_batches, time_horizon)
    np.shape(X)
    np.shape(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle = False, test_size = 0.2, train_size = 0.8)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
        
     
    model = tf.keras.Sequential()
   
    model.add(tf.keras.Input(shape = (np.shape(X_train)[1], np.shape(X_train)[2], )))
    model.add(tf.keras.layers.LSTM(128, return_sequences = True)), # , return_sequences = True when stacking LSTM
    model.add(tf.keras.layers.LSTM(128, return_sequences = False)), # , return_sequences = True when stacking LSTM
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    
    model.compile(loss=tf.losses.MeanSquaredError(),
                 optimizer=tf.optimizers.Adam(lr=0.1),
                 metrics=[tf.metrics.MeanAbsoluteError()])
    
    history = model.fit(X_train, 
                        Y_train,
                        batch_size = 100,
                        epochs = 1000,
                        )

    pred = model.predict(X_test)    
    
    import matplotlib.pyplot as plt
    plt.plot(pred[:100], color = 'blue', marker = '.', markersize=2, linewidth=0)
    plt.plot(Y_test[:100], color = 'red', marker = '.', markersize=2, linewidth=0)
    plt.show()
    def get_LSTM_window(df_data, nb_batches = 100, time_horizon = 5, time_target = 1, offset = 0):
        # Shape (Nbdays, TimeSteps, FeaturesPerStep).
        X, X_tmp, Y, Y_tmp = [], [], [], []
        for d in np.array_split(df_data, 27) :
            for i in range(len(d) - time_horizon - offset):
                tmp_X = list(d.iloc[i:i + time_horizon].values)
                tmp_Y = list(d.iloc[i + time_horizon + offset : i + time_horizon + offset + time_target].values)

                X_tmp.append(tmp_X)
                Y_tmp.append(tmp_Y[0])
                
            X.append(X_tmp)
            Y.append(Y_tmp)
        return X, Y