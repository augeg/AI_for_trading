# AI_model
import tensorflow as tf
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



    