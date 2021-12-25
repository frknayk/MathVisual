import os
from datetime import datetime
import pandas as pd
import keras
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
import tensorflow as tf

class OccModel:
    def __init__(self) -> None:
        self.model = None
        self.history = None

    def set_model(self, model):
        self.model = model
    
    def train(self,X_train,y_train,train_config):
        # training the model
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=train_config['batch_size'],
            epochs=train_config['epochs'],
            validation_split=train_config['validation_split'],
            shuffle=train_config['shuffle']
        )

    def plot_loss(self):
        plt.figure(1)
        plt.plot(self.history.history['loss'],label='loss_train')
        plt.plot(self.history.history['val_loss'],label='loss_validation')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()

    def plot_accuracy(self):
        plt.figure(1)
        plt.plot(self.history.history['val_accuracy'], label='acc_validation')
        plt.plot(self.history.history['accuracy'], label='acc_train', color="red")
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()

    def show_performance(self,X_data,y_data,prefix=""):
        score, acc = self.model.evaluate(X_data, y_data)
        print('{0}|score: {1}'.format(prefix, score))
        print('{0}|accuracy: {1}'.format(prefix, acc))

    def predict(self,sample):
        return self.model.predict(sample)

    def save_model(self, path_save=None):
        path_current = os.getcwd()
        path_folder = path_current+'/checkpoints'
        path_model = path_folder+"/models"
        path_csv = path_folder+"/histories/"
        try:
            os.mkdir(path_folder)
            os.mkdir(path_model)
            os.mkdir(path_csv)
        except FileExistsError:
            pass
        except Exception as e:
            print(e)
            return
        # Save model
        if path_save is None:
            # Trim date
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(' ','__')
            date = date.replace(':','_')
            date = date.replace('-','_')
            path_save = date
        path_model = path_model + '/' + path_save +'.h5'
        self.model.save(path_model)
        # Save history
        path_csv = path_csv + '/' + path_save + '.csv'      
        hist_df = pd.DataFrame(self.history.history) 
        with open(path_csv, mode='w') as f:
            hist_df.to_csv(f)

    def load_model(self, path:str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(path)
            return True
        except Exception as E:
            print(E)
            return False
