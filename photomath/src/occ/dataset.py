import os 
import random
from cv2 import data
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2

symbols_list = ['zero','one','two','three','four','five','six','seven','eight','nine','minus','plus','equal','div','decimal','times']

class DatasetLoader:
    """Dataset loader class for numbers+symbols dataset
    ---
    Download the dataset from here :https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset
    """
    def __init__(self, dataset_path:str) -> None:
        self.train_image_paths, self.train_label_paths = self.__set_train_images(dataset_path) 
        self.test_image_paths, self.test_label_paths = self.__set_test_images(dataset_path)
        self.__preprocess_data()

    def create_train_test_data(self):
        """Create train/test dataset

        Returns
        -------
        X_train,X_test
            Returns train and test images respectively
        """
        X_train, X_test = self.__create_train_test_data(self.train_image_paths, self.test_image_paths)
        X_train, X_test = self.__preprocess_data(X_train, X_test)
        return X_train, X_test

    def create_labels(self):
        """Create train/test labels

        Returns
        -------
        y_train, y_test
            Returns training and testing labels respectively
        """
        y_train, y_test = self.__create_labels(self.train_label_paths, self.test_label_paths)
        return y_train, y_test

    def show_random_sample(self):
        """Plot a random dataset sample"""
        random_idx = random.randint(0,self.train_image_paths.__len__())
        image = cv2.imread(self.train_image_paths[random_idx])
        plt.imshow(image)
        plt.title("Label: " + self.train_label_paths[random_idx])
        plt.show()

    def __set_test_images(self, path_dataset):
        """Set test image names and train label names  

        Returns
        -------
        bool
            True if lists are set correctly
        """
        test_label_paths = []
        test_image_paths = []
        eval_path = path_dataset+'/eval'
        for symbols_dir in os.listdir(eval_path):
            if symbols_dir.split()[0] in symbols_list:
                for image in os.listdir(eval_path + "/" + symbols_dir):
                    test_label_paths.append(symbols_dir.split()[0])
                    test_image_paths.append(eval_path + "/" + symbols_dir + "/" + image)
        return test_label_paths, test_image_paths

    def __set_train_images(self, path_dataset):
        """Set train image names and train label names  

        Returns
        -------
        bool
            True if lists are set correctly
        """
        train_label_paths = []
        train_image_paths = []
        train_path = path_dataset+'/train'
        for symbols_dir in os.listdir(train_path):
            if symbols_dir.split()[0] in symbols_list:
                for image in os.listdir(train_path + "/" + symbols_dir):
                    train_label_paths.append(symbols_dir.split()[0])
                    train_image_paths.append(train_path + "/" + symbols_dir + "/" + image)
        return train_label_paths, train_image_paths
    
    def __create_train_test_data(self, train_image_paths:str, test_image_paths:str):
        """loading the images from the path"""
        X_train_ = []
        X_test_ = []
        for path in train_image_paths:    
            img = cv2.imread(path)
            img = cv2.resize(img, (100, 100))
            img = np.array(img)
            X_train_.append(img)
        for path in test_image_paths:    
            img = cv2.imread(path)
            img = cv2.resize(img, (100, 100))
            img = np.array(img)     
            X_test_.append(img)
        return X_train_, X_test_
    
    def __create_labels(self, train_label_paths:str, test_label_paths:str):
        """Encode labels"""
        label_encoder = preprocessing.LabelEncoder()
        y_train_temp = label_encoder.fit_transform(train_label_paths)
        y_test_temp = label_encoder.fit_transform(test_label_paths)
        y_train_ = keras.utils.np_utils.to_categorical(y_train_temp, 16)
        y_test_ = keras.utils.np_utils.to_categorical(y_test_temp, 16)
        return y_train_, y_test_

    def __preprocess_data(self,X_train, X_test):
        """label encoding the 16 symbols"""
        # Normalize data
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        return X_train, X_test


# fig, axs= plt.subplots(2,5, figsize=[24,12])
# count=0
# for i in range(2):    
#     for j in range(5):  
#         image = cv2.imread(test_image[count + count*100])
#         img = cv2.resize(image, (100, 100))        
#         img = np.array(img)
#         img = np.expand_dims(img, axis=0)
#         img = img.astype('float32')
#         img /= 255
#         pred = model.predict(img)        
#         result = np.argsort(pred)  
#         result = result[0][::-1]
#         final_label = label_encoder.inverse_transform(np.array(result))
#         axs[i][j].imshow(image)
#         axs[i][j].set_title(str("Prediction: " + final_label[0]), fontsize = 14)        
#         count += 1
# plt.suptitle("All predictions are shown in title", fontsize = 18)
# plt.show()
