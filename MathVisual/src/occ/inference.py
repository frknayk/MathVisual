import os
import argparse
import numpy as np
import cv2
from MathVisual.src.occ.model import OccModel
from MathVisual.src.occ.dataset import DatasetLoader, symbols_list
import tensorflow as tf

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test trained model with dataset')
    parser.add_argument('-path', '--model-path', default="MathVisual/src/occ/model_200epoch_custom.h5")
    parser.add_argument('-dataset', '--dataset-path')
    args = parser.parse_args()

    # Create the dataset loader object
    path_dataset = args.dataset_path
    dataset_loader = DatasetLoader(path_dataset)
    X_train,X_test = dataset_loader.create_train_test_data()
    y_train, y_test = dataset_loader.create_labels()
    y_train_labels, y_test_labels = dataset_loader.get_labels()

    # Load model
    occ_model = OccModel()
    occ_model.load_model(args.model_path)

    # Create label encoder
    label_encoder = dataset_loader.set_label_encoder(symbols_list)

    # Select an image
    random_indices = np.random.random_integers(0,y_train_labels.__len__(),(y_train_labels.__len__(),))
    for idx in random_indices:
        label_true = y_train_labels[idx]
        img_orig = X_train[idx]
        img = (np.expand_dims(img_orig,0))
        predictions_array = occ_model.predict(img)
        result = np.argsort(predictions_array)  
        result = result[0][::-1]
        final_label = label_encoder.inverse_transform(np.array(result))[0]
        cv2.imshow("{0}".format(final_label), img_orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

