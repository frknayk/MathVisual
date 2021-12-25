import os
import argparse
import numpy as np
import cv2
from photomath.src.occ.model import OccModel
from photomath.src.occ.dataset import DatasetLoader, symbols_list
import tensorflow as tf

"""===USAGE=== 
Run from project folder path:
python photomath/src/occ/inference.py -p ABSOLUTE_PATH_TO_DATASET
"""

def predict(occ_model, img:np.ndarray,label_encoder):
    """Predict label of image

    Parameters
    ----------
    model_path : str
        [description]
    img : np.ndarray
        [description]
    """
    # Read a image and preprocess for inference
    img = X_test[0]
    img = cv2.resize(img, (100, 100))        
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img /= 255
    # Get prediction
    prediction = occ_model.predict(img)
    result = np.argsort(prediction)  
    result = result[0][::-1]
    final_label = label_encoder.inverse_transform(np.array(result))
    return final_label

if __name__ == '__main__':
    # Parse arguments
    path_dataset = "/home/{0}/Downloads/archive/".format(os.getlogin())
    path_model = "/home/{0}/Furkan/coding/repos_me/PhotoMath/checkpoints/models/model_200epoch_custom.h5".format(os.getlogin())
    parser = argparse.ArgumentParser(description='Trained model absolute path')
    parser.add_argument('-p', '--model-path', default=path_model)
    args = parser.parse_args()

    # Create the dataset loader object
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

