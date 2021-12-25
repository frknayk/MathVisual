import argparse
import os
from MathVisual.src.occ.model import OccModel
from MathVisual.src.occ.dataset import DatasetLoader
from MathVisual.src.occ.models.custom_model import model as model_custom
# from MathVisual.src.occ.models.resnet_50 import model as model_resnet


"""===USAGE=== 
Run from project folder path:
    python MathVisual/src/occ/train.py -p ABSOLUTE_PATH_TO_DATASET
"""

if __name__ == '__main__':
    path_default = "/home/{0}/Downloads/archive/".format(os.getlogin())
    parser = argparse.ArgumentParser(description='Dataset absolute path')
    parser.add_argument('-p', '--train-path', default=path_default)
    args = parser.parse_args()
    # Create the dataset loader object
    dataset_loader = DatasetLoader(args.train_path)
    X_train,X_test = dataset_loader.create_train_test_data()
    y_train, y_test = dataset_loader.create_labels()
    # Create the vision model object
    occ_model = OccModel()
    occ_model.set_model(model_custom)
    train_config = {
        'batch_size': 50,
        'epochs': 200,
        'validation_split': 0.2,
        'shuffle': True}
    occ_model.train(X_train,y_train,train_config)
    occ_model.show_performance(X_test,y_test,"Train")
    occ_model.show_performance(X_test,y_test,"Test")
    occ_model.save_model("model_200epoch_custom")
    occ_model.plot_accuracy()
    occ_model.plot_loss()