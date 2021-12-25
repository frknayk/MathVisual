import argparse
import os
from MathVisual.src.occ.model import OccModel
from MathVisual.src.occ.dataset import DatasetLoader

if __name__ == '__main__':
    path_default = "/home/{0}/Downloads/archive/".format(os.getlogin())
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument('-dataset', '--path-dataset', default=path_default)
    parser.add_argument('-vision', '--model-type',default="model_custom")
    parser.add_argument('-model', '--model-name',default="model_name_default")
    parser.add_argument('-batch', '--batch-size', default=50)
    parser.add_argument('-epochs', '--num_epoch', default=200)
    parser.add_argument('-ratio', '--val_split_ratio', default=0.2)
    parser.add_argument('-shuffle', '--shuffle_data', default=True)
    args = parser.parse_args()

    # Import vision model
    model = None
    if args.model_type == "custom":
        from MathVisual.src.occ.models.custom_model import model
    elif args.model_type == "resnet50":
        from MathVisual.src.occ.models.resnet_50 import model
    else:
        print("Wrong model type is entered.")
        print("Please either enter custom or resnet50 or name of your custom model")

    # Create the dataset loader object
    dataset_loader = DatasetLoader(args.path_dataset)
    X_train,X_test = dataset_loader.create_train_test_data()
    y_train, y_test = dataset_loader.create_labels()

    # Create the vision model object
    occ_model = OccModel()
    occ_model.set_model(model)
    train_config = {
        'batch_size': args.batch_size,
        'epochs': args.num_epoch,
        'validation_split': args.val_split_ratio,
        'shuffle': args.shuffle_data}
    occ_model.train(X_train,y_train,train_config)

    # Save trained model
    occ_model.save_model(args.model_name)

    # Model evaluation
    occ_model.show_performance(X_test,y_test,"Train")
    occ_model.show_performance(X_test,y_test,"Test")
    occ_model.plot_accuracy()
    occ_model.plot_loss()
