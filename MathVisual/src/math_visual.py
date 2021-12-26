import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections
from MathVisual.src.occ.model import OccModel
from MathVisual.src.occ.dataset import create_label_encoder, symbols_list
from MathVisual.src.solver import solve, label_numeric_to_number, label_expression_to_str
from MathVisual.src.text_detector import DetectorText
from MathVisual.src.utils import Bcolors, logging, CustomFormatter

logger = logging.getLogger("MathVisual")
# logging.basicConfig(level=logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


def solve_print_result(results_sorted):
    labels = []
    for idx, img_label in results_sorted.items(): 
        labels.append(img_label[0])
    if labels.__len__() < 3:
        return
    num_1 = labels[0]
    num_2 = labels[2]
    expression = labels[1]
    final_result = solve(num_1,num_2,expression)
    if final_result is None:
        return
    equation_str = "{0} {1} {2} = {3}".format(
        label_numeric_to_number(num_1),
        label_expression_to_str(expression),
        label_numeric_to_number(num_2),
        final_result)
    return equation_str

def plot_equation(results_dict, final_equation:str):
    # Plot sorted images by order
    fig = plt.figure()
    # Final Image created from stacked images
    # num_1 EXPRESSION(+,-,/) num_2 = final_result
    image_one_big = np.zeros((100,500,3))
    # One big image creator loop
    idx_first = 0
    idx_last = 100
    for idx, img_label in results_dict.items():
        image_one_big[:,idx_first:idx_last,:] = cv2.resize(img_label[1], (100, 100))
        idx_first = idx_first + 100
        idx_last = idx_last + 100
    # Plot equation
    plt.title(final_equation,fontsize = 18)
    plt.imshow(image_one_big)
    plt.show()

def predict_equation(cropped_images_bbox, occ_model, label_encoder):
    # Dictionary that maps images and their X coordinate
    # To sort by X coordiantes later
    results = {}
    for idx,image in cropped_images_bbox:
        # Preprocess raw image
        img = cv2.resize(image, (100, 100)) 
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img /= 255
        # Get model prediction
        pred = occ_model.predict(img)
        result = np.argsort(pred)  
        result = result[0][::-1]
        final_label = label_encoder.inverse_transform(np.array(result))        
        # Add predicted label to dict with detection's X coordinate
        results[idx]=(final_label[0],image)
    return results

def plot_cropped_images(results_dict):
    for idx,label_image in results_dict.items():
        # Preprocess raw image
        img = cv2.resize(label_image[1], (100, 100))
        plt.title(label_image[0])
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test trained model with dataset')
    parser.add_argument('-path', '--model-path', default="checkpoints/model_200epoch_custom.h5")
    parser.add_argument('-image', '--image-path', default="MathVisual/src/figures/2021_12_25_0wi_Kleki.png")
    args = parser.parse_args()

    # Create the dataset loader object
    path_model = args.model_path
    path_image = args.image_path

    # Load model
    occ_model = OccModel()
    occ_model.load_model(args.model_path)

    # Read image
    img_read = cv2.imread(path_image)

    # Text detector
    text_detector = DetectorText(kernel_size_=(80,80))
    cropped_images_bbox = text_detector.detect_text(img_read)
    label_encoder = create_label_encoder(symbols_list)

    # Get predictions dict
    results_dict = predict_equation(cropped_images_bbox, occ_model, label_encoder)

    # # To tune kernel size uncomment this line
    # # until see all characters correctly!
    # plot_cropped_images(results_dict)

    # Sort images by X coordinate
    results_dict = collections.OrderedDict(sorted(results_dict.items()))

    # Print result
    final_equation_str = solve_print_result(results_dict)

    if final_equation_str is None:
        import sys
        logger.error("Could not detect a valid equation")
        sys.exit()
        
    # Plot equation with prediction
    plot_equation(results_dict, final_equation_str)
    
