import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from MathVisual.src.occ.model import OccModel
from MathVisual.src.occ.dataset import create_label_encoder, symbols_list
from MathVisual.src.solver import solve
from MathVisual.src.text_detector import DetectorText


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

    # Create label encoder to decode label
    label_encoder = create_label_encoder(symbols_list)

    results = {}

    # TODO: Show images stacked
    fig = plt.figure()
    for idx in range(cropped_images_bbox.__len__()):
        image = cropped_images_bbox[idx]['img']
        img = cv2.resize(image, (100, 100)) 
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img /= 255

        pred = occ_model.predict(img)
        result = np.argsort(pred)  
        result = result[0][::-1]
        final_label = label_encoder.inverse_transform(np.array(result))

        print("LABEL: {0}/Coordinate:{1}\n".format(final_label[0],cropped_images_bbox[idx]['coordinates']['x']))
        results[cropped_images_bbox[idx]['coordinates']['x']]=final_label[0]

        # plt.imshow(image)
        # plt.title("{0} / {1}".format(
        #     final_label[0],cropped_images_bbox[idx]['coordinates']['x']), 
        #     fontsize = 18)        
        # plt.show()

    # TODO: STR TO NUM FUNCTION
    # TODO: MOVE THIS TO FUNCTION HERE
    # sort by key
    import collections
    od = collections.OrderedDict(sorted(results.items()))
    labels = []
    for idx, label in od.items(): 
        labels.append(label)
    print("{0} {1} {2} is {3}".format(
        labels[0],
        labels[2],
        labels[1],
        solve(labels[0],labels[2],labels[1])))