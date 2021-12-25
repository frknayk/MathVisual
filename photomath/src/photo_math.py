import random
from photomath.src.text_detection.text_detector import DetectorText

fig = plt.figure()

# Single image detect
img_read = cv2.imread(
    "/home/anton/Furkan/coding/repos_me/PhotoMath/photomath/src/text_detection/figures/handwritten-numbers.jpg")

dim_img = (100,100)
text_detector = DetectorText(kernel_size_=(3, 3))
cropped_images_bbox_ = text_detector.detect_text(img_read)

num_samples = cropped_images_bbox_.__len__()

for idx in range(num_samples):
    image = cropped_images_bbox_[idx]['img']

    img = cv2.resize(image, (100, 100)) 
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img /= 255

    pred = model.predict(img)        
    result = np.argsort(pred)  
    result = result[0][::-1]
    final_label = label_encoder.inverse_transform(np.array(result))

    plt.imshow(image)
    plt.title("Prediction:{0}".format(final_label[0]), fontsize = 18)        
    plt.show()