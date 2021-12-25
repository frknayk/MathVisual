# import packages
import cv2
import numpy as np


class DetectorText:
    """Preprocess and image and return each detected texts:
        1. cropped image
        2. cropped image's bounding box coordinates
    """

    def __init__(self,
                 kernel_size_: tuple = (20, 20),
                 dilation_iterations_: int = 1,
                 debug_: bool = False) -> None:
        self.kernel_size = kernel_size_
        self.dilation_iterations = dilation_iterations_
        self.debug = debug_

    def __find_contours(self, img_: np.ndarray) -> list:
        """Preprocess image and find contours in the image

        Args:
            img_ (np.ndarray): Read image

        Returns:
            list: List of contours
        """
        # Convert the image to gray scale
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=self.dilation_iterations)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        return contours

    @staticmethod
    def __draw_bounding_boxes(img_: np.ndarray, contours: list) -> list:
        """Draw bounding boxes around contours

        Args:
            img_ (np.ndarray): [description]
            contours (list) : [description]

        Returns:
            list: [description]
        """
        # List of cropped images and their bbox coordinates
        list_cropped_images_bbox = []
        # Creating a copy of image
        im2 = img_.copy()
        # Looping through the identified contours
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # # Drawing a rectangle on copied image
            # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = None
            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]
            cropped_dict = {
                'img': cropped,
                'coordinates': {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                },
                'rect': rect
            }
            list_cropped_images_bbox.append(cropped_dict)
        return list_cropped_images_bbox

    def detect_text(self, img: np.ndarray) -> list:
        contours = self.__find_contours(img)
        cropped_images_bbox = self.__draw_bounding_boxes(img, contours)
        return cropped_images_bbox

def example_pytesseract():
    import pytesseract
    img_read = cv2.imread("MathVisual/src/figures/handwritten-numbers.jpg")
    text_detector = DetectorText(kernel_size_=(3, 3))
    cropped_images_bbox_ = text_detector.detect_text(img_read)
    for idx_, cropped_dict_ in enumerate(cropped_images_bbox_):
        print(idx_)
        cropped_img = cropped_dict_['img']
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped_img,
                                           config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789+-x/()=")
        text = text if len(text) < 2 else text[0]
        cv2.imshow("Character: '{0}' ".format(text), cropped_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Finished.")
