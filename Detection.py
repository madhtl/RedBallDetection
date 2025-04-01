import cv2
import numpy as np
import sys


def imshow(title, image, scale=0.2):
    if img is None:
        sys.exit("Could not read the image.")
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized_image = cv2.resize(image, new_size)
    cv2.imshow(title, resized_image)
    cv2.waitKey(0)  # wait forever
    cv2.destroyAllWindows()


lower_red1 = np.array([15-15, 50, 20])
upper_red1 = np.array([15, 255, 255])

lower_red2 = np.array([180-15, 50, 20])
upper_red2 = np.array([180, 255, 255])


def detect_red_ball(img):
    # Convert image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create masks for two ranges of red
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    # Combine both red masks
    mask = cv2.bitwise_or(mask1, mask2)

    # remove noise
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    imshow("Mask after noise removal", mask)

    # Segment the image using the mask
    seg_img = cv2.bitwise_and(img, img, mask=mask)
    imshow("Segmented Image", seg_img)
    return seg_img

#def centroidcalc(img):
def binarizeimg(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_OTSU)[1]
    imshow("Binary_img", binr)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1)
    imshow("opening", opening)
    closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=3)
    imshow("closing", closing)


# Load the image and detect the red ball
img = cv2.imread("RedBallPhoto.jpeg")
#imshow("Original Image", img)
#detect_red_ball(img)
binarizeimg(img)
