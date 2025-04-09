import cv2
import numpy as np
import sys

def imshow(title, img, scale = 0.2):
    if img is None:
        sys.exit("Could not read the image.")
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    resized_image = cv2.resize(img, new_size)
    cv2.imshow(title, resized_image)
    cv2.waitKey(0)  # wait forever
    cv2.destroyAllWindows()


lower_red1 = np.array([15-15, 109, 70])
upper_red1 = np.array([8, 255, 255])

lower_red2 = np.array([180-8, 110, 70])
upper_red2 = np.array([180, 255, 255])

def centroid(mask, img):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #imshow("The img_gray", img_gray)
    #thresh = cv2.threshold(img_gray, 20,255 , cv2.THRESH_BINARY)[1]
    M = cv2.moments(mask)
    print(M)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX, cY), 400, (255, 0, 0), -1)
        cv2.putText(img, "Centroid", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 9)
    except:
        pass
    return img
    #imshow("Centroid Detected", img)

def detect_red_ball(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    #imshow("Masks combined", mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    largest_contour = max(contours, key=cv2.contourArea)
    print(largest_contour)

    kernel = np.ones((4,4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #imshow("Mask after noise removal", mask)

    cv2.drawContours(img, [largest_contour], -1, (255, 0, 0), 5, cv2.LINE_AA)
    imshow("temp_output", img)
    return img, mask
'''
def binarizeimg(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)[1]
    imshow("Binary_img", binr)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1)
    imshow("opening", opening)
    closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=3)
    imshow("closing", closing)
    return opening
'''

#image = cv2.imread("RedBallPhoto.jpeg")
#processed_img, mask = detect_red_ball(image)
#centroid(mask, processed_img)



