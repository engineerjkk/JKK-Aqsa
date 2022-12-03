import cv2  # opencv
import numpy as np
from google.colab.patches import cv2_imshow

def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1

    cv2.fillPoly(mask, vertices, color)

    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, aa=1, bb=1., cc=0.):
    return cv2.addWeighted(initial_img, aa, img, bb, cc)


image = cv2.imread('road.png')
height, width = image.shape[:2]

kernel_size = 3
low_threshold = 70
high_threshold = 210
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)  # Blur
canny_img = cv2.Canny(blur_img, low_threshold, high_threshold)  # Canny edge

vertices = np.array(
    [[(0, height), (width / 2 - 70, height / 2 + 0), (width / 2 + 70, height / 2 + 0), (width - 0, height)]],
    dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices)  # ROI

hough_img = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)

result = weighted_img(hough_img, image)  # overlap
cv2_imshow(result)
cv2.waitKey(0)