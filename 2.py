import sys
import numpy as np
import cv2 as cv
from utils import *

def bilinear_interpolation(image,x,y):
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1

    if x0 >= image.shape[1] - 1 or y0 >= image.shape[0] - 1:
        return 0  # Return 0 for out-of-bound points

    dx, dy = x - x0, y - y0
    # print(image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1])
    # Bilinear interpolation equation
    interpolated_value = np.zeros(image.shape[2], dtype=np.float32)
    for i in range(image.shape[2]):
        interpolated_value[i] = (1 - dx) * (1 - dy) * image[y0, x0, i] + \
            dx * (1 - dy) * image[y0, x1, i] + \
            (1 - dx) * dy * image[y1, x0, i] + \
            dx * dy * image[y1, x1, i]

    return interpolated_value.astype(np.uint8)


def backward_warping(original_image, M, output_shape):
    output_image = np.zeros(output_shape, dtype=np.uint8)

    # Compute the inverse transformation matrix
    M_inv = np.linalg.inv(M)

    # Iterate through each pixel in the output image
    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            # Map the (x, y) coordinates back to the original image
            original_coords = np.dot(M_inv, np.array([x, y, 1]))
            original_x, original_y = original_coords[:2] / original_coords[2]

            # Perform bilinear interpolation
            output_image[y, x] = bilinear_interpolation(original_image, original_x, original_y)

    return output_image

if __name__ == '__main__':
    original_image = cv.imread(sys.argv[1])  # 'images/2-my_img.png'

    # Get the four corners of the original image (mouse_click_example.py)
    ## 1 4 ##
    ## 2 3 ##
    corner_points1 = np.float32([[186, 678], [652, 3630], [2768, 3484], [2495, 289]])
    corner_points2 = np.float32([[0, 0], [0, 4032], [3024, 4032], [3024, 0]])
    
    # Define the transformation matrix
    H = direct_linear_transformer(corner_points1, corner_points2)
    # H = cv.getPerspectiveTransform(corner_points1, corner_points2)
    print('H: ', H)
    
    # Perform backward warping
    warped_image = backward_warping(original_image, H, original_image.shape)
    # warped_image = cv.warpPerspective(original_image, H, (original_image.shape[1], original_image.shape[0]), flags=cv.INTER_LINEAR)

    
    print("Show the original image and the warped image")
    cv.imshow('Original Image', original_image)
    cv.imshow('Warped Image', warped_image)

    # Wait for a key press and close the OpenCV windows
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # print("Result saved as 'images/2-warped_image.png'")
    # cv.imwrite('images/2-warped_image.png', warped_image)
