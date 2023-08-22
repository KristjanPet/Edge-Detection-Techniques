import cv2
import numpy as np
import math
from numba import jit

@jit(nopython=True)
def Multiplier(matrix, y, x, picture):  # Function for matrix multiplication
    result = 0
    for i in range(-1, 2):
        for j in range(-1, 2):  # Iterate through both matrices
            try:  # Check if it goes out of the image boundaries
                pixel = picture[y+i][x+j]
            except:
                pixel = 0
            result += matrix[i][j] * pixel  # Calculate the average
    return result


@jit(nopython=True)
def Roberts(picture):
    height = picture.shape[0]
    width = picture.shape[1]
    
    v_mask = [[0, 0, 0],
              [1, 0, 0],
              [0, -1, 0]]

    h_mask = [[0, 0, 0],
              [0, 1, 0],
              [-1, 0, 0]]

    result_image = np.zeros(picture.shape, dtype=np.uint8)  # Create a temporary image filled with zeros

    for x in range(width):
        for y in range(height):  # Iterate through the entire image
            vertical = Multiplier(v_mask, y, x, picture)  # Apply vertical mask
            horizontal = Multiplier(h_mask, y, x, picture)  # Apply horizontal mask
            result_image[y][x] = math.sqrt(vertical**2 + horizontal**2)  # Absolute value

    return result_image

@jit(nopython=True)
def Prewitt(picture):
    height = picture.shape[0]
    width = picture.shape[1]
    
    v_mask = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]

    h_mask = [[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]]

    result_image = np.zeros(picture.shape, dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            vertical = Multiplier(v_mask, y, x, picture)
            horizontal = Multiplier(h_mask, y, x, picture)
            result_image[y][x] = math.sqrt(vertical**2 + horizontal**2)

    return result_image

@jit(nopython=True)
def Sobel(picture):
    height = picture.shape[0]
    width = picture.shape[1]
    
    v_mask = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]

    h_mask = [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]

    result_image = np.zeros(picture.shape, dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            vertical = Multiplier(v_mask, y, x, picture)
            horizontal = Multiplier(h_mask, y, x, picture)
            result_image[y][x] = math.sqrt(vertical**2 + horizontal**2)

    return result_image


# Load grayscale images
imgDark = cv2.imread('LightDog.png', cv2.IMREAD_GRAYSCALE)
imgLight = cv2.imread('DarkDog.png', cv2.IMREAD_GRAYSCALE)

# Display the original images side by side
cv2.imshow('image', np.hstack([imgLight, imgDark]))

# Apply median blur to reduce noise
medianDark = cv2.medianBlur(imgDark, 7)
medianLight = cv2.medianBlur(imgLight, 7)


# Display edge detection results for Prewitt, Roberts, Sobel, and Canny
cv2.imshow('Prewitt', np.hstack([Prewitt(medianLight), Prewitt(medianDark)]))
cv2.imshow('Roberts', np.hstack([Roberts(medianLight), Roberts(medianDark)]))
cv2.imshow('Sobel', np.hstack([Sobel(medianLight), Sobel(medianDark)]))
cv2.imshow('Canny', np.hstack([cv2.Canny(medianLight, 100, 200), cv2.Canny(medianDark, 100, 200)]))

cv2.waitKey(0)
cv2.destroyAllWindows()
