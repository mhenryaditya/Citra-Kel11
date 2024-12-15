import numpy as np
import cv2

def improve_segmentation(binary):
    kernel = np.ones((18, 18), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Sure background area
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)

    return cleaned, sure_bg