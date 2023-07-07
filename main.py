
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import diameter_calc



def enhance_image(image, clahe):
    # Проверка количества каналов в изображении
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Преобразование изображения в оттенки серого и увеличение контраста
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # # Применение адаптивного порогового значения для выделения вен
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        
        return blurred
    else:
        return gray


image = cv2.imread('hand_qwe.png')

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение фильтра Гаусса для сглаживания изображения
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Применение адаптивного порогового значения для выделения вен
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
thresh = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





diameter_calc.diameter(thresh);