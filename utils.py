import cv as cv
import numpy as np
# from tensorflow.keras.models import load_model


# 1 - Preprocesamiento
def preprocess(img):
    """
    Realiza el preprocesamiento de una imagen para facilitar la detección de contornos.

    Parámetros:
    - img (numpy.ndarray): La imagen de entrada en formato BGR.

    Retorna:
    - numpy.ndarray: La imagen después de aplicar el preprocesamiento, con contornos resaltados.
    """

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    noise = cv.GaussianBlur(gray_img, (5, 5), 1)
    imgThreshold = cv.adaptiveThreshold(noise, 255, 1, 1, 11, 2)
    print(type(imgThreshold))
    return imgThreshold




#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver
