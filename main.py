# detectar contornos
# detectar sudoku
# clasificar los numeros
# encontrar solucion unica
# mostrar solucion unica

import cv2 as cv
import numpy as np
# import tensorflow as tf
from utils import *

rutaImagen = "Rsc/sudoku01.jpg"
witdth = 450
height = 450


# 1) procesar imagen
img = cv.imread(rutaImagen)
img = cv.resize(img, (witdth, height))

blankImg = np.zeros((witdth, height, 3), np.uint8)
imgThreshold = preprocess(img)



