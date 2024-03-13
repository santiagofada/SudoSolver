# detectar contornos
# detectar sudoku
# clasificar los numeros
# encontrar solucion unica
# mostrar solucion unica

# import cv2 as cv
# import numpy as np
# import tensorflow as tf
import cv2

from utils import *

rutaImagen = "Rsc/sudoku01.jpg"
width = 450
height = 450


# 1) procesar imagen
img = cv.imread(rutaImagen)
img = cv.resize(img, (width, height))

blankImg = np.zeros((width, height, 3), np.uint8)
imgThreshold = preprocess(img)


# 2) detectar bordes y contornos
imgBordes = img.copy()
# Lo importante esta en cv.RETR_EXTERNAL para detectar los bordes externos
bordes, jerarquia = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(imgBordes, bordes, -1, (0, 255, 0), 3)


# 3) Encontrar el contorno mas grande para que este sea el tablero del sudoku
imgGranBorde = img.copy()
mayor_borde, area = mayorBorde(bordes)
if mayor_borde.size != 0:
    # a veces open CV lee los puntos en otro orden, por eso debemos reordenarlos para que esten en el orden valido
    mayor_borde = reordenar(mayor_borde)

    cv2.drawContours(imgGranBorde, mayor_borde, -1, (0, 0, 255), 25)

    # preparar puntos para la transformaicon
    input_pts = np.array(mayor_borde, dtype="float32")
    output_pts = np.array([[0, 0], [width, 0], [0,height], [width, height]], dtype="float32")

    matriz = cv.getPerspectiveTransform(input_pts, output_pts)
    imgTransformada = cv.warpPerspective(img, matriz, (width, height), flags=cv2.INTER_LINEAR)

    imgDigitosDetectados = blankImg.copy()
    imgTransformada = cv.cvtColor(imgTransformada, cv.COLOR_BGR2GRAY)












imageArray = ([img, imgThreshold, imgBordes, imgGranBorde],
              [imgTransformada, blankImg, blankImg, blankImg])


stackedImage = stackImages(imageArray, 1)
cv.imshow('Stacked Images', stackedImage)

cv.waitKey(0)
