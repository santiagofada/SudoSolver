import cv2 as cv
import numpy as np
# from tensorflow.keras.models import load_model


# 1 - Preprocesamiento
def preprocess(img):
    """
    Realiza el preprocesamiento de una imagen para facilitar la detección de contornos.
    Me lleva la imagen a una escala de blanco y negro, y se encarga de las sombras.

    Parámetros:
    - img (numpy.ndarray): La imagen de entrada en formato BGR.

    Retorna:
    - numpy.ndarray: La imagen después de aplicar el preprocesamiento, con contornos resaltados.
    """

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    noise = cv.GaussianBlur(gray_img, (5, 5), 1)
    imgThreshold = cv.adaptiveThreshold(noise, 255, 1, 1, 11, 2)
    return imgThreshold


# 3 - Encontrar el mayor borde para que ese sea el sudoku
def mayorBorde(contours):
    """
        Encuentra el contorno o borde con el área máxima y con cuatro esquinas, posiblemente representando un sudoku.

        Parámetros:
        - contours (list): Lista de bordes detectados en una imagen.

        Retorna:
        - np.ndarray: El contorno con el área máxima y cuatro esquinas.
        - float: El área correspondiente al contorno encontrado.
    """
    borde = np.array([])
    area_maxima = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)

            # solo buscamos formas rectangulares o cuadradas por eso la segunda condicion
            if area > area_maxima and len(approx) == 4:
                borde = approx
                area_maxima = area
    return borde, area_maxima


#### 3 - Reorder points for Warp Perspective
def reordenar(puntos):
    """
    Reordena los puntos para facilitar la transformación.

    Parámetros:
    - puntos (numpy.ndarray): Un arreglo de puntos de esquina, originalmente en formato (4, 1, 2).


    Retorna:
    - numpy.ndarray: Un arreglo de puntos reordenados, manteniendo el formato (4, 1, 2).
    """
    puntos = puntos.reshape((4, 2))
    nuevos_puntos = np.zeros((4, 1, 2), dtype=np.int32)

    add = puntos.sum(1)

    nuevos_puntos[0] = puntos[np.argmin(add)]
    nuevos_puntos[3] =puntos[np.argmax(add)]

    diff = np.diff(puntos, axis=1)

    nuevos_puntos[1] =puntos[np.argmin(diff)]
    nuevos_puntos[2] = puntos[np.argmax(diff)]

    return nuevos_puntos


# 6 - Mostrar todas las imagenes juntas
def stackImages(imgArray, scale):
    """
    Apila imágenes tanto horizontal como verticalmente y devuelve la imagen resultante, asi como un escalado.

    Parámetros:
    - imgArray (list): Una lista de imágenes (matrices) a apilar.
    - scale (float): Factor de escala para redimensionar las imágenes.

    Retorna:
    - numpy.ndarray: La imagen apilada resultante.
    """

    filas = len(imgArray)
    columnas = len(imgArray[0])
    filasAvailable = isinstance(imgArray[0], list)

    ancho = imgArray[0][0].shape[1]
    alto = imgArray[0][0].shape[0]

    if filasAvailable:
        for x in range(filas):
            for y in range(columnas):
                imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)

                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)

        blankImg = np.zeros((alto, ancho, 3), np.uint8)
        hor = [blankImg]*filas
        hor_con = [blankImg]*filas

        for x in range(0, filas):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])

        ver = np.vstack(hor)
    else:
        for x in range(filas):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)

            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        ver = hor
    return ver
