import numpy as np
import cv2 as cv
from keras.models import load_model
from training import preproces
###############
width = 640
height = 480
###############


captura = cv.VideoCapture(0)
captura.set(3,width)
captura.set(4,height)

model = load_model('modelo_entrenado.h5')

while True:
    success, img = captura.read()

    img = np.asarray(img)
    img = cv.resize(img, (32, 32))
    img = preproces(img)
    cv.imshow("Processsed Image", img)
    img = img.reshape(1,32,32,1)

    y = model.predict(img,verbose=None)
    prob = np.max(y)

    if prob > 0.95:
        print(np.argmax(y, axis=1), prob)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    #img = img.reshape(1, 32, 32, 1)

    cv.waitKey(1)