import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.layers import Dense
from keras.optimizers import Adam
import pickle

##########
path = "C:/Users/Usuario/Proyectos/SudoSolver/assets/data"
test_ratio = 0.2
validation_ratio = 0.2
#########


# 1)- Carga de datos
imagenes = []  # lista con todas las imagenes
clases = []  # lista con la correspondiente clase de cada imagen

carpetas = os.listdir(path)
nClases = len(carpetas)

for folder in range(nClases):
    numbers = os.listdir(f"{path}/{folder}")
    for number in numbers:
        img = cv.imread(f"{path}/{folder}/{number}")
        img = cv.resize(img, (32, 32))

        imagenes.append(img)
        clases.append(folder)
    # print(folder, end=" ")
# print(" ")
# print(f"Total imagenes: {len(imagenes)}")

imagenes = np.array(imagenes)
clases = np.array(clases)

# 2)- Separacion de los datos
X_train, X_test, y_train, y_test = train_test_split(imagenes, clases, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)
# print(X_train.shape)
# print(X_test.shape)
# print(X_validation.shape)


# 3)- Histograma de los datos de entrenamiento
"""
muestra = []
for i in range(nClases):
    muestra.append(len(np.where(y_train == i)[0]))


fig, ax = plt.subplots()
ax.bar(range(nClases), muestra)
ax.set_title("Elementos muestrados de cada clase")
plt.xlabel("Case")
plt.ylabel("Cantidad de imagenes")
plt.show()
"""


# 4)- Preprocesamiento de imagenes


def preproces(imagen):
    n, m, _ = imagen.shape

    imagen = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)  # llevar a gris
    imagen = cv.equalizeHist(imagen)
    imagen = imagen / 255  # normalizar
    imagen = imagen.reshape((n, m, 1))
    return imagen


# img = preproces(X_train[42])
# img = cv.resize(img, (300, 300))
# cv.imshow("PreProcessed", img)
# cv.waitKey(0)

# llevamos los numeros a una escala de grises
# por lo que cada imagen es una matriz (32, 32, 1) en lugar de (32, 32, 3)
X_train = np.array(list(map(preproces, X_train)))
X_test = np.array(list(map(preproces, X_test)))
X_validation = np.array(list(map(preproces, X_validation)))

# Aplicar transformaciones a las imagenes
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)
dataGenerator.fit(X_train)

y_train = to_categorical(y_train, nClases)
y_test = to_categorical(y_test, nClases)
y_validation = to_categorical(y_validation, nClases)


def clasificador():
    categories = 10
    n_filters = 60
    size_filter1 = (5, 5)
    size_filter2 = (3, 3)
    pool_size = (2, 2)
    neurons = 512
    dropout = 0.3

    model = Sequential([
        Conv2D(n_filters, size_filter1, input_shape=(32, 32, 1), activation="relu"),
        Conv2D(n_filters, size_filter1, activation="relu"),
        MaxPooling2D(pool_size=pool_size),

        Conv2D(n_filters / 2, size_filter2, activation="relu"),
        Conv2D(n_filters / 2, size_filter2, activation="relu"),
        MaxPooling2D(pool_size=pool_size),

        Dropout(dropout),

        Flatten(),
        Dense(neurons, activation="relu"),

        Dropout(dropout),

        Dense(categories, activation="softmax")
    ])
    model.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = clasificador()
print(model.summary())

batch_size = 50
epochs = 10
steps_per_epoch = 2000

history = model.fit(dataGenerator.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=(X_validation, y_validation),
                              shuffle=1,
                              )
#### SAVE THE TRAINED MODEL
pickle_out= open("model_trained.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()