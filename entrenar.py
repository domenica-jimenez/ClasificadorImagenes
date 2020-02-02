import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
K.clear_session()

data_entrenamiento = './Data/Entrenamiento'
data_validacion = './Data/Validación'

epocas = 20 #Número de veces que vamos a íterar sobre el set de datos durante el entrenamiento
altura, longitud = 100, 100 #Tamaño a definir en pixeles de las imagenes para entrenar
batch_size = 32 #Número de imagenes a procesar en cada paso
pasos = 1000 #Número de veces en que se va a procesar la información en cada época
pasos_validacion = 200 #En cada época
#Número de filtros de cada convolución
filtrosConv1 = 32
filtrosConv2 = 64
size_filtro1 = (3,3)
size_filtro2 = (2,2)
size_pool = (2,2)
clases = 2
lr = 0.0005 #Que tan grandes van a ser los ajustes de la red para acercarse a una solución óptima

#Preprocesamiento de imágenes

entrenamiento_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.3, zoom_range=0.3, horizontal_flip=True)
validacion_datagen = ImageDataGenerator(rescale=1./255)
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(data_entrenamiento, target_size=(altura, longitud), batch_size=batch_size, class_mode='categorical')
imagen_validacion = validacion_datagen.flow_from_directory(data_validacion, target_size=(altura, longitud), batch_size=batch_size, class_mode='categorical')

#Crear la red convolucional

cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, size_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=size_pool))
cnn.add(Convolution2D(filtrosConv2, size_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=size_pool))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)

dir='./modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
