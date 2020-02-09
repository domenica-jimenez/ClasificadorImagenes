import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x) #[[1, 0, 0]]
    resultado = arreglo[0] #[0, 0, 1]
    respuesta = np.argmax(resultado) #2
    if respuesta == 0:
        print("Gato")
    elif respuesta == 1:
        print("Perro")

    return respuesta


#predict('gatito.jpeg')
