# Cerranet v2: Multiclass cerrado

# Pacotes
import tensorflow as tf
from tensorflow import keras

# Acessando os dados e dividindo 20 por cento para validacao
img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                   validation_split=0.25)

# Separando dados para as variaveis de treinamento e validacao
x_train = img.flow_from_directory('dataset/train/',
                                  target_size=(256, 256),
                                  batch_size=64,
                                  class_mode="categorical",
                                  subset="training")

x_valida = img.flow_from_directory('dataset/train/',
                                   target_size=(256, 256),
                                   batch_size=64,
                                   class_mode="categorical",
                                   subset="validation")

# Modelo sequencial
cerranet = keras.models.Sequential()

# Camadas convolucionais/ Maxpooling/ Dropout
cerranet.add(keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))
cerranet.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
cerranet.add(keras.layers.Dropout(0.15))

cerranet.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
cerranet.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
cerranet.add(keras.layers.Dropout(0.15))

cerranet.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
cerranet.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
cerranet.add(keras.layers.Dropout(0.15))

cerranet.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
cerranet.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
cerranet.add(keras.layers.Dropout(0.15))

cerranet.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
cerranet.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
cerranet.add(keras.layers.Dropout(0.15))

cerranet.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
cerranet.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
cerranet.add(keras.layers.Dropout(0.15))

# Camada que converte matrizes para vetores
cerranet.add(keras.layers.Flatten())

# Camadas Ocultas/Dropout
cerranet.add(keras.layers.Dense(units=256, activation='relu'))
cerranet.add(keras.layers.Dropout(0.15))

cerranet.add(keras.layers.Dense(units=128, activation='relu'))
cerranet.add(keras.layers.Dropout(0.15))

# Camada de Saida
cerranet.add(keras.layers.Dense(4, activation='softmax'))

# Compilador: Calcula a taxa de perda; a metrica da validacao; otimizacao da fucao de custo usando SGD
cerranet.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics='accuracy')

# Impressao da topologia do modelo
cerranet.summary()

# Controlando a parada do treinamento
cedo = keras.callbacks.EarlyStopping(monitor='loss', patience=2)

# Treinamento
history = cerranet.fit(x_train, epochs=20, callbacks=cedo, validation_data=x_valida)

cerranet.save_weights('cerranetBETA2.h5')
configs = cerranet.to_json()
with open('cerranetBETA2.json2', 'w') as json_file:
    json_file.write(configs)

cerranet.save_weights('cerranetBETA2.hdf5')
