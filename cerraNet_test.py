# CerraNet v2: assessment of model

# Library
import os
import numpy as np
import pandas as pd
from glob import glob
from tensorflow import keras
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json


# Loading model and weight
net = open('pesos/cerranetBETA2.json2', 'r')
cnnNet = net.read()
net.close()
cerranet = keras.models.model_from_json(cnnNet)
cerranet.load_weights('pesos/cerranetBETA2.h5')


# Listas
overall_accuracy = []
forest = []
fire = []
deforest = []
agriculture = []


# FOREST CLASS FORECAST
print('FOREST CLASS FORECAST')

# Loading the test dataset
testSet_forest = glob(os.path.join(os.getcwd(), 'dataset/test8m/forest/*.tiff'))

# Variable initialized
i = 0
j = 0
k = 0
l = 0

# Reading the test dataset
for img in testSet_forest:
    ia = image.load_img(img, target_size=(256, 256))
    imgplot = ia
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    forecast = cerranet.predict(ia)
    pred = pd.DataFrame(forecast, dtype='float32')

    if pred.loc[0, 3] > pred.loc[0, 0] and pred.loc[0, 3] > pred.loc[0, 1] and pred.loc[0, 3] > pred.loc[0, 2]:
        i+=1
       # print('Forest')
        plt.imshow(imgplot)
        plt.title('Forest')
        #plt.show()

    elif pred.loc[0, 0] > pred.loc[0, 1] and pred.loc[0, 0] > pred.loc[0, 2] and pred.loc[0, 0] > pred.loc[0, 3]:
        j+=1
       # print('Agriculture')
        plt.imshow(imgplot)
        plt.title('Agriculture')
       # plt.show()

    elif pred.loc[0, 1] > pred.loc[0, 0] and pred.loc[0, 1] > pred.loc[0, 2] and pred.loc[0, 1] > pred.loc[0, 3]:
        k+=1
      #  print('Deforest')
        plt.imshow(imgplot)
        plt.title('Deforest')
       # plt.show()

    elif pred.loc[0, 2] > pred.loc[0, 0] and pred.loc[0, 2] > pred.loc[0, 1] and pred.loc[0, 2] > pred.loc[0, 3]:
        l+=1
       # print('Fire')
        plt.imshow(imgplot)
        plt.title('Fire')
       # plt.show()

forest.append(i)
forest.append(j)
forest.append(k)
forest.append(l)

# FIRE CLASS FORECAST
print('FIRE CLASS FORECAST')

# Loading the test dataset
testSet_fire = glob(os.path.join(os.getcwd(), 'dataset/test8m/fire/*.tiff'))

# Variable reinitialized
i = 0
j = 0
k = 0
l = 0

# Reading the test dataset
for img in testSet_fire:
    ia = image.load_img(img, target_size=(256, 256))
    imgplot = ia
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    forecast = cerranet.predict(ia)
    pred = pd.DataFrame(forecast, dtype='float32')

    if pred.loc[0, 3] > pred.loc[0, 0] and pred.loc[0, 3] > pred.loc[0, 1] and pred.loc[0, 3] > pred.loc[0, 2]:
        i+=1
      #  print('Forest')
        plt.imshow(imgplot)
        plt.title('Forest')
        #plt.show()

    elif pred.loc[0, 0] > pred.loc[0, 1] and pred.loc[0, 0] > pred.loc[0, 2] and pred.loc[0, 0] > pred.loc[0, 3]:
        j+=1
       # print('Agriculture')
        plt.imshow(imgplot)
        plt.title('Agriculture')
        #plt.show()

    elif pred.loc[0, 1] > pred.loc[0, 0] and pred.loc[0, 1] > pred.loc[0, 2] and pred.loc[0, 1] > pred.loc[0, 3]:
        k+=1
       # print('Deforest')
        plt.imshow(imgplot)
        #plt.title('Deforest')
        #plt.show()

    elif pred.loc[0, 2] > pred.loc[0, 0] and pred.loc[0, 2] > pred.loc[0, 1] and pred.loc[0, 2] > pred.loc[0, 3]:
        l+=1
       # print('Fire')
        #plt.imshow(imgplot)
        #plt.title('Fire')
        #plt.show()

fire.append(i)
fire.append(j)
fire.append(k)
fire.append(l)


# DEFOREST CLASS FORECAST
print('DEFOREST CLASS FORECAST')

# Loading the test dataset
testSet_deforest = glob(os.path.join(os.getcwd(), 'dataset/test8m/deforest/*.tiff'))

# Variable reinitialized
i = 0
j = 0
k = 0
l = 0

# Reading the test dataset
for img in testSet_deforest:
    ia = image.load_img(img, target_size=(256, 256))
    imgplot = ia
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    forecast = cerranet.predict(ia)
    pred = pd.DataFrame(forecast, dtype='float32')

    if pred.loc[0, 3] > pred.loc[0, 0] and pred.loc[0, 3] > pred.loc[0, 1] and pred.loc[0, 3] > pred.loc[0, 2]:
        i+=1
       # print('Forest')
        plt.imshow(imgplot)
        plt.title('Forest')
        #plt.show()

    elif pred.loc[0, 0] > pred.loc[0, 1] and pred.loc[0, 0] > pred.loc[0, 2] and pred.loc[0, 0] > pred.loc[0, 3]:
        j+=1
       # print('Agriculture')
        plt.imshow(imgplot)
        plt.title('Agriculture')
        #plt.show()

    elif pred.loc[0, 1] > pred.loc[0, 0] and pred.loc[0, 1] > pred.loc[0, 2] and pred.loc[0, 1] > pred.loc[0, 3]:
        k+=1
        #print('Deforest')
        plt.imshow(imgplot)
        plt.title('Deforest')
        #plt.show()

    elif pred.loc[0, 2] > pred.loc[0, 0] and pred.loc[0, 2] > pred.loc[0, 1] and pred.loc[0, 2] > pred.loc[0, 3]:
        l+=1
        #print('Fire')
        plt.imshow(imgplot)
        plt.title('Fire')
       # plt.show()

deforest.append(i)
deforest.append(j)
deforest.append(k)
deforest.append(l)


# AGRICULTURE CLASS FORECAST
print('AGRICULTURE CLASS FORECAST')

# Loading the test dataset
testSet_agriculture = glob(os.path.join(os.getcwd(), 'dataset/test8m/agriculture/*.tiff'))

# Variable reinitialized
i = 0
j = 0
k = 0
l = 0

# Reading the test dataset
for img in testSet_agriculture:
    ia = image.load_img(img, target_size=(256, 256))
    imgplot = ia
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    forecast = cerranet.predict(ia)
    pred = pd.DataFrame(forecast, dtype='float32')

    if pred.loc[0, 3] > pred.loc[0, 0] and pred.loc[0, 3] > pred.loc[0, 1] and pred.loc[0, 3] > pred.loc[0, 2]:
        i+=1
        #print('Forest')
        plt.imshow(imgplot)
        plt.title('Forest')
        #plt.show()


    elif pred.loc[0, 0] > pred.loc[0, 1] and pred.loc[0, 0] > pred.loc[0, 2] and pred.loc[0, 0] > pred.loc[0, 3]:
        j+=1
        #print('Agriculture')
        plt.imshow(imgplot)
        plt.title('Agriculture')
        #plt.show()


    elif pred.loc[0, 1] > pred.loc[0, 0] and pred.loc[0, 1] > pred.loc[0, 2] and pred.loc[0, 1] > pred.loc[0, 3]:
        k+=1
        #print('Deforest')
        plt.imshow(imgplot)
        plt.title('Deforest')
        #plt.show()


    elif pred.loc[0, 2] > pred.loc[0, 0] and pred.loc[0, 2] > pred.loc[0, 1] and pred.loc[0, 2] > pred.loc[0, 3]:
        l+=1
        #print('Fire')
        plt.imshow(imgplot)
        plt.title('Fire')
        #plt.show()


agriculture.append(i)
agriculture.append(j)
agriculture.append(k)
agriculture.append(l)

# ASSESSMENT OF MODEL
print('Report of learning:')

# Overall accuracy
sumAllClass = sum(forest) + sum(agriculture) + sum(deforest) + sum(fire)
sumCorrectClassification = forest[0] + agriculture[1] + deforest[2] + fire[3]
overall_accuracy = round((sumCorrectClassification*100)/sumAllClass,2)

# F1-Score
# tp = true positive
tp = sumCorrectClassification

# fp = false positive
fp_forest = agriculture[0] + deforest[0] + fire[0]
fp_agriculture = forest[1] + deforest[1] + fire[1]
fp_deforest = forest[2] + agriculture[2] + fire[2]
fp_fire = forest[3] + agriculture[3] + deforest[3]
fp = fp_forest + fp_deforest + fp_agriculture + fp_fire

# fn = false negative
fn_forest = forest[1] + forest[2] + forest[3]
fn_agriculture = agriculture[0] + agriculture[2] + agriculture[3]
fn_deforest = deforest[0] + deforest[1] + deforest[3]
fn_fire = fire[0] + fire[1] + fire[2]
fn = fn_forest + fn_deforest + fn_agriculture + fn_fire

# precision metric
precision = tp/(tp+fp)

# recall metric
recall = tp/(tp+fn)

# f1-score
f1Score = 2 * (precision * recall) / (precision + recall)

# Accuracy Forest
accuracyForest = round(forest[0]*100/sum(forest),2)
print('1 Forest subset')
print('1.1 Accuracy: ', accuracyForest)
print('1.2 Classifications: ')
print('- Forest: ', forest[0])
print('- Deforest: ', forest[2])
print('- Agriculture: ', forest[1])
print('- Fire: ', forest[3])
print('- Correct classification: ', forest[0])
print('- Incorrect classification: ', forest[1]+forest[2]+forest[3])

# Accuracy Agriculture
accuracyAgriculture = round(agriculture[1]*100/sum(agriculture),2)
print('2 Agriculture subset')
print('2.1 Accuracy: ', accuracyAgriculture)
print('2.2 Classifications: ')
print('- Forest: ', agriculture[0])
print('- Deforest: ', agriculture[2])
print('- Agriculture: ', agriculture[1])
print('- Fire: ', agriculture[3])
print('- Correct classification: ', agriculture[1])
print('- Incorrect classification: ', agriculture[0]+agriculture[2]+agriculture[3])


# Accuracy Deforest
accuracyDeforest = round(deforest[2]*100/sum(deforest),2)
print('3 Deforest subset')
print('3.1 Accuracy: ', accuracyDeforest)
print('3.2 Classifications: ')
print('- Forest: ', deforest[0])
print('- Deforest: ', deforest[2])
print('- Agriculture: ', deforest[1])
print('- Fire: ', deforest[3])
print('- Correct classification: ', deforest[2])
print('- Incorrect classification: ', deforest[0]+deforest[1]+deforest[3])

# Accuracy Fire
accuracyFire = round(fire[3]*100/sum(fire),2)
print('4 Fire subset')
print('4.1 Accuracy: ', accuracyFire)
print('4.2 Classifications: ')
print('- Forest: ', fire[0])
print('- Deforest: ', fire[2])
print('- Agriculture: ', fire[1])
print('- Fire: ', fire[3])
print('- Correct classification: ', fire[3])
print('- Incorrect classification: ', fire[0]+fire[1]+fire[2])


# Over all
print('5 Overall performance')
print('5.1 Accuracy: ', overall_accuracy)
print('5.2 Precision: ', precision*100)
print('5.3 Recall: ', recall*100)
print('5.4 F1-Score: ', f1Score*100)
