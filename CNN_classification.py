import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras 
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, InputLayer, Reshape, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import random
import cv2
import os
import math
import time
import pickle

start_time = time.time()

model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, 1, 3, activation='relu'))
model.add(Conv2D(128, 1, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(256, 1, 3, activation='relu'))
model.add(Conv2D(256, 1, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

#model.load_weights("/home/path/to/file/exempl98%ROC.hdf5")


#image_directory = '/home/path/to/file/Images_decoupes/'
image_directory = '/home/path/to/file//New-data-test/data/'
train_subdirs = 'train'
validation_subdirs = 'validation'

test_subdirs = 'test' 

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])


list_input=[]
list_label=[]
train=os.path.join(image_directory,train_subdirs)
#sain=os.path.join(train,"Sain_Training2")
#tumeur=os.path.join(train,"Tumeur_Training2")
#sain=os.path.join(train,"strr512")
#tumeur=os.path.join(train,"ttr512")
#sain=os.path.join(train,"S-RG")
#tumeur=os.path.join(train,"T-RG")
sain=os.path.join(train,"Sain_train")
tumeur=os.path.join(train,"Tumeur_train")
#sain=os.path.join(train,"strr512-2")
#tumeur=os.path.join(train,"ttr512-2")
#sain=os.path.join(train,"Sgreen-train")
#tumeur=os.path.join(train,"Tgreen-train")
#sain=os.path.join(train,"****-train")
#tumeur=os.path.join(train,"Tgreen-train")
S=len(os.listdir(sain))
T=len(os.listdir(tumeur))
print(S,T)
for i in range(S) :
    image=cv2.imread(os.path.join(sain,os.listdir(sain)[i]))
    list_input.append(image)
    list_label.append(0)


for i in range(T) :
    image=cv2.imread(os.path.join(tumeur,os.listdir(tumeur)[i]))
    list_input.append(image)
    list_label.append(1)
    
input_train = np.array(list_input)
output_train =[]
output_train = np.array(list_label)
output_train = np.reshape(output_train,(S+T,1))

list_input_val=[]
list_label_val=[]
validation=os.path.join(image_directory,validation_subdirs)
#sain=os.path.join(validation,"Sain_Validate2")
#tumeur=os.path.join(validation,"Tumeur_Validate2")
#sain=os.path.join(validation,"svrr512")
#tumeur=os.path.join(validation,"tvr512")
#sain=os.path.join(validation,"Val-S-RG")
#tumeur=os.path.join(validation,"Val-T-RG")
sain=os.path.join(validation,"Sain_val")
tumeur=os.path.join(validation,"Tumeur_val")
#sain=os.path.join(validation,"svrr512-3")
#tumeur=os.path.join(validation,"tvr512-3")
#sain=os.path.join(validation,"Sgreen-val")
#tumeur=os.path.join(validation,"Tgreen-val")
S=len(os.listdir(sain))
T=len(os.listdir(tumeur))
for i in range(S) :
    image=cv2.imread(os.path.join(sain,os.listdir(sain)[i]))
    list_input_val.append(image)
    list_label_val.append(0)

for i in range(T) :
    image=cv2.imread(os.path.join(tumeur,os.listdir(tumeur)[i]))
    list_input_val.append(image)
    list_label_val.append(1)

input_val = np.array(list_input_val)
output_val = np.array(list_label_val)
output_val = np.reshape(output_val,(S+T,1))

checkpoint = tf.keras.callbacks.ModelCheckpoint("model-3.hdf5",monitor='val_loss',save_best_only=True, save_weights_only=True,mode='min')
history=model.fit(input_train, output_train,validation_data=(input_val,output_val),callbacks=[checkpoint], batch_size=32, epochs=100,verbose=2)

model.load_weights("/home/path/to/file/model-3.hdf5")


list_input_test=[]
list_label_test=[]
test=os.path.join(image_directory,test_subdirs)
#sain=os.path.join(test,"Sain_Test2")
#tumeur=os.path.join(test,"Tumeur_Test2")
#sain=os.path.join(test,"strr512")
#tumeur=os.path.join(test,"ttr512")
#sain=os.path.join(test,"test-S-RG")
#tumeur=os.path.join(test,"test-T-RG")
sain=os.path.join(test,"Sain_test")
tumeur=os.path.join(test,"Tumeur_test")
#sain=os.path.join(test,"strr512-1")
#tumeur=os.path.join(test,"ttr512-1")
#sain=os.path.join(test,"Sgreen-test")
#tumeur=os.path.join(test,"Tgreen-test")
S=len(os.listdir(sain))
T=len(os.listdir(tumeur))
print(S,T)
for i in range(S) :
    image=cv2.imread(os.path.join(sain,os.listdir(sain)[i]))
    list_input_test.append(image)
    list_label_test.append(0)

for i in range(T) :
    image=cv2.imread(os.path.join(tumeur,os.listdir(tumeur)[i]))
    list_input_test.append(image)
    list_label_test.append(1)
    
input_test = np.array(list_input_test)
output_test =[]
output_test = np.array(list_label_test)
output_test = np.reshape(output_test,(S+T,1))

output_test_predict=model.predict(input_test)

output_test_predict=model.predict(input_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(output_test, output_test_predict)



from sklearn.metrics import auc

fpr_keras, tpr_keras, thresholds_keras =roc_curve(output_test, output_test_predict)

roc = {
	"fpr_keras":   fpr_keras,
	"tpr_keras":   tpr_keras,
	"thresholds_keras":   thresholds_keras,
	}

with open('/home/path/to/file/New-data-test/datas-3', 'wb') as fichier:
	mon_pickler = pickle.Pickler(fichier)
	mon_pickler.dump(roc)
###########################################
fichier = open("/home/path/to/file/New-data-test/results-RGB-3.txt", "w")
history_dict =history.history 
fichier.write(str(history_dict))



print("Temps d execution : %s secondes ---" % (time.time() - start_time))
test_labels_c=model.predict(input_test)


TP=0
TN=0
FP=0
FN=0
seuil=0.5

for i in range(S+T) :
	if test_labels_c[i]>=seuil :
		test_labels_c[i]=1
		if output_test[i]==1 :
			TP+=1
		else : 
			FP+=1
	else :
		test_labels_c[i]=0
		if output_test[i]==0 :
			TN+=1
		else :
			FN+=1
fichier.write("TP : ")
fichier.write(str(TP))
fichier.write(" TN : ")
fichier.write(str(TN))
fichier.write(" FP : ")
fichier.write(str(FP))
fichier.write(" FN : ")
fichier.write(str(FN))
fichier.write(" la specificite ou precision : ")
fichier.write(str(TP/(TP+FP)))
fichier.write(" la sensitivite ou rappel : ")
fichier.write(str(TP/(TP+FN)))
fichier.write(" Accuracy : ")
fichier.write(str((TP+TN)/(TP+TN+FN+FP)))

print('TP : ', TP)
print('TN : ', TN)
print('FP : ', FP)
print('FN : ', FN)
print('la specificite ou precision : ', TP/(TP+FP))
print('la sensitivite ou rappel : ', TP/(TP+FN))
print(" Accuracy : ", (TP+TN)/(TP+TN+FN+FP))
fichier.close()
"""
Interpretation with GradCam heatmap chapter 
"""
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.IMREAD_GRAYSCALE):
                        #IMREAD_GRAYSCALE / COLORMAP_VIRIDIS
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

#image = cv2.imread('/home/bouchra/home/path/to/file/image.tif')
image = cv2.resize(image, (512, 512))
image = image.astype('uint8') / 255
image = np.expand_dims(image, axis=0)

preds = model.predict(image) 
i = np.argmax(preds[0])

icam = GradCAM(model, i, 'conv2d_5')  #change name layer according to the corresponding last layer of you architecture
heatmap = icam.compute_heatmap(image)
heatmap = cv2.resize(heatmap, (512, 512))

image = cv2.imread('/home/bouchra/home/path/to/file/image.tif')
image = cv2.resize(image, (512, 512))
print(heatmap.shape, image.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)

ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)

print ("image.png - python Debug")
cv2.imwrite('image_13.png', output)




