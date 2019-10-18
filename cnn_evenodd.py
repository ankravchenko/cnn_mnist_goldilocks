# TensorFlow and tf.keras
import tensorflow as tf
mnist = tf.keras.datasets.mnist

import random
from datetime import datetime
from tensorflow import keras

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, MaxPooling2D
ttt=tf.layers.Conv2D

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


from random import randrange
import sys
import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#train_images = train_images / 255.0
#test_images = test_images / 255.0

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()


spacing1 = numpy.zeros(28)
spacing=spacing1[:,None]

spacing1 = numpy.zeros(28*29)
filler = spacing1.reshape(28,29)


#(28,57 digit)


#train_images=pickle.load(open("trainimages.p", "rb"))
#train_labels=pickle.load(open("trainlabels.p", "rb"))

#test_images=pickle.load(open("testimages.p", "rb"))
#test_labels=pickle.load(open("testlabels.p", "rb"))

train_images_1digit=numpy.zeros(shape=(60000,84,84))
train_labels_1digit=numpy.zeros(60000,dtype=int)
train_images_2digit=numpy.zeros(shape=(60000,84,84))
train_labels_2digit=numpy.zeros(60000,dtype=int)

empty_digit=numpy.zeros(28*28)


spacing2=numpy.zeros(14*56)
filler2=spacing2.reshape(14,56)
spacing1=numpy.zeros(14*28)
filler1=spacing1.reshape(28,14)
#new2digit_half=numpy.concatenate((train_images[i],train_images[j]),axis=1)
#new2digit=numpy.concatenate((filler2,new2digit_half,filler2),axis=0)

for i in range(0,60000):
	j=2
	pos1x=random.randint(0,2)
	pos1y=random.randint(0,2)
	while True:
		pos2x=random.randint(0,2)
		pos2y=random.randint(0,2)
		if (pos1x,pos1y)!=(pos2x,pos2y):
			break	
	train_images_2digit[i,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = train_images[i]
	train_images_2digit[i,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = train_images[j]
	train_labels_2digit[i]=(train_labels[i]+train_labels[j])%2
	#print(new2digit_label)


	pos0x=random.randint(0,2)
	pos0y=random.randint(0,2)
	train_images_1digit[i,pos0y*28:pos0y*28+28,pos0x*28:pos0x*28+28] = train_images[i]
	train_labels_1digit[i]=train_labels[i]

number_of_sets=numpy.unique(train_labels_2digit).size
print('Unique 2-digit numbers:', number_of_sets)


test_images_1digit=numpy.zeros(shape=(10000,84,84))
test_labels_1digit=numpy.zeros(10000,dtype=int)
test_images_2digit=numpy.zeros(shape=(10000,84,84))
test_labels_2digit=numpy.zeros(10000,dtype=int)

j=0
for k in range(0,10000):
	if test_labels[k]==train_labels[2]:
		j=k;
		break

for i in range(0,10000):
	pos1x=random.randint(0,2)
	pos1y=random.randint(0,2)
	while True:
		pos2x=random.randint(0,2)
		pos2y=random.randint(0,2)
		if (pos1x,pos1y)!=(pos2x,pos2y):
			break	
	test_images_2digit[i,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = test_images[i]
	test_images_2digit[i,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = test_images[j]

	test_labels_2digit[i]=(test_labels[i]+test_labels[j])%2
	

	pos0x=random.randint(0,2)
	pos0y=random.randint(0,2)
	test_images_1digit[i,pos0y*28:pos0y*28+28,pos0x*28:pos0x*28+28] = test_images[i]
	test_labels_1digit[i]=test_labels[i]


number_of_test_sets=numpy.unique(test_labels_2digit).size
print('Unique 2-digit numbers in test set:', number_of_test_sets, 'digit:', numpy.unique(test_labels_2digit))





#train mode 1 - goldilocks



goldilocks_train_images1=numpy.zeros(shape=(10000,84,84))
goldilocks_train_labels1=numpy.zeros(10000,dtype=int)
goldilocks_train_images2=numpy.zeros(shape=(10000,84,84))
goldilocks_train_labels2=numpy.zeros(10000,dtype=int)


normal_train_images1=numpy.zeros(shape=(10000,84,84))
normal_train_labels1=numpy.zeros(10000,dtype=int)
normal_train_images2=numpy.zeros(shape=(10000,84,84))
normal_train_labels2=numpy.zeros(10000,dtype=int)

for i in range(0,10000):
	goldilocks_train_images1[i] = train_images_1digit[i]
	goldilocks_train_labels1[i] = train_labels_1digit[i]
	normal_train_images1[i] = train_images_2digit[i]
	normal_train_labels1[i] = train_labels_2digit[i]

for i in range(0,10000):
	goldilocks_train_images2[i] = train_images_2digit[i]
	goldilocks_train_labels2[i] = train_labels_2digit[i]
	normal_train_images2[i] = train_images_2digit[i]
	normal_train_labels2[i] = train_labels_2digit[i]

#train mode 2 - starting hard


#layer settings. how many for the hierarchy to kick in and which ones for cnn


#################################1-digit test#########################
#train_images_1digit = train_images_1digit / 255.0
#test_images_1digit = test_images_1digit / 255.0

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(56, 56)),
#  tf.keras.layers.Dense(512, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#])
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#model.fit(train_images_1digit, train_labels_1digit, epochs=5)
#test_loss, test_acc = model.evaluate(test_images_1digit, test_labels_1digit)

#print('Test accuracy, 1-digit learning:', test_acc) #97% acc result 

########################################################

normal_train_images1 = normal_train_images1 / 255.0
normal_train_images2 = normal_train_images2 / 255.0
goldilocks_train_images1 = goldilocks_train_images1 / 255.0
goldilocks_train_images2 = goldilocks_train_images2 / 255.0
test_images_2digit = test_images_2digit / 255.0
test_images_1digit = test_images_1digit / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images_1digit[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels_1digit[i])
plt.savefig("train_1digit.png")

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_2digit[i], cmap=plt.cm.binary)
    plt.xlabel(test_labels_2digit[i])
plt.savefig("test_2digit.png")

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(goldilocks_train_images2[i], cmap=plt.cm.binary)
    plt.xlabel(goldilocks_train_labels2[i])
plt.savefig("goldilocks_2digit.png")

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(normal_train_images2[i], cmap=plt.cm.binary)
    plt.xlabel(normal_train_labels2[i])
plt.savefig("normaltraining_2digit.png")

normal_train_images1 = normal_train_images1.reshape(10000,84,84,1)
normal_train_images2 = normal_train_images2.reshape(10000,84,84,1)
goldilocks_train_images1 = goldilocks_train_images1.reshape(10000,84,84,1)
goldilocks_train_images2 = goldilocks_train_images2.reshape(10000,84,84,1)
test_images_2digit = test_images_2digit.reshape(10000,84,84,1)
test_images_1digit = test_images_1digit.reshape(10000,84,84,1)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(84,84,1)))
model.add(AveragePooling2D())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.add(Dense(2, activation='softmax'))
model.add(Dense(2, activation='softmax'))

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(56, 56)),
#  tf.keras.layers.Dense(512, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(number_of_sets+1, activation=tf.nn.softmax)
#])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())


model.fit(normal_train_images1, normal_train_labels1, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(test_images_2digit, test_labels_2digit)
print('Test accuracy, 2-digit half learning:', test_acc)

model.fit(normal_train_images2, normal_train_labels2, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
test_loss, test_acc = model.evaluate(test_images_2digit, test_labels_2digit)

print('Test accuracy, 2-digit primitive learning:', test_acc)


#test_loss, test_acc = model.evaluate(test_images_1digit, test_labels_1digit)

#print('Test accuracy on 1 digit, 2-digit normal learning:', test_acc)


#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(84,84,1)))
model.add(AveragePooling2D())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(56, 56)),
#  tf.keras.layers.Dense(512, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(number_of_sets+1, activation=tf.nn.softmax)
#])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(goldilocks_train_images1, goldilocks_train_labels1, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])


#test_loss, test_acc = model.evaluate(test_images_2digit, test_labels_2digit)

#print('Test accuracy, Goldilocks half learning:', test_acc)


model.add(Dense(2, activation='softmax'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())

model.fit(goldilocks_train_images2, goldilocks_train_labels2, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
test_loss, test_acc = model.evaluate(test_images_2digit, test_labels_2digit)

print('Test accuracy, Goldilocks learning:', test_acc)




