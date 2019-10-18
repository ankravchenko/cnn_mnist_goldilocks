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

#tensorboard logs
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


from random import randrange
import sys
import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)

print(tf.__version__)

#load MNIST set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#load MNIST set from disk:
#train_images=pickle.load(open("trainimages.p", "rb"))
#train_labels=pickle.load(open("trainlabels.p", "rb"))
#test_images=pickle.load(open("testimages.p", "rb"))
#test_labels=pickle.load(open("testlabels.p", "rb"))


def random_pos():
	pos1x=random.randint(0,2)
	pos1y=random.randint(0,2)
	while True:
		pos2x=random.randint(0,2)
		pos2y=random.randint(0,2)
		if (pos1x,pos1y)!=(pos2x,pos2y):
			break	
	return pos1x,pos2x,pos1y,pos2y

#unfinished and not used at the moment
def scatter_2_digits(pos1x,pos2x,pos1y,pos2y,i,j):
	new_digit=numpy.zeros(84*84)
	new_digit[pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = train_images[i]
	new_digit[pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = train_images[j]
	ew_digit_label=train_labels[i]
	return new_digit, new_digit_label


def main():
	train_images_1digit=numpy.zeros(shape=(60000,84,84))
	train_labels_1digit=numpy.zeros(60000,dtype=int)
	train_images_2digit=numpy.zeros(shape=(60000,84,84))
	train_labels_2digit=numpy.zeros(60000,dtype=int)

	test_images_1digit=numpy.zeros(shape=(10000,84,84))
	test_labels_1digit=numpy.zeros(10000,dtype=int)
	test_images_2digit=numpy.zeros(shape=(10000,84,84))
	test_labels_2digit=numpy.zeros(10000,dtype=int)

	empty_digit=numpy.zeros(28*28)

	#create new training set, digits scattered in 3x3 square
	for i in range(0,60000):
		#2-digit training set
		j=2 #first digit is fixed for now. if goldilocks is better, it will be better on this set as well
		pos1x,pos2x,pos1y,pos2y=random_pos()
		train_images_2digit[i,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = train_images[i]
		train_images_2digit[i,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = train_images[j]
		train_labels_2digit[i]=train_labels[i]
		
		#1-digit training set
		pos0x=random.randint(0,2)
		pos0y=random.randint(0,2)
		train_images_1digit[i,pos0y*28:pos0y*28+28,pos0x*28:pos0x*28+28] = train_images[i]
		train_labels_1digit[i]=train_labels[i]

	number_of_sets=numpy.unique(train_labels_2digit).size
	print('Unique 2-digit numbers:', number_of_sets)


	#fixed digit in test set should be the same as in training set
	j=0
	for k in range(0,10000):
		if test_labels[k]==train_labels[2]:
			j=k;
			break

	#create new test set, digits scattered in 3x3 square
	for i in range(0,10000):
		#2-digit test set
		pos1x,pos2x,pos1y,pos2y=random_pos()
		test_images_2digit[i,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = test_images[i]
		test_images_2digit[i,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = test_images[j]
		test_labels_2digit[i]=test_labels[i]
		#1-digit test set
		pos0x=random.randint(0,2)
		pos0y=random.randint(0,2)
		test_images_1digit[i,pos0y*28:pos0y*28+28,pos0x*28:pos0x*28+28] = test_images[i]
		test_labels_1digit[i]=test_labels[i]


	number_of_test_sets=numpy.unique(test_labels_2digit).size
	print('Unique 2-digit numbers in test set:', number_of_test_sets)


	#breaking sets into 2 parts for better logging. only using part of the set atm to make it harder for NN

	#train mode 1 - goldilocks
	goldilocks_train_images1=train_images_1digit[:10000]
	goldilocks_train_labels1=train_labels_1digit[:10000]
	goldilocks_train_images2=train_images_2digit[10000:20000]
	goldilocks_train_labels2=train_labels_2digit[10000:20000]

	#train mode 2 - regular
	normal_train_images1=train_images_2digit[:10000]
	normal_train_labels1=train_labels_2digit[:10000]
	normal_train_images2=train_images_2digit[10000:20000]
	normal_train_labels2=train_labels_2digit[10000:20000]


##################1-digit test, old code####################
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

	#normalize
	normal_train_images1 = normal_train_images1 / 255.0
	normal_train_images2 = normal_train_images2 / 255.0
	goldilocks_train_images1 = goldilocks_train_images1 / 255.0
	goldilocks_train_images2 = goldilocks_train_images2 / 255.0
	test_images_2digit = test_images_2digit / 255.0
	test_images_1digit = test_images_1digit / 255.0

############plot dataset for debug############
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

####################################

	normal_train_images1 = normal_train_images1.reshape(10000,84,84,1)
	normal_train_images2 = normal_train_images2.reshape(10000,84,84,1)
	goldilocks_train_images1 = goldilocks_train_images1.reshape(10000,84,84,1)
	goldilocks_train_images2 = goldilocks_train_images2.reshape(10000,84,84,1)
	test_images_2digit = test_images_2digit.reshape(10000,84,84,1)
	test_images_1digit = test_images_1digit.reshape(10000,84,84,1)

	#create model
	model = Sequential()
	model.add(Conv2D(64, kernel_size=3, name='conv1', activation='relu', input_shape=(84,84,1)))
	model.add(AveragePooling2D(name='pool1'))
	model.add(Conv2D(32, kernel_size=3, name='conv2', activation='relu'))
	model.add(AveragePooling2D(name='pool2'))
	model.add(Flatten())
	model.add(Dense(10, activation='softmax'))

	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
      	        metrics=['accuracy'])

	print(model.summary())

	
	model.fit(normal_train_images1, normal_train_labels1, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = model.evaluate(test_images_2digit, test_labels_2digit)
	print('Test accuracy, 2-digit half learning:', test_acc)

	model.layers.pop() #doesn't work for some reason, code below is a workaround
	model2 = Model(model.input, model.layers[-1].output)
	model2.get_layer(name='conv1').trainable = False
	model2.get_layer(name='pool1').trainable = False
	model2.get_layer(name='conv2').trainable = False
	model2.get_layer(name='pool2').trainable = False


	print(model2.summary())
	
	model2.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
      	        metrics=['accuracy'])
	
	new_model = Sequential()
	new_model.add(model2)
	new_model.add(Dense(10, activation='softmax'))

	new_model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])


	print(new_model.summary())

	new_model.fit(normal_train_images2, normal_train_labels2, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = new_model.evaluate(test_images_2digit, test_labels_2digit)

	print('Test accuracy, 2-digit primitive learning:', test_acc)

	test_loss, test_acc = new_model.evaluate(test_images_1digit, test_labels_1digit)

	print('Test accuracy on 1 digit, primitive learning:', test_acc)

#test_loss, test_acc = model.evaluate(test_images_1digit, test_labels_1digit)

#print('Test accuracy on 1 digit, 2-digit normal learning:', test_acc)


	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(64, kernel_size=3, name='conv1', activation='relu', input_shape=(84,84,1)))
	model.add(AveragePooling2D(name='pool1'))
	model.add(Conv2D(32, kernel_size=3, name='conv2', activation='relu'))
	model.add(AveragePooling2D(name='pool2'))
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

	model.fit(goldilocks_train_images1, goldilocks_train_labels1, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=	[tensorboard_callback])

	test_loss, test_acc = model.evaluate(test_images_1digit, test_labels_1digit)

	print('Test accuracy on 1 digit, half learning:', test_acc)

	test_loss, test_acc = model.evaluate(test_images_2digit, test_labels_2digit)

	print('Test accuracy, Goldilocks half learning:', test_acc)

	model.layers.pop() 
	model2 = Model(model.input, model.layers[-1].output)

	print(model2.summary())

	new_model = Sequential()
	new_model.add(model2)
	new_model.add(Dense(10, activation='softmax'))

	new_model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])


	print(new_model.summary())

	new_model.fit(goldilocks_train_images2, goldilocks_train_labels2, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = new_model.evaluate(test_images_2digit, test_labels_2digit)

	print('Test accuracy, Goldilocks learning:', test_acc)


main()

test_loss, test_acc = new_model.evaluate(test_images_1digit, test_labels_1digit)

print('Test accuracy on 1 digit, Goldilocks learning:', test_acc)


