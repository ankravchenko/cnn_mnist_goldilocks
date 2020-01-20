#python 3.5, Tensorflow 1.14

import sys
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

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	

def load_data(): # use as a global variable maybe? less hassle
	#check for pickled and load pickled
	#load MNIST set
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	
	#pickle
	return train_images, train_labels, test_images, test_labels
	
#generate1digit_dataset(n)
#generate2digit(n, digitlist)

def generate_cnn_model():
#create model
	model = Sequential()
	model.add(Conv2D(64, kernel_size=3, name='conv1', activation='relu', input_shape=(84,84,1)))
	model.add(AveragePooling2D(name='pool1'))
	model.add(Conv2D(32, kernel_size=3, name='conv2', activation='relu'))
	model.add(AveragePooling2D(name='pool2'))
	model.add(Flatten(name='flatten'))
	model.add(Dense(10, activation='softmax', name='dense1'))


	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
      	        metrics=['accuracy'])

	return model

def ten_random_2digits():
	digitlist=[];
	while len(digitlist)<10:
		x=random.randint(10,99)
		reversex=int(str(x)[::-1])
		if (x not in digitlist)&(reversex not in digitlist):
			digitlist.append(x);
	return digitlist

def random_pos():
	pos1x=random.randint(0,2)
	pos1y=random.randint(0,2)
	while True:
		pos2x=random.randint(0,2)
		pos2y=random.randint(0,2)
		if (pos1x,pos1y)!=(pos2x,pos2y):
			break	
	return pos1x,pos2x,pos1y,pos2y

def generate_1digit_set(n):
	train_images_1digit=numpy.zeros(shape=(n,84,84)) #FIXME change array name, they are not necessarily training images
	train_labels_1digit=numpy.zeros(n,dtype=int)
	for d in range(0,n):
		pos0x=random.randint(0,2)
		pos0y=random.randint(0,2)
		train_images_1digit[d,pos0y*28:pos0y*28+28,pos0x*28:pos0x*28+28] = train_images[d]
		train_labels_1digit[d]=train_labels[d]		
	return train_images_1digit, train_labels_1digit

def generate_2digit_set(n, twodigit_labels):
	train_images_2digit=numpy.zeros(shape=(n,84,84))
	train_labels_2digit=numpy.zeros(n,dtype=int)
	for l in twodigit_labels:
			l_count=twodigit_labels.index(l);
			j_label= l//10 #1st digit label
			i_label= l%10 #2nd digit label
			i_list = numpy.where(train_labels == i_label)[0]
			j_list = numpy.where(train_labels == j_label)[0]
			for d in range(0,int(n/10)): #FIXME this is only for 10 categories
				i = i_list[random.randint(0,i_list.size-1)]
				j = j_list[random.randint(0,j_list.size-1)]
				pos1x,pos2x,pos1y,pos2y=random_pos()
				train_images_2digit[l_count*int(n/10)+d,pos1y*28:pos1y*28+28,pos1x*28:pos1x*28+28] = train_images[i]
				train_images_2digit[l_count*int(n/10)+d,pos2y*28:pos2y*28+28,pos2x*28:pos2x*28+28] = train_images[j]
				train_labels_2digit[l_count*int(n/10)+d]=l
		

	number_of_sets=numpy.unique(train_labels_2digit).size
	print('Unique 2-digit numbers:', number_of_sets)
	print(numpy.unique(train_labels_2digit))
	return train_images_2digit, train_labels_2digit

def generate_2digit_set_50categories(n): #FIXME incorprate into a previous one with changeable number of catgories
	train_images_2digit=numpy.zeros(shape=(n,84,84))
	train_labels_2digit=numpy.zeros(n,dtype=int)
	

	for i in range(0,n):
		j=random.randint(0,30000)
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


	return train_images_2digit, train_labels_2digit

def main(): 

	print(tf.__version__)
	#command line arguments: 
	n1 = int(sys.argv[1]) #number of examples in the 1st training phase (2n total after full training)
	n2 = int(sys.argv[2]) #number of examples in a training phase (2n total after full training)
	task1_type=sys.argv[3] #phase1
	task2_type=sys.argv[4] #phase2
	print("task1_type=",task1_type)
	print("task2_type=",task2_type)
	architecture=sys.argv[5] #same/add/replace - how do we handle ANN's output layers

	twodigit_labels=ten_random_2digits() #FIXME only works for 10 output categories
	(test_images_1digit, test_labels_1digit) = generate_1digit_set(10000)
	(test_images_2digit, test_labels_2digit) = generate_2digit_set(10000,twodigit_labels)

	if task1_type == "digit1":  
			(goldilocks_phase1_train_images, goldilocks_phase1_train_labels) = generate_1digit_set(n1)
	elif task1_type == "digit2":  
			(goldilocks_phase1_train_images, goldilocks_phase1_train_labels) = generate_2digit_set(n1,twodigit_labels)
			#dothething

	if task2_type == "digit1":  #not really going to use this one, but just in case
			(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_1digit_set(n2)
			(regular_phase1_train_images, regular_phase1_train_labels) = generate_1digit_set(n1)
			(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
	elif task2_type == "digit2": #normal scattered 2-digit set
			(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set(n2,twodigit_labels)
			(regular_phase1_train_images, regular_phase1_train_labels) = generate_2digit_set(n1,twodigit_labels)
			(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
	elif task2_type == "digit2_1stfixed":  
			#FIXME write this part
			(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set(n2,twodigit_labels)
	elif task2_type == "oddeven50": 
			(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set_50categories(n2) 
			(regular_phase1_train_images, regular_phase1_train_labels) = generate_2digit_set_50categories(n1)
			(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
			for l in range(0,10000):
				test_labels_2digit[l] = (test_labels_2digit[l]//10+test_labels_2digit[l]%10)%2
	elif task2_type == "oddeven":  
			(goldilocks_phase2_train_images, goldilocks_phase2_train_labels) = generate_2digit_set(n2,twodigit_labels)
			(regular_phase1_train_images, regular_phase1_train_labels) = generate_2digit_set(n1,twodigit_labels)
			(regular_phase2_train_images, regular_phase2_train_labels) = (goldilocks_phase2_train_images, goldilocks_phase2_train_labels)
			#label modification, FIXME change it into a lambda operator or a function for clarity	
			for l in range(0,n1):
				regular_phase1_train_labels[l] = (regular_phase1_train_labels[l]//10+regular_phase1_train_labels[l]%10)%2
			for l in range(0,n2):
				regular_phase2_train_labels[l] = (regular_phase2_train_labels[l]//10+regular_phase2_train_labels[l]%10)%2
				goldilocks_phase2_train_labels[l] = (goldilocks_phase2_train_labels[l]//10+goldilocks_phase2_train_labels[l]%10)%2
			for l in range(0,10000):
				test_labels_2digit[l] = (test_labels_2digit[l]//10+test_labels_2digit[l]%10)%2

	

	#normalize training set
	regular_phase1_train_images = regular_phase1_train_images / 255.0
	regular_phase2_train_images = regular_phase2_train_images  / 255.0
	goldilocks_phase1_train_images = goldilocks_phase1_train_images / 255.0
	goldilocks_phase2_train_images = goldilocks_phase2_train_images / 255.0
	test_images_2digit = test_images_2digit / 255.0
	test_images_1digit = test_images_1digit / 255.0

	########################################DEBUG###############################################################
	#FIXME put it into a separate function too
	#test output

	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(test_images_1digit[i], cmap=plt.cm.binary)
		plt.xlabel(test_labels_1digit[i])
	plt.savefig("new_test_1digit.png")

	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(test_images_2digit[i], cmap=plt.cm.binary)
		plt.xlabel(test_labels_2digit[i])
	plt.savefig("new_test_2digit.png")

	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(goldilocks_phase1_train_images[i], cmap=plt.cm.binary)
		plt.xlabel(goldilocks_phase1_train_labels[i])
	plt.savefig("new_goldilocks_2digit_phase1.png")

	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(goldilocks_phase2_train_images[i], cmap=plt.cm.binary)
		plt.xlabel(goldilocks_phase2_train_labels[i])
	plt.savefig("new_goldilocks_2digit_phase2.png")

	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(regular_phase1_train_images[i], cmap=plt.cm.binary)
		plt.xlabel(regular_phase1_train_labels[i])
	plt.savefig("new_regular_phase1.png")


	########################################DEBUG###############################################################
	# reshape
	regular_phase1_train_images = regular_phase1_train_images.reshape(n1,84,84,1)
	regular_phase2_train_images = regular_phase2_train_images.reshape(n2,84,84,1)
	goldilocks_phase1_train_images = goldilocks_phase1_train_images.reshape(n1,84,84,1)
	goldilocks_phase2_train_images = goldilocks_phase2_train_images.reshape(n2,84,84,1)
	test_images_2digit = test_images_2digit.reshape(10000,84,84,1)
	test_images_1digit = test_images_1digit.reshape(10000,84,84,1)

	#nn training

	#create model
	model_regular = generate_cnn_model()


	#FIXME this is temporal for oddeven task
	model_regular.add(Dense(2, activation='softmax'))
	model_regular.add(Dense(2, activation='softmax'))
	model_regular.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


	model_goldilocks = generate_cnn_model()
	#print(model.summary())

	

	#regular training phase 1
	model_regular.fit(regular_phase1_train_images, regular_phase1_train_labels, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = model_regular.evaluate(test_images_2digit, test_labels_2digit)
	print('Test accuracy, 2-digit half learning:', test_acc)

	model_goldilocks.fit(goldilocks_phase1_train_images, goldilocks_phase1_train_labels, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = model_goldilocks.evaluate(test_images_2digit, test_labels_2digit)
	print('Test accuracy, goldilocks half learning:', test_acc)
	test_loss, test_acc = model_goldilocks.evaluate(test_images_1digit, test_labels_1digit)
	print('Test accuracy, goldilocks half learning on 1 digit:', test_acc)

	#nn adjustment
	#FIXME proper branching

	if ((architecture=="add") & (task2_type=="oddeven50") or (architecture=="add") & (task2_type=="oddeven")):
		model_goldilocks.add(Dense(2, activation='softmax'))
		model_goldilocks.add(Dense(2, activation='softmax'))
		model_goldilocks.compile(optimizer='adam',
              		loss='sparse_categorical_crossentropy',
              		metrics=['accuracy'])

	#regular training phase 2
	model_regular.fit(regular_phase2_train_images, regular_phase2_train_labels, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = model_regular.evaluate(test_images_2digit, test_labels_2digit)
	print('Test accuracy, 2-digit primitive learning:', test_acc)
	#test_loss, test_acc = model_regular.evaluate(test_images_1digit, test_labels_1digit)
	#print('Test accuracy on 1 digit, primitive learning:', test_acc)


	model_goldilocks.fit(goldilocks_phase2_train_images, goldilocks_phase2_train_labels, epochs=3)#, validation_data=(test_images_2digit, test_labels_2digit), callbacks=[tensorboard_callback])
	test_loss, test_acc = model_goldilocks.evaluate(test_images_2digit, test_labels_2digit)
	print('Test accuracy, 2-digit goldilocks learning:', test_acc)

main()
