import tensorflow as tf
import pickle
import numpy as np
import cv2
from keras.utils import np_utils
from glob import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.996):
      print("\nReached 99.6% accuracy so canceling training!")
      self.model.stop_training = True

callbacks = myCallback()

def get_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(os.listdir('gestures'))



image_x, image_y = get_size()


with open('train_images', 'rb') as f:
	train_images = np.array(pickle.load(f))
with open('train_labels', 'rb') as f:
	train_labels = np.array(pickle.load(f), dtype = np.int32)

with open('val_images', 'rb') as f:
	val_images = np.array(pickle.load(f))
with open('val_labels', 'rb') as f:
	val_labels = np.array(pickle.load(f), dtype = np.int32)

train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
#print(len(train_labels))
train_labels = np_utils.to_categorical(train_labels)
#print(len(val_labels))
val_labels = np_utils.to_categorical(val_labels)

num_of_classes = get_num_of_classes()

print(num_of_classes)

model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(16, (2, 2), activation = 'relu', input_shape = (image_x, image_y, 1)),
			tf.keras.layers.MaxPooling2D(3, 3),
			tf.keras.layers.Conv2D(32, (2, 2), activation = 'relu'),
			tf.keras.layers.MaxPooling2D(3, 3),
			tf.keras.layers.Conv2D(64, (2, 2), activation = 'relu'),
			tf.keras.layers.MaxPooling2D(3, 3),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(128, activation = 'relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(num_of_classes, activation = 'softmax')
		])

model.summary()

model.compile(
		optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),
		loss = 'categorical_crossentropy',
		metrics = ['accuracy']
	)	



history = model.fit(
		train_images,
		train_labels,
		validation_data = (val_images, val_labels),
		epochs = 25,
		#steps_per_epoch = 800,
		batch_size = 20,
		callbacks = [callbacks]
	)




import h5py
model.save("trainedModel.h5")



import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')	
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
