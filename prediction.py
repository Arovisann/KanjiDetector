# make a prediction for a new image.
from numpy import argmax
import tensorflow as tf
from keras import preprocessing
import keras.preprocessing.image 
from keras.models import load_model


# load and prepare the image
def load_image(filename):
	# load the image
	img = tf.keras.preprocessing.image.load_img(filename, grayscale=False, target_size=(224, 224))
	# convert to array
	img = tf.keras.preprocessing.image.img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 224, 224, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample_image.png')
	# load model
	model = load_model('keras_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)
 
# entry point, run the example
run_example()