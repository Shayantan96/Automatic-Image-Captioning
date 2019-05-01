from os import listdir
from pickle import dump
        from keras.applications.vgg16 import VGG16
        from keras.preprocessing.image import load_img
        from keras.preprocessing.image import img_to_array
        from keras.applications.vgg16 import preprocess_input
        from keras.models import Model

def xtract_ft(directory):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print(model.summary())
	features = dict()
	for name in listdir(directory):
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		image_id = name.split('.')[0]
		features[image_id] = feature
		print('>%s' % name)
	return features

dr = 'Flicker8k_Dataset'
features = xtract_ft(dr)
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))
