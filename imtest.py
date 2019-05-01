from keras.preprocessing.sequence import pad_sequences
from numpy import argmax
from pickle import load
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def wordidmap(integer,tokenizer):
	for w,i in tokenizer.word_index.items():
		if i==integer:
			return w
	return None

def gensentence(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = wordidmap(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

def xtract_ft(fname):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image= load_img(fname,target_size = (224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


lmax=34

tokenizer=load(open('tokenizer.pkl','rb'))
model = load_model('modelw.h5')     

pic = xtract_ft('2847615962_c330bded6e.jpg')      ###########################################################
description = gensentence(model, tokenizer, pic, lmax)
print(description)









