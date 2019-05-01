from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from keras.utils import to_categorical
import h5py

def docld(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

def setld(filename):
	doc = docld(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def cdescld(filename, dataset):
	doc = docld(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = 'S ' + ' '.join(image_desc) + ' E'
			descriptions[image_id].append(desc)
	return descriptions

def picload(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features


def linewise(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

def encode(descriptions):
	lines = linewise(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def makeseq(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			seq = tokenizer.texts_to_sequences([desc])[0]
			for i in range(1, len(seq)):
				in_seq, out_seq = seq[:i], seq[i]
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

def max_length(descriptions):
	lines = linewise(descriptions)
	return max(len(d.split()) for d in lines)
  
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = setld(filename)
train_descriptions = cdescld('descriptions.txt', train)
train_features = picload('features.pkl', train)
tokenizer = encode(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(train_descriptions)
print(" Train Dataset length is "+str(len(train)))
x1_train, x2_train, y_train = makeseq(tokenizer, max_length, train_descriptions, train_features)


filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = setld(filename)
test_descriptions = cdescld('descriptions.txt', test)
test_features = picload('features.pkl', test)
print(" Test Dataset length is "+str((len(test))))
x1_test, x2_test, y_test = makeseq(tokenizer, max_length, test_descriptions, test_features)


hf = h5py.File('data.h5', 'w')

hf.create_dataset('x1_train', data=x1_train)
hf.create_dataset('x2_train', data=x2_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x1_test', data=x1_test)
hf.create_dataset('x2_test', data=x2_test)
hf.create_dataset('y_test', data=y_test)

hf.close()

print("Done")