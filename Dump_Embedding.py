from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from pickle import dump

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

def max_length(descriptions):
	lines = linewise(descriptions)
	return max(len(d.split()) for d in lines)

train = setld('Flickr8k_text/Flickr_8k.testImages.txt')
train_descriptions = cdescld('descriptions.txt', train)
lines = linewise(train_descriptions)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
vocab_size = len(tokenizer.word_index) + 1

embeddings_index = dict()
f = open('Embedding/glove.6B.50d.txt', encoding = 'Utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = zeros((vocab_size, 50))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
dump(embedding_matrix, open('embedding_matrixtest50.pkl', 'wb'))
print(embedding_matrix.shape)
print("done")