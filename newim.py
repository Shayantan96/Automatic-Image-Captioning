from pickle import dump
from keras.preprocessing.text import Tokenizer

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

def encode(descriptions):
	lines = linewise(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


f='Flickr8k_text/Flickr_8k.trainImages.txt'
train=setld(f)
t_desc=cdescld(('descriptions.txt'),train)
tokenizer=encode(t_desc)
dump(tokenizer,open('tokenizer.pkl','wb'))
print("dumping done")





