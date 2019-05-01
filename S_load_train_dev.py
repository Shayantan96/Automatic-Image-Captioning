from pickle import load
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

emb_dim=300

hf = h5py.File("data.h5", 'r')
x1_train =np.array( hf.get("x1_train"))
x2_train = np.array(hf.get("x2_train"))
y_train = np.array(hf.get("y_train"))
x1_test = np.array(hf.get("x1_test"))
x2_test = np.array(hf.get("x2_test"))
y_test = np.array(hf.get("y_test"))
hf.close()

print("Starting Model Creation...")
embedding_matrix = load(open('embedding_matrix/embedding_matrix'+str(emb_dim)+'.pkl','rb'))
#embedding_matrix=load(open('embedding_matrix/embedding_matrix50.pkl','rb'))

def createmodel(vsize,lmax):
    ip1=Input(shape=(4096,))
    f1=Dropout(0.25)(ip1)
    f2=Dense(emb_dim,activation='relu')(f1)
    ip2=Input(shape=(lmax,))
    s1=Embedding(vsize,emb_dim, weights=[embedding_matrix], input_length=lmax, trainable=False)(ip2)
    s2=Dropout(0.25)(s1)
    s3=LSTM(emb_dim)(s2)
    d1=add([f2, s3])
    d2=Dense(emb_dim,activation='relu')(d1)
    op=Dense(vsize,activation='softmax')(d2)
    mod=Model(inputs=[ip1 ,ip2],outputs=op)
    mod.compile(loss='categorical_crossentropy',optimizer='adam')
    print(mod.summary())
    return mod


tokenizer=load(open('tokenizer.pkl','rb'))
vocab_size = len(tokenizer.word_index) + 1 

M=createmodel(vocab_size, 34)
filepath = 'model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
M.fit([x1_train, x2_train], y_train, epochs=10, verbose=1, callbacks=[checkpoint], 
      validation_data=([x1_test, x2_test], y_test))
