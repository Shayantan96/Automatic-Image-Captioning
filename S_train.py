import numpy as np
import h5py
from keras.models import load_model
#from keras import model
from keras.callbacks import ModelCheckpoint

hf = h5py.File("data.h5", 'r')
x1_train =np.array( hf.get("x1_train"))
x2_train = np.array(hf.get("x2_train"))
y_train = np.array(hf.get("y_train"))
x1_test = np.array(hf.get("x1_test"))
x2_test = np.array(hf.get("x2_test"))
y_test = np.array(hf.get("y_test"))
hf.close()
'''x1_train = np.array(f[list(f.keys())[0]])
x2_train = np.array(f[list(f.keys())[1]])
y_train = np.array(f[list(f.keys())[2]])
x1_test = np.array(f[list(f.keys())[3]])
x2_test = np.array(f[list(f.keys())[4]])
y_test = np.array(f[list(f.keys())[5]])'''


new_model = load_model('model.h5') 

filepath = 'model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
new_model.fit([x1_train, x2_train], y_train, epochs=10, verbose=1, callbacks=[checkpoint], 
              validation_data=([x1_test, x2_test], y_test))



