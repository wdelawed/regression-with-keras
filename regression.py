from keras.datasets import boston_housing 
from keras import layers
from keras import models 
import matplotlib.pyplot  as plt 
import numpy as np 

(train_data, train_labels) , (test_data, test_labels) = boston_housing.load_data() 

mean = train_data.mean(axis=0) 
train_data -= mean 

std = train_data.std(axis=0) 
train_data /= std 

test_data -= mean 
test_data /= std 

model = models.Sequential() 
model.add(layers.Dense(64, activation='relu',input_shape=(13,)))
model.add(layers.Dense(1))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

val_x = train_data[:100]
val_y = train_labels[:100]

partial_x = train_data[100:]
partial_y = train_labels[100:] 

history = model.fit(partial_x,partial_y,batch_size=20, epochs=100 , validation_data=(val_x, val_y))

history_dic = history.history 

val_loss = history_dic['val_loss'] 
loss = history_dic['loss'] 

epochs = range(1,len(loss)+1)

plt.plot(epochs,val_loss,'b',Label="Validation Loss")
plt.plot(epochs,loss,'bo',Label="Training Loss") 
plt.title("Losses Vs Epochs plot")
plt.xlabel("Epoch")
plt.ylabel("Loss") 
plt.legend()
plt.show()


val_acc = history_dic['val_mae'] 
acc = history_dic['mae'] 

plt.clf() 

plt.plot(epochs, val_acc, 'b', Label="Validation MAE")
plt.plot(epochs, acc, 'bo', Label="Training MAE") 
plt.title("MAE Vs Epochs plot")
plt.xlabel("Epochs")
plt.ylabel("Accuracy") 
plt.legend()
plt.show()
