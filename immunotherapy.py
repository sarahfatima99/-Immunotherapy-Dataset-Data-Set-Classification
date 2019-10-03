#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
import numpy


# In[3]:


seed=7
numpy.random.seed(seed)


# In[52]:


dataframe=pandas.read_csv("Immune.csv",header=None)


# In[53]:


dataframe


# In[56]:


dataset=dataframe.values
x=dataset[1:,0:7].astype(float) #dataset[row(:) to row , colum:to co]
y=dataset[1:,7]
print(y)


# In[45]:


from keras import optimizers 
from keras.models import Sequential
from keras.layers import Dense 
def build_model():
    model =Sequential()
    model.add(Dense(50, activation='relu', input_shape=(7,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  ) 
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[46]:



numpy.random.shuffle(dataset)
xtrain=x[:50]
#print(traindata)
ytrain=y[:50]
#print(ytrain)
k=4
num_val_samples = len(xtrain)//k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = xtrain[i * num_val_samples: (i + 1) * num_val_samples]
    print(xtrain.shape)
    val_targets = ytrain[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = numpy.concatenate(
    [xtrain[:i * num_val_samples],
    xtrain[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = numpy.concatenate(
    [ytrain[:i * num_val_samples],
    ytrain[(i + 1) * num_val_samples:]],
    axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=5,verbose=0)
    results = model.evaluate(val_data,val_targets)
    


# In[57]:


xtest=x[50:]
ylabel=y[50:]
print(y)
def build_model():
    model =Sequential()
    model.add(Dense(50, activation='relu', input_shape=(7,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  ) 
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = build_model()
model.fit(xtest, ylabel , epochs=250, batch_size=5 , verbose=1)


# In[ ]:




