
# coding: utf-8

# In[1]:


from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[2]:


print(x.shape)


# In[ ]:


from keras.applications.vgg16 import VGG16

from keras.layers import Flatten,Dense,Dropout
from keras.models import Model

from keras.utils import to_categorical
import cv2
import numpy as np


# In[ ]:


model_vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (48,48,3))

# vgg的层设为“不可训练的”
for layer in model_vgg.layers:
    layer.trainable = False
        
model = Flatten()(model_vgg.output)
model = Dense(4096, activation = 'relu')(model)
model = Dense(4096, activation = 'relu')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation = 'softmax')(model)

model_mnist_VGG = Model(inputs = model_vgg.input, outputs = model, name = 'vgg16')


# In[ ]:


# model_mnist.summary()


# In[ ]:


from keras.optimizers import SGD

sgd = SGD(lr = 0.05, decay = 1e-5)
model_mnist_VGG.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


# (X_train, y_train),(X_test, y_test) = mnist.load_data()

# 训练集中前10000张数据，测试集中前1000张数据
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:1000], y_test[:1000]

#用 opencv 把图像处理一下，大小和通道
#X_train = [cv2.cvtColor]

print(x_train.shape)
print(y_train.shape)


# In[ ]:


# 将 训练集图像 改变大小
#X_train = X_train.reshape((10000, 28, 28, 1))
#X_test = X_test.reshape((1000, 28, 28, 1))
x_train = [cv2.cvtColor(cv2.resize(i,(48,48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i,(48,48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[ ]:


model_mnist_VGG.fit(x_train, y_train, epochs = 20, batch_size=64)


# In[ ]:


result = model_mnist_VGG.evaluate(x_test, y_test)
print('\n Test Acc:', result[1])


# In[ ]:




