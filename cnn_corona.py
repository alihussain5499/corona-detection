
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:41:03 2020

@author: ali hussain
"""



from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('D:/keras-covid-19/dataset',target_size=(64,64),batch_size=32,class_mode='binary')


from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D
 
from keras.layers import Flatten

from keras.layers import Dense


classifier=Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit_generator(training_set,samples_per_epoch=50,nb_epoch=25,nb_val_samples=20)


from keras.models import load_model

classifier.save('D:/keras-covid-19/dataset/my_model_cnn.h5')

model=load_model('D:/keras-covid-19/dataset/my_model_cnn.h5')

import numpy as np

from keras.preprocessing import image

test_image=image.load_img('D:/keras-covid-19/dataset/normal/person1558_bacteria_4066.jpeg',target_size=(64,64))



print(test_image)

test_image=image.img_to_array(test_image)
print(test_image)

print(test_image[0,0,:].size)

test_image=np.expand_dims(test_image,axis=0)
print(np.ndim(test_image))

print(np.shape(test_image))

result=model.predict(test_image)
print("Prediction ",result)

if result>0.5:
    result=1
elif result<0.5:
    result=0
else:
    print("Not Predictable")
    

print(result)

#print(training_set.class_indices)

if (result==1):
    print("Normal")
elif result==0:
    print("Covid")
else:
    print("Given image is neither Normal nor Covid ")
    
    
