import cv2, numpy as np, os

#parameters
working_dir = '/home/stephen/Desktop/shapes/data/'
os.chdir(working_dir)
img_size = 60 #size of image fed into model

def flatten(dimData, images):
    images = np.array(images)
    images = images.reshape(len(images), dimData)
    images = images.astype('float32')
    images /=255
    return images

#-------------get train/test data-----------------
#get data
folders, labels, images = ['triangle', 'star', 'square', 'circle'], [], []
for folder in folders:
    print folder
    for path in os.listdir(os.getcwd()+'/'+folder):
        img = cv2.imread(folder+'/'+path,0)
        #cv2.imshow('img', img)
        #cv2.waitKey(1)
        images.append(cv2.resize(img, (img_size, img_size)))
        labels.append(folders.index(folder))
    
#break data into training and test sets
to_train= 0
train_images, test_images, train_labels, test_labels = [],[],[],[]
for image, label in zip(images, labels):
    if to_train<5:
        train_images.append(image)
        train_labels.append(label)
        to_train+=1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0

#-----------------keras time --> make the model
from keras.utils import to_categorical

#flatten data
dataDim = np.prod(images[0].shape)
train_data  = flatten(dataDim, train_images)
test_data = flatten(dataDim, test_images)

#change labels to categorical
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#determine the number of classes
classes = np.unique(train_labels)
nClasses  = len(classes)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#three layers
#activation function: both
#neurons: 256
model = Sequential()
model.add(Dense(256, activation = 'tanh', input_shape = (dataDim,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size = 256, epochs=50, verbose=1,
                    validation_data=(test_data, test_labels_one_hot))

#test model
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
#save model
model.save('/home/stephen/Desktop/shapes/shapesmodel.h5')
