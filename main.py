import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage import data, color, io
from skimage.transform import rescale,resize
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import joblib

# prep data
image_dir = 'C:\\Users\\dasak\HackAi\\clf-data\\clf-data'
im_types = ['empty', 'not_empty']

dataset = []
labels = []
for im_type_index,im_type in enumerate(im_types):
     for file in os.listdir(os.path.join(image_dir,im_type)):
         im_path = os.path.join(image_dir,im_type, file)
         image = io.imread(im_path)
         image = resize(image,(20,20)) #change dimensions if
         dataset.append(image.flatten())
         labels.append(im_type_index)
        #  print()

dataset = np.asarray(dataset)
labels = np.asarray(labels)
# print(labels[:100])
# l = np.array()
    

#gather training and test ds
x_train,x_test,y_train,y_test = train_test_split(dataset, labels, test_size=.2, shuffle =True, stratify=labels)

# train classifier
classifier = SVC()
parameters = [{'gamma':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]
grid_search = GridSearchCV(classifier,parameters)
grid_search.fit(x_train,y_train)



# test performance
estimator = grid_search.best_estimator_
y_pred = estimator.predict(x_test)
score = accuracy_score(y_pred,y_test)
print('{}% of samples were correctly classified'.format(str(score*100)))

# joblib.dump(estimator,'parking_model5.pkl')


