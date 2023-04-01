# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:15:14 2023

@author: Bharathi/dell
"""

"""
KNN ALGORITHM-Its used for classification and regression predictive problems.
Non-Parametric Method.
Instance Based Learning Algorithm.
Its a lazy learning algorithm works well with small dataset.
"""
# =============================================================================
################################# Required packages ############################

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier  

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

###############################################################################

# Importing data

from sklearn.datasets import load_iris

iris=load_iris()

# Get to know the Dataset

iris.data

iris.feature_names

iris.target

iris.target_names

#Data Pre-processing for KNN

# Storing the input(independent) values in y
x=iris.data

# Storing the output(dependent) values in y
y=iris.target

# Splitting the data into train and test(test=30%)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=4)


# KNN

# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 3)  

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

# Performance metric check
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())


"""
Effect of K value on classifier
"""
# Calculating error for K values between 1 and 20

Misclassified_sample = []

for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)


ax=sns.lineplot(x=range(1,20),y=Misclassified_sample)
ax.set(ylim=(0, 2),xlim=(0, 20))


"""
CONCLUSION:
    KNN Classifier predicts the Iris Dataset with the accuracy of 97.7%
    K Value from 3 to 20 gives the same accuracy.(K=3,is optimum)"""
    
       














