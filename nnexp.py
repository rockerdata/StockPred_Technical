# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 2
# Creating ANN model

import keras
from  keras.models import Sequential
from keras.layers import Dense

# Initializing the layers
classifier = Sequential()

# Adding input layer and hidden layer
classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', activation='relu' ))

# Adding second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu' ))

# Adding output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid' ))

# ANN Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit Model
classifier.fit(X_train, y_train, batch_size=10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


classifier.predict(sc.transform(X_test))


def build_classifier():
        # Initializing the layers
            classifier = Sequential()
                
                    # Adding input layer and hidden layer
                        classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', activation='relu' ))
                            
                                # Adding second hidden layer
                                    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu' ))
                                        
                                            # Adding output layer
                                                classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid' ))
                                                    
                                                        # ANN Compile
                                                            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                                                                
                                                                    return classifier
                                                                    
                                                                from keras.wrappers.scikit_learn import KerasClassifier
                                                                from sklearn.model_selection import cross_val_score
                                                                from sklearn.model_selection import GridSearchCV

                                                                classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, nb_epoch=100)
                                                                accuracies = cross_val_score(classifier, X_train, y_train, cv=6)

                                                                params = {'batch_size':[20,32,50],
                                                                                  'nb_epoch':[200,300,500],
                                                                                            'optimizer':}



