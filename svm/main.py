# import scikit-lean dataset library
from sklearn import datasets

# load dataset
cancer = datasets.load_breast_cancer()

# print dataset information
print("\n=== Dataset Information ===")
print("Dataset Name: Breast Cancer Wisconsin (Diagnostic) Dataset")
print("\n=== Features Information ===")
print("Number of Features:", len(cancer.feature_names))
print("Feature Names:", cancer.feature_names)
print("\n=== Target Information ===")
print("Target Names:", cancer.target_names)
print("Target Values (0: malignant, 1: benign):", cancer.target)
print("\n=== Data Shape ===")
print("Shape:", cancer.data.shape)
print("\n=== Sample Data (First 5 records) ===")
print(cancer.data[0:5])
print("\n=== Description ===")
print(cancer.DESCR)

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)


# print data(feature)shape
print("Shape: ", cancer.data.shape)

# print the cancer data features (top 5 records)
print(cancer.data[0:5])


# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25, random_state=123) # 75% training and 25% test


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


