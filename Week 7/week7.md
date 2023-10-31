
# Feature Selection & Class Imbalance

## Objectives
- Use filter methods for feature selection, using the VarianceThreshold and SelectKBest selectors in Scikit Learn
- Use wrapper methods for feature selection, such as Recursive Feature Elimination
- Understand the problem of class imbalance and use the imblearn package for resampling, including oversampling, undersampling and SMOTE.
- Use the imshow function from matplotlib to visualise images stored as a Numpy array
  
## Dataset
The dataset for this task is attached to the main Week 7 folder

```
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 %matplotlib inline
```

## File import

Now I can use the genfromtxt function to create a Numpy array from the csv file. I can then define X and y from this Numpy array (the first 784 columns are the features, and the last column is the class label)
```
mnist = np.genfromtxt('MNIST_Shortened.csv', delimiter=',', skip_header=1)
# Define X and y
X = mnist[:,0:784] y = mnist[:,-1]
# Check dimensions of X
X.shape
#X has dimensions (6000, 784).
```

## Exploring the data

```
# I'll use the Numpy reshape function along with the matplotlib imshow function to visualise
plt.imshow(X[0].reshape(28,28),cmap='gray_r')

#I can use the Numpy randint function to select 100 images at random from the dataset and display these using imshow.
plt.figure(figsize=(8, 12))
 
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(X[np.random.randint(0,6000)].reshape(28,28),cmap='gray_r')
    plt.axis('off')
plt.show()

```

## training and test data
```
#I'll use the train_test_split function to split the data into a training dataset and a test dataset.
# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
```
## Variance
```
#using the VarianceThreshold selector from Scikit Learn to remove any features which have no variance
from sklearn.feature_selection import VarianceThreshold 
variance_selector = VarianceThreshold(threshold=0)
X_train_fs = variance_selector.fit_transform(X_train) 
X_test_fs = variance_selector.transform(X_test)
print(f"{X_train.shape[1]-X_train_fs.shape[1]} features have been removed, {X_train_fs.shape[1]} features remain")

#result: 118 features have been removed, 666 features remain 
```

## Checking dropped features
```
# We can use the get_support function to see which features have been dropped
selected_features = variance_selector.get_support()
selected_features = selected_features.reshape(28,28)
# Visualise which pixels have been dropped
sns.heatmap(selected_features,cmap='rocket')
'''The black pixels in the grid are those which have been dropped due to zero variance. Unsurprisingly, these are the pixels around the edges and the corners of the image, given that the digits are centered in the middle of the 28x28 image.'''
```

## Training and Evaluating a Model
```
#instantiate a new RandomForestClassifier model, and then fit it to the data with the selected features.
 
rf_selectedfeatures = RandomForestClassifier()
rf_selectedfeatures.fit(X_train_fs, y_train)

#Predicting
'''
Having fitted the model, we can then use the predict() method to make predictions on the test dataset, and then use the accuracy_score and confusion_matrix functions to evaluate it. We use the seaborn heatmap function to visualise the confusion matrix.'''
 
# Make predictions on the test data
y_pred = rf_selectedfeatures.predict(X_test_fs)
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)*100:.2f}%") 
cm = confusion_matrix(y_test,y_pred)
ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class",fontsize=12) 
plt.ylabel("True Class",fontsize=12) 
plt.title("Confusion Matrix",fontsize=12)
plt.show()

### Accuracy Score: 91.22%
'''So, with less than a quarter of the features in the training data, we have achieved a model accuracy of over 90%.'''
```
## Addressing Class Imbalance
```
#Oversampling and undersampling
# We can implement both oversampling and undersampling using the Imbalanced Learn library.

#Installing a library
pip install -U imbalanced-learn

# this is the second file in the week 7 folder
cardio_data = pd.read_csv('Cardiotocographic.csv')
cardio_data.head() cardio_data.tail()


#To carry out oversampling, we can use RandomOverSampler. You can run the code below to carry out random oversampling. We can then use a countplot to visualise the results.

from imblearn .over_sampling import RandomOverSampler

resampler = RandomOverSampler(random_state=0)
X_train_oversampled, y_train_oversampled = resampler.fit_resample(X_train, y_train)

sns.countplot(x=y_train_oversampled)

#To carry out undersampling we follow the same process, except in this case we want to undersample classes 0 and 1. We use the RandomUnderSampler to achieve this.
from imblearn .under_sampling import RandomUnderSampler

resampler = RandomUnderSampler(random_state=0)
X_train_undersampled, y_train_undersampled = resampler.fit_resample(X_train, y_train)

sns.countplot(x=y_train_undersampled)
```
