# Introduction to Neural Networks with Keras

## Objectives
- Use Google Colab and make use of a GPU for running your code
- Use Keras to build and train a neural network
- Develop neural network with appropriate parameters
- Analyse neural network performance using common metrics including accuracy, precision, recall and F1 score.
- 
## Dataset
The file for today's workshop is saved on Blackboard as Cardiotocographic.csv. It has
the following variables.
BPM Beat per minutes
APC Accelerations per second
FMPS Fetal movement per second
UCPS Uterine contractions per second DLPS Light declaration per second
SDPS Severe declaration per second
PDPS Prolonged declaration per second ASTV % of abnormal short term Variability MSTV Mean of short term Variability ALTV % of abnormal long term Variability MLTV Mean of long term Variability
                        4
  Width Width of FHR Histogram
Min Min Width of FHR Histogram Max Max Width of FHR Histogram
Cardiotocography is a recording of the fetal heart rate obtained by ultrasound and is used in pregnancy to assess fetal well‐being. We therefore want to use these variables to predict the target variable (NSP), which is the Fetal State Class code - or in other words, the diagnosis.
There are three values this can take:
• N = Normal (1)
• S = Suspect (2)
• P = Pathologic (3)
A machine learning model which can predict the diagnosis based on the cardiotocographic data could aid clinicians in making an accurate diagnosis. The aim of today's workshop is to investigate whether we can build an artificial neural network to support clinicians in screening pregnancies to identify those which potentially have issues so that they can receive further medical attention.


Libraries to use for this task, here is the code

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

## File import
```
cardio_data = pd.read_csv('Cardiotocographic.csv')
// Checking the first 10 record
cardio_data.head()
```

## Exploring the data

```
cardio_data.describe()
cardio_data.info()
cardio_data.shape()
cardio_data['NSP'].value_counts()
//We can see we have imbalanced classes with 77.8% of the observations belong to the Normal class. We can also use the seaborn countplot //to visualise this.
sns.countplot(cardio_data, x="NSP")

```

## Dividing the data into training and test 
```

X = cardio_data.drop('NSP',axis = 1)
y = cardio_data['NSP'] - 1

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```

#Building and training the neural network
```
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(9,activation='relu',input_shape=(14,)))
model.add(tf.keras.layers.Dense(3,activation='softmax'))
```
