import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Multi-Class Perceptron Learning Model

dataFileName = 'mnist_data.txt'
dataSet = pd.read_csv(dataFileName, header = None,sep=" ")
dataSet.insert(loc=len(dataSet.columns),column=len(dataSet.columns),value=1,allow_duplicates=True)

labelFileName = 'mnist_labels.txt'
labelSet = pd.read_csv(labelFileName, header = None)
labelSet.rename({0:'class'}, axis='columns',inplace=True)

weightMatrix = pd.DataFrame(index=np.arange(len(labelSet['class'].unique())), columns=np.arange(len(dataSet.columns)))
weightMatrix.fillna(value=0.0,inplace=True)

## Splitting Data

xTrain, xTest, yTrain, yTest = train_test_split(dataSet, labelSet, test_size=0.5)

## Getting # iteration and learning rate value from user

iterationValue = int(input("Enter number of iteration: "))
learningRate = float(input("Enter the learning rate: "))

## Training Model

for i in range(iterationValue):
  change = False
  for index,row in xTrain.iterrows():
    tempWeight = weightMatrix
    predictedY = tempWeight.dot(row).idxmax(axis = 0)
    correctY = yTrain.loc[index,'class']
    if(predictedY != correctY):
      change = True
      weightMatrix.loc[correctY,:] = weightMatrix.loc[correctY,:] + learningRate * row
      weightMatrix.loc[predictedY,:] = weightMatrix.loc[predictedY,:] - learningRate * row
  if(change == False):
    break

## Training Set

predictedOutput = pd.DataFrame(index=list(xTrain.index), columns={'class'})
for index,row in xTrain.iterrows():
  tempWeight = weightMatrix
  predictedOutput.loc[index,'class'] = tempWeight.dot(row).idxmax(axis = 0)

## Calculating Accuracy on Training Set

accuracy = accuracy_score(yTrain, predictedOutput)
print("Accuracy: ", accuracy)
print("classification_report:\n",classification_report(yTrain, predictedOutput))

## Test Set

predictedOutput = pd.DataFrame(index=list(xTest.index), columns={'class'})
for index,row in xTest.iterrows():
  tempWeight = weightMatrix
  predictedOutput.loc[index,'class'] = tempWeight.dot(row).idxmax(axis = 0)

## Calculating Accuracy on Test Set

accuracy = accuracy_score(yTest, predictedOutput)
print("Accuracy: ", accuracy)
print("classification_report:\n",classification_report(yTest, predictedOutput))