# Practical Machine Learning: Prediction Assignment Writeup
## Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self-movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The goal of this project was to build and train a machine learning algorithm to recognize the manner in which a particular exercise (lifting a barbell) is performed by a subject based on data collected from accelerometers (on the belt, forearm, arm, and dumbell) attached to the person performing the exercises. The data set consists of data from six different participants and the outcome is classified into five different categories, either correctly or incorrectly - 4 variations of common mistakes. The main objective is to implement a classifier that correctly categorizes 20 samples provided as a testing set.


## Data Processing
Loading the required R libraries:

```r
library(caret);library(kernlab)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

And let's fix the seed to make the experience reproductive


```r
set.seed(32323)
```

Let's load the train datasets.


```r
# Load the training dataset:
dataset  <- read.csv("pml-training.csv", na.strings=c("NA",""), strip.white=TRUE)
dim(dataset)
```

```
## [1] 19622   160
```

We have 19622 training examples composed of the recording of 160 measures. 

## Cleaning the data and Selection of features.

Removing of  NAs values (present in a very large in number), Changing classe to factor and Removing columns with high correlation.
In summary, many columns have been discarded. We have reduced the training set to 19622 training examples composed of the recording of 53 usable measures. These remaining feature variables are used to predict the variable "classe".


```r
isNA <- apply(dataset, 2, function(x) { sum(is.na(x)) })

dataset <- subset(dataset[, which(isNA == 0)], 
                    select=-c(X, user_name, new_window, num_window, 
                              raw_timestamp_part_1, raw_timestamp_part_2, 
                              cvtd_timestamp))
dim(dataset)
```

```
## [1] 19622    53
```


## Model Building

Creation of the training set and validation set

In order to measure the performances of our future predictor we split the dataset into a training set and a validation set. The first one will be used to train the predictor, while the second one will be used to assess the performances.
The split used is: 75% for the training set and 25% for the validation set.


```r
# Spliting the dataset:
inTrain <- createDataPartition(dataset$classe, p=0.75, list=FALSE)
train_set <- dataset[inTrain,]
valid_set <- dataset[-inTrain,]

folds <- createFolds(y=dataset$classe,k=10,
                     list=TRUE,returnTrain=TRUE)
sapply(folds,length)
```

```
## Fold01 Fold02 Fold03 Fold04 Fold05 Fold06 Fold07 Fold08 Fold09 Fold10 
##  17660  17658  17660  17658  17661  17659  17662  17661  17660  17659
```
## Training some predictors and cross validating

We first try an powerful yet easy to parametrize predictor to have an idea of what performance could be obtained. We train a Random Forest Classifier.


```r
# Random forest classifier using cross validation contol
ctrl <- trainControl(allowParallel=TRUE, method="cv", number=4)
model <- train(classe ~ ., data=train_set, model="rf", trControl=ctrl)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
predictor <- predict(model, newdata=valid_set)
```

Let's now assess the generalization performance of this classifier by computing the error made on the validation set.


```r
# Error on valid_set:
sampleError <- sum(predictor == valid_set$classe) / length(predictor)
sampleError
```

```
## [1] 0.9955
```


## Confusion Matrix
The confusion matrix allows visualization of the performance of an machine learning algorithm - typically a supervised learning. Each column of the matrix represents the instances in a predicted class, while each row represents the instances in an actual (reference) class.


```r
confusionMatrix(valid_set$classe, predictor)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    0    1    0    0
##          B    9  937    3    0    0
##          C    0    0  853    2    0
##          D    0    0    4  800    0
##          E    0    0    1    2  898
```

```r
confusionMatrix(valid_set$classe, predictor)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9955         0.9943         0.9932         0.9972         0.2861 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```
Therefore, the out of sample error estimated with cross-validation is **0.9955**.We can consider this classifier as good.

# Classification for test_set
Finally we apply the model to predict the classe on the testing set.


```r
# Classification for test_set:
dataset_test <- read.csv("pml-testing.csv", na.strings=c("NA",""), strip.white=T)
dataset_test <- subset(dataset_test[, which(isNA == 0)], 
                        select=-c(X, user_name, new_window, num_window,
                                  raw_timestamp_part_1, raw_timestamp_part_2,
                                 cvtd_timestamp))
```

Let's see what are the predictions for the test set. 


```r
# Prediction on the test set:
predictions <- predict(model, newdata=dataset_test)
print(predictions)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

We have the following prediction: 
B A B A A E D B A A B C B A E E A B B B -which has been validated 100% accurately by programming submission.

## Conclusion

Model is perfect for the given testing data so any further analysis is not necessary.

