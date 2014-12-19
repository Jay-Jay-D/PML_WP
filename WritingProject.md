# Practical Machine Learning
Prediction Assignment Writeup  


## Practical Machine Learnig

This is the use of machine learning methods applied to the [human activity recognition](http://groupware.les.inf.puc-rio.br/har) data, particularly, the Weight Lifting Exercises Data set

First, we load the needed libraries and the data from the files.

```r
library(caret)
set.seed(1234)
data <- read.csv(pml.training)
inTrain = createDataPartition(data$classe, p = 0.8)[[1]]
training = data[ inTrain,]
testing = data[-inTrain,]
validation <- read.csv(pml.testing)
```

## Cleaning Data 
The data has 160 variables. 
After a first (and quick) visual inspect, it seems that there is some variables with lots of NA, others with almost zero variance.  
I drop the first five variables too, because are just for ordering the data.

```r
drop.var <- c(names(training)[1:5])
for(nm in names(training)) {if (sum(is.na(training[, nm])) > 15000)  
                           {drop.var <- c(drop.var, nm)}}
training <- training[, !(names(training) %in% drop.var)]
dim(training)
```

```
## [1] 15699    88
```
This code drops the first five variables and all the variables with more than 15000 NA (>95%). Now the working variables are 87 plus ```classe```.

## Cleaning Data | (continuation)
Now I drop the variables with variance near to zero.

```r
zv <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[,!zv$nzv]
dim(training)
```

```
## [1] 15699    54
```
The final working variables are 53 plus ```classe```.

## Running the Algotirhms
Now as first try, I run two models using the ```train``` default options:  
- A Random Forest  
- A Boosting model  
And I use the ```resample``` command to compare both results

```r
mod.rf <- train(classe~., method='rf', data=training)
mod.bst <- train(classe~., method='gbm', data=training, verbose=FALSE)
results <- resamples(list(RandomForest=mod.rf, Boosting=mod.bst))
```
The ```train``` command uses a random sub-sampling with replacement (bootstrap with 25 repetitions) as default method for choosing the best parameters for the model.
The ```resamples``` shows in a very intuitive way the results of the iterations and compares the models.

## Results

```r
bwplot(results)
```
![](Rplot12.png)

## Results | (continuation)

```r
summary(results)
```
```
Models: RandomForest, Boosting 
Number of resamples: 25 

Accuracy 
               Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
RandomForest 0.9939  0.9955 0.9967 0.9964  0.9976 0.9984    0
Boosting     0.9783  0.9831 0.9841 0.9841  0.9855 0.9876    0

Kappa 
               Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
RandomForest 0.9923  0.9944 0.9958 0.9955  0.9969 0.9980    0
Boosting     0.9725  0.9786 0.9800 0.9799  0.9817 0.9844    0
```

## Results | (continuation)
- Both models have excellent performance in the training set.  
  
- Particularly the Random Forest model, has mean Accuracy **and** Kappa > 99,5%!  
  
- Given the better performance, I choose the Random Forest model for the predictions.

## Estimation of out of sample error | for the Random Forest model
- The minimum Accuracy in the ```resamples``` result is ```0.9923```

```r
mod.rf$results
```
```
  mtry  Accuracy     Kappa  AccuracySD     KappaSD
1    2 0.9928344 0.9909352 0.001221010 0.001544460
2   27 0.9964354 0.9954911 0.001295815 0.001639253
3   53 0.9929649 0.9911010 0.001943789 0.002457841
```
- The best Accuracy mean minus three times the standard deviation (~ 99,7%) is ```0.992548```.
- As [the bootstrap can provide accurate measures of BOTH the bias and variance of the true error estimate](http://research.cs.tamu.edu/prism/lectures/pr/pr_l13.pdf), one can expect an Accuracy > 0.99 for the Random Forest model in the testing set. 

## Testing the model


```r
pred.rf <- predict(mod.rf, newdata=testing)
confusionMatrix(pred.rf, testing$classe)
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1116    3    0    0    0
         B    0  755    2    0    2
         C    0    1  682    0    0
         D    0    0    0  643    0
         E    0    0    0    0  719
  
# continue...
```

## Testing the model
```
    Overall Statistics
                                         
               Accuracy : 0.998          
                 95% CI : (0.996, 0.9991)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9974    
  
Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9947   0.9971   1.0000   0.9972
Specificity            0.9989   0.9987   0.9997   1.0000   1.0000
Pos Pred Value         0.9973   0.9947   0.9985   1.0000   1.0000
Neg Pred Value         1.0000   0.9987   0.9994   1.0000   0.9994
...
```

## Conclusion
- As expected, the Accuracy is > 0.99. Moreover, Accuracy and Kappa are very close to 1.  
- That mean the model has a very high prediction capability. 
- I feel lucky, because the "first try" was actually the only one; but it took almost three hours in my good ol' machine... in parallel!

# Thank you!
