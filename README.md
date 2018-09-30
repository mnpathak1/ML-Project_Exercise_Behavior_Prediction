# Exercise behavior prediction

## Report synopsis of the project
Purpose of this project is to predict the exercise behavior in the 20 test cases applying machine learning tools in R. In this project, two datasets are provided. The large training dataset is partitioned to build and test prediction models to predict exercise behavior. Then the model is applied to the test dataset of 20 cases to predict exercise behavior of them. In building the model, "Decision Tree" and "Random Forest" methods were evaluated and "Random Forest" method is considered for prediction due to higher accuracy (99.9%). Before applying any model, both training and testing datasets are cleaned removing near zero variables, removing columns with significant (>60%) NAs and other redundant columns and matching the column classes of both training and testing datasets.

## Project Background
Using devices such as Apple watch, Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har
