eeg_raw = read.csv("EEG.csv")

# aggregate every 10 rows
window = 10
aggreData = aggregate(eeg_raw,list(rep(0:(nrow(eeg_raw)%/%window+1),each=window,len=nrow(eeg_raw))),mean)
eeg_noID = data.frame(aggreData[,-c(1,2,3,15)])

# data normalization
library("e1071")
library(caret)
library(dplyr)
library(class)
set.seed(123)
preObj = preProcess(eeg_noID[,-12], method=c("BoxCox"))
dataNom = predict(preObj, eeg_noID)

# spilt training and testing set
set.seed(3286)
random_vector = runif(length(dataNom$SelfDefinedConfusion))
train = random_vector < 0.7
test = !train
train_num = sum(train == TRUE)
test_num = sum(train == FALSE)
train_set = dataNom[train,]
test_set = dataNom[test,]

# Do Logistic Regression
glm.fit = glm(SelfDefinedConfusion~., data = train_set, family = "binomial")
summary(glm.fit)
probs = predict(glm.fit, test_set, type = "response")
pred = rep(0, length(probs))
pred[probs>0.5] = 1
mean(pred == test_set$SelfDefinedConfusion)
# 58.77%

# Do SVM
svm.fit = svm(SelfDefinedConfusion~., data = train_set)
plot(svm.fit)
summary(svm.fit)
probs = predict(svm.fit, test_set)
pred = rep(0, length(probs))
pred[probs>0.5] = 1
mean(pred == test_set$SelfDefinedConfusion)
# 69.68%

# RandomForest
#install.packages('randomForest', repos="http://cran.r-project.org")
require(randomForest)
require(MASS)
set.seed(101)
#tree.fit = tree(SelfDefinedConfusion~., data = train_set)
rf.fit = randomForest(SelfDefinedConfusion~., data = train_set, ntree = 400)
plot(rf.fit)
summary(rf.fit)
probs = predict(rf.fit, test_set)
pred = rep(0, length(probs))
pred[probs>0.5] = 1
mean(pred == test_set$SelfDefinedConfusion)
# 69.41%

# Boosting
#install.packages('gbm', repos="http://cran.r-project.org")
require(gbm)
boost.eeg=gbm(SelfDefinedConfusion~.,data=train_set,distribution="gaussian",n.trees=6000,shrinkage=0.01,interaction.depth=4)
summary(boost.eeg)
n.trees=seq(from=100,to=10000,by=100)
predmat=predict(boost.eeg,newdata=test_set,n.trees=6000)
pred = rep(0, length(predmat))
pred[predmat>0.5] = 1
mean(pred == test_set$SelfDefinedConfusion)
# 65.96%
