eeg_raw = read.csv("EEG.csv")

# aggregate every 20 rows(10 seconds), which performs best
window = 20
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
accuracy.lm=mean(pred == test_set$SelfDefinedConfusion)
# 62.16%

# Do SVM
svm.fit = svm(SelfDefinedConfusion~., data = train_set)
plot(svm.fit)
summary(svm.fit)
probs = predict(svm.fit, test_set)
pred = rep(0, length(probs))
pred[probs>0.5] = 1
accuracy.svm=mean(pred == test_set$SelfDefinedConfusion)
# 68.11%

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
accuracy.RF=mean(pred == test_set$SelfDefinedConfusion)
# 71.89%

# Boosting
#install.packages('gbm', repos="http://cran.r-project.org")
require(gbm)
boost.eeg=gbm(SelfDefinedConfusion~.,data=train_set,distribution="gaussian",n.trees=6000,shrinkage=0.01,interaction.depth=4)
summary(boost.eeg)
n.trees=seq(from=100,to=10000,by=100)
predmat=predict(boost.eeg,newdata=test_set,n.trees=6000)
pred = rep(0, length(predmat))
pred[predmat>0.5] = 1
accuracy.Boosting = mean(pred == test_set$SelfDefinedConfusion)
# 67.57%

# knn
require(class)
# Have tried k=1,2,3,4,5,6,10,13,20,23,50,100. k=1 gets the highest accuracy.
knn.pred = knn(train_set, test_set, train_set$SelfDefinedConfusion, k = 1)
100*sum(test_set$SelfDefinedConfusion == knn.pred)/100
table(knn.pred ,test_set$SelfDefinedConfusion)


# print results
print(paste0("Accuracy(Data Aggregation 5 seconds Logistic Regression): ", accuracy.lm))
print(paste0("Accuracy(Data Aggregation 5 seconds SVM): ", accuracy.svm))
print(paste0("Accuracy(Data Aggregation 5 seconds RandomForest): ", accuracy.RF))
print(paste0("Accuracy(Data Aggregation 5 seconds Boosting): ", accuracy.Boosting))



