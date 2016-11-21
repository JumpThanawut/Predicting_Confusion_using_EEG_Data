eeg_raw = read.csv("EEG.csv")

# aggregate every 10 rows(5 seconds) for each Subject and Video, which performs best
window = 10
mylist = split(eeg_raw, eeg_raw$SubjectID)
aggreData = data.frame()
for(one in mylist){
    one = data.frame(one)
    sublist=split(one, one$VideoID)
    for(subone in sublist){
      chunk = aggregate(subone,list(rep(0:(nrow(subone)%/%window+1),each=window,len=nrow(subone))),mean)
      aggreData = rbind(aggreData, chunk)
    }
}

eeg_noID = data.frame(aggreData[,-c(0,1,2,3,15)])
eeg_noID = eeg_noID[!eeg_noID$SelfDefinedConfusion != 0 || 1]

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
# 67.68%

# Do SVM
svm.fit = svm(SelfDefinedConfusion~., data = train_set)
plot(svm.fit)
summary(svm.fit)
probs = predict(svm.fit, test_set)
pred = rep(0, length(probs))
pred[probs>0.5] = 1
accuracy.svm=mean(pred == test_set$SelfDefinedConfusion)
# 76.77%

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
# 76.26%

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
# 71.07%

# KNN
require(class)
knn.pred = knn(train_set, test_set, train_set$SelfDefinedConfusion, k = 5)
accuracy.knn = sum(test_set$SelfDefinedConfusion == knn.pred)/nrow(test_set)
table(knn.pred ,test_set$SelfDefinedConfusion)
# 62.37%

# print results
print(paste0("Accuracy(Data Aggregation 5 seconds Logistic Regression): ", accuracy.lm))
print(paste0("Accuracy(Data Aggregation 5 seconds SVM): ", accuracy.svm))
print(paste0("Accuracy(Data Aggregation 5 seconds RandomForest): ", accuracy.RF))
print(paste0("Accuracy(Data Aggregation 5 seconds Boosting): ", accuracy.Boosting))
print(paste0("Accuracy(Data Aggregation 5 seconds KNN): ", accuracy.knn))



