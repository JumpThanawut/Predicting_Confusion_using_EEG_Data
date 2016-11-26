# Read EEG data. The data have 12811 examples with 15 features.
eeg_raw = read.csv("../data/EEG.csv")
# Remove pre-defined confusion. We will use self-defined confusion as a target.
eeg_self = subset(eeg_raw, select=-c(PreDefinedConfusion))

# See correlation of features and target. Max is 0.15. Min is -0.12. No strong correlation of features and target has been found.
cor(subset(eeg_self, select=-c(SelfDefinedConfusion)), eeg_self$SelfDefinedConfusion)

# aggregate every 10 rows(5 seconds) for each Subject and Video, which performs best
window = 10
listBySubjectID = split(eeg_raw, eeg_raw$SubjectID)
aggreData = data.frame()
for(eeg_rawVector in listBySubjectID){
  eeg_rawVector = data.frame(eeg_rawVector)
  listBySubjectIDVideoID=split(eeg_rawVector, eeg_rawVector$VideoID)
  for(subVector in listBySubjectIDVideoID){
    chunk = aggregate(subVector,list(rep(0:(nrow(subVector)%/%window+1),each=window,len=nrow(subVector))),mean)
    aggreData = rbind(aggreData, chunk)
  }
}

# Remove Group index
eeg_aggregated = aggreData[,-c(1)]
eeg_aggregated = eeg_aggregated[!eeg_aggregated$SelfDefinedConfusion != 0 || 1]

# Cross-validation preparation
n = nrow(eeg_self)
fold = 5
getCVIndex = function(n, fold) {
  set.seed(1)
  cv_set = rep(0,n)
  for(i in 1:n) {
    cv_set[i] = i%%fold + 1
  }
  cv_set = sample(cv_set, n)
  return(cv_set)
}
cv_set = getCVIndex(n, fold)

# Logistic Regression
# Accuracy of 5-fold Logistic Regression = 60.01%
logistic_regression.accuracy = rep(0, fold)
n_fold = rep(0, fold)
for (i in 1:fold) {
  train = eeg_self[cv_set != i,]
  test = eeg_self[cv_set == i,]
  n_fold[i] = nrow(test)
  logistic_regression.model = glm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, family = "binomial")
  #print(summary(logistic_regression.model))
  logistic_regression.prob = predict(logistic_regression.model, newdata = test, type = "response")
  logistic_regression.pred = ifelse(logistic_regression.prob > 0.5, 1, 0)
  logistic_regression.accuracy[i] = mean(logistic_regression.pred == test$SelfDefinedConfusion)
  #print(paste0("Accuracy(Logistic Regression): ", logistic_regression.accuracy[i]))
}
print(paste0(paste0("Accuracy of 5-fold Logistic Regression = ", format(round(weighted.mean(logistic_regression.accuracy, n_fold)*100, 2), nsmall = 2)), "%"))

# Logistic Regression without Subject 6.
# Accuracy of 5-fold Logistic Regression without Subject 6 = 60.72%. It is 0.71% higher than the one with Subject 6.
logistic_regression.accuracy = rep(0, fold)
n_fold = rep(0, fold)
for (i in 1:fold) {
  train = eeg_self[cv_set != i,]
  test = eeg_self[cv_set == i,]
  train = train[train$SubjectID != 6,]
  test = test[test$SubjectID != 6,]
  n_fold[i] = nrow(test)
  logistic_regression.model = glm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, family = "binomial")
  #print(summary(logistic_regression.model))
  logistic_regression.prob = predict(logistic_regression.model, newdata = test, type = "response")
  logistic_regression.pred = ifelse(logistic_regression.prob > 0.5, 1, 0)
  logistic_regression.accuracy[i] = mean(logistic_regression.pred == test$SelfDefinedConfusion)
  #print(paste0("Accuracy(Logistic Regression): ", logistic_regression.accuracy[i]))
}
print(paste0(paste0("Accuracy of 5-fold Logistic Regression without Subject 6 = ", format(round(weighted.mean(logistic_regression.accuracy, n_fold)*100, 2), nsmall = 2)), "%"))

# Visualization
features = c("SubjectID", "VideoID", "Attention", "Meditation", "Raw", "Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2", "PreDefinedConfusion", "SelfDefinedConfusion")
for (feature in 3:13) {
  pdf(paste0(paste0("../result/visualization/plot_", features[feature]),".pdf"), paper = "USr", height = 8, width = 10)
  for (subject in 0:9) {
    plot(1:nrow(eeg_raw[eeg_raw$SubjectID == subject & eeg_raw$VideoID == 0,]), eeg_raw[eeg_raw$SubjectID == subject & eeg_raw$VideoID == 0,features[feature]], main = paste0(paste0(features[feature]," of Subject"), subject), xlab = "Time", ylab = features[feature], type="l", col="#00000000", lwd = 0.5)
    for (i in 0:9) { 
      if (eeg_raw[eeg_raw$SubjectID == subject & eeg_raw$VideoID == i,]$SelfDefinedConfusion[1] == 1) {
        color = "#FF0000FF"
      }
      else {
        color = "#00CC00FF"
      }
      lines(1:nrow(eeg_raw[eeg_raw$SubjectID == subject & eeg_raw$VideoID == i,]), eeg_raw[eeg_raw$SubjectID == subject & eeg_raw$VideoID == i,features[feature]], type="l", col=color, lwd = 0.5) 
    } 
  }
  dev.off()
}

# Leave one subject-video out cross validation.
n_example = 0
n_correct_example = 0
n_correct_example_lda = 0
n_correct_example_qda = 0
n_correct_example_knn = 0
n_correct_example_tree = 0
n_correct_example_rf = 0
n_correct_example_svm = 0
n_correct_example_nn = 0

uniqueSubjectID = unique(eeg_self$SubjectID)
for (subjectID in uniqueSubjectID) {
  uniqueVideoID = unique(eeg_self[eeg_self$SubjectID == subjectID,]$VideoID)
  for (videoID in uniqueVideoID) {
    n_example = n_example + 1
    train = eeg_self[(eeg_self$SubjectID != subjectID) | (eeg_self$VideoID != videoID),]
    test = eeg_self[(eeg_self$SubjectID == subjectID) & (eeg_self$VideoID == videoID),]
    train_aggre = eeg_aggregated[(eeg_aggregated$SubjectID != subjectID) | (eeg_aggregated$VideoID != videoID),]
    test_aggre = eeg_aggregated[(eeg_aggregated$SubjectID == subjectID) & (eeg_aggregated$VideoID == videoID),]
    real_class = test[1,]$SelfDefinedConfusion
    real_aggreClass = test_aggre[1,]$SelfDefinedConfusion
    
    # logistic regression
    logistic_regression.model = glm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, family = "binomial")
    logistic_regression.prob = predict(logistic_regression.model, newdata = test, type = "response")
    logistic_regression.pred = ifelse(logistic_regression.prob > 0.5, 1, 0)
    predict_class = ifelse(mean(logistic_regression.pred) > 0.5, 1, 0)
    if (real_class == predict_class) {
      n_correct_example = n_correct_example + 1
      print(paste0(subjectID, videoID))
    }
    
    # linear discriminant analysis
    require(MASS)
    linear_discriminant_analysis.model = lda(SelfDefinedConfusion~.-SubjectID-VideoID, data = train)
    linear_discriminant_analysis.pred = predict(linear_discriminant_analysis.model, newdata = test)
    predict_class_lda = linear_discriminant_analysis.pred$class
    if (real_class == predict_class_lda) {
      n_correct_example_lda = n_correct_example_lda + 1
      print(paste0(subjectID, videoID))
    }

    # quadratic discriminant analysis
    quadratic_discriminant_analysis.model = qda(SelfDefinedConfusion~.-SubjectID-VideoID, data = train)
    quadratic_discriminant_analysis.pred = predict(quadratic_discriminant_analysis.model, newdata = test)
    predict_class_qda = quadratic_discriminant_analysis.pred$class
    if (real_class == predict_class_qda) {
      n_correct_example_qda = n_correct_example_qda + 1
      print(paste0(subjectID, videoID))
    }

    # k nearest neighbor
    require(class)
    # Have tried k=1,2,3,4,5,6,10,13,20,23,50,100. k=1 gets the highest accuracy.
    knn.pred = knn(train, test, train$SelfDefinedConfusion, k = 1)
    predict_class_knn = knn.pred
    if (real_class == predict_class_knn) {
      n_correct_example_knn = n_correct_example_knn + 1
      print(paste0(subjectID, videoID))
    }

    # decision tree
    require(ISLR)
    require(tree)
    decision_tree.model = tree(as.factor(SelfDefinedConfusion)~.-SubjectID-VideoID, data = train)
    decision_tree.pred = predict(decision_tree.model, test, type = "class")
    cv.decision_tree = cv.tree(decision_tree.model,FUN = prune.misclass)
    prune.decision_tree = prune.misclass(decision_tree.model, best = 10)
    decision_tree.pred = predict(prune.decision_tree, test, type = "class")

    predict_class_tree = decision_tree.pred
    if (real_class == predict_class_tree) {
      n_correct_example_tree = n_correct_example_tree + 1
      print(paste0(subjectID, videoID))
    }

    # random forest
    require(randomForest)
    rf.model = randomForest(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, mtry=5, ntree=100, cutoff = 2, importance=TRUE)
    rf.probs = predict(rf.model,newdata = test)
    rf.pred = rep(0, length(rf.probs))
    rf.pred[rf.probs>=0.5] = 1
    predict_class_rf = rf.pred
    if (real_class == predict_class_rf) {
      n_correct_example_rf = n_correct_example_rf + 1
      print(paste0(subjectID, videoID))
    }

    # support vector machine
    require("e1071")
    svm.model = svm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, kernal = "poly")
    svm.probs = predict(svm.model, test)
    svm.pred = rep(0, length(svm.probs))
    svm.pred[svm.probs>=0.5] = 1
    # svm.pred = predict(svm.model, test)
    predict_class_svm = svm.pred
    if (real_class == predict_class_svm) {
      n_correct_example_svm = n_correct_example_svm + 1
      print(paste0(subjectID, videoID))
    }

    # neural network
    require("nnet")
    ideal_train = class.ind(train$SelfDefinedConfusion)
    neural_network.model = nnet(train[-1-2-14], ideal_train, size=10, softmax=TRUE)
    neural_network.pred = predict(neural_network.model, test, type = "class")
    prediction_class_nn = neural_network.pred

    if (real_class == prediction_class_nn) {
      n_correct_example_nn = n_correct_example_nn + 1
      print(paste0(subjectID, videoID))
    }
  }
}

accuracy = n_correct_example * 100 / n_example
accuracy_lda = n_correct_example_lda * 100 / n_example
accuracy_qda = n_correct_example_qda * 100 / n_example
accuracy_knn = n_correct_example_knn * 100 / n_example
accuracy_tree = n_correct_example_tree * 100 / n_example
accuracy_rf = n_correct_example_rf * 100 / n_example
accuracy_svm = n_correct_example_svm * 100 / n_example
accuracy_nn = n_correct_example_nn * 100 / n_example

print(paste0("Accuracy(Leave One Subject-Video Out Logistic Regression): ", accuracy))
# 63
print(paste0("Accuracy(Leave One Subject-Video Out Linear Discriminant Analysis Regression): ", accuracy_lda))
# 52
print(paste0("Accuracy(Leave One Subject-Video Out Quadratic Discriminant Analysis Regression): ", accuracy_qda))
# 50
print(paste0("Accuracy(Leave One Subject-Video Out K Nearest Neighbor): ", accuracy_knn))
# 61
print(paste0("Accuracy(Leave One Subject-Video Out Decision Tree): ", accuracy_tree))
# 56
print(paste0("Accuracy(Leave One Subject-Video Out Random Forest): ", accuracy_rf))
# 53
print(paste0("Accuracy(Leave One Subject-Video Out Support Vector Machine): ", accuracy_svm))
# 53
print(paste0("Accuracy(Leave One Subject-Video Out Neural Network): ", accuracy_nn))
# 55

# Cross-validation preparation with aggregated data
aggre_eeg_self = subset(eeg_aggregated, select=-c(PreDefinedConfusion))
n = nrow(aggre_eeg_self)
fold = 5
getCVIndex = function(n, fold) {
  set.seed(1)
  cv_set = rep(0,n)
  for(i in 1:n) {
    cv_set[i] = i%%fold + 1
  }
  cv_set = sample(cv_set, n)
  return(cv_set)
}
cv_set = getCVIndex(n, fold)

# Logistic Regression with aggregated data
lm.fit = function(aggre_eeg_self, cv_set){
  logistic_regression.accuracy = rep(0, fold)
  n_fold = rep(0, fold)
  for (i in 1:fold) {
    train = aggre_eeg_self[cv_set != i,]
    test = aggre_eeg_self[cv_set == i,]
    n_fold[i] = nrow(test)
    logistic_regression = glm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, family = "binomial")
    #print(summary(logistic_regression.model))
    logistic_regression.prob = predict(logistic_regression, newdata = test, type = "response")
    logistic_regression.pred = rep(0, length(logistic_regression.prob))
    logistic_regression.pred[logistic_regression.prob>0.5] = 1
    logistic_regression.accuracy[i] = mean(logistic_regression.pred == test$SelfDefinedConfusion)
    print(paste0("Accuracy(Logistic Regression): ", logistic_regression.accuracy[i]))
  }
  accuracy.lm = format(round(weighted.mean(logistic_regression.accuracy, n_fold)*100, 2), nsmall = 2)
  return(accuracy.lm)
}

# SVM with aggregated data
svm.fit = function(aggre_eeg_self, cv_set){
  svm.accuracy = rep(0, fold)
  n_fold = rep(0, fold)
  for (i in 1:fold) {
    train = aggre_eeg_self[cv_set != i,]
    test = aggre_eeg_self[cv_set == i,]
    n_fold[i] = nrow(test)
    svm = svm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train)
    svm.prob = predict(svm, newdata = test)
    svm.pred = rep(0, length(svm.prob))
    svm.pred[svm.prob>0.5] = 1
    svm.accuracy[i] = mean(svm.pred == test$SelfDefinedConfusion)
    print(paste0("Accuracy(SVM): ", svm.accuracy[i]))
  }
  accuracy.svm = format(round(weighted.mean(svm.accuracy, n_fold)*100, 2), nsmall = 2)
  return(accuracy.svm)
}

# RandomForest with aggregated data
rf.fit = function(aggre_eeg_self, cv_set){
  rf.accuracy = rep(0, fold)
  n_fold = rep(0, fold)
  for (i in 1:fold) {
    train = aggre_eeg_self[cv_set != i,]
    test = aggre_eeg_self[cv_set == i,]
    n_fold[i] = nrow(test)
    rf = randomForest(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, ntree = 400)
    rf.prob = predict(rf, newdata = test)
    rf.pred = rep(0, length(rf.prob))
    rf.pred[rf.prob>0.5] = 1
    rf.accuracy[i] = mean(rf.pred == test$SelfDefinedConfusion)
    print(paste0("Accuracy(RandomForest): ", rf.accuracy[i]))
  }
  accuracy.RF = format(round(weighted.mean(rf.accuracy, n_fold)*100, 2), nsmall = 2)
  return(accuracy.RF)
}

# Boosting with aggregated data
bst.fit = function(aggre_eeg_self, cv_set){
  bst.accuracy = rep(0, fold)
  n_fold = rep(0, fold)
  for (i in 1:fold) {
    train = aggre_eeg_self[cv_set != i,]
    test = aggre_eeg_self[cv_set == i,]
    n_fold[i] = nrow(test)
    bst = gbm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, distribution="gaussian",n.trees=6000,shrinkage=0.01,interaction.depth=4)
    n.trees=seq(from=100,to=10000,by=100)
    predmat=predict(bst,newdata=test,n.trees=6000)
    bst.pred = rep(0, length(predmat))
    bst.pred[predmat>0.5] = 1
    bst.accuracy[i] = mean(bst.pred == test$SelfDefinedConfusion)
    print(paste0("Accuracy(Boosting): ", bst.accuracy[i]))
  }
  accuracy.Boosting = format(round(weighted.mean(bst.accuracy, n_fold)*100, 2), nsmall = 2)
  return(accuracy.Boosting)
}

# KNN with aggregated data
knn.fit = function(aggre_eeg_self, cv_set){
  knn.accuracy = rep(0, fold)
  n_fold = rep(0, fold)
  for (i in 1:fold) {
    train = aggre_eeg_self[cv_set != i,]
    test = aggre_eeg_self[cv_set == i,]
    train = subset(train, select=-c(SubjectID,VideoID))
    test = subset(test, select=-c(SubjectID,VideoID))
    n_fold[i] = nrow(test)
    knn.pred = knn(train, test, train$SelfDefinedConfusion, k = 5)
    knn.accuracy[i] = sum(test$SelfDefinedConfusion == knn.pred)/nrow(test)
    print(paste0("Accuracy(KNN): ", knn.accuracy[i]))
  }
  accuracy.knn = format(round(weighted.mean(knn.accuracy, n_fold)*100, 2), nsmall = 2)
  return(accuracy.knn)
}

print(paste0(paste0("Accuracy of 5-fold LogisticRegression with aggregated data = ", accuracy.lm), "%"))
# 65.20%
print(paste0(paste0("Accuracy of 5-fold SVM with aggregated data = ", accuracy.svm), "%"))
# 68.11%
print(paste0(paste0("Accuracy of 5-fold RandomForest with aggregated data = ", accuracy.RF), "%"))
# 74.61%
print(paste0(paste0("Accuracy of 5-fold Boosting with aggregated data = ", accuracy.Boosting), "%"))
# 71.02%
print(paste0(paste0("Accuracy of 5-fold KNN with aggregated data = ", accuracy.knn), "%"))
# 63.13%

# data normalization
library("e1071")
library(caret)
library(dplyr)
library(class)
set.seed(123)
preObj = preProcess(eeg_self[,-c(1,2,12)], method=c("BoxCox"))
norm_eeg = predict(preObj, eeg_self)

# algorithms with normalized data
accuracy.lm.norm = lm.fit(norm_eeg, cv_set)
accuracy.knn.norm = knn.fit(norm_eeg, cv_set)
accuracy.svm.norm = svm.fit(norm_eeg, cv_set)
accuracy.rf.norm = rf.fit(norm_eeg, cv_set)
accuracy.bst.norm = bst.fit(norm_eeg, cv_set)

# Normalization on aggregated data
preObj = preProcess(eeg_aggregated[,-c(12)], method=c("BoxCox"))
normAggre_eeg = predict(preObj, eeg_aggregated)

# algorithms with normalized$aggregated data
accuracy.lm.normAggre = lm.fit(normAggre_eeg, cv_set)
accuracy.knn.normAggre = knn.fit(normAggre_eeg, cv_set)
accuracy.svm.normAggre = svm.fit(normAggre_eeg, cv_set)
accuracy.rf.normAggre = rf.fit(normAggre_eeg, cv_set)
accuracy.bst.normAggre = bst.fit(normAggre_eeg, cv_set)

# Add more features related to delta
eeg_aggregated$d_attention = NA
eeg_aggregated$d_meditation = NA
eeg_aggregated$d_raw = NA
eeg_aggregated$d_delta = NA
eeg_aggregated$d_theta = NA
eeg_aggregated$d_alpha1 = NA
eeg_aggregated$d_alpha2 = NA
eeg_aggregated$d_beta1 = NA
eeg_aggregated$d_beat2 = NA
eeg_aggregated$d_gamma1 = NA
eeg_aggregated$d_gamma2 = NA

for (i in 2:nrow(eeg_aggregated)){
  eeg_aggregated[i,]$d_attention = eeg_aggregated[i, 3] - eeg_aggregated[i-1, 3]
  eeg_aggregated[i,]$d_meditation = eeg_aggregated[i, 4] - eeg_aggregated[i-1, 4]
  eeg_aggregated[i,]$d_raw = eeg_aggregated[i, 5] - eeg_aggregated[i-1, 5]
  eeg_aggregated[i,]$d_delta = eeg_aggregated[i, 6] - eeg_aggregated[i-1,6]
  eeg_aggregated[i,]$d_theta = eeg_aggregated[i, 7] - eeg_aggregated[i-1,7]
  eeg_aggregated[i,]$d_alpha1 = eeg_aggregated[i, 8] - eeg_aggregated[i-1,8]
  eeg_aggregated[i,]$d_alpha2 = eeg_aggregated[i, 9] - eeg_aggregated[i-1,9]
  eeg_aggregated[i,]$d_beta1 = eeg_aggregated[i, 10] - eeg_aggregated[i-1,10]
  eeg_aggregated[i,]$d_beat2 = eeg_aggregated[i, 11] - eeg_aggregated[i-1,11]
  eeg_aggregated[i,]$d_gamma1 = eeg_aggregated[i, 12] - eeg_aggregated[i-1,12]
  eeg_aggregated[i,]$d_gamma2 = eeg_aggregated[i, 13] - eeg_aggregated[i-1,13]
}

# Remove first row
eeg_aggregated = eeg_aggregated[-1,]
eeg_aggregated = eeg_aggregated[, -14]

# cross validation
n = nrow(eeg_aggregated)
fold = 5
getCVIndex = function(n, fold) {
  set.seed(1)
  cv_set = rep(0,n)
  for(i in 1:n) {
    cv_set[i] = i%%fold + 1
  }
  cv_set = sample(cv_set, n)
  return(cv_set)
}
cv_set = getCVIndex(n, fold)

# algorithms with aggregated d_data
accuracy.lm.d_aggre = lm.fit(eeg_aggregated, cv_set)
accuracy.knn.d_aggre = knn.fit(eeg_aggregated, cv_set)
accuracy.svm.d_aggre = svm.fit(eeg_aggregated, cv_set)
accuracy.rf.d_aggre = rf.fit(eeg_aggregated, cv_set)
accuracy.bst.d_aggre = bst.fit(eeg_aggregated, cv_set)

# algorithms with aggregated$normalization d_data
accuracy.lm.d_aggre = lm.fit(eeg_aggregated, cv_set)
accuracy.knn.d_aggre = knn.fit(eeg_aggregated, cv_set)
accuracy.svm.d_aggre = svm.fit(eeg_aggregated, cv_set)
accuracy.rf.d_aggre = rf.fit(eeg_aggregated, cv_set)
accuracy.bst.d_aggre = bst.fit(eeg_aggregated, cv_set)
