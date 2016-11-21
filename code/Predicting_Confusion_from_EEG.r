# Read EEG data. The data have 12811 examples with 15 features.
eeg_raw = read.csv("../data/EEG.csv")
# Remove pre-defined confusion. We will use self-defined confusion as a target.
eeg_self = subset(eeg_raw, select=-c(PreDefinedConfusion))

# See correlation of features and target. Max is 0.15. Min is -0.12. No strong correlation of features and target has been found.
cor(subset(eeg_self, select=-c(SelfDefinedConfusion)), eeg_self$SelfDefinedConfusion)

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
n_correct_example_svm = 0
n_correct_example_nn = 0

uniqueSubjectID = unique(eeg_self$SubjectID)
for (subjectID in uniqueSubjectID) {
  uniqueVideoID = unique(eeg_self[eeg_self$SubjectID == subjectID,]$VideoID)
  for (videoID in uniqueVideoID) {
    n_example = n_example + 1
    train = eeg_self[(eeg_self$SubjectID != subjectID) | (eeg_self$VideoID != videoID),]
    test = eeg_self[(eeg_self$SubjectID == subjectID) & (eeg_self$VideoID == videoID),]
    real_class = test[1,]$SelfDefinedConfusion
    
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
# 53
print(paste0("Accuracy(Leave One Subject-Video Out Support Vector Machine): ", accuracy_svm))
# 53
print(paste0("Accuracy(Leave One Subject-Video Out Neural Network): ", accuracy_nn))
# 55