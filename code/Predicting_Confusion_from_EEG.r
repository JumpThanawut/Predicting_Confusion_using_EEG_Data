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
uniqueSubjectID = unique(eeg_self$SubjectID)
for (subjectID in uniqueSubjectID) {
  uniqueVideoID = unique(eeg_self[eeg_self$SubjectID == subjectID,]$VideoID)
  for (videoID in uniqueVideoID) {
    n_example = n_example + 1
    train = eeg_self[(eeg_self$SubjectID != subjectID) | (eeg_self$VideoID != videoID),]
    test = eeg_self[(eeg_self$SubjectID == subjectID) & (eeg_self$VideoID == videoID),]
    logistic_regression.model = glm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, family = "binomial")
    logistic_regression.prob = predict(logistic_regression.model, newdata = test, type = "response")
    logistic_regression.pred = ifelse(logistic_regression.prob > 0.5, 1, 0)
    real_class = test[1,]$SelfDefinedConfusion
    predict_class = ifelse(mean(logistic_regression.pred) > 0.5, 1, 0)
    if (real_class == predict_class) {
      n_correct_example = n_correct_example + 1
      print(paste0(subjectID, videoID))
    }
  }
}
accuracy = n_correct_example * 100 / n_example
print(paste0("Accuracy(Leave One Subject-Video Out Logistic Regression): ", accuracy))



