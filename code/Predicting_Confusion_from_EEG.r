# Read EEG data. The data have 12811 examples with 15 features.
eeg_raw = read.csv("../data/EEG.csv")
# Remove pre-defined confusion. We will use self-defined confusion as a target.
eeg_self = subset(eeg_raw, select=-c(PreDefinedConfusion))

# See correlation of features and target. Max is 0.15. Min is -0.12. No strong correlation of features and target has been found.
cor(subset(eeg_self, select=-c(SelfDefinedConfusion)), eeg_self$SelfDefinedConfusion)

# Cross-validation preparation
n = nrow(eeg_self)
fold = 10
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
logistic_regression.accuracy = rep(0, fold)
for (i in 1:fold) {
  train = eeg_self[cv_set == i,]
  test = eeg_self[cv_set != i,]
  logistic_regression.model = glm(SelfDefinedConfusion~.-SubjectID-VideoID, data = train, family = "binomial")
  summary(logistic_regression.model)
  logistic_regression.prob = predict(logistic_regression.model, newdata = test, type = "response")
  logistic_regression.pred = ifelse(logistic_regression.prob > 0.5, 1, 0)
  logistic_regression.accuracy[i] = mean(logistic_regression.pred == test$SelfDefinedConfusion)
  #print(paste0("Accuracy(Logistic Regression): ", logistic_regression.accuracy[i]))
}
print(paste0(paste0("Accuracy of 10-fold Logistic Regression = ", format(round(mean(logistic_regression.accuracy)*100, 2), nsmall = 2)), "%"))
