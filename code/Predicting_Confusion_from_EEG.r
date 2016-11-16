# Read EEG data. The data have 12811 examples with 15 features.
eeg_raw = read.csv("../data/EEG.csv")
# Remove pre-defined confusion. We will use self-defined confusion as a target.
eeg_self = subset(eeg_raw, select=-c(PreDefinedConfusion))

# See correlation of features and target. Max is 0.15. Min is -0.12. No strong correlation of features and target has been found.
cor(subset(eeg_self, select=-c(SelfDefinedConfusion)), eeg_self$SelfDefinedConfusion)

# Cross-validation preparation
n = nrow(eeg_self)
getCVIndex = function(n, fold) {
  set.seed(1)
  cv_set = 1:n
  for(i in 1:n) {
    cv_set[i] = i%%10 + 1
  }
  cv_set = sample(cv_set, n)
  return(cv_set)
}
cv_set = getCVIndex(n, 10)

