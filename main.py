# CS445 Program 2
# Christopher Juncker
#
# "In this homework you will use Gaussian Naive Bayes to classify the Spambase data from the
# "UCI ML repository, which can be found here:
# "https://archive.ics.uci.edu/ml/datasets/spambase
#
# (Spambase data not included in commit, to save space.)
#
#


# "1. Create training and test set:
# "    Split the data into a training and test set. Each of these should have about 2,300 instances,
# "    and each should have about 40% spam, 60% not-spam, to reflect the statistics of the full
# "    data set. Since you are assuming each feature is independent of all others, here it is not
# "    necessary to standardize the features.


# "2. Create probabilistic model. (Write your own code to do this.)
# "    • Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in
# "      the training data. As described in part 1, P(1) should be about 0.4.
# "
# "    • For each of the 57 features, compute the mean and standard deviation in the
# "      training set of the values given each class. If any of the features has zero standard
# "      deviation, assign it a “minimal” standard deviation (e.g., 0.0001) to avoid a divide-by-
# "      zero error in Gaussian Naive Bayes.


# "3. Run Naive Bayes on the test data. (Write your own code to do this.)
# "    • Use the Gaussian Naive Bayes algorithm to classify the instances in your test
# "      set, using
# "
# "          P(xᵢ|cⱼ) = N(xᵢ;μ₍ᵢ,cⱼ₎,σ₍ᵢ,cⱼ₎), where N(x;μ,σ) = 1/√(2πσ) × e^-[(x - μ)/2σ²]
# "
# "      Because a product of 58 probabilities will be very small, we will instead use the
# "      log of the product. Recall that the classification method is:
# "
# "          class₍NB₎(x) = argmax₍class₎[P(class) Πᵢ P(xᵢ|class)]
# "
# "      Since
# "
# "          argmax₍z₎ f(z) = argmax₍z₎ log(z)
# "
# "      we have:
# "
# "          class₍NB₎(x) = argmax₍class₎ log[P(class) Πᵢ P(xᵢ|class)]
# "
# "          = class₍NB₎(x) = argmax₍class₎[log P(class) + log P(xᵢ|class) + … + + P(xₙ|class)]
# "


# "include […] your results:
# "  the accuracy, precision, and recall on the test set,
# "  as well as a confusion matrix for the test set.


def main():
    print("Program 2")


if __name__ == '__main__':
    main()
