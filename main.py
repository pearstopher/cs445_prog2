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

import pandas as pd
import numpy as np


# "1. Create training and test set:
# "    Split the data into a training and test set. Each of these should have about 2,300 instances,
# "    and each should have about 40% spam, 60% not-spam, to reflect the statistics of the full
# "    data set. Since you are assuming each feature is independent of all others, here it is not
# "    necessary to standardize the features.
class Data:
    DATA = "./spambase/spambase.data"
    NAMES = "./spambase/spambase.names"

    def __init__(self):
        self.train, self.train_truth, \
            self.test, self.test_truth = self.read_data()
        self.names = self.read_names()

    def read_data(self):
        # get all of the data
        all_data = pd.read_csv(self.DATA).to_numpy()

        # mix up all of the data since it is divided into spam/not spam
        np.random.shuffle(all_data)

        # the last element of each row is the truth value
        truth_values = all_data[:, -1:]

        # and the first 56 are the attribute values
        attributes = all_data[:, :-1]

        # there are 4600 items total, which will be separated into 2 stacks of 2300
        end = len(truth_values)  # or attributes, they're the same length
        mid = int(end/2)
        return attributes[0:mid], truth_values[0:mid], \
            attributes[mid:end], truth_values[mid:end]

    def read_names(self):
        # get all the attribute names
        names = pd.read_csv(self.NAMES, comment="|").to_numpy()
        return names


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

    data = Data()
    print(len(data.train))
    print(len(data.train_truth))
    print(len(data.test))
    print(len(data.test_truth))
    print(len(data.names))
    print(len(data.train_truth[89]))
    print(data.train)
    print(data.test_truth)



if __name__ == '__main__':
    main()
