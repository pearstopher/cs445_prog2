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
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


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

        # and the first 57 are the attribute values
        attributes = all_data[:, :-1]

        # there are 4600 items total, which will be separated into 2 stacks of 2300
        end = len(truth_values)  # or len(attributes), they're the same length
        mid = int(end/2)
        return attributes[0:mid], truth_values[0:mid], \
            attributes[mid:end], truth_values[mid:end]

    def read_names(self):
        # get all the attribute names
        return pd.read_csv(self.NAMES, comment="|").to_numpy()


# "2. Create probabilistic model. (Write your own code to do this.)
# "    • Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in
# "      the training data. As described in part 1, P(1) should be about 0.4.
# "
# "    • For each of the 57 features, compute the mean and standard deviation in the
# "      training set of the values given each class. If any of the features has zero standard
# "      deviation, assign it a “minimal” standard deviation (e.g., 0.0001) to avoid a divide-by-
# "      zero error in Gaussian Naive Bayes.
class Model(Data):

    def __init__(self):
        super().__init__()
        # for each of these arrays, indexes [0] and [1] represent classes 0 and 1
        self.train_prior = self.compute_priors()
        self.train_mean = self.compute_means()
        self.train_std = self.compute_stds()

    def compute_priors(self):
        train_count = np.count_nonzero(self.train_truth)
        train_total = len(self.train_truth)

        train_prior = (train_count/train_total,
                       (train_total - train_count)/train_total)

        # t_p[0] = prior probability of 0
        # t_p[1] = prior probability of 1
        return train_prior

    def compute_means(self):
        train_means = np.zeros((2, len(self.train[0])))
        length = np.zeros((2, 1))

        for t, tt in zip(self.train, self.train_truth):
            train_means[int(tt)] += t
            length[int(tt)] += 1
        train_means /= length

        # [0] = mean of class 0
        # [1] = mean of class 1
        return train_means

    def compute_stds(self):
        split = [[], []]
        train_std = np.zeros((2, 57))

        for t, tt in zip(self.train, self.train_truth):
            split[int(tt)].append(t)

        train_std[0] = np.std(split[0], axis=0)
        train_std[1] = np.std(split[1], axis=0)

        # (make sure the standard deviation is never zero)
        train_std = np.where(train_std > 0.0001, train_std, 0.0001)

        # [0] = nonzero standard deviation of class 0
        # [1] = nonzero standard deviation of class 1
        return train_std


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
class NaiveBayes(Model):
    def __init__(self):
        super().__init__()
        self.test_probabilities = self.compute_probabilities()
        self.test_predictions = self.predict_classes()

    def compute_probabilities(self):
        probabilities = [[], []]
        for x in range(len(self.test)):
            for j in range(2):
                probabilities[j].append([self.attribute_class_probability(x, i, j) for i in range(57)])
        return probabilities

    # "P(xᵢ|cⱼ) = N(xᵢ;μ₍ᵢ,cⱼ₎,σ₍ᵢ,cⱼ₎)
    # "
    # "# " where μi,c is the mean of feature i given the class c, and σi,c is
    # # " the standard deviation of feature i given the class c
    #
    def attribute_class_probability(self, x, i, j):  # x_i, c_j
        p = self.probability_density(self.test[x][i], self.train_mean[j][i], self.train_std[j][i])

        # just like earlier, need to set some sort of nonzero minimum for this
        return p if p > 0.0001 else 0.0001

    # "N(x;μ,σ) = 1/√(2πσ) × e^-[(x - μ)/2σ²]
    # "
    # "Note: N is the probability density function, but can be used analogously to
    # "probability in Naive Bayes calculations.
    @staticmethod
    def probability_density(x, mu, sigma):
        return (1 / math.sqrt(2*math.pi*sigma)) * math.exp(-((abs(x - mu))/(2*(sigma**2))))

    # "class₍NB₎(x) = argmax₍class₎ log[P(class) Πᵢ P(xᵢ|class)]
    def predict_classes(self):
        classes = []
        for p0, p1 in zip(self.test_probabilities[0], self.test_probabilities[1]):
            total = [math.log(self.train_prior[0]), math.log(self.train_prior[1])]

            for probability in p0:
                total[0] += math.log(probability)
            for probability in p1:
                total[1] += math.log(probability)

            classes.append(0) if total[0] > total[1] else classes.append(1)

        return classes


def main():

    loops = 1
    accuracy = 0
    precision = 0
    recall = 0
    nb = None

    for i in range(loops):
        print(i+1)
        nb = NaiveBayes()
        accuracy += accuracy_score(nb.test_truth, nb.test_predictions)
        precision += precision_score(nb.test_truth, nb.test_predictions)
        recall += recall_score(nb.test_truth, nb.test_predictions)

    accuracy /= loops
    precision /= loops
    recall /= loops

    # "include […] your results:
    # "  the accuracy, precision, and recall on the test set,
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # "  as well as a confusion matrix for the test set.
    #
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    clf = SVC(random_state=0)
    clf.fit(nb.test_truth, nb.test_predictions)
    cm = confusion_matrix(nb.test_truth, nb.test_predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    display.plot(cmap="RdPu")
    plt.show()


if __name__ == '__main__':
    main()
