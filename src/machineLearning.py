# -*- coding: utf-8 -*-
from __future__ import print_function

from numpy import array
import random
from collections import Counter

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.externals.six import StringIO
import pydot    # Draws DecisionTree maps

from corpus import Corpus
from features import Feature, createFeatures, extractFeatures
from features import showFeatureOccurrence, createBagOfWordsDictionary

from features import posStarPolarityDiscrepancy, negStarPolarityDiscrepancy, scareQuotes, scareQuotesNegative, positiveNGramPlusPunctuation, negativeNGramPlusPunctuation, positiveStreak, negativeStreak, ellipsisPlusPunctuation, positiveNGramPlusPunctuation, negativeNGramPlusPunctuation

from performance import showPerformance, showMeanPerformance
from defaultConfig import CORPUS_PATH


def applyML(trainingSetFilename, testSetFilename=None, setPath=CORPUS_PATH):
    """
    Uses machine learning approach to classify sentences.

    """
    # TODO: Add condition to create corpus, if no file exists.
    print("Training the classifiers using the set at '{path}{file}'".format(
                                                    path=setPath,
                                                    file=trainingSetFilename))

    trainingSet = Corpus(trainingSetFilename, corpusPath=CORPUS_PATH)

    # trainingSet = Corpus.loadCorpus(filename="shuffled_set.pk")

    # for each in trainingSet.reviewIDs[0:10]:
    #     print(each)
    # print()

    # Get the ids - which are ordered ironic, regular - and shuffle them.
    ids = trainingSet.reviewIDs
    random.seed(44)
    random.shuffle(ids)
    # for each in ids[0:10]:
    #     print(each)
    # print()

    reviews = trainingSet.reviews

    if not testSetFilename == None:
        testSet = Corpus(testSetFilename, corpusPath=CORPUS_PATH)
        reviews = dict(**trainingSet.reviews, **testSet.reviews)
        bowDictionary = createBagOfWordsDictionary(reviews)
    else:
        bowDictionary = None

    print("Extracting features...")
    trainFeatures, trainFeatureVectors = extractFeatures(ids,
                                            trainingSet.reviews,
                                            bowDictionary=bowDictionary)

    trainTargets = []
    trainData = []
    stars = []

    trainGold = trainingSet.goldStandard
    for ID, g in trainGold.items():
        trainTargets.append(g)
        trainData.append(trainFeatureVectors[ID])
        stars.append(trainingSet.reviews[ID].stars)
    #for i, vec in enumerate(data):
    #    print(targets[i], " | ", vec)

    featureCount = sum([sum(v) for v in trainData])
    # print("Feature found: ", featureCount, "times.")

    trainTargets = array(trainTargets)
    trainData = array(trainData)

    classifiers = [DecisionTreeClassifier(),
                    SVC(kernel="linear"),
                    SVC(),
                    LinearSVC(),
                    MultinomialNB(),
                    GaussianNB(),
                    RandomForestClassifier(),
                    LogisticRegression(),]

    # Cross validation
    if testSetFilename == None:
        for c in classifiers:
            applyCrossValidation(c, trainData, trainTargets)

            # Show star distribution for each classifier
            # applyCrossValidation(c, trainData, trainTargets, stars=stars)

        # scores = cross_validation.cross_val_score(classifier, array(data),
        #                                         array(targets), cv=10)
        # print(scores)

    else:
        print("Testing the classifiers using the set at '{path}{file}'".format(
                                                    path=CORPUS_PATH,
                                                    file=testSetFilename))

        # testSet = Corpus(testSetFilename, corpusPath=CORPUS_PATH)
        # testSet = Corpus.loadCorpus(filename="test_set.pk")

        # Create bag of words dictionary that contains words of all reviews
        # bowDictionary = createBagOfWordsDictionary(
        #                     trainingSet.reviews + testSet.reviews)


        print("Extracting features...")
        testFeatures, testFeatureVectors = extractFeatures(testSet.reviewIDs,
                                                    testSet.reviews,
                                                    bowDictionary=bowDictionary)

        testData = []
        testTargets = []

        testGold = testSet.goldStandard
        for ID, g in testGold.items():
            testTargets.append(g)
            testData.append(testFeatureVectors[ID])

        testData = array(testData)
        testTargets = array(testTargets)

        for c in classifiers:
            applyClassifier(c, trainData, trainTargets, testData, testTargets)

        # print("\n\n\n\n\nApplying Decision Tree Classifier:\n")
        # applyDecisionTree(trainData, trainTargets, testData, testTargets,
        #                 featureNames=[str(f.name) for f in trainFeatures])
        # applyDecisionTree(array(data), array(targets))
        # applySVM(data, targets)
        # applyNaiveBayes(data, targets)
        # applyNaiveBayes2(data, targets)



def applyML2(trainingSetFilename, testSetFilename=None, setPath=CORPUS_PATH):
    """
    Uses machine learning approach to classify sentences.
    Implements a truly simple 'Leave One Out' function.
    """
    # TODO: Add condition to create corpus, if no file exists.
    print("Training the classifiers using the set at '{path}{file}'".format(
                                                    path=setPath,
                                                    file=trainingSetFilename))

    #trainingSet = Corpus(trainingSetFilename, corpusPath=CORPUS_PATH)
    # trainingSet = Corpus.loadCorpus(filename=trainingSetFilename)
    # trainingSet = Corpus.loadCorpus(filename="training_and_validation_set.pk")
    trainingSet = Corpus.loadCorpus(filename="shuffled_set.pk")


    # for each in trainingSet.reviewIDs[0:10]:
    #     print(each)
    # print()

    # Get the ids - which are ordered ironic, regular - and shuffle them.
    ids = trainingSet.reviewIDs
    random.seed(44)
    random.shuffle(ids)
    # for each in ids[0:10]:
    #     print(each)
    # print()

    # Falls das -new flag nicht gesetzt ist ODER es keine Datei zum laden gibt,
    # erstelle den Corpus neu.


    print("Extracting features...")
#    trainFeatures, trainFeatureVectors = extractFeatures(trainingSet.reviewIDs,
#                                                trainingSet.reviews)


    featureConfig = {
        "minus Imba": { u"Positive Quotes": (u"\"..\"", scareQuotes),
                        u"Negative Quotes": (u"\"--\"", scareQuotesNegative),
                        u"Pos&Punctuation": (u"w+!?", positiveNGramPlusPunctuation),
                        u"Neg&Punctuation": (u"w-!?", negativeNGramPlusPunctuation),
                        u"Positive Hyperbole": (u"3w+", positiveStreak),
                        u"Negative Hyperbole": (u"3w-", negativeStreak),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
                        u"Positive&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                        u"Negative&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
        "minus Quotes": {u"Positive Imbalance": (u"w-\u2605 ",
                            posStarPolarityDiscrepancy),
                        u"Negative Imbalance": (u"w+\u2606 ",
                            negStarPolarityDiscrepancy),
                        u"Pos&Punctuation": (u"w+!?", positiveNGramPlusPunctuation),
                        u"Neg&Punctuation": (u"w-!?", negativeNGramPlusPunctuation),
                        u"Positive Hyperbole": (u"3w+", positiveStreak),
                        u"Negative Hyperbole": (u"3w-", negativeStreak),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
                        u"Positive&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                        u"Negative&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
        "minus Pos/Neg&Punctuation": {u"Positive Imbalance": (u"w-\u2605 ",
                            posStarPolarityDiscrepancy),
                        u"Negative Imbalance": (u"w+\u2606 ",
                            negStarPolarityDiscrepancy),
                        u"Positive Quotes": (u"\"..\"", scareQuotes),
                        u"Negative Quotes": (u"\"--\"", scareQuotesNegative),
                        u"Positive Hyperbole": (u"3w+", positiveStreak),
                        u"Negative Hyperbole": (u"3w-", negativeStreak),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
                        u"Positive&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                        u"Negative&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
        "minus Hyperbole": {u"Positive Imbalance": (u"w-\u2605 ",
                            posStarPolarityDiscrepancy),
                        u"Negative Imbalance": (u"w+\u2606 ",
                            negStarPolarityDiscrepancy),
                        u"Positive Quotes": (u"\"..\"", scareQuotes),
                        u"Negative Quotes": (u"\"--\"", scareQuotesNegative),
                        u"Pos&Punctuation": (u"w+!?", positiveNGramPlusPunctuation),
                        u"Neg&Punctuation": (u"w-!?", negativeNGramPlusPunctuation),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
                        u"Positive&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                        u"Negative&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
        "minus Ellipsis and Punctuation": {u"Positive Imbalance": (u"w-\u2605 ",
                            posStarPolarityDiscrepancy),
                        u"Negative Imbalance": (u"w+\u2606 ",
                            negStarPolarityDiscrepancy),
                        u"Positive Quotes": (u"\"..\"", scareQuotes),
                        u"Negative Quotes": (u"\"--\"", scareQuotesNegative),
                        u"Pos&Punctuation": (u"w+!?", positiveNGramPlusPunctuation),
                        u"Neg&Punctuation": (u"w-!?", negativeNGramPlusPunctuation),
                        u"Positive Hyperbole": (u"3w+", positiveStreak),
                        u"Negative Hyperbole": (u"3w-", negativeStreak),
                        u"Positive&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                        u"Negative&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
        "minus Pos/Neg&Ellipsis": {u"Positive Imbalance": (u"w-\u2605 ",
                            posStarPolarityDiscrepancy),
                        u"Negative Imbalance": (u"w+\u2606 ",
                            negStarPolarityDiscrepancy),
                        u"Positive Quotes": (u"\"..\"", scareQuotes),
                        u"Negative Quotes": (u"\"--\"", scareQuotesNegative),
                        u"Pos&Punctuation": (u"w+!?", positiveNGramPlusPunctuation),
                        u"Neg&Punctuation": (u"w-!?", negativeNGramPlusPunctuation),
                        u"Positive Hyperbole": (u"3w+", positiveStreak),
                        u"Negative Hyperbole": (u"3w-", negativeStreak),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
        },
        "minus Pos": {  u"Negative Imbalance": (u"w+\u2606 ",
                            negStarPolarityDiscrepancy),
                        u"Negative Quotes": (u"\"--\"", scareQuotesNegative),
                        u"Neg&Punctuation": (u"w-!?", negativeNGramPlusPunctuation),
                        u"Negative Hyperbole": (u"3w-", negativeStreak),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
                        u"Negative&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
        "minus Neg": {u"Positive Imbalance": (u"w-\u2605 ",
                            posStarPolarityDiscrepancy),
                        u"Positive Quotes": (u"\"..\"", scareQuotes),
                        u"Pos&Punctuation": (u"w+!?", positiveNGramPlusPunctuation),
                        u"Positive Hyperbole": (u"3w+", positiveStreak),
                        u"Ellipsis and Punctuation": (u"..?!", ellipsisPlusPunctuation),
                        u"Positive&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        },
    }



    for name, config in featureConfig.items():
        print("\n"*5, name)
        print("-"*60)
        for each in config:
            print(each)
        print()



        trainFeatures, trainFeatureVectors = extractFeatures(ids,
                                                trainingSet.reviews, config)

        trainTargets = []
        trainData = []

        trainGold = trainingSet.goldStandard
        for ID, g in trainGold.items():
            trainTargets.append(g)
            trainData.append(trainFeatureVectors[ID])
        #for i, vec in enumerate(data):
        #    print(targets[i], " | ", vec)

        featureCount = sum([sum(v) for v in trainData])
        print("Feature found: ", featureCount, "times.")


        classifiers = [DecisionTreeClassifier(),
                        SVC(kernel="linear"),
                        SVC(),
                        LinearSVC(),
                        MultinomialNB(),
                        GaussianNB(),
                        RandomForestClassifier(),
                        LogisticRegression(),]

        # Cross validation
        if testSetFilename == None:
            for c in classifiers:
                applyCrossValidation(c, trainData, trainTargets)

            # scores = cross_validation.cross_val_score(classifier, array(data),
            #                                         array(targets), cv=10)
            # print(scores)

        else:
            print("Testing the classifiers using the set at '{path}{file}'".format(
                                                        path=CORPUS_PATH,
                                                        file=testSetFilename))

            testSet = Corpus(testSetFilename, corpusPath=CORPUS_PATH)
            # testSet = Corpus.loadCorpus(filename="test_set.pk")

            print("Extracting features...")
            testFeatures, testFeatureVectors = extractFeatures(testSet.reviewIDs,
                                                    testSet.reviews)

            testData = []
            testTargets = []

            testGold = testSet.goldStandard
            for ID, g in testGold.items():
                testTargets.append(g)
                testData.append(testFeatureVectors[ID])

            for c in classifiers:
                applyClassifier(c, trainData, trainTargets, testData, testTargets)


def applyCrossValidation(classifier, data, targets, k=10, stars=None):
    """
    Uses k fold cross validation to test classifiers.

    Which information is most interesting by cross validation -- mean?
    """
    print("\nUsing {k} fold cross validation with {c}...".format(
        c=classifier, k=k))

    goldStandards = []
    classifications = []

    if stars is not None:
        starDistribution = []

    kf = KFold(n_splits=k)
    for train_indices, test_indices in kf.split(data, y=targets):
        trainData = [d for i, d in enumerate(data) if i in train_indices]
        trainTargets = [d for i, d in enumerate(targets) if i in train_indices]

        testData = [d for i, d in enumerate(data) if i in test_indices]
        testTargets = [d for i, d in enumerate(targets) if i in test_indices]
        if stars is not None:
            testStars = [s for i,s in enumerate(stars) if i in test_indices]

        model = classifier.fit(trainData, trainTargets)
        classification = list(model.predict(testData))

        goldStandards.append(testTargets)
        classifications.append(classification)
        if stars is not None:
            starDistribution.append(testStars)

    # Star-rating to category distribution
    if stars is not None:
        starsFlat = [s for distr in starDistribution for s in distr]
        classificationFlat = [c for cls in classifications for c in cls]
        goldFlat = [g for gold in goldStandards for g in gold]

        print("Actual star distribution:")
        showStarDistribution(starsFlat, goldFlat)
        print("Star distribution according to classification:")
        showStarDistribution(starsFlat, classificationFlat)

    showMeanPerformance(goldStandards, classifications)


def showStarDistribution(stars, targets):
    """
    Computes a star distribution,
    given a list of stars and a list of targets.
    """
    assert len(stars) == len(targets)
    distribution = Counter(zip(stars, targets))
    print(distribution)
    print("\tIronic:\tRegular:")
    for i in range(1, 6):
        print("*"*i, ":\t", distribution[(float(i), 1)], "\t",
                            distribution[(float(i), 0)] )
    for i in range(1, 6):
        print("\\filledlargestar"*i, "&", distribution[(float(i), 1)], "&",
                            distribution[(float(i), 0)], "\\\\")

def trainModel(classifier, targets, data):
    """Train the classifier."""
    pass

def classify(classfier, data):
    """Classify the given data."""
    pass


# ---- Machine Learning approaches ----
def applyClassifier(classifier, trainData, trainTargets, testData, testTargets):
    """Train and classify using a Support Vector Machine."""
    model = classifier.fit(trainData, trainTargets)

    classification = model.predict(testData)

    print("\nUsing {0}".format(classifier))
    showPerformance(testTargets, classification)

def applySVM(trainData, trainTargets, testData, testTargets):
    """Train and classify using a Support Vector Machine (linear kernel)."""
    svm = SVC(kernel="linear")
    model = svm.fit(trainData, trainTargets)

    classification = [model.predict(d)[0] for d in testData]

    print("\nUsing a Support Vector Machine:")
    showPerformance(testTargets, classification)

def applyNaiveBayes(trainData, trainTargets, testData, testTargets):
    """Train and classify using Naive Bayes."""
    gnb = GaussianNB()
    model = gnb.fit(trainData, trainTargets)

    classification = [model.predict(d)[0] for d in testData]

    print("\nUsing Naive Bayes:")
    showPerformance(testTargets, classification)

def applyNaiveBayes2(trainData, trainTargets, testData, testTargets):
    """Train and classify using Naive Bayes."""
    gnb = MultinomialNB()
    model = gnb.fit(trainData, trainTargets)

    classification = [model.predict(d)[0] for d in testData]

    print("\nUsing Naive Bayes:")
    showPerformance(testTargets, classification)

def applyDecisionTree(trainData, trainTargets, testData, testTargets, featureNames):
    """Train and classify using a Decision Tree and prints the decision Tree."""
    decisionTree = DecisionTreeClassifier()
    model = decisionTree.fit(trainData, trainTargets)

    # Create graph description of the Decision Tree
    dot_data = StringIO()
    #export_graphviz(model, out_file=dot_data, max_depth=5)
    print("Feature names:", featureNames)
    export_graphviz(model, out_file=dot_data, feature_names=featureNames,
                    max_depth=5)
    export_graphviz(model, out_file="DecisionTree.dot", feature_names=featureNames,
                    max_depth=5)
    #with open("DecisionTree.dot", 'r') as dotFile:
    #    dotFile.write(exportFile)
    # Create PDF from dot
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #path = "/Users/konstantin/Documents/University/Bachelorthesis/paper/src/DecisionTree.dot"
    #graph = pydot.graph_from_dot_file(path)
    #graph.write_pdf("DecisionTree.pdf")


    classification = [model.predict(d)[0] for d in testData]

    print("\nUsing a Decision Tree:")
    showPerformance(testTargets, classification)



# ---- Test basic ML tools ----
def test():
    """Test some basic ML behaviour."""
    pass

if __name__ == '__main__':
    test()
