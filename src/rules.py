# -*- coding: utf-8 -*-
from __future__ import print_function
from corpus import Corpus
from features import Feature, createFeatures, extractFeatures
from features import showFeatureOccurrence
from defaultConfig import CORPUS_PATH
from performance import showPerformance

# TODO:
# Major refactoring.
# Change names of the features or better yet load from a dictionary.

def classify(features, featureVectors, decisiveFeatureNames=None):
    """
    Classifies reviews as ironic or regular based on rules.

        What is a rule?
        What distinguishes a rule from a feature?

    Apply rules:
        -If one of the following features is present, classify as ironic.
    """
    if decisiveFeatureNames == None:
        decisiveFeatureNames = ["Positive star polarity discrepancy",
                                #"Negative star polarity discrepancy",
                                #"Streak of Positive Words",
                                "Streak of Negative Words",
                                #"Positive Ppunctuation",
                                "Negative Ppunctuation",
                                "Negative word and ellipsis",
                                "Scare quotes",
                                "Negative Scare quotes",
                                "Question Mark",
                                "Interrobang",
                                "Ellipsis and Punctuation",
                                "Interjection",
                                #"Emoticon Happy",
                                #"Emoticon Laughing",
                                #"Emoticon Winking",
                                #"Emotion Tongue",
                                #"Onomatopoeia",
                                #"LoLAcroym",
                                #"GrinAcronym",
                                ]

    featureNames = [f.name for f in features]

    decisiveFeatureIndices = [featureNames.index(name)
                                for name in decisiveFeatureNames]

    return {ID: any([vector[i] for i in decisiveFeatureIndices])
            for ID, vector in featureVectors.items()}

def simpleClassify(featureVectors):
    """Classifies a review as ironic based on rules."""
    return {ID: any(vector) for ID, vector in featureVectors.items()}

def applySingleRules(IDsFilename):
    """
    Should originally just apply one rule.
    Is now used to apply one feature to the given corpus.
    So it basically shows how often each feature occurs in ironic and regular
    reviews.
    """
    print("Using the set at '{path}{file}'".format(path=CORPUS_PATH,
                                                    file=IDsFilename))

    print("Creating reviews...(this may take a while)")
    dataSet = Corpus(IDsFilename, corpusPath=CORPUS_PATH)
    print("Loading reviews...")
#   dataSet = Corpus.loadCorpus(filename="training_set.pk")
    # dataSet = Corpus.loadCorpus(filename="training_and_validation_set.pk")


    print("Extracting features...")
    features, featureVectors = extractFeatures(dataSet.reviewIDs,
                                                dataSet.reviews)

    showFeatureOccurrence(features, featureVectors)

    gold = dataSet.goldStandard

    # decisiveFeatureNames = ["Scare quotes",
    #                         "Positive star polarity discrepancy",
    #                         "Negative star polarity discrepancy",
    #                         "Positive Ppunctuation",
    #                         "Negative Ppunctuation",
    #                         "Streak of Positive Words",
    #                         "Ellipsis and Punctuation",
    #                         "Emoticon Happy", "Emoticon Laughing",
    #                         "Emoticon Winking", "Emotion Tongue",
    #                         "LoLAcroym", "GrinAcronym", "Onomatopoeia",
    #                         "Interrobang"]

    decisiveFeatureNames = [f.name for f in features]

    for d in decisiveFeatureNames:
        classification = classify(features, featureVectors, [d])

        targets = []
        cls = []

        for ID, g in gold.items():
            targets.append(g)
            cls.append(classification[ID])

        print("\nClassifying by rule: ", d)

        showPerformance(targets, cls)


def applyRules(IDsFilename):
    """Uses rule based approach to classify the reviews from the given set."""
    print("Using the set at '{path}{file}'".format(path=CORPUS_PATH,
                                                    file=IDsFilename))

    print("Creating reviews...(this may take a while)")
    dataSet = Corpus(IDsFilename, corpusPath=CORPUS_PATH)

    # print("Loading reviews...")
    # dataSet = Corpus.loadCorpus(filename="training_set.pk")

    print("Extracting features...")
    features, featureVectors = extractFeatures(dataSet.reviewIDs,
                                                dataSet.reviews)

    gold = dataSet.goldStandard
    classification = classify(features, featureVectors)

    showFeatureOccurrence(features, featureVectors, gold, classification)

    targets = []
    cls = []

    for ID, g in gold.items():
        targets.append(g)
        cls.append(classification[ID])

    showPerformance(targets, cls)


# def applySingleRule():
#     """Uses rule based approach to classify sentences."""
#     # TODO: Use one rule. i:i+1
#     # TODO: Still in use?
#     reviewPairIDs, reviewIronicIDs, reviewRegularIDs, reviews = readCorpus(
#         path=WORKING_COPY_PATH, filename=SET_FILENAMES[2])

#     print("Ironic:", len(reviewIronicIDs))
#     print("Regular:", len(reviewRegularIDs))

#     IDLists = [reviewIronicIDs, reviewRegularIDs]

#     features, result = extractFeatures(IDLists[0]+IDLists[1], reviews)

#     for i,f in enumerate(features):
#         classification = [(reviews[ID].ironic, any(vec[i:i+1]))
#                             for ID, vec in result.items()]
#         print(f)
#         #showFeatureOccurrence(reviews, features, result, classification)
#         showPerformance(reviews, classification)
#         print(calcPerformance)


# ---- Tests for basic functions ----
def testClassification():
    """Tests if the classify function works properly"""
    names = ["1_1_TEST", "1_2_Test"]
    featureVectors = {name[0]: [False]*24,
                        name[1]: [False]*24}

    featureVectors[name[0]][3] = True
    featureVectors[name[0]][4] = True
    featureVectors[name[1]][3] = True
    decisiveFeatureNames = ["scareQuote", "PosStarPolarityDiscrepancy"]

    classify(createFeatures(), featureVectors, decisiveFeatureNames)

if __name__ == '__main__':
    testClassification()
