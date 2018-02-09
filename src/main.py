# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import codecs
from timeit import timeit
from defaultConfig import CORPUS_PATH, REVIEW_IDS_FILENAME
from defaultConfig import IRONIC_REVIEWS_PATH, REGULAR_REVIEWS_PATH
from defaultConfig import WORKING_COPY_PATH, SET_FILENAMES
from corpus import Corpus, readCorpus, readReviews, showStatistics
from features import extractFeatures, createFeatures, showFeatureOccurrence
from performance import showPerformance
from rules import applyRules, applySingleRules
from rules import classify as ruleClassify
from machineLearning import applyML
from setGenerator import shuffledSet, createSets


# Corpus Mode
# ------------------------------------------------------------------------------
def runCorpusMode(arguments):
    print("Loading corpus. This may take awhile.")
    requestedInformation = {"stats": showCorpusStatistics,
                            "reviews": showDetailedReviews,}
    duration = timeit(requestedInformation[arguments.info], number=1)
    showDuration(duration)

def showCorpusStatistics(IDsFilename=REVIEW_IDS_FILENAME):
    """Shows statistics about the corpus."""
    corpus = Corpus(IDsFilename)
    corpus.statistics()

def showDetailedReviews(IDsFilename=REVIEW_IDS_FILENAME):
    """Shows details of all given reviews."""
    corpus = Corpus(IDsFilename)
    for ID, review in corpus.reviews.items():
        print("{id}:\n{details}".format(id=ID, details=review.showDetails()))


# Features Mode
# ------------------------------------------------------------------------------
def runFeaturesMode(arguments):
    print("Loading corpus. This may take awhile.")
    requestedInformation = {"export": exportFeatures,
                            "show": showFeatures,}
    duration = timeit(requestedInformation[arguments.info], number=1)
    showDuration(duration)

def exportFeatures():
    corpus = Corpus(SET_FILENAMES[3])
    features, featureVectors = extractFeatures(corpus.reviewIDs, corpus.reviews,
                                                features=None, createARFF=True)

def showFeatures(IDsFilename=REVIEW_IDS_FILENAME):
    corpus = Corpus(IDsFilename)
    features, featureVectors = extractFeatures(corpus.reviewIDs, corpus.reviews,
                                                features=None)

    showFeatureOccurrence(features, featureVectors)

    # applySingleRules(SET_FILENAMES[3])

# Interactive Mode
# ------------------------------------------------------------------------------
def runInteractiveMode(arguments):
    print(arguments.text, "\nNothing happens yet. :-)")
    # TODO:
    # * Create review from text. Review class needs to be extended.
    # * Save/load learned model and classify a given input string


# Rules Mode
# ------------------------------------------------------------------------------
def runRulesMode(arguments):
    """Determines which set is classified by the rule-based approach."""
    setFilenames = {"training": SET_FILENAMES[0],
                    "test": SET_FILENAMES[1],
                    "validation": SET_FILENAMES[2],}

    duration = timeit(lambda: applyRules(setFilenames[arguments.set.lower()]),
                number=1),
    showDuration(duration)


# Machine learning
# ------------------------------------------------------------------------------
def runMachineLearningMode(arguments):
    """
    Determines which set will be used as the validation set, i.e. the set the
    classifiers will be tested against. Currently are the following values
    available
    * 'test' and
    * 'cross-validation'.
    """
    if arguments.set.lower() == "cross-validation":
        duration = timeit(lambda: applyML(SET_FILENAMES[3]),
                        number=1)
    elif arguments.set.lower() == "test":
        duration = timeit(lambda: applyML(SET_FILENAMES[0], SET_FILENAMES[1]),
                        number=1)
    showDuration(duration)


# Set mode
# ------------------------------------------------------------------------------
def runSetMode(arguments):
    duration = timeit(generateSets, number=1),
    showDuration(duration)

def generateSets():
    """
    Generate a shuffled set for cross-validation and training and test sets.
    """
    shuffledSet(randomSeed=44)
    createSets(setSizes=[90, 10])

# Test Mode
# ------------------------------------------------------------------------------
def runTestMode(arguments):
    """Invokes the to be tested functionality."""
    tests = {"ARFF": testARFFExport,
            "BOW": testBagOfWords,
            "corpus": testCorpus,
            "features": testFeatures,
            "ml": testML,
            "reviews": testReviews,
            "rules": testRules,}
    duration = timeit(tests[arguments.mode], number=1),
    showDuration(duration)

def createTestReviews():
    """For faster testing, create a small corpus."""
    ironicIDs = ["1_5_R4F7L5HVAZZHU",
                "1_13_R3JBEPN242VR6U",
                "1_7_R19YXRYPPILTJ5",
                "1_9_R3POR9QS2KZGI8"]
    regularIDs = ["47_17_R2A767BWWBGBTK"]

    reviews = {}
    reviews.update(readReviews(ironicIDs,
                                CORPUS_PATH + IRONIC_REVIEWS_PATH,
                                True))
    reviews.update(readReviews(regularIDs,
                                CORPUS_PATH + REGULAR_REVIEWS_PATH,
                                False))
    return ironicIDs, regularIDs, reviews

def testCorpus():
    """Tests if the corpus can be loaded and displayed."""
    entireCorpus = Corpus("training_and_validation_set.txt")
    entireCorpus.statistics()

def testFeatures():
    """Tests if the features work on the corpus."""
    ironicIDs, regularIDs, reviews = createTestReviews()
    features, featureVectors = extractFeatures(ironicIDs + regularIDs, reviews)
    showFeatureOccurrence(features, featureVectors)

def testBagOfWords():
    """Tests if bag of word feature works correctly."""
    ironicIDs, regularIDs, reviews = createTestReviews()

    print("-"*8, "Create dictionary of all used words.", "-"*8)
    bowDictionary  = createBagOfWordsDictionary(reviews)
    print("Number of words overall", sum([len(r.words) for r in reviews.values()]))
    print("Unique words", len(bowDictionary ))
    # When are two Tokens equal? - lowercase word or token, i.e. + pos + polarity
    for word in bowDictionary :
        print(word.__repr__())

    print("-"*8, "Test review's words and sets.", "-"*8)
    print("Words:", len(reviews[ironicIDs[2]].words))
    print("Unique words:", len(set(reviews[ironicIDs[2]].words)))
    bag = fillBagOfWords(bowDictionary , reviews[ironicIDs[2]])
    print(bag)
    print("Feature vector length:", len(bag))
    print("-"*8, "Create feature vector.", "-"*8)

    print("-"*8, "Verify that bag of word works correctly.", "-"*8)
    print(verifyBagOfWords(reviews[ironicIDs[2]], bowDictionary ))

    print("-"*8, "No final result present yet.", "-"*8)

def testRules():
    """Uses rule based approach to classify reviews."""
    ironicIDs, regularIDs, reviews = createTestReviews()
    features, featureVectors = extractFeatures(ironicIDs + regularIDs, reviews)

    gold = {ID: reviews[ID].ironic for ID in ironicIDs + regularIDs}
    classification = ruleClassify(features, featureVectors)

    showFeatureOccurrence(features, featureVectors, gold, classification)
    showPerformance(gold, classification)

def testML():
    """Uses machine learning based approach to classify reviews."""
    print(("Test if machine learning approach functions as intended."
        "-- Nothing happens yet."))

def testReviews():
    """Tests, if the reviews with the given IDs can be loaded and displayed."""
    ironicIDs, regularIDs, reviews = createTestReviews()
    showDetailedReviews(reviews)

def testStats():
    """Test the corpus' statistics function."""
    ironicIDs, regularIDs, reviews = createTestReviews()
    showStatistics(ironicIDs,regularIDs, reviews, ["Ironic", "Regular"])

def testARFFExport():
    """ Tests the functionality to export the features as a valid ARFF file."""
    ironicIDs, regularIDs, reviews = createTestReviews()
    reviewIDs = ironicIDs + regularIDs
    # for review in reviews.values():
    #     print(review)

    features, featureVectors = extractFeatures(reviewIDs, reviews,
                                                features=None, createARFF=True)

    # corpus = Corpus("file_pairing.txt")
    # print("Reviews:", len(corpus.reviews))

    # features, featureVectors = extractFeatures(corpus.reviewIDs, corpus.reviews,
    #                                             features=None, createARFF=True)


# Auxiliary functions
# ------------------------------------------------------------------------------
def showDuration(duration):
    print("-"*60, "\nElapsed time: {duration}s".format(duration=duration))


# Main function
# ------------------------------------------------------------------------------

# Application texts: Name, usage and help messages.
APPLICATION_NAME = "Irony Detector"
APPLICATION_DESCRIPTION = "Detects irony in amazon reviews."

# Main function: Controls the general application behaviour.
def main():
    # Top-level parser
    commandParser = argparse.ArgumentParser(prog=APPLICATION_NAME,
                                            description=APPLICATION_DESCRIPTION)
    subParsers = commandParser.add_subparsers(title="Commands",
                                            description="""The following
                                                        commands can be invoked.
                                                        """,
                                            help="Valid commands.",
                                            dest="command",)

    # Corpus command parser:
    corpusParser = subParsers.add_parser("corpus",
                                        help="""Show details about the entire
                                                corpus.""")
    corpusParser.set_defaults(func=runCorpusMode)
    corpusParser.add_argument("info",
                            choices=["stats", "reviews"],
                            default="stats",
                            help="Valid commands.")

    # Feature command parser:
    featureParser = subParsers.add_parser("feature",
                                        help="""Shows how often each feature
                                                is found for ironic and
                                                regular reviews in the
                                                training_and_validation_set.""")
    featureParser.set_defaults(func=runFeaturesMode)
    featureParser.add_argument("info",
                                choices=["export", "show"],
                                default="show",
                                help="""Show specific features or export the
                                        features as an arff file.""")

    # Interactive command parser:
    interactiveParser = subParsers.add_parser("interactive",
                                            help="""The interactive mode
                                                classifies a given sentence
                                                using a saved model.""")
    interactiveParser.set_defaults(func=runInteractiveMode)
    interactiveParser.add_argument("text",
                                    type=str,
                                    help="""Text to classify, e.g.
                                        \"What a great product ;-)\".""")

    # Machine Learning approach command parser:
    mlParser = subParsers.add_parser("ml",
                                    help="""Use the machine learning approach
                                            to classify reviews.""")
    mlParser.set_defaults(func=runMachineLearningMode)
    mlParser.add_argument("set",
                        choices=["cross-validation", "test",],
                        default="cross-validation",
                        help="""Set used for the validation step.""")


    # Rule-based approach command parser:
    # rulesParser = subParsers.add_parser("rules",
    #                                     help="""Use the rule based approach
    #                                             to classify reviews.""")
    # rulesParser.set_defaults(func=determineTrainingSet)
    # rulesParser.add_argument("set",
    #                         choices=["training", "validation", "test"],
    #                         default="training",
    #                         help="""Set used for the classification.""")


    # Set command parser:
    setParser = subParsers.add_parser("sets",
                                    help="""Divide the corpus into training,
                                            validation and test set.""")
    setParser.set_defaults(func=lambda: runSetMode(arguments))


    # Test command parser:
    # testParser = subParsers.add_parser("test", help="""Test basic functionality
    #                                                 of the application.""")
    # testParser.set_defaults(func=runTestMode)
    # testParser.add_argument("mode",
    #                         choices=["ARFF", "BOW", "features", "corpus",
    #                                 "ml", "reviews", "rules",],
    #                         default="review",
    #                         help="""Choose the functionality you want
    #                                 to test.""")


    # Parse given arguments and call the function specified by the command:
    arguments = commandParser.parse_args()

    if arguments.command in ["corpus", "feature", "interactive", "ml", "rules",
                            "test"]:
        arguments.func(arguments)
    else:
        arguments.func()

if __name__ == '__main__':
    main()
