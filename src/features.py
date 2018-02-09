# -*- coding: utf-8 -*-
from __future__ import print_function
import re
from defaultConfig import REGEX_FEATURE_CONFIG
from review import Review

class Feature(object):
    """
    Basic feature class. A feature has a name and some function that extracts
    information from a review.
    """
    def __init__(self, name, short=None, function=None):
        self.name = name.encode("utf-8")
        if not function == None:
            self.extract = function
        if not short == None:
            self.short = short[:4].encode("utf-8")
        else:
            self.short = name[:4].encode("utf-8")

    def __repr__(self):
        return "Feature({name}, {function})".format(name=self.name,
                                                function=self.extract.__name__)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return u"{name}({short})".format(name=self.name, short=self.short)

    def extract(self, review):
        """Extract the information from the review."""
        pass


class RegularExpressionFeature(Feature):
    """A feature that is based on regular expressions."""
    def __init__(self, name, short=None, regEx=r""):
        Feature.__init__(self, name, short)
        self.regEx = re.compile(regEx, flags=re.UNICODE|re.VERBOSE)

    def __repr__(self):
        return "RegExFeature({name})".format(name=self.name)

    def extract(self, review):
        """Search for the regular expression in the review's text."""
        if self.regEx.findall(review.text):

            # if self.name == "Interjektion": #and not self.name == "ExclamationMark":
            #     print(self.name, ":", review.ironic, review.ID)
            #     for m in self.regEx.finditer(review.text):
            #         print(m.span())
            #         print(review.text[m.span()[0]:m.span()[1]])

            return 1
        else:
            return 0


# ---- Feature functions (They replace the extract method in features) ----
def starRating(review):
    """Returns the star rating."""
    return [1 if i+1 == int(review.stars) else 0 for i in range(5)]

def posStarPolarityDiscrepancy(review):
    """
    Returns True if the star rating is 1.0 or 2.0 but the polarity is positive.
    """
    if review.stars < 3.0 and review.polarity == "positive":
        return 1
    else:
        return 0

def negStarPolarityDiscrepancy(review):
    """
    Returns True if the star rating is 4.0 or 5.0 but the polarity is negative.
    """
    if review.stars > 3.0 and review.polarity == "negative":
        return 1
    else:
        return 0

# TODO: check if adverbs are good indicators as well.
def scareQuotes(review):
    """
    Searches for scare quotes surrounding one or two positive noun,
    adjective or adverb.
    """
    nounCategories = ["NN", "NNS", "NNP", "NNPS"]
    adjCategories = ["JJ", "JJR", "JJS"]
    advCategories = ["RB", "RBR", "RBS"]

    polarityCategories = nounCategories + adjCategories #+ advCategories
    openingMarks = [u"\"", u"`", u"``", u"*"]    # * are not tokenized.
    closingMarks = [u"\"", u"'", u"''", u"*"]

    numberOfWords = len(review.words)
    for i in range(numberOfWords):
        # Opening quotes
        if review.words[i].text in openingMarks:
            # One on the two words in quotes is a positive adjective of noun
            if (i + 3 < numberOfWords-1 and
                review.words[i+3].text in closingMarks):
                if (review.words[i+1].polarity == "positive" and
                review.words[i+1].pos in polarityCategories or
                review.words[i+2].polarity == "positive" and
                review.words[i+2].pos in polarityCategories):
                    return 1

            # The word in quotes is a positive adjective of noun
            if (i + 2 < numberOfWords-1 and
                review.words[i+2].text in closingMarks):
                if (review.words[i+1].polarity == "positive" and
                review.words[i+1].pos in polarityCategories):
                    return 1
    return 0

def scareQuotesNegative(review):
    """
    Searches for scare quotes surrounding one or two negative noun,
    adjective or adverb.
    """
    nounCategories = ["NN", "NNS", "NNP", "NNPS"]
    adjCategories = ["JJ", "JJR", "JJS"]
    advCategories = ["RB", "RBR", "RBS"]

    polarityCategories = nounCategories + adjCategories #+ advCategories
    openingMarks = [u"\"", u"`", u"``", u"*"]    # * are not tokenized.
    closingMarks = [u"\"", u"'", u"''", u"*"]

    numberOfWords = len(review.words)
    for i in range(numberOfWords):
        # Opening quotes
        if review.words[i].text in openingMarks:
            # One on the two words in quotes is a negative adjective of noun
            if (i + 3 < numberOfWords-1 and
                review.words[i+3].text in closingMarks):
                if (review.words[i+1].polarity == "negative" and
                review.words[i+1].pos in polarityCategories or
                review.words[i+2].polarity == "negative" and
                review.words[i+2].pos in polarityCategories):
                    return 1

            # The word in quotes is a negative adjective of noun
            if (i + 2 < numberOfWords-1 and
                review.words[i+2].text in closingMarks):
                if (review.words[i+1].polarity == "negative" and
                review.words[i+1].pos in polarityCategories):
                    return 1
    return 0


def positiveNGramPlusPunctuation(review, n=4, pattern=r"(!!|!\?|\?!)"):
    """
    Searches (4-Gram+) followed by the given pattern.
    """
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    numberOfWords = len(review.words)

    for i in range(numberOfWords-1):
        if exclamation.findall(review.words[i].text + review.words[i+1].text):
            # Get (maximal) the 4 previous words
            previousWords = review.words[(i-n if i > (n-1) else 0):i]
            positiveWords = [w for w in previousWords
                            if w.polarity == "positive"]
            negativeWords = [w for w in previousWords
                            if w.polarity == "negative"]
            if positiveWords and not negativeWords:
                return 1
    return 0

def negativeNGramPlusPunctuation(review, n=4, pattern=r"(!!|!\?|\?!)"):
    """Searches a (4-Gram-) followed by the given pattern."""
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    numberOfWords = len(review.words)

    for i in range(numberOfWords-1):
        if exclamation.findall(review.words[i].text + review.words[i+1].text):
            # Get (maximal) the 4 previous words
            previousWords = review.words[((i-n) if i > (n-1) else 0):i]
            positiveWords = [w for w in previousWords
                            if w.polarity == "positive"]
            negativeWords = [w for w in previousWords
                            if w.polarity == "negative"]
            if not positiveWords and negativeWords:
                return 1
    return 0

def ellipsisPlusPunctuation(review, pattern=r"(!!|!\?|\?!|\?)"):
    """
    Searches for an ellipsis followed by the given pattern.
    """
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    ellipsis = re.compile(r"(\.\.|\. \. \.)$")
    numberOfWords = len(review.words)

    for i in range(numberOfWords-1):
        if exclamation.findall(review.words[i].text + review.words[i+1].text):
            # Get the 2 previous words
            previousWords = review.words[(i-2 if i > 1 else 0):i]

            if ellipsis.findall("".join([w.text for w in previousWords])):
                return 1
    return 0


# def interjectionPlusPunctuation(review, pattern=r"(!!|!\?|\?!|\?)"):
#     """Searches for an interjection followed by the given pattern."""
#     exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
#     numberOfWords = len(review.words)

#     for i in range(numberOfWords-1):
#         if exclamation.findall(review.words[i].text + review.words[i+1].text):
#             print
#             # Get the 2 previous words
#             previousWords = review.words[(i-2 if i > 1 else 0):i]

#             if ellipsis.findall("".join([w.text for w in previousWords])):
#                 return 1
#     return 0


def positiveStreak(review, length=3):
    """Searches for streaks of positive words."""
    numberOfWords = len(review.words)

    for i in range(numberOfWords-length):
        if review.words[i].polarity == "positive":
            if all([True if w.polarity == "positive" else False
                        for w in review.words[i+1:i+length]]):
                return 1
    return 0

def negativeStreak(review, length=3):
    """Searches for streaks of negative words."""
    numberOfWords = len(review.words)

    for i in range(numberOfWords-length):
        if review.words[i].polarity == "negative":
            if all([True if w.polarity == "negative" else False
                        for w in review.words[i+1:i+length]]):
                return 1
    return 0

# ---- Bag of words ----
def createBagOfWordsDictionary(reviews):
    """Create the dictionary of all words."""
    dictionary = {}
    index = 0

    for review in reviews.values():
        for each in review.words:
            word = each.text.lower()
            if not word in dictionary:
                dictionary[word] = index
                index += 1

    # print("Unigram index:", index)
    return dictionary

def fillBagOfWords(bowDictionary, review):
    """Fill a bag with words of one review."""
    # initialise with zeros
    bag = [0] * len(bowDictionary )
    for word in review.words:
        bag[bowDictionary[word.text.lower()]] = 1
    return bag

def verifyBagOfWords(review, bowDictionary ):
    initialWords = set(review.words)
    bag = fillBagOfWords(bowDictionary, review)
    indices = [i for i, v in enumerate(bag) if v == 1]
    unpackedWords = {word for i, v in enumerate(bag) if v == 1
                    for word, index in bowDictionary .items() if index == i}
    print("Initial words:", len(initialWords), "Bag words:", len(unpackedWords))
    return not initialWords - unpackedWords


def createBagOfBigramsDictionary (reviews):
    """Create a dictionary of bigrams."""
    dictionary = {}
    index = 0

    for review in reviews.values():
        # Transform bigrams of tokens to strings.
        bigrams = [(word1.text.lower(), word2.text.lower())
                for word1, word2 in review.bigrams]
        for bigram in bigrams:
            if not bigram in dictionary:
                dictionary[bigram] = index
                index += 1

    print("Bigram index:", index)
    return dictionary

def fillBagOfBigrams(bigramDictionary, review):
    """Fill a bag with bigrams of one review."""
    # initialise with zeros
    bag = [0] * len(bigramDictionary )
    bigrams = [(word1.text.lower(), word2.text.lower())
                for word1, word2 in review.bigrams]
    for bigram in bigrams:
        bag[bigramDictionary[bigram]] = 1
    return bag


def sentiment(review):
        """Returns a vector for the sentiment of a given review."""
        # result = [0] * 7
        # polarity = len(review.positiveWords) - len(review.negativeWords)
        # if polarity < -10:
        #     result[0] = 1
        # elif polarity < -5:
        #     result[1] = 1
        # elif polarity < 0:
        #     result[2] = 1
        # elif polarity == 0:
        #     result[3] = 1
        # elif polarity > 10:
        #     result[6] = 1
        # elif polarity > 5:
        #     result[5] = 1
        # elif polarity > 0:
        #     result[4] = 1

        # result = [0] * 3
        # polarity = len(review.positiveWords) - len(review.negativeWords)
        # if polarity < -1:
        #     result[0] = 1
        # elif polarity > 1:
        #     result[2] = 1
        # else:
        #     result[1] = 1

        result = [0] * 1
        polarity = len(review.positiveWords) - len(review.negativeWords)
        if polarity > 0:
            result[0] = 1


        # print("polarity:", result, "with", ", ".join(review.positiveWords),
        #     "as positive words and", ", ".join(review.negativeWords),
        #     "as negative words.")
        return result



# ---- Create Features and extract information from reviews. ----
def createFeatures(featureConfig=None):
    """Returns a list of features created from configurations."""
    if featureConfig is None:
        featureConfig = {u"Positive Imbalance": (u"w-\u2605 ",
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
                        u"Pos&Ellipsis": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                        u"Neg&Ellipsis": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
        }

    features = []

    for name, config in featureConfig.items():
        features.append(Feature(name, config[0], config[1]))

    # Add regularExpression features
    for name, config in REGEX_FEATURE_CONFIG.items():
        features.append(RegularExpressionFeature(name, config[0], config[1]))

    return features

def extractFeatures(reviewIDs, reviews, features=None, createARFF=False, bowDictionary=None):
    """Returns lists of the used features and its result."""
    if features == None:
        features = createFeatures()
    else:
        features = createFeatures(features)

    if bowDictionary is None:
        bowDictionary = createBagOfWordsDictionary(reviews)
    # print("Unigrams dictionary size:", len(bowDictionary))
    # bigramDictionary = createBagOfBigramsDictionary(reviews)
    # print("Bigrams dictionary size:", len(bigramDictionary))

    featureVectors = {}
    for ID in reviewIDs:
        review = reviews[ID]
        featureVectors[ID] = []

        # Irony specific features
        for feature in features:
            featureVectors[ID].append(feature.extract(reviews[ID]))
        #print("Specific Features:", len(features))

        # Star Rating
        featureVectors[ID].extend(starRating(review))

        # Sentiment
        featureVectors[ID].extend(sentiment(review))

        # Bag of words
        featureVectors[ID].extend(fillBagOfWords(bowDictionary, review))

        # Bag of bigrams
        # featureVectors[ID].extend(fillBagOfBigrams(bigramDictionary, review))

    # Save extracted features in a file
    if createARFF:
        # As attributes add the specific features, the star rating,
        # the sentiment, and all unigrams.
        attributes = []
        attributes.extend(["\"{name}\"".format(name=feature.name)
                            for feature in features])
        attributes.extend(["\"1 star\"", "\"2 stars\"", "\"3 stars\"",
                            "\"4 stars\"", "\"5 stars\""])
        attributes.append("\"positive sentiment\"")
        attributes.extend(["\"word={word}\"".format(word=word)
            for word in sorted(bowDictionary, key=bowDictionary.get)])

        # attributes.extend(["\"bigram={bigram1}_{bigram2}\"".format(
        #                     bigram1=bigram[0], bigram2=bigram[1])
        #     for bigram in sorted(bigramDictionary, key=bigramDictionary.get)])

        categories= {ID: "ironic" if reviews[ID].ironic else "regular"
                    for ID in reviewIDs}

        createARFFFile(attributes, featureVectors, categories)

        print("Features:", len(featureVectors[reviewIDs[0]]))

    return features, featureVectors


def createARFFFile(features, data, categories, filename="./features.arff"):
    """Create an ARFF file, that contains the extracted features."""
    relation = "@RELATION irony"
    attributes = []
    for feature in features:
        attributes.append("@ATTRIBUTE {name}\tNUMERIC".format(name=feature))
    attributes.append("@ATTRIBUTE class {ironic, regular}")
    # print("\n".join(attributes))

    with open(filename, "w") as arffFile:
        arffFile.write(relation + "\n")
        arffFile.write("\n".join(attributes))
        arffFile.write("\n@DATA\n")
        for ID, featureVector in data.items():
            # print("\n", ID, "\n".join([str(each) for each in zip(attributes, featureVector)]))
            arffFile.write(",".join(str(value) for value in featureVector) + "," + categories[ID] + "\n")



def showFeatureOccurrence(features, featureVectors, gold=None, classification=None):
    """Shows the features' occurrence."""
    MAX_ID_LENGTH = 23
    MAX_NAME_LENGTH = 4
    MAX_FEATURES = 29

    print("Using the following features:")
    print(", ".join(["{0} ({1})".format(f.name, f.short) for f in features]))

    headline = "ID \t\t\tCorrect | {0}".format(
                                    " ".join([f.short + " "*(4 - len(f.short))
                                            for f in features[:MAX_FEATURES]]))
    print(headline)
    for ID, vec in featureVectors.items():
        print("{0}{1}{2}\t| {3}".format(ID,
                    "\t"*(MAX_ID_LENGTH/len(ID)),
                    "Yes " if gold and gold[ID]==classification[ID] else "___",
                    " ".join(["Yes " if v == 1 else "_"*4 for v in vec[:MAX_FEATURES]])))

    print(headline)
    vec = [vector[:MAX_FEATURES] for vector in featureVectors.values()]

    if not classification == None and not gold == None:
        correct = sum([1 if gold and gold[ID] == p else 0
                        for ID, p in classification.items()])
    else:
        correct = 0

    occurrences = [sum([1 if v[i] else 0 for v in vec])
                    for i in range(len(features[:MAX_FEATURES]))]
    print("Summation\t\t{0}\t| {1}".format(correct,
        " ".join([" "*(4-len(str(s))) + str(s) for s in occurrences])))


# ---- Functions to test basic behaviour. ----
def testFeatures():
    """Tests basic feature functions."""
    raw = u"""<STARS>2.0</STARS>
    <TITLE>House</TITLE>
    <DATE>October 22, 2013</DATE>
    <AUTHOR>Customer</AUTHOR>
    <PRODUCT>House</PRODUCT>
    <REVIEW>
    Really!? "nice" super house..!?! good . . . good good good Or was it really that bad?! nice good nice bad.....
    </REVIEW>"""

    r = Review(rawReview=raw, ironic=True)
    print(r)
    print([w for w in r.words])
    print("Scarequotes:", scareQuotes(r))
    print("positiveNGramPlusPunctuation:", positiveNGramPlusPunctuation(r))
    print("negativeNGramPlusPunctuation:", negativeNGramPlusPunctuation(r))
    print("positiveStreak:", positiveStreak(r, length=4))
    print("Ellipsis+Punctuation:", ellipsisPlusPunctuation(r))
    print("Star rating:", starRating(r))


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


if __name__ == '__main__':
    #testFeatures()
    createARFFFile(["Feature 1", "Feature 2", "Feature 3", "Feature 4"],
                    {1: [1, 1, 0, 1], 2: [1, 0, 0, 1], 3: [0, 1, 0, 1]},
                    {1: "ironic", 2: "ironic", 3: "regular"})
