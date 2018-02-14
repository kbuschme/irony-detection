# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import Counter
from re import sub
import codecs
import pickle

#import matplotlib.pyplot as plt

from defaultConfig import CORPUS_PATH, IRONIC_REVIEWS_PATH, REGULAR_REVIEWS_PATH
from defaultConfig import REVIEW_IDS_FILENAME, IRONIC_UTTERANCES_FILENAME
from review import Review


class Corpus(object):
    """Represents a corpus."""

    def __init__(self, IDsFilename,
                corpusPath=CORPUS_PATH,
                ironicReviewsPath=IRONIC_REVIEWS_PATH,
                regularReviewsPath=REGULAR_REVIEWS_PATH,
                utterancesFile=IRONIC_UTTERANCES_FILENAME):
        # Save File locations
        self.IDsFilename = IDsFilename
        self.corpusPath = corpusPath
        self.ironicReviewPath = ironicReviewsPath
        self.regularReviewsPath = regularReviewsPath
        self.utterancesFile = utterancesFile

        # Load IDs
        self._ironicReviewIDs = []
        self._regularReviewIDs = []
        self._pairReviewIDs = []
        self.loadIDs()

        # Load reviews
        self.reviews = {}
        self.loadReviews()

        self.saveCorpus()

    def __repr__(self):
        description = """Corpus('{0}', ironicReviewsPath='{1}',
                        regularReviewsPath='{2}', utterancesFile='{3}')"""
        return description.format(
                            self.IDsFilename,
                            self.ironicReviewPath,
                            self.regularReviewsPath,
                            self.utterancesFile).encode('utf-8')

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        description = "Corpus with {0} ironic and {1} regular (total {2}) reviews."
        numberOfIronic = len(self.ironicReviewIDs)
        numberOfRegular = len(self.regularReviewIDs)
        return description.format(numberOfIronic,
                                    numberOfRegular,
                                    numberOfIronic + numberOfRegular)

    def loadIDs(self):
        """Loads the IDs for pairs, ironic and regular utterances."""
        location = self.corpusPath + self.IDsFilename
        # 'utf-8-sig' removes leading Byte Order Mark (BOM)
        with codecs.open(location, 'r', encoding='utf-8-sig') as idsFile:
            rawReviewIDs = idsFile.readlines()
        for rawID in rawReviewIDs:
            IDParts = rawID.split()

            if IDParts[0] == "PAIR:":
                self._pairReviewIDs.append((IDParts[1], IDParts[3]))
            elif IDParts[0] == "IRONIC:":
                self._ironicReviewIDs.append(IDParts[1])
            elif IDParts[0] == "REGULAR:":
                self._regularReviewIDs.append(IDParts[1])

    def loadReviews(self):
        """Loads all reviews."""
        self.reviews.update(readReviews(self.ironicReviewIDs,
                                    self.corpusPath + self.ironicReviewPath,
                                    ironic=True))
        self.reviews.update(readReviews(self.regularReviewIDs,
                                    self.corpusPath + self.regularReviewsPath,
                                    ironic=False))

    def saveCorpus(self, path=None, filename=None):
        """Save a Corpus object to disk."""
        if path == None:
            path = self.corpusPath
        if filename == None:
            # Delete file extension
            filename = sub(r"\.[^\.]+$", "", self.IDsFilename)

        with open(path + filename + ".pk", 'wb') as dataFile:
            pickle.dump(self, dataFile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadCorpus(path=CORPUS_PATH, filename="training_set.pk"):
        """Load a Corpus object from disk."""
        with open(path + filename, 'rb') as dataFile:
            return pickle.load(dataFile)

    @property
    def ironicReviews(self):
        """Returns all ironic reviews."""
        return {ID: review for ID, review in  self.reviews.items()
                            if ID in self.ironicReviewIDs}

    @property
    def regularReviews(self):
        """Returns all regular reviews."""
        return {ID: review for ID, review in  self.reviews.items()
                            if ID in self.regularReviewIDs}

    @property
    def reviewIDs(self):
        """Returns the IDs for all reviews."""
        #return self.reviews.keys()
        return self.ironicReviewIDs + self.regularReviewIDs


    @property
    def ironicReviewIDs(self):
        """Returns the IDs for all ironic reviews."""
        return self._ironicReviewIDs + [i for i,r in self._pairReviewIDs]

    @property
    def regularReviewIDs(self):
        """Returns the IDs for all regular reviews."""
        return self._regularReviewIDs + [r for i,r in self._pairReviewIDs]

    @property
    def goldStandard(self):
        """Returns the IDs and the corresponding category (ironic/regular)."""
        return {ID: 1 if self.reviews[ID].ironic else 0 for ID in self.reviewIDs}

    def starDistribution(self):
        """Returns the star distribution of the corpus."""
        return calcStarDistribution(self.reviewIDs, self.reviews)

    def ironicStarDistribution(self):
        """Returns the star distribution for the ironic reviews."""
        return calcStarDistribution(self.ironicReviewIDs, self.reviews)

    def regularStarDistribution(self):
        """Returns the star distribution for the ironic reviews."""
        return calcStarDistribution(self.regularReviewIDs, self.reviews)

    def mostCommonWords(self, reviews=None, amount=100):
        """Returns the most common words of the corpus."""
        # TODO: Use nltk FreqDist instead!
        if reviews == None:
            reviews = self.reviews

        return Counter([word.text.lower() for ID, review in reviews.items()
                            for word in review.words]).most_common(amount)

    def wordDistribution(self, reviews=None, amount=100):
        """
        Returns a list with the ratio of the most common words to all words.
        """
        if reviews == None:
            reviews = self.reviews

        wordSum = sum([len(review.words) for ID, review in reviews.items()])
        return [(word, amount/float(wordSum))
                for word, amount in self.mostCommonWords(reviews=reviews,
                                                        amount=amount)]

    def polarityDistribution(self, reviews):
        """Show how the polarity is distributed across the corpus."""

        for each in reviews.values():
            print("Positive:", len(each.positiveWords),
                "\tNegative:", len(each.negativeWords), "\t",
                self.sentiment(each))

        positiveDistribution = Counter([len(r.positiveWords) for r in reviews.values()])
        negativeDistribution = Counter([len(r.negativeWords) for r in reviews.values()])

        print("Positive Distribution:")
        for positiveCount, occurence in sorted(positiveDistribution.items(), key=lambda i: i[0]):
            print(positiveCount, ":", occurence)

        print("Negative Distribution:")
        for negativeCount, occurence in sorted(negativeDistribution.items(), key=lambda i: i[0]):
            print(negativeCount, ":", occurence)


        polarity = Counter([len(r.positiveWords)-len(r.negativeWords) for r in reviews.values()])
        print("Polarity Distribution:")
        for negativeCount, occurence in sorted(polarity.items(), key=lambda i: i[0]):
            print(negativeCount, ":", occurence)

    def sentiment(self, review):
        """Returns a vector for the sentiment of a given review."""
        result = [0] * 7
        polarity = len(review.positiveWords) - len(review.negativeWords)
        if polarity < -10:
            result[0] = 1
        elif polarity < -5:
            result[1] = 1
        elif polarity < 0:
            result[2] = 1
        elif polarity == 0:
            result[3] = 1
        elif polarity > 10:
            result[6] = 1
        elif polarity > 5:
            result[5] = 1
        elif polarity > 0:
            result[4] = 1
        return result

    def printStarDistribution(self):
        """Prints the star distribution of ironic and regular reviews."""
        ironicDist = self.ironicStarDistribution()
        regularDist = self.regularStarDistribution()

        ratings = sorted(set([*ironicDist.keys(), *regularDist.keys()]),
                         reverse=True)

        print(u"\t      Ironic: Regular:")
        for rating in ratings:
            ironicCount = ironicDist[rating] if rating in ironicDist else 0
            regularCount = regularDist[rating] if rating in regularDist else 0
            print(u"{0}{1}:\t{2}\t{3}\treviews".format(u"\u2605 "*int(rating),
                                                u"\u2606 "*int(5-rating),
                                                ironicCount, regularCount))

    def printWordDistribution(self, amount=100):
        """Prints the word distribution of the ironic and regular reviews."""
        ironicDist = self.wordDistribution(reviews=self.ironicReviews,
                                            amount=amount)
        regularDist = self.wordDistribution(reviews=self.regularReviews,
                                            amount=amount)
        print("The {amount} most common words:".format(amount=amount))
        print("Ironic:\t\t\t\t Regular:")
        print("word\t\tratio\t\t word\t\tratio")
        for i in range(amount):
            ironic = "{0}\t\t{1:.5f}".format(ironicDist[i][0], ironicDist[i][1])
            regular = "{0}\t\t{1:.5f}".format(regularDist[i][0], regularDist[i][1])
            print(ironic, "\t", regular)


    def printPolarityDistribution(self):
        print("Polarity Distribution (all):")
        self.polarityDistribution(self.reviews)

        print("Polarity Distribution (irony):")
        self.polarityDistribution(self.ironicReviews)

        print("Polarity Distribution (regular):")
        self.polarityDistribution(self.regularReviews)


    def statistics(self):
        """Shows statistics about the corpus."""
        numberOfReviews = len(self.reviews)
        numberOfIronicReviews = len(self.ironicReviews)
        numberOfRegularReviews = len(self.regularReviews)
        print(("The corpus contains\n{ironicTotal} ironic reviews and\n"
                "{regularTotal} regular reviews, so\n"
                "{total} reviews in total.\n").format(
                    ironicTotal=numberOfIronicReviews,
                    regularTotal=numberOfRegularReviews,
                    total=numberOfReviews))

        self.printStarDistribution()

        wordCountsAll = [len(v.words) for k,v in self.reviews.items()]
        wordCountsIronic = [len(v.words) for k,v in self.ironicReviews.items()]
        wordCountsRegular = [len(v.words) for k,v in self.regularReviews.items()]

        averageWordCountAll = sum(wordCountsAll) / float(len(wordCountsAll))
        averageWordCountIronic = sum(wordCountsIronic) / float(len(wordCountsIronic))
        averageWordCountRegular = sum(wordCountsRegular) / float(len(wordCountsRegular))

        print("\nWords on average:",
                "\nIronic reviews:\t", averageWordCountIronic,
                "\nRegular reviews:",  averageWordCountRegular,
                "\nAll reviews:\t",    averageWordCountAll)

        # Distinct words
        dictionaryAll = {w.text.lower() for r in self.reviews.values() for w in r.words}
        dictionaryIronic = {w.text.lower() for r in self.ironicReviews.values() for w in r.words}
        dictionaryRegular = {w.text.lower() for r in self.regularReviews.values() for w in r.words}

        print("\nNumber of distinct words:",
            "\nIronic reviews:\t", len(dictionaryIronic),
            "\nRegular reviews:",  len(dictionaryRegular),
            "\nAll reviews:\t",    len(dictionaryAll))

        # Distinct words
        inBothDicts = dictionaryIronic & dictionaryRegular
        justInDictIronic = dictionaryIronic - dictionaryRegular
        justInDictRegular = dictionaryRegular - dictionaryIronic

        print("\n{both} words occur in both ironic and regular reviews".format(
                both=len(inBothDicts)),
            "\n{ironic} words occur only in ironic reviews and".format(
                ironic=len(justInDictIronic)),
            "\n{regular} words occur only in regular reviews.\n".format(
                regular=len(justInDictRegular)))

        # 100 most common words for ironic and regular reviews
        self.printWordDistribution()

        # self.printPolarityDistribution()

# ---- Basic corpus reading functions ----
def readIDs(filename, fileEncoding='utf-8-sig'):
    """Returns lists of IDs for pairs, ironic and regular reviews from the
    given file.
    """
    reviewPairIDs = []
    reviewIronicIDs = []
    reviewRegularIDs = []

    # 'utf-8-sig' removes leading Byte Order Mark
    with codecs.open(filename, 'r', encoding=fileEncoding) as idsFile:
        rawReviewIDs = idsFile.readlines()

    for rawID in rawReviewIDs:
        IDParts = rawID.split()
        if IDParts[0] == "PAIR:":
            reviewPairIDs.append((IDParts[1], IDParts[3]))
        elif IDParts[0] == "IRONIC:":
            reviewIronicIDs.append(IDParts[1])
        elif IDParts[0] == "REGULAR:":
            reviewRegularIDs.append(IDParts[1])

    return reviewPairIDs, reviewIronicIDs, reviewRegularIDs

def readReviews(reviewIDs, folder, ironic):
    """Returns a dictionary containing reviews to the given IDs."""
    return {reviewID: Review(filename="{0}{1}.txt".format(folder, reviewID),
                            ironic=ironic) for reviewID in reviewIDs}

def readIronicUtterances(reviews, filename, fileEncoding='utf-8-sig'):
    """Reads the ironic utterances for the given reviews."""
    # TODO: Expand and incorporate the ironic utterances in the reviews.
    with codecs.open(filename, 'r', encoding=fileEncoding) as utterancesFile:
        count = 0
        for line in utterancesFile.readlines():
            ID, utterance = line.partition("\t")[::2]
            # Remove trailing newline character.
            #if utterance.endswith("\n"):
            #    utterance = utterance[:-1]
            # Remove leading and trailing double quotes and replace
            # double double (sic) quotes within.
            #utterance = utterance[1:-1].replace('""', '"')

            utterance = utterance.strip('\n\s')#.replace('""', '"')
            reviews[ID].ironicUtterance = utterance


            if utterance.strip() in reviews[ID].text:
                count += 1
            else:
                print("Text: ", reviews[ID])
                print("+"*60)
                print(utterance)
                print("\n\n")

        print("Found", count, "utterances in their reviews!")

def readCorpus(path=CORPUS_PATH, filename=REVIEW_IDS_FILENAME):
    """Returns the reviews and lists of IDs for regular, ironic and pairs
    from the given files."""
    reviewPairIDs, reviewIronicIDs, reviewRegularIDs = readIDs(path + filename)

    reviews = {}
    reviews.update(readReviews([i for i,r in reviewPairIDs],
                    CORPUS_PATH + IRONIC_REVIEWS_PATH, ironic=True))
    reviews.update(readReviews([r for i,r in reviewPairIDs],
                    CORPUS_PATH + REGULAR_REVIEWS_PATH, ironic=False))
    reviews.update(readReviews(reviewIronicIDs,
                    CORPUS_PATH + IRONIC_REVIEWS_PATH, ironic=True))
    reviews.update(readReviews(reviewRegularIDs,
                    CORPUS_PATH + REGULAR_REVIEWS_PATH, ironic=False))
    return reviewPairIDs, reviewIronicIDs, reviewRegularIDs, reviews

def clacGoldStandard(IDs, reviews):
    """Returns the gold standard of the given IDs and reviews."""
    return {ID: reviews[ID].ironic for ID in IDs}


# ----- Basic statistical functions -----
def calcWordCounts(IDs, reviews):
    """Returns a counter of the number of words."""
    return Counter([reviews[ID].numberOfWords() for ID in IDs])

def calcMeanWordCounts(IDs, wordCounts):
    """Returns the mean number of words of the review for the given IDs."""
    total = sum([length*count
                    for length, count in wordCounts.items()])
    return total/float(len(IDs))

def calcWordDistribution(IDs, reviews):
    """Returns a counter of the number of occurrences of the words."""
    return Counter([str(word).lower()
                    for ID in IDs
                    for word in reviews[ID].words])

def calcWordDistributionRatio(IDs, reviews):
    """Returns a counter of the ratio of the words to all words."""
    wordSum = sum([length*count
                    for length, count in calcWordCounts(IDs, reviews).items()])
    return Counter({word: count/float(wordSum)
                    for word, count
                    in calcWordDistribution(IDs, reviews).items()})

def calcStarDistribution(IDs, reviews):
    """
    Returns a dictionary of the number of reviews with a certain star rating.
    """
    return dict(Counter([reviews[ID].stars for ID in IDs]))


def showStatistics(IDLists, reviews, labels=[], plot=False):
    """Prints some statistics about the reviews of the given IDs."""
    if not any(isinstance(IDs, list) for IDs in IDLists):
        IDLists = [IDLists]

    for index, IDs in enumerate(IDLists):
        wordCounts = calcWordCounts(IDs, reviews)
        meanWordCounts = calcMeanWordCounts(IDs, wordCounts)

        if not len(labels) == len(IDLists):
            labels.append("{0}. list".format(str(index + 1)))

        print("{0} contains {1} reviews".format(labels[index], len(IDs)))
        print("with {0} words on average.".format(meanWordCounts))

        starDist = sorted(calcStarDistribution(IDs, reviews).items(),
                          key=lambda stars_amount: stars_amount[0],
                          reverse=True)
        for rating, count in starDist:
            print(u"{0}{1}: {2} reviews".format(u"\u2605 "*int(float(rating)),
                                                u"\u2606 "*int(5-float(rating)),
                                                count))

        print("")

        # if plot:
        #     barwidth = 0.8
        #     plt.bar([float(rating) for rating in starDist.keys()], starDist.values(), width=barwidth, bottom=None, hold=None)
        #     plt.ylabel('Star Rating')
        #     plt.title('Star rating distribution of reviews from {0}'.format(labels[index]))
        #     plt.xticks([v+barwidth/2. for v in range(1,6)], (u'\u2605', u'\u2605'*2, u'\u2605'*3, u'\u2605'*4, u'\u2605'*5))
        #     plt.yticks(range(0,300,30))
        #     plt.show()

    amount = 100
    wordRatios = [calcWordDistributionRatio(IDs, reviews).most_common(amount)
                    for IDs in IDLists]
    print("The", amount, "most common words are:")
    print("\t\t\t\t".join(labels))
    for i in range(0, 100):
        print("\t\t".join(["{0}: {1}".format(ratio[i][0], ratio[i][1])
                            for ratio in wordRatios]))


def testCorpus():
    """Test, if the Corpus class works properly."""
    pass

def main():
    pass

if __name__ == '__main__':
    main()
