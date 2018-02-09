# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs
from re import finditer, findall, escape
from datetime import datetime
from HTMLParser import HTMLParser   # for Python 3 use html.parser

import nltk.data
from nltk.tokenize import TreebankWordTokenizer

from defaultConfig import NEGATIVE_WORDS_FILENAME, POSITIVE_WORDS_FILENAME

def loadPolarityLexicon(filenames, categories):
    """
    Returns a dictionary containing the words from the given files
    and their corresponding category.
    """
    assert len(filenames) == len(categories)
    polarityLexicon = {}
    for filename, category in zip(filenames, categories):
        with codecs.open(filename, 'r', encoding='latin-1') as wordsFile:
            polarityLexicon.update({w.strip(): category
                                for w in wordsFile.readlines()
                                if w.strip() and not w.strip().startswith(";")})
    return polarityLexicon

# TODO: Is this still in use?
def loadPolarityWords(filename):
    """Returns a list containing the words from the given file."""
    with codecs.open(filename, 'r', encoding='latin-1') as wordsFile:
        return [w.strip()
                for w in wordsFile.readlines()
                if w.strip() and not w.strip().startswith(";")]


# ---- Review, Sentence and Token classes ----
class Review(object):
    """Represents a review and its meta data."""

    HTMLParser = HTMLParser()
    sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    wordTokenizer = TreebankWordTokenizer()

    def __init__(self, rawReview=None, filename=None, ironic=None):
        if filename is not None:
            self.parseFromFile(filename, ironic)
        elif rawReview is not None:
            self.parse(rawReview, ironic)
        else:
            self.product = ""
            self.title = ""
            self.text = ""
            self.stars = 0.0
            self.author = ""
            self.date = Datetime()
            self.ironic = None
            self.wordSpans = []
            self.wordPolarity = []
            self.sentenceSpans = []

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        printout = u"{0}, {1} by {2} ({3})\n{4}{5} {6}\n{7}\n{8}\n"
        return printout.format(self.title, self.date.strftime("%B %d, %Y"),
                                self.author,
                                u"ironic" if self.ironic else u"regular",
                                u"\u2605 "*int(float(self.stars)),
                                u"\u2606 "*int(5-float(self.stars)),
                                self.product,
                                "-"*60, self.text)

    def parseFromFile(self, filename, ironic, fileEncoding='latin-1'):
        with codecs.open(filename, 'r', encoding=fileEncoding) as reviewFile:
            self.parse(reviewFile.read(), ironic)

    def parse(self, rawReview, ironic):
        self.ironic = ironic
        # Parse "XML"-Style.
        self.stars = float(findBetween(rawReview, "<STARS>", "</STARS>"))
        self.author = findBetween(rawReview, "<AUTHOR>", "</AUTHOR>")
        self.product = findBetween(rawReview, "<PRODUCT>", "</PRODUCT>")
        # Date is inconsistent and has possibly one of the three formats.
        try:
            self.date = datetime.strptime(
                                findBetween(rawReview, "<DATE>", "</DATE>"),
                                "%B %d, %Y")
        except ValueError:
            try: self.date = datetime.strptime(
                                findBetween(rawReview, "<DATE>", "</DATE>"),
                                "%d %B %Y")
            except ValueError:
                self.date = datetime.strptime(
                                findBetween(rawReview, "<DATE>", "</DATE>"),
                                "%d %b %Y")
        self.title = findBetween(rawReview, "<TITLE>", "</TITLE>")

        # Is it a good practice to strip the review text?
        self.text = findBetween(rawReview, "<REVIEW>", "</REVIEW>").strip()
        self.text = self.preprocess(self.text)

        self.sentences = self.tokenizeSentences(self.text)

    @property
    def words(self):
        return [w for s in self.sentences for w in s.words]

    @property
    def bigrams(self):
        words = self.words
        return zip(words[:-1], words[1:])

    @property
    def positiveWords(self):
        return [p for s in self.sentences for p in s.positiveWords]

    @property
    def negativeWords(self):
        return [n for s in self.sentences for n in s.negativeWords]

    @property
    def polarity(self):
        """Returns the review's polarity."""
        if len(self.negativeWords) > len(self.positiveWords):
            return "positive"
        elif len(self.negativeWords) > len(self.positiveWords):
            return "negative"
        else:
            return "neutral"

    def preprocess(self, text):
        """Preprocesses the given text by unescaping HTML entities."""
        return self.HTMLParser.unescape(text)

    def tokenizeSentences(self, text):
        return [Sentence(text)
                for text in self.sentenceTokenizer.tokenize(self.text,
                                                    realign_boundaries=True)]

    def numberOfWords(self):
        return len(self.words)

    def numberOfSentences(self):
        return len(self.sentences)

    # TODO: delete?
    def analysePolarity(self, words):
        self.positivePolarity = []
        self.negativePolarity = []
        for word in words:
            self.positivePolarity += [p for p in self.positiveWords
                                        if p == word]
            self.negativePolarity += [n for n in self.negativeWords
                                        if n == word]

    def showDetails(self):
        """Show a detailed review representation."""
        print(this)
        print("Sentences:\n", "\n".join(
                ["Sentence({s})".format(s=str(s)) for s in self.sentences]))
        # print("\nWords:\n", "--".join([str(w) for w in self.words]))
        print("\nTokens: {tokens}".format(tokens=self.words))
        print("\nPos words:", self.positiveWords)
        print("Neg words:", self.negativeWords)
        # print("Pos word positions:", [s.positiveWordPositions
        #                                 for s in self.sentences])
        # print("Neg word positions:", [s.negativeWordPositions
        #                                 for s in self.sentences])

class Sentence(object):
    """Represents a single sentence."""
    POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    wordTagger = nltk.data.load(POS_TAGGER)
    wordTokenizer = TreebankWordTokenizer()
    __slots__ = ['text', 'words']

    def __init__(self, text):
        start = None
        end = None
        self.text = text
        self.words = self.tokenizeWords(self.text)

    def __repr__(self):
        return "Sentence({0})".format(self.text).encode('utf-8')

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.text

    def tokenizeWords(self, text):
        return [Token(w,p)
               for w,p in self.wordTagger.tag(self.wordTokenizer.tokenize(text))]

    @property
    def positiveWords(self):
        return [w.text for w in self.words if not w.positiveScore == 0]

    @property
    def positiveWordPositions(self):
        return [i for i, w in enumerate(self.words) if not w.positiveScore == 0]

    @property
    def negativeWords(self):
        return [w.text for w in self.words if not w.negativeScore == 0]

    @property
    def negativeWordPositions(self):
        return [i for i, w in enumerate(self.words) if not w.negativeScore == 0]

class Token(object):
    """Represents a single token."""

    polarityLexicon = loadPolarityLexicon([POSITIVE_WORDS_FILENAME,
                                        NEGATIVE_WORDS_FILENAME],
                                        ["positive", "negative"])

    __slots__ = ['text', 'pos', 'positiveScore', 'negativeScore']

    def __init__(self, text, pos=None):
        start = None
        end = None
        self.text = text
        self.pos = pos
        if (self.text in self.polarityLexicon and
                self.polarityLexicon[self.text] == "positive"):
            self.positiveScore = 1
        else:
            self.positiveScore = 0

        if (self.text in self.polarityLexicon and
                self.polarityLexicon[self.text] == "negative"):
            self.negativeScore = 1
        else:
            self.negativeScore = 0

    def __repr__(self):
        return "Token({0}, {1}, {2})".format(self.text,
                                        self.pos,
                                        self.polarity).encode('utf-8')

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.text

    def __eq__(self, other):
        return (self.text == other.text and self.pos == other.pos and
                self.positiveScore == other.positiveScore and
                self.negativeScore == other.negativeScore)

    def __hash__(self):
        return hash(self.__repr__())

    @property
    def polarity(self):
        if self.positiveScore > self.negativeScore:
            return "positive"
        if self.positiveScore < self.negativeScore:
            return "negative"
        else:
            return "neutral"

# ---- Auxiliary functions used by the review class ----
def findBetween(text, start, end):
    """Returns the text contained between the two given delimiters."""
    startPos = text.find(start)
    endPos = text.find(end)
    if not startPos == -1 and not endPos == -1:
        return text[startPos+len(start):endPos]
    else:
        return None

def findBetweenTag(tag, text):
    # TODO: Refine!
    import re
    """Returns the text contained between the two given delimiters."""
    p = re.compile('<REVIEW>(.*?)<\/REVIEW>', re.DOTALL)
    return p.findall(text)




# ---- Testing functions ----
def testToken():
    """Test, if the Token class works properly."""
    words = [("Test", "neutral"), ("Gibberish", "neutral"),
                ("jibber-jabber", "neutral"),
                ("correct", "positive"), ("nicely", "positive"),
                ("funny", "negative"), ("weird", "negative")]
    for word, targetPolarity in words:
        token = Token(word)
        if not token.text == word:
            testFailed(token.text, word, "Word")
        if not token.polarity == targetPolarity:
            testFailed(token.polarity, targetPolarity, "Polarity")

def testSentence():
    """Test, if the Sentence class works properly."""
    phrases = [("Mary had a little lamb.",
                    ["Mary", "had", "a", "litle", "lamb", "."]),
                ("Does \"this\" work?",
                    ["Does", "``", "this", "''", "work", "?"]),
    ]
    for phrase, target in phrases:
        sentence = Sentence(phrase)
        if not all(word1 == word2
                    for word1, word2 in zip([str(word)
                            for word in sentence.words], target)):
            testFailed([word for word in sentence.words], target, "Sentence")

def testReview():
    """Test, if the Token class works properly."""
    pass

def testFailed(found, expected, label=""):
    print(("{label} test failed:\tfound '{found}'\t "
                    " expected '{expected}'.").format(
                    label=label, found=found, expected=expected))

def main():
    testToken()
    testSentence()
    testReview()

if __name__ == '__main__':
    main()
