# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import random

from corpus import readIDs
from defaultConfig import RANDOM_SEED
from defaultConfig import TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE
from defaultConfig import WORKING_COPY_PATH, SET_FILENAMES
from defaultConfig import CORPUS_PATH, REVIEW_IDS_FILENAME


def divideData(data, setSizes):
    """
    Returns a list of list from the data accordingly to the given percentages.
    """
    assert not sum(setSizes) > 100
    random.seed(RANDOM_SEED)
    random.shuffle(data)

    result = []
    numberOfSets = len(setSizes)
    offset = 0
    for setNumber in range(numberOfSets):
        if setNumber == numberOfSets-1:
            result.append(data[offset:])
        else:
            end = offset + len(data)*setSizes[setNumber]/100
            result.append(data[offset:end])
            offset = end
    return result

def setToString(IDLabelSet):
    """Returns a list of string representations for every ID, label pair."""
    return ["{0}:\t{1}".format(label.upper(), ID) for ID, label in IDLabelSet]

def createIDLabelSet(data, label):
    """Returns a list of ID, label pairs."""
    return [(ID, label) for ID in data]

def read(filename=CORPUS_PATH + REVIEW_IDS_FILENAME, encoding='utf-8-sig'):
    """Returns a list of IDs from the given file."""
    with codecs.open(filename, 'r', encoding=encoding) as idsFile:
        return [ID.strip() for ID in idsFile.readlines()]

def saveSet(set, filename):
    """Saves the given set in a file."""
    with codecs.open(filename, 'w', encoding='utf-8') as setFile:
        setFile.writelines("\n".join(set))

def createSets(setSizes=[TRAINING_SET_SIZE, TEST_SET_SIZE, VALIDATION_SET_SIZE]):
    """Reads IDs, creates and saves randomly shuffled subsets of these."""
    reviewPairIDs, reviewIronicIDs, reviewRegularIDs = readIDs(CORPUS_PATH
                                                        + REVIEW_IDS_FILENAME)

    reviewIDs = setToString(createIDLabelSet(reviewIronicIDs, "ironic")) 
    reviewIDs += setToString(createIDLabelSet(reviewRegularIDs, "regular")) 
    reviewIDs += setToString(createIDLabelSet([r for i,r in reviewPairIDs], 
                                        "regular")) 
    reviewIDs += setToString(createIDLabelSet([i for i,r in reviewPairIDs], 
                                        "ironic"))

    sets = divideData(reviewIDs, setSizes)

    for i in range(len(sets)):
        saveSet(sets[i], CORPUS_PATH + SET_FILENAMES[i])


def shuffledSet(setFilename=REVIEW_IDS_FILENAME, path=CORPUS_PATH, randomSeed=RANDOM_SEED):
    """Reads IDs and saves a shuffled version of it."""
    reviewPairIDs, reviewIronicIDs, reviewRegularIDs = readIDs(path + 
                                                                setFilename)

    reviewIDs = setToString(createIDLabelSet(reviewIronicIDs, "ironic")) 
    reviewIDs += setToString(createIDLabelSet(reviewRegularIDs, "regular")) 
    reviewIDs += setToString(createIDLabelSet([r for i,r in reviewPairIDs], 
                                        "regular")) 
    reviewIDs += setToString(createIDLabelSet([i for i,r in reviewPairIDs], 
                                        "ironic"))

    random.seed(randomSeed)
    random.shuffle(reviewIDs)

    # for each in reviewIDs[1:20]:
    #     print(each)    

    saveSet(reviewIDs, path + "shuffled_set.txt")

def main():
    """Statistics about the corpus:
    Paired reviews:      331
    Ironic reviews:      437    (106)
    Regular reviews:     817    (486)
    All reviews:        1254
    """
    # createSets()
    shuffledSet(randomSeed=44)

if __name__ == '__main__':
    main()