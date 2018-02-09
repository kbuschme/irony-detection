# -*- coding: utf-8 -*-
from __future__ import print_function

from numpy import mean, std
from math import sqrt

# TODO:
# * Modify to work with python 3
#   * calcPerformance broken in python 3 - See list comprehension with zip.


# ---- Functions to calculate performances ----
def calcPerformance(gold, prediction):
    """
    Calculates performance of the prediction.
    Gold and prediction can be lists or lists of lists.
    """
    assert(len(gold) == len(prediction))
    # If gold and prediction are lists of lists.
    if (all(isinstance(l, list) for l in gold) and
            all(isinstance(l, list) for l in prediction)):
        return [calcPerformance(g, p)
                for g, p in zip(gold, prediction)]
    else:
        result = zip(gold, prediction)

        tp = sum([1 if g and p else 0
                    for g, p in result])
        tn = sum([1 if not g and not p else 0
                    for g, p in result])
        fp = sum([1 if not g and p else 0
                    for g, p in result])
        fn = sum([1 if g and not p else 0
                    for g, p in result])

        precision = float(tp)/(tp + fp) if not (tp + fp) == 0 else 1.0
        recall = float(tp)/(tp + fn) if not (tp + fn) == 0 else 1.0
        accuracy = float(tp + tn)/float(tp + fp + fn + tn)
        # if (not (precision == None and recall == None) and
        #         not (precision == 0 and recall == 0)):
        if precision is None or recall is None:
            fScore = None
        elif precision == 0 and recall == 0:
            fScore = 0.0
        else:
            fScore = 2*precision*recall/(precision + recall)

        return {"truePositive": tp, "trueNegative": tn, "falsePositive": fp,
                "falseNegative": fn, "precision": precision, "recall": recall,
                "accuracy": accuracy, "F-score": fScore}

def calcMeanPerformance(goldStandards, prediction):
    """Calculates the mean performances."""
    assert(len(goldStandards) == len(prediction))

    measures = {"truePositive", "trueNegative", "falsePositive",
                "falseNegative", "precision", "recall","accuracy", "F-score"}

    performances = calcPerformance(goldStandards, prediction)

    for number, each in enumerate(performances):
       print("{number}. fold:\n".format(number=number), each)

    report = {}
    for m in measures:
        total = sum([p[m] if p[m] is not None else 0 for p in performances])
        count = sum([1 if p[m] is not None else 0 for p in performances])
        if not count == 0:
            report[m] = total/float(count)
        else:
            report[m] = 0

    return report


# ---- Functions to show performances ----
def showPerformance(gold, classification, label=["ironic", "regular"]):
    """Shows the classification's performance."""
    performance = calcPerformance(gold, classification)
    print("# Predictions:", len(classification))
    showEvaluation(performance)
    showScores(performance)

def showMeanPerformance(golds, classifications, label=["ironic", "regular"]):
    """Shows the classification's mean performance."""
    print("Below shows means:")
    performances = calcPerformance(golds, classifications)
    showEvaluation(calcMeanPerformance(golds, classifications))
    showMeanScores(performances)

def showEvaluation(performance, label=["ironic", "regular"]):
    """Shows a the result of the classification."""
    # print("True positive:\t{0}".format(performance["truePositive"]))
    # print("True negative:\t{0}".format(performance["trueNegative"]))
    # print("False positive:\t{0}".format(performance["falsePositive"]))
    # print("False negative:\t{0}".format(performance["falseNegative"]))

    # Table form of prediction/gold
    print("\t\t\tGold: {positive}\tGold: {negative}".format(positive=label[0],
                                                            negative=label[1]))
    print("Predict: {positive}\t\t{tp} (tp)\t\t{fp} (fp)".format(
                                            positive=label[0],
                                            tp=performance["truePositive"],
                                            fp=performance["falsePositive"]))
    print("Predict: {negative}\t{fn} (fn)\t\t{tn} (tn)".format(
                                            negative=label[1],
                                            fn=performance["falseNegative"],
                                            tn=performance["trueNegative"]))

def showMeanEvaluation(performances, label=["ironic", "regular"]):
    """Shows a the result of the classification."""
    # TODO: Add range (x to y) or delete function

    meanPerformance = calcMeanPerformance(performances)
    showEvaluation(meanPerformances, label)
    # # Table form of prediction/gold
    # print("\t\t\tGold: {positive}\tGold: {negative}".format(positive=label[0],
    #                                                         negative=label[1]))
    # print("Predict: {positive}\t\t{tp} (tp)\t\t{fp} (fp)".format(
    #                                     positive=label[0],
    #                                     tp=meanPerformance["truePositive"],
    #                                     fp=meanPerformance["falsePositive"]))
    # print("Predict: {negative}\t{fn} (fn)\t\t{tn} (tn)".format(
    #                                     negative=label[1],
    #                                     fn=meanPerformance["falseNegative"],
    #                                     tn=meanPerformance["trueNegative"]))

def showScores(performance):
    """Shows performance measurements precision, recall, accuracy, f-Score."""
    print("Precision:\t{0}".format(performance["precision"]))
    print("Recall:\t\t{0}".format(performance["recall"]))
    print("Accuracy:\t{0}".format(performance["accuracy"]))
    print("F-Score:\t{0}".format(performance["F-score"]))

def showMeanScores(performances):
    """
    Shows mean performance measurements precision, recall, accuracy, f-Score.
    """
    precisions = [p["precision"] for p in performances]
    recalls = [p["recall"] for p in performances]
    accuracys = [p["accuracy"] for p in performances]
    fScores = [p["F-score"] for p in performances]

    cleanPrecisions = [p["precision"] for p in performances if p["precision"] is not None]
    cleanRecalls = [p["recall"] for p in performances if p["recall"] is not None]
    cleanAccuracys = [p["accuracy"] for p in performances if p["accuracy"] is not None]
    cleanFScores = [p["F-score"] for p in performances if p["F-score"] is not None]


    print("Precision\t{0}\tstd: {1}".format(mean(cleanPrecisions), std(cleanPrecisions)),
        "\t({0} to {1})".format(min(precisions), max(precisions)))
    print("Recall:\t\t{0}\tstd: {1}".format(mean(cleanRecalls), std(cleanRecalls)),
        "\t({0} to {1})".format(min(recalls), max(recalls)))
    print("Accuracy\t{0}\tstd: {1}".format(mean(accuracys), std(accuracys)),
        "\t({0} to {1})".format(min(cleanAccuracys), max(cleanAccuracys)))
    print("F-Score:\t{0}\tstd: {1}".format(mean(cleanFScores), std(cleanFScores)),
        "\t({0} to {1})".format(min(fScores), max(fScores)))


def test():
    """Test basic clac and show functions."""
    gold1 = [1, 0, 1, 0, 1, 1]
    clas1 = [1, 0, 0, 1, 1, 0]
    gold2 = [1, 0, 1, 0, 1, 1]
    clas2 = [1, 0, 0, 1, 0, 0]
    golds = [gold1, gold2]
    clas = [clas1, clas2]
    for b in calcPerformance(golds, clas):
        print(b)
    showPerformance(gold1, clas1)
    showPerformance(gold2, clas2)
    showMeanPerformance(golds, clas)

def test2():
    gold1 = [1, 0, 1, 0, 1, 1]
    clas1 = [0, 1, 0, 1, 0, 0]
    golds = [gold1]
    clas = [clas1]
    #for b in calcPerformance(golds, clas):
    #    print(b)
    showPerformance(gold1, clas1)
    showMeanPerformance(golds, clas)

if __name__ == '__main__':
    test2()
