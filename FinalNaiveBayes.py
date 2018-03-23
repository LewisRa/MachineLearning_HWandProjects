# prog.py --- 
# Rachel Lewis t
# Filename: prog.py


# Code:
import math
import matplotlib.pyplot as plt


def loadDataByIndex(filename):

    out = {}
    with open(filename, 'r') as f:
        i = 1
        line = True
        while(line):
            line = f.readline()
            if(line == '\n' or line == ''):
                continue
            out[i] = line[:-1]
            i += 1

    return out


def loadLabels(filename, labelByIndex):

    out = {}
    with open(filename, 'r') as f:
        i = 1
        line = True
        while(line):
            line = f.readline()
            if(line == '\n' or line == ''):
                continue
            out[i] = labelByIndex[int(line[:-1])]
            i += 1

    return out


def loadData(filename, vocabByIndex):

    out = {}
    with open(filename, 'r') as f:
        i = 1
        line = True
        while(line):
            line = f.readline()
            if(line == '\n' or line == ''):
                continue
            items = [int(j) for j in line[:-1].split()]

            if(items[0] not in out):
                out[items[0]] = {vocabByIndex[items[1]]: items[2]}
            else:
                out[items[0]][vocabByIndex[items[1]]] = items[2]
            i += 1

    return out


def getValuedVocabDict(vocabByIndex, value):
    out = {}
    for key in vocabByIndex:
        out[vocabByIndex[key]] = value
    return out

def getFuncdVocabDict(vocabByIndex, fun):
    out = {}
    for key in vocabByIndex:
        out[vocabByIndex[key]] = fun()
    return out


def computeProbability(tokenCounts, word, alpha, denominator):
    return ((tokenCounts[word] + alpha) * 1.0)/denominator



def getDenominator(tokenCounts, word, alpha):
    denominator = 0
    for token in tokenCounts:
        denominator += tokenCounts[token] + alpha
    return denominator


def getValuedLabelDict(labelByIndex, value):
    out = {}
    for key in labelByIndex:
        out[labelByIndex[key]] = value
    return out


def printConfMatrix(predictions, testLabels, labelByIndex):
    mat = {}
    for key in labelByIndex:
        mat[labelByIndex[key]] = {}
        for inkey in labelByIndex:
            mat[labelByIndex[key]][labelByIndex[inkey]] = 0

    for did in testLabels:
        mat[testLabels[did]][trainLabels[did]] += 1

    out = ''
    for label in labelByIndex.values():
        out += "%s"%label+'\t'
    print ("True\Pred\t"+out)

    for label in labelByIndex.values():
        out = "%30s"%label+'\t'
        for lab in mat[label]:
            out += "%d\t" % (mat[label][lab])
        print (out)


labelByIndex = loadDataByIndex("newsgrouplabels.txt")
vocabByIndex = loadDataByIndex("vocabulary.txt")
testLabels = loadLabels("test.label", labelByIndex)
trainLabels = loadLabels("train.label", labelByIndex)
testData = loadData("test.data", vocabByIndex)
trainData = loadData("train.data", vocabByIndex)

emptyVocabDict = getValuedVocabDict(vocabByIndex, 0)

Py = {}
tokensInLabel = {}
denominator = {}

for docId in trainLabels:
    if(trainLabels[docId] not in Py):
        Py[trainLabels[docId]] = 1
        tokensInLabel[trainLabels[docId]] = emptyVocabDict.copy()
        denominator[trainLabels[docId]] = 0

        for word in trainData[docId]:
            tokensInLabel[trainLabels[docId]][word] += trainData[docId][word]
            denominator[trainLabels[docId]] += trainData[docId][word]
    else:
        Py[trainLabels[docId]] += 1
        for word in trainData[docId]:
            tokensInLabel[trainLabels[docId]][word] += trainData[docId][word]
            denominator[trainLabels[docId]] += trainData[docId][word]

for label in Py:
    Py[label] /= (len(trainLabels) * 1.0)


for alpha in [1./len(vocabByIndex)]:

    denominatorMod = {}
    tokenGivenLabel = getFuncdVocabDict(vocabByIndex, dict)

    for label in denominator:
        denominatorMod[label] = denominator[label] + (alpha * len(tokenGivenLabel))

    for word in tokenGivenLabel:
        for label in Py:
            tokenGivenLabel[word][label] = computeProbability(tokensInLabel[label],
                                                              word, alpha,
                                                              denominatorMod[label])

    predictions = {}

    for docId in testData:
        scores = getValuedLabelDict(labelByIndex, 0)

        for label in scores:
            scores[label] = math.log(Py[label])

        den = sum(testData[docId].values())
        for label in scores:
            for word in testData[docId]:

                temp = testData[docId][word]
                scores[label] += math.log(tokenGivenLabel[word][label]) * temp

        predictions[docId] = max(scores, key=scores.get)

    num = 0.0
    for docId in predictions:
        #print (predictions[docId], testLabels[docId])
        if(predictions[docId] == testLabels[docId]):
            num += 1.0

    print ("Alpha: %f\tAccuracy: %f\n\n"%(alpha, num/len(predictions)))

    printConfMatrix(predictions, testLabels, labelByIndex)
    print ("\n\n")


testList =[1.00000000e-05, 1.83298071e-05, 3.35981829e-05, 6.15848211e-05,
       1.12883789e-04, 2.06913808e-04, 3.79269019e-04, 6.95192796e-04,
       1.27427499e-03, 2.33572147e-03, 4.28133240e-03, 7.84759970e-03,
       1.43844989e-02, 2.63665090e-02, 4.83293024e-02, 8.85866790e-02,
       1.62377674e-01, 2.97635144e-01, 5.45559478e-01, 1.00000000e+00]

alphas = []
accuracy = []

for alpha in testList:

    denominatorMod = {}
    tokenGivenLabel = getFuncdVocabDict(vocabByIndex, dict)

    for label in denominator:
        denominatorMod[label] = denominator[label] + (alpha * len(tokenGivenLabel))

    for word in tokenGivenLabel:
        for label in Py:
            tokenGivenLabel[word][label] = computeProbability(tokensInLabel[label],
                                                              word, alpha,
                                                              denominatorMod[label])

    predictions = {}

    for docId in testData:
        scores = getValuedLabelDict(labelByIndex, 0)

        for label in scores:
            scores[label] = math.log(Py[label])

        den = sum(testData[docId].values())
        for label in scores:
            for word in testData[docId]:

                temp = testData[docId][word]
                scores[label] += math.log(tokenGivenLabel[word][label]) * temp

        predictions[docId] = max(scores, key=scores.get)

    num = 0.0
    for docId in predictions:
        #print (predictions[docId], testLabels[docId])
        if(predictions[docId] == testLabels[docId]):
            num += 1.0

    print ("Alpha: %f\tAccuracy: %f"%(alpha, num/len(predictions)))

    alphas.append(alpha)
    accuracy.append(num/len(predictions))

plt.plot(alphas, accuracy)
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("Accuracy of Naive Bayes")
plt.title("Accuracy vs Alpha for Multinomial Naive Bayes")
plt.show()


"""fp = open ('20newsgroupWorking.txt', "w")
for i in range (num):
    fp.write (" % s% s \ n "% (train_label [i], "  " .join (train_data [i])))
fp.close ()"""
