# this code is the automated round robin implementation of TextRank


from methods_main4 import DataSet , computeTermPDF , pageRankClass
import pandas as pd
import time
from collections import Counter , OrderedDict
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

start = time.time()

df_iter = pd.read_csv("testPermutation.txt")


all_precision = []
all_recall = []
all_fscore = []

for iter in range(0, df_iter.shape[0]):
#for iter in range(0, 1):
    print(iter)
    # path associate with target data
    #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
    path = "C:/userOne/AutomaticKeyphraseExtraction-master/data/"
    # initialse class with path pointer
    dataClass = DataSet(path)

    # pull out the text  - sectionsDict : text store in a dictionary partitioned by keys to sections
    # for the moment we have only pulled out section relating to references to extract references
    dataClass.extractText()

    #dataDict = dataClass.meta_dataset.files.apply(dataClass.extractSections)
    dataClass.dataset['rawDict'] = dataClass.meta_dataset.files.apply(dataClass.extractSections)
    #>>>>>>>>>>>>>>>>>>>>>>>>
    # we will look to first cleaning the dictionary heads
    # method loops over dictionary or sections and extracts refernces
    # returns as a key <ref num> value <string>
    dataClass.dataset['refs'] = dataClass.dataset.rawDict.apply(dataClass.methodRefsExtract)

    #>>>>>>>>>>>>>>>>>>>>>>
    dataClass.dataset['rawDict'] = dataClass.dataset.rawDict.apply(dataClass.cleanKeys)
    # drop references from text
    dataClass.dataset.rawDict.apply(dataClass.dropReferences)

    # clean the dict
    allTermArrayCount = dataClass.extractVocab(list(dataClass.dataset.rawDict))
    pdf = computeTermPDF(allTermArrayCount)
    pdf.calculateProbTerm()


    #tester = list(dataClass.dataset['refs'])

    # takes out unwanted terms in references
    dataClass.dataset['refs'] = dataClass.cleanRefs(dataClass.dataset.refs, pdf)

    # creates on string of the docs
    dataClass.dataset['stringDocs'] = dataClass.dataset.rawDict.apply(dataClass.concatDict)


    # assign the permutations
    permutations1 = df_iter.iloc[iter]
    #permutations1 = [False, True, True, True, True]

    #permutations1 = dataClass.convertToBool(permutations1)

    print(permutations1)
    ## event booleans
    dataClass.stopwordRemove = permutations1[0]
    fillRefferences = permutations1[1]
    fillAcc = permutations1[2]
    applyStemming = permutations1[3]
    setDeliminators = permutations1[4]

    # 1
    if fillAcc:
        print("applying acc")
        dataClass.dataset['accDict'] = dataClass.dataset['stringDocs'].apply(dataClass.extractAccronymnsFromText)
        dataClass.expandAccronymnsInText()

    # 2
    if fillRefferences:
        print("applying ref")
        dataClass.dataset['stringDocs'] = dataClass.ALL_fillOutReference(dataClass)

    # 3.  instance of deliminators takes an either or situation
    if setDeliminators:
        print("applying delim")
        #dataClass.dataset.stringDocs = dataClass.dataset.stringDocs.apply(dataClass.creatDeliminators)
        dataClass.dataset['processDocs']  = dataClass.dataset.stringDocs.apply(dataClass.splitCorpus)


    #print(dataClass.dataset.processDocs[0])


    if not setDeliminators:
        dataClass.dataset['processDocs']  = dataClass.dataset.stringDocs.apply(dataClass.wrapTextInArray)


    dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(dataClass.cleanSentences)




    #extracting the targetTerms
    dataClass.extractTargetTerms()

    # 4
    if applyStemming:
        print("applying stemming")
        #text =dataClass.stem_Doc(dataClass.dataset['processDocs'][0])
        dataClass.dataset['keyTerms'] = dataClass.dataset['keyTerms'].apply(dataClass.stem_array)



    allIndex = []
    precision = 0
    recall = 0
    fscore = 0
    for index in range(len(list(dataClass.dataset['processDocs']))):
    #for index in range(0, 1):

        print("at stage {}".format(index))

        testerDoc = dataClass.dataset['processDocs'][index]

        PR = pageRankClass(testerDoc)

        # as far as here it is good
        PR.constructGraph(testerDoc, stem = applyStemming)
        print("number of nodes : " + str(len(PR.graph.nodes())))


        PR.createPhrasese()

        #print(len(PR.textRankDict.items()))
        tempDict  = {}


        #print(PR.posCorp)
        docKeys = dataClass.dataset.keyTerms[index]
        indexLoc = dataClass.extractKeyOrderedrank(PR.textRankDict , docKeys)

        allIndex.append(indexLoc)

        y_pred = dict(list(PR.textRankDict.items())[:15])
        print(y_pred)
        y_true = docKeys

        precision_instance , recall_instance, fscore_instance = dataClass.calculateFscore( y_pred, y_true)
        precision += precision_instance
        recall += recall_instance
        fscore += fscore_instance
        # print(10*"*")
        # print(precision , recall, fscore)
        # print(10*"-")

    eval_sum = sum([1 for terms in dataClass.dataset.keyTerms if len(terms) > 0])
    print(" p , r , f {} {} {} ".format(precision/eval_sum , recall/eval_sum, fscore/eval_sum))
    all_precision.append(precision/eval_sum)
    all_recall.append(recall/eval_sum)
    all_fscore.append(fscore/eval_sum)

print(all_fscore)
df_iter['precision'] = all_precision
df_iter['recall'] = all_recall
df_iter['fscore'] = all_fscore

df_iter.to_csv("results_TextRank_recheck.csv")

print(10*"*")
print((time.time() - start)/60)

# find the sum of docs that have actual answers
