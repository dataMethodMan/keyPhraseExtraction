# class will examine
# expanding references - pos , tfidf, phraseA , phraseB , stopwords , no stopwords, graph = 4 , directed undirected

from methods_main2 import *
from methods_main4 import DataSet , computeTermPDF , pageRankClass , tfidfClass
import pandas as pd
import time
from collections import Counter , OrderedDict
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import time
start = time.time()


# open up the permutations the class takes
df = pd.read_csv("testPermutation.txt")




#
start = time.time()
# path associate with target data
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
#path = "C:/userOne/AutomaticKeyphraseExtraction-master/data/"

# initialse class with path pointer
dataClass = DataSet(path)

all_precision = []
all_recall = []
all_fscore = []

#for iter in range(0, df.shape[0]):
for iter in range(0, 1):
    print(iter)

    # initialise the tfidf class
    tf = tfidfClass()
    # pull out the text  - sectionsDict : text store in a dictionary partitioned by keys to sections
    # for the moment we have only pulled out section relating to references to extract references
    dataClass.extractText()

    permutations1 = df.iloc[iter]

    permutations1 = dataClass.convertToBool(permutations1)


    ## event booleans
    tf.stopwordRemove = permutations1[0]
    fillRefferences = permutations1[1]
    fillAcc = permutations1[2]
    applyStemming = permutations1[3]
    setDeliminators = permutations1[4]


    #dataDict = dataClass.meta_dataset.files.apply(dataClass.extractSections)
    dataClass.dataset['rawDict'] = dataClass.meta_dataset.files.apply(dataClass.extractSections)
    #>>>>>>>>>>>>>>>>>>>>>>>>
    # we will look to first cleaning the dictionary heads
    # method loops over dictionary or sections and extracts refernces
    # returns as a key <ref num> value <string>
    dataClass.dataset['refs'] = dataClass.dataset.rawDict.apply(dataClass.methodRefsExtract)

    dataClass.extractText()

    ##################
    # handling references
    # extracts alll of the references from text
    dataClass.dataset['refs'] = dataClass.dataset.rawDict.apply(dataClass.methodRefsExtract)
    # lower cases the keys in the dictionary -> prep for ref drop
    dataClass.dataset['rawDict'] = dataClass.dataset.rawDict.apply(dataClass.cleanKeys)
    # drop references from text
    dataClass.dataset.rawDict.apply(dataClass.dropReferences)


    # --> further extract all the terms so that we can assess which refs to keep
    # clean the dict
    allTermArrayCount = dataClass.extractVocab(list(dataClass.dataset.rawDict))
    pdf = computeTermPDF(allTermArrayCount)
    pdf.calculateProbTerm()
    # takes out unwanted terms in references
    # now dataset is without reference which are filtered and stored in another column
    dataClass.dataset['refs'] = dataClass.cleanRefs(dataClass.dataset.refs, pdf)
    ##################

    # creates on string of the docs
    dataClass.dataset['stringDocs'] = dataClass.dataset.rawDict.apply(dataClass.concatDict)


    if setDeliminators:
        print("applying delims")
        dataClass.dataset.stringDocs = dataClass.dataset.stringDocs.apply(dataClass.creatDeliminators)



    #
    # # ---------> one can add in the references here

    if fillRefferences:
        print("applying ref")
        dataClass.dataset['stringDocs'] = dataClass.ALL_fillOutReference(dataClass)
    #
    # ####################
    # # handling acronymns
    dataClass.dataset['accDict'] = dataClass.dataset['stringDocs'].apply(dataClass.extractAccronymnsFromText)
    # ---------> one can add in the Acronymns here
    if fillAcc:
        print("applying acc")
        dataClass.expandAccronymnsInText()
    # ####################
    #
    # # # clean the corpus --> returns an array of array tokens
    # #dataClass.dataset['processDocs'] = dataClass.dataset.stringDocs.apply(dataClass.cleanSentences)
    #
    #
    dataClass.dataset['processDocs'] = dataClass.dataset.stringDocs.apply(dataClass.cleanSent)


    dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(tf.processSent)

    #######################
    #extracting the targetTerms
    dataClass.extractTargetTerms()

    #######################

    # apply stemming to docset
    if applyStemming:
        print("applying stemming")
        #text =dataClass.stem_Doc(dataClass.dataset['processDocs'][0])
        dataClass.dataset.processDocs = dataClass.dataset.processDocs.apply(dataClass.stem_Doc)
        dataClass.dataset['keyTerms'] = dataClass.dataset['keyTerms'].apply(dataClass.stem_array)


    #dataClass.dataset.stringDocs[0] = tf.removeSingles(dataClass.dataset.stringDocs[0])
    #print(dataClass.dataset.processDocs[0])




    #the tfidf portion of the method
    ####################
    # creating tfidf
    tfidf_matrix, tfidf_vectoriser = tf.applyTFidfToCorpus(list(dataClass.dataset.processDocs), failSafe = False)
    df = tf.ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, title ="tfidf_.pkl", failSafe = False)



    ####################


    # generate the phrases so that they are similar to the ones in the previous section
    # takes the corpus as if there are no deliminators
    dataClass.dataset['corpus'] = dataClass.dataset.stringDocs.apply(dataClass.wrapTextInArray)

    #clean these
    dataClass.dataset['corpus'] = dataClass.dataset.corpus.apply(dataClass.cleanSentences)
    # create potential phrases from stems.
    #print(dataClass.dataset['corpus'][0])


    precision = 0
    recall = 0
    fscore = 0
    allIndex = []

    #for index in range(0, 3):
    for index in range(len(list(dataClass.dataset['processDocs']))):
        text = dataClass.dataset.corpus[index]

        PR = pageRankClass(text)

        PR.posCorp = PR.extractPosTags(text)
        # rename keyTerms to match old code implementation
        #dataClass.dataset .rename(columns = {'keyTerms':'targetTerms'}, inplace = True)
        # not just yet ^^^^^

        df1 = df[df.doc_id_list == index]
        termsDict = dict(zip(list(df1.term_list), list(df1.term_idf_list)))
        # df contains 4 grams  --> remove only single instances

        singletons = {}
        ngrams = {}
        for key, values in termsDict.items():
            if len(key.split()) == 1:
                singletons[key] = values
            else:
                ngrams[key] = values

        singletons = dict(sorted(singletons.items(), key=lambda x: x[1], reverse = False))

        # set the values for the terms
        PR.textRankDict = singletons

        # create the phrases
        PR.createPhrasese()
        #PR.textRankDict = dict(sorted(PR.textRankDict.items(), key=lambda x: x[1], reverse = False))
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

    # find the sum of docs that have actual answers
    eval_sum = sum([1 for terms in dataClass.dataset.keyTerms if len(terms) > 0])

    print(" p , r , f {} {} {} ".format(precision/eval_sum , recall/eval_sum, fscore/eval_sum))
    all_precision.append(precision/eval_sum )
    all_recall.append(recall/eval_sum)
    all_fscore.append(fscore/eval_sum)

print(all_fscore)
# df['precision'] = all_precision
# df['recall'] = all_recall
# df['fscore'] = all_fscore

print(10*"*")
print((time.time() - start)/60)
