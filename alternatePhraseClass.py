# this method is the one following extractingKeyPhrases 9 / 8 and will look at phrase generation


#from methods_main2 import *
from methods_main4 import DataSet , computeTermPDF , pageRankClass , tfidfClass
import pandas as pd
import time
from collections import Counter , OrderedDict
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import time
start = time.time()


# open up the permutations the class takes
df_iter = pd.read_csv("testPermutation.txt")

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


textRank = True
tfidfRank = False
#for mover in range(0, df_iter.shape[0]):
if textRank:
    thresher = 0.0002

if tfidfRank:
    thresher = 0.001

for i in range(0, 1):
        # initialise the tfidf class
        tf = tfidfClass()


        dataClass.refThreshold = thresher
        print("threshold set to {}".format(str(dataClass.refThreshold)))

        dataClass.extractText()

        #print("turning off stemming for now - should be presaved ")


        #dataClass.stopwordRemove = True
        if tfidfRank:
            tf.stopwordRemove = True

        if textRank:
            dataClass.stopwordRemove = False


        fillRefferences = True
        fillAcc = True
        applyStemming = True
        setDeliminators = True


        #dataDict = dataClass.meta_dataset.files.apply(dataClass.extractSections)
        dataClass.dataset['rawDict'] = dataClass.meta_dataset.files.apply(dataClass.extractSections)
        #>>>>>>>>>>>>>>>>>>>>>>>>
        # we will look to first cleaning the dictionary heads
        # method loops over dictionary or sections and extracts refernces
        # returns as a key <ref num> value <string>


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

        # ####################
        # # handling acronymns
        # ---------> one can add in the Acronymns here
        if fillAcc:
            dataClass.dataset['accDict'] = dataClass.dataset['stringDocs'].apply(dataClass.extractAccronymnsFromText)

            print("applying acc")
            dataClass.expandAccronymnsInText()
        # ####################

        if fillRefferences:
            print("applying ref")
            dataClass.dataset['stringDocs'] = dataClass.ALL_fillOutReference(dataClass)

        if setDeliminators:
            print("applying delims")
            print("unchanging how delimins is done, check that it does not affect tfidf")
            if tfidfRank:
                dataClass.dataset.stringDocs = dataClass.dataset.stringDocs.apply(dataClass.creatDeliminators)
            if textRank:
                dataClass.dataset['processDocs']  = dataClass.dataset.stringDocs.apply(dataClass.splitCorpus)

        # ---------> one can add in the references here

        #

        #
        # # clean the corpus --> returns an array of array tokens
        #dataClass.dataset['processDocs'] = dataClass.dataset.stringDocs.apply(dataClass.cleanSentences)

        if tfidfRank:
            print("applying tfidf processing for passing to statistics generator")
            dataClass.dataset['processDocs'] = dataClass.dataset.stringDocs.apply(dataClass.cleanSent)
            dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(tf.processSent)

            print("separating text for phrase generation")

        dataClass.dataset['phraseText'] = dataClass.dataset.stringDocs.apply(dataClass.cleanSent)
        dataClass.dataset['phraseText'] = dataClass.dataset['phraseText']
        dataClass.dataset['phraseText']  = dataClass.dataset['phraseText'].apply(dataClass.cleanCorpus2)

            #print("converted it to vsm here, if needs another format")
            #dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(dataClass.cleanCorpus2)




        if textRank:
            #dataClass.dataset['processDocs']  = dataClass.dataset.stringDocs.apply(dataClass.splitCorpus)
            dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(dataClass.cleanSentences)



        #######################
        #extracting the targetTerms
        dataClass.extractTargetTerms()

        #######################

        # apply stemming to docset
        if applyStemming:
            print("applying stemming")
            print("leaving this altered stemming but will need to revisit")
            if tfidfRank:
                dataClass.dataset.processDocs = dataClass.dataset.processDocs.apply(dataClass.stem_Doc)
            dataClass.dataset['keyTerms'] = dataClass.dataset['keyTerms'].apply(dataClass.stem_array)

        #the tfidf portion of the method
        ####################
        # creating tfidf
        if tfidfRank:
            tfidf_matrix, tfidf_vectoriser = tf.applyTFidfToCorpus(list(dataClass.dataset.processDocs), failSafe = True)
            df = tf.ExtractSalientTerms(tfidf_vectoriser, tfidf_matrix, title ="tfidf_.pkl", failSafe = True)

        ####################
        #
        # # generate the phrases so that they are similar to the ones in the previous section
        # # takes the corpus as if there are no deliminators
        # dataClass.dataset['corpus'] = dataClass.dataset.stringDocs.apply(dataClass.wrapTextInArray)
        # #dataClass.dataset.stringDocs = dataClass.dataset.stringDocs.apply(dataClass.creatDeliminators)
        # #clean these
        # dataClass.dataset['corpus'] = dataClass.dataset.stringDocs.apply(dataClass.cleanSentences)
        # # create potential phrases from stems.


        precision = 0
        recall = 0
        fscore = 0
        allIndex = []
        # classicWay and newWay are booleans that direct it to do the phrases or the old standard
        # this is specifically to check textRank - which should be true, tfidf , has no old way as testing is done
        # set classicWay to true to generate graph
        # set new way to true to pass graph to new phrase generation 
        classicWay = True
        newWay = True
        #for index in range(0,1):

        for index in range(dataClass.dataset.shape[0]):
            print("stage {}".format(index))
            text = dataClass.dataset['phraseText'][index]

            if tfidfRank:
                df1 = df[df.doc_id_list == index]
                termsDict = dict(zip(list(df1.term_list), list(df1.term_idf_list)))



            t = dataClass.stem_Doc(text)
            dataClass.createAjoinedPhrases(t)

            all_terms = []
            for array in t:
                all_terms.extend(array)
            all_terms = dict(Counter(all_terms))
            dataClass.phraseDict.update(all_terms)

            if textRank:
                testerDoc = dataClass.dataset['processDocs'][index]
                # #print(testerDoc)
                PR = pageRankClass(testerDoc)
                # # this method populates the textRank dict of values
                PR.constructGraph(testerDoc, stem = True)
                # # for key, value in termsDict.items():
                # #     print(key, value)


                if classicWay:
                    PR.createPhrasese()

                    print(len(PR.textRankDict.items()))
                    PR.PhraseCandidates = dict(sorted(PR.PhraseCandidates.items(), key=lambda x: x[1], reverse = True))
                    print("running classic approach")
                    y_pred = dict(list(PR.textRankDict.items())[:15])
                    termsDict = PR.textRankDict



            if newWay:
                # method for iterating over dict and assigning values to each term in array
                rankedPhrases = dataClass.rankPhrasesCorpus(termsDict, dataClass.phraseDict)
                # for key, value in rankedPhrases.items():
                #     print(key, value)
                # for phrase , value in rankedPhrases.items():
                #     print(phrase , value)
                # ranking the phrases
                for phrase, value in rankedPhrases.items():
                    #print(phrase, value)
                    if phrase in dataClass.phraseDict:
                        #print(dataClass.phraseDict[phrase])
                        rankedPhrases[phrase] = dataClass.phraseDict[phrase] * rankedPhrases[phrase]


                rankedPhrases = dict(sorted(rankedPhrases.items(), key=lambda x: x[1], reverse = True))

            # our targets
            docKeys = dataClass.dataset.keyTerms[index]

            #print(docKeys)
            #print(rankedPhrases)
            indexLoc = dataClass.extractKeyOrderedrank(rankedPhrases, docKeys)
            print(indexLoc)

            allIndex.append(indexLoc)
            indexLoc = dataClass.rankLocationIndex(allIndex)
            #indexLoc = dataClass.extractKeyOrderedrank(PR.rankedPhrases , docKeys)
            print(indexLoc)


            y_pred = dict(list(rankedPhrases.items())[:15])
            #print(y_pred)
            y_true = docKeys

            precision_instance , recall_instance, fscore_instance = dataClass.calculateFscore( y_pred, y_true)
            precision += precision_instance
            recall += recall_instance
            fscore += fscore_instance

            # reset data dict
            dataClass.phraseDict = {}

        # find the sum of docs that have actual answers
        eval_sum = sum([1 for terms in dataClass.dataset.keyTerms if len(terms) > 0])

        print(" p , r , f {} {} {} ".format(precision/eval_sum , recall/eval_sum, fscore/eval_sum))
        all_precision.append(precision/eval_sum )
        all_recall.append(recall/eval_sum)
        all_fscore.append(fscore/eval_sum)


df = pd.DataFrame()


print(10*"*")
print((time.time() - start)/60)


# running tfidf with alternative phrase generation
#  p , r , f 4.519230769230764 12.330655455655457 6.430052720428534
# [140, 88, 152, 94, 635, 143]

# page rank
# p , r , f 4.326923076923074 12.173434204684206 6.217984973881821
#[129, 100, 155, 100, 630, 138]

# revisiting with better preprocessing
# - first rerun will include the references
# - rerun with no references included  in phrase generation , but still attached as values
# seems it is the same regardless of delims
#[144, 92, 153, 92, 615, 156]
# p , r , f 4.679487179487174 12.783379814629816 6.664407419886208

# totally without references
# [141, 81, 137, 97, 640, 156]
#  p , r , f 4.583333333333329 12.401834276834277 6.509561886476159
# with counting metric
# [260, 336, 166, 64, 283, 143]
#  p , r , f 8.814102564102562 23.91927516927516 12.50064439090812
#####

#[246, 375, 180, 60, 253, 138]
#  p , r , f 8.333333333333334 22.5578588078588 11.802673999573887
# alt way - with variant delimins - check above and below tomorrow
# [243, 368, 178, 45, 281, 137]
#  p , r , f 8.237179487179487 22.292291042291044 11.664080951498077

###
# by the size
# [259, 258, 147, 50, 396, 142]
#  p , r , f 8.557692307692307 23.04490648240648 12.13270536354061

###
# textRank appraoch - parameters agument
# [190, 258, 363, 107, 14, 320]
#  p , r , f 6.314102564102561 17.84500222000222 9.08477395266693
