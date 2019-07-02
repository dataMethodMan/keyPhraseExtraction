# class will examine
# expanding references - pos , tfidf, phraseA , phraseB , stopwords , no stopwords, graph = 4 , directed undirected

from methods_main2 import *
from methods_main4 import DataSet , computeTermPDF , pageRankClass
import pandas as pd
import time
from collections import Counter , OrderedDict
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

start = time.time()
# path associate with target data
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
# initialse class with path pointer
dataClass = DataSet(path)
methods = mainMethods(path)
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



tester = list(dataClass.dataset['refs'])

# takes out unwanted terms in references
dataClass.dataset['refs'] = dataClass.cleanRefs(dataClass.dataset.refs, pdf)

# creates on string of the docs
dataClass.dataset['stringDocs'] = dataClass.dataset.rawDict.apply(dataClass.concatDict)

# method takes the columns refs and stringDocs and expands refs in all docs
dataClass.dataset['stringDocs'] = dataClass.ALL_fillOutReference(dataClass)


#print(dataClass.dataset.stringDocs[0])

# process doc so it is an array of arrays
dataClass.dataset['processDocs']  = dataClass.dataset.stringDocs.apply(dataClass.splitCorpus)

#accDict = dataClass.extractAccronymnsFromText(text)

# expand accronyms

# loops over docArrayStrings and creates a dictionary
dataClass.dataset['accDict'] = dataClass.dataset['stringDocs'].apply(dataClass.extractAccronymnsFromText)
print(dataClass.dataset.accDict[1])

dataClass.expandAccronymnsInText()

# # #tester = dataClass.dataset.processDocs[0]
# # # clean the corpus --> returns an array of array tokens
# dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(dataClass.cleanSentences)
#
# count = 0
#
# allIndex = []
# for index in range(len(list(dataClass.dataset['processDocs']))):
# #for index in range(0, 1):
#
#     print("at stage {}".format(index))
#
#     testerDoc = dataClass.dataset['processDocs'][index]
#
#     PR = pageRankClass(testerDoc)
#
#     # as far as here it is good
#     PR.constructGraph(testerDoc)
#     #print(PR.graph.nodes())
#
#     PR.createPhrasese()
#
#     #print(PR.posCorp)
#     docKeys = dataClass.dataset.keyTerms[index]
#     indexLoc = dataClass.extractKeyOrderedrank(PR.textRankDict , docKeys)
#     allIndex.append(indexLoc)
#
#     print(list(PR.textRankDict.items())[:20])
#
# indexLoc = dataClass.rankLocationIndex(allIndex)
# print(indexLoc)
# dataClass.plotIndexResults(indexLoc)

print(10*"-*-")
print((time.time() - start)/60)
