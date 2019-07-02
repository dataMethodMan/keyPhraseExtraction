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



# process doc so it is an array of arrays
dataClass.dataset['processDocs']  = dataClass.dataset.stringDocs.apply(dataClass.splitCorpus)
text =  dataClass.dataset['processDocs'][0]
#print(len(text))



# #tester = dataClass.dataset.processDocs[0]
# # clean the corpus --> returns an array of array tokens
dataClass.dataset['processDocs'] = dataClass.dataset.processDocs.apply(dataClass.cleanSentences)
text =  dataClass.dataset['processDocs'][0]
print(10*"=")
print(text)
dataClass.extractTargetTerms()
termKeys = dataClass.dataset['keyTerms'][0]
dictTemp = {}
for vector in text:
    tester = " ".join(vector)
    for key in termKeys:
        if key in tester:
            if key in list(dictTemp.keys()):
                dictTemp[key] += 1
            else:
                dictTemp[key] = 1
            print(tester)
            print(10*"-+-")
print(dictTemp)


# #dataClass.createConsecutivePhrases()
#
#
# # create the pageRank
# #result = dataClass.constructGraph(testerDoc)
#
# # extract keyPhrases
# # extract keys terms and phrases , filters text and returns a column keyTerms
dataClass.extractTargetTerms()
# #d   = {'dispersers': 12.599054, 'sources': 11.84855, 'extractor': 9.604512, 'source': 7.98054}
#
# for term in dataClass.dataset['keyTerms']:
#     print(term)
#
# present = 0
# absent = 0
# for i in range(1):
#     for term in dataClass.dataset['keyTerms'][i]:
#         if term in  dataClass.dataset.stringDocs[i].lower():
#             present = present + 1
#
#         else:
#             absent = absent + 1
#             print(i)
#             print(10*"-")
#             print(term)
#
# print("present: " , present )
# print("absent: " , absent )


#d = OrderedDict(d)
#print(d)
count = 0

allIndex = []
for index in range(len(list(dataClass.dataset['processDocs']))):
#for index in range(0, 1):

    print("at stage {}".format(index))

    testerDoc = dataClass.dataset['processDocs'][index]

    PR = pageRankClass(testerDoc)

    # as far as here it is good
    PR.constructGraph(testerDoc)
    #print(PR.graph.nodes())

    PR.createPhrasese()

    #print(PR.posCorp)
    docKeys = dataClass.dataset.keyTerms[index]
    indexLoc = dataClass.extractKeyOrderedrank(PR.textRankDict , docKeys)
    allIndex.append(indexLoc)

    print(list(PR.textRankDict.items())[:20])

indexLoc = dataClass.rankLocationIndex(allIndex)
print(indexLoc)
dataClass.plotIndexResults(indexLoc)

print(10*"-*-")
print((time.time() - start)/60)
