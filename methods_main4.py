# toward the final version will contain more formally constructed code and should phase out
# methods_main2
import pandas as pd
import os
import codecs
import re
import operator
import string
from collections import Counter
import networkx as nx
import itertools
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import matplotlib.pyplot as plt
import numpy as np



class DataSet(object):
    """ represents the class object of the target Text
        Attributes:
            dataset:        df of the target Data
            meta_dataset:   df of Data afterProcessing
            path:           string location of targt directory
    """

    def __init__(self, path):
        # path to target directory
        self.path = path
        # df that holds data
        self.dataset = pd.DataFrame()
        # df that holds metaData on the class
        self.meta_dataset = pd.DataFrame()
        # dict to hold phrases
        self.phraseDict = {}

    def calculateFscore(self, y_pred , y_true ):


        y_pred = list(y_pred.keys())
        y_true = y_true
        print(y_true)

        #precision = (number of true positives)/ number of positives predicted
        correct = [1 for x in y_true if x in y_pred ]
        correct = sum(correct)
        precision = float(correct/len(y_pred))

        # recall = correct/len(y_pred)
        recall = float(correct/len(y_true))

        # fscore = 2 *( precision * recall)/(precision + recall)
        try:
            fscore = 2*(precision*recall)/(precision + recall)
        except:
            fscore = 0

        print(fscore)

        return precision*100 , recall*100 , fscore*100




    def wrapTextInArray(self, text):
        return [text]

    def createAjoinedPhrases(self, text):
        print(len(text))
        # length of phrases to be generated
        for t in text:
            while len(t) > 1:
                t = self.generatePhraseFromArray(t)

    def generatePhraseFromArray(self, vector):
        #phrase = " ".join(vector)
        vec = vector[:4]
        sliding_window = 4
        if len(vec) < sliding_window:
            sliding_window = len(vec)
        index = 2
        while index < sliding_window + 1:
            phrase = vec[:index]
            index = index + 1
            phrase = " ".join(phrase)
            if phrase in self.phraseDict.keys():
                self.phraseDict[phrase] += 1
            else:
                self.phraseDict[phrase] = 1


        return vector[1:]



    def expandAccronymnsInText(self):
        # method takes the dictionary created in extractAccronymnsFromText
        # loops over processDocs -- arrays of string and expands accronyms
        for i in range(self.dataset.shape[0]):
            dictt = self.dataset.accDict[i]
            #print(dictt)
            text = self.dataset.stringDocs[i]
            text = self.fillOutReference(text, dictt)
            self.dataset.stringDocs[i] = text


    def extractAccronymnsFromText(self, text):
        # input is a string list of sentences
        accronymDict = {}
        # loop over the sentences in text
        text = text.split(". ")
        for t in text:
            if "(" in t:
                tester = self.extractCandidateSubstring1(t)
                # if not blank add to dict
                if tester[0] != '' and len(tester[0]) > 1:
                    phrase = [x.lower() for x in tester[1]]
                    accronymDict[tester[0]] = " ".join(phrase)
                    #liste.append(tester)
        return accronymDict


    def extractCandidateSubstring1(self, match):
        #print("this one")
        pattern = '\((.*?)\)'
        candidate = ""
        substring = ""
        #match = match.strip("\n")
        match = match.split(" ")
        for i in range(0, len(match)):
            cand = re.search(pattern, match[i])
            if cand:
                candidate = cand.group(1)
                # check that it is longer than 1
                if len(candidate) > 1:
                    # check and remove for non capital mix
                    #print(candidate)
                    if(self.lookingAtAcroynms(candidate)):
                        candidate = self.removeNonCapitals(candidate)
                    j = len(candidate)
                    substring = match[i-j:i]
                    #print(substring)
                    # check if accronym is present
                    wordsAccro = self.returnPotentAccro(substring)
                    if candidate.lower() == wordsAccro.lower():
                        # return the correct accro and definition
                        return (candidate, substring)

        # no accronym found return blank , will be filtered out
        return("", "")

    def lookingAtAcroynms(self, accro):
        # case one check if accroynm has an append s
        bool = False
        for s in accro[:1]:
            if s.isupper:
                bool = True
        return bool

        # check of the main lettes match
    def returnPotentAccro(self, substring):
        firsts = ""
        for s in substring:
            if(len(s) > 0):
                firsts = firsts + s[0]
        return firsts

    def removeNonCapitals(self, accro):
        string = ""
        for s in accro:
            if s.isupper():
                string = string + s
        return string

    def createConsecutivePhrases(self):
        # iterate over corpus and extract connected terms
        print(self.dataset.columns)
        doc_dict = {}
        index = 0
        sliding_window = 4
        term = ""

        doc = self.dataset.processDocs[0]

        for array in doc:
            while len(array) > 1:
                array = self.reduceVector(array, sliding_window)
            print()



        # remove singles and parse text


    def reduceVector(self, vector, sliding_window):
        processArray = [vector[:sliding_window]]
        print(processArray)
        return vector[sliding_window:]



    def extractKeyOrderedrank(self, docDict, docKeyTerms):
        # sort the dictionary to allow for phrases being added
        #docDict = dict(sorted(docDict.items(), key=lambda x: x[1], reverse = True))



        index = [x for x in range(1 , len(docDict.items())+ 1)]

        # assign index to rank
        docDict = dict(zip(docDict.keys(), index))
        indexLoc = []
        #print(docKeyTerms)
        temp = {}
        for key in docKeyTerms:
            if key in docDict.keys():
                indexLoc.append(docDict[key])
                temp[key] = docDict[key]
            else:
                indexLoc.append(-1)
                temp[key] = - 1

        print(temp)
        return indexLoc


    def rankLocationIndex(self, indexList):
        cups = [15, 100, 500, 1000, 0]
        notPresent = 0
        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        for docIndex in indexList:
            for index in docIndex:
                if index == -1:
                    notPresent = notPresent + 1
                elif index < cups[0]:
                    zero = zero + 1
                elif index < cups[1]:
                    one = one + 1
                elif index < cups[2]:
                    two = two + 1
                elif index < cups[3]:
                    three = three + 1
                else:
                    four = four + 1

        return [zero, one, two, three, four, notPresent]


    def extractTargetTerms(self):
        self.dataset['keywords'] = self.meta_dataset.handle.apply(self.extractKeyWordFiles)
        self.dataset['competition_terms'] = self.meta_dataset.handle.apply(self.extractKeyWordFilesTerms)

        # remove single bad instance
        self.dataset.keywords[47] = []
        # concatenate the results
        self.dataset['keyTerms'] = self.dataset.keywords + self.dataset.competition_terms




    def extractKeyWordFiles(self, text):
        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        path = self.path + str(text) + "/" + str(text) + ".kwd"
        try:
            text = self.extractContent(path, removeLines = False)
            text = [ x.strip().lower() for x in text.split("\n") if len(x)> 1]
        except:
            text = []
        return text

    def extractKeyWordFilesTerms(self, text):
        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        path = self.path + str(text) + "/" + str(text) + ".term"
        try:
            text = self.extractContent(path, removeLines = False)
            # cleans unwanted text
            text = [x.strip().lower() for x in text.split("\n") if len(x)> 1]
        except:
            #print("keyword absent: " + str(text))
            text = []
        return text


    def cleanSentences(self, text):
        allSent = []
        bool = False
        for line in text:
            line = self.cleanSent(line)
            line = [x for x in line.split() if len(x) > 1]
            if len(line)  > 1:
                allSent.append(line)
        return allSent

    def splitCorpus(self, text):
        # first separate by sentence
        # remove newline metatag
        text = " ".join(text.split("\n"))
        # break into sentences
        text = text.split(". ")
        # further split into commas
        commaArray = self.divideByIndicator(text, ",")
        collonArray = self.divideByIndicator(commaArray, ":")
        semiCollonArray = self.divideByIndicator(collonArray, ";")

        return semiCollonArray

    def divideByIndicator(self, array, indicator):
        # accepts as argument array and indicator ; returns array of arrays separated by indicator
        chunkArray = []
        for sent in array:
            sent1 = sent.split(indicator)
            chunkArray.extend(sent1)
        return chunkArray


    def ALL_fillOutReference(self, dataClass):
        stringDocsArray = []
        for index, row in dataClass.dataset.iterrows():
            #print(index , dataClass.meta_dataset.files[index])
            text = self.fillOutReference(dataClass.dataset.stringDocs[index], dataClass.dataset.refs[index])
            #print(text)
            stringDocsArray.append(text)
        return stringDocsArray



    def fillOutReference(self, text, ref_dict):
        # takes as arguemnt string and dictionary and expands reference numbers
        for key , value in ref_dict.items():
            value = "." +  value + "."
            if value in text:
                text = text.replace( key, value)
        return text


    def concatDict(self, dictt):
        # takes as argument dictionary; returns array of values
        docString = ""
        for value in dictt.values():
            docString = docString + " " + value.strip()
        return docString

    def cleanRefs(self, textDictList, pdf):
        # takes in a reference list for a doc , cleans it , filters it and returns a dict
        # ref [x] as key and ''
        cleanedDictList = []
        for textDict in textDictList:
            refDict = {}
            for k , v in textDict.items():
                v = self.cleanSent(v)
                v = self.extractSalientTerm(v, pdf)
                v = [w for w in v if not w.isdigit() ]
                refDict[k] = " ".join(v)
            cleanedDictList.append(refDict)
        return cleanedDictList


    def extractSalientTerm(self, text, pdf):
        # method takes in a string text and returns an array of only salient terms
        keeps = []
        for t in text.split():
            if pdf.prob(t) > 0.0001:
                keeps.append(t)
        return keeps


    # convert all to word tokens
    def TokeniseDoc(self, textDict):
        termList = []
        for v in textDict.values():
            termList.extend(v.split())
        return termList

    def dropReferences(self , text):
        try:
            del text['references']
        except:
            i = 1
            try:
                del text['bibliography']
            except:
                #print("no bib to drop")
                i = 1

        return text

    def extractVocab(self, corpusList):
        # method extracts and cleans docs , returns a list of cleaned documents
        cleanedCorpusList = self.iterateCleanDictionay(corpusList)

        # list
        allTermArray = []
        for dict1 in cleanedCorpusList:
            # iterate over the dictionaries and create an array of vocab
            allTermArray.extend(self.TokeniseDoc(dict1))

        # remove stop words and singles
        allTermArray = [x for x in allTermArray if len(x) > 1 and x not in stop]
        # count all of the resulting terms
        allTermArrayCount = dict(Counter(allTermArray))

        return allTermArrayCount


    def iterateCleanDictionay(self, corpusList):
        # clean the dict
        cleanCorpusDictList = []
        for dictt in corpusList:
            dictC = self.processDict(dictt)
            cleanCorpusDictList.append(dictC)
        return cleanCorpusDictList



    def cleanKeys(self, data):
        data = self.cleanDictionaryHeaders(data)
        return data

    def methodRefsExtract(self, data):
        dict = self.cleanDictionaryHeaders(data)
        refs = self.retriveRefsFromTitle(dict)
        refs = self.extractIndividualReferences(refs)

        return refs


    def cleanDictionaryHeaders(self, text):
        # takes in dictionary as argument and returns dictionary with cleaned keys
        tempDict = {}
        for key, value in text.items():
            tempDict[self.cleanSent(key)] = value
        return tempDict

    def retriveRefsFromTitle(self, text):
        # looks for refs or bib
        # else returns none -- 4 errors seen in the data
        try:
            return(text['references'])
        except:
            #print("no references section")
            i  =  1

        try:
            return(text['bibliography'])
        except:
            i = 1

        return None


    def extractText(self):
        # populates meta class with ref to directory files
        self.extractFileNames(self.path)
        # extract xml data and assign to meta class | takes as input required file type
        self.extractFileContent("xml")


    def extractFileNames(self, path):
        listee = []
        dirs =  os.listdir(path)
        for d in dirs:
            path1 = path + d
            if(os.path.isdir(path1)):
                listee.append(int(d))
        listee.sort()
        # assigns the value to the class def
        self.meta_dataset = pd.DataFrame({"handle" : listee})
        return listee , dirs

    def extractFileContent(self, contentType):
        # checks that intend return type
        if contentType == "xml":
            # populates column with directory path per file
            self.meta_dataset['fileNames'] = self.meta_dataset.handle.apply(self.extractXMLFiles)
        # loops over fileNames and returns the content
        self.meta_dataset['files'] = self.meta_dataset.fileNames.apply(self.extractContent)
        ### breaks the xml into sections allowing us to compartmentalise analysis
        #self.meta_dataset['sectionsDict'] = self.meta_dataset.files.apply(self.extractSections)

        # code written to examine raw text and pull out content after keyword identified
        #self.meta_dataset['ref'] = self.meta_dataset.files.apply(self.extractRefernces)


    def extractXMLFiles(self, text):
        # given a directory name returns an instance of tha files of a particular format found within
        return self.path + str(text) + "/" + str(text) + ".xml"

    def extractContent(self, text, removeLines = True):
        """ given a file name reads the text within """
        with codecs.open(text, 'r', encoding='utf8', errors="ignore") as file:
            lines = file.read()
            # removes the newlines rife within the text
            if removeLines:
                lines = " ".join(lines.split("\n"))
        return lines


    def extractSections(self, text):
        ## method to extract sections
        sectionsArray = []
        # identifies sections
        for result in re.findall('<SECTION(.*?)</SECTION>', text, re.S):
            sectionsArray.append(result)

        # further extract headers (will be used as keys)
        sectionHeaders= []
        for section in sectionsArray:
            for result in re.findall('header=(.*?)>', section):
                sectionHeaders.append(result)

        # append key headers to text
        sectionDict = dict(zip(sectionHeaders, sectionsArray))
        return sectionDict

    def extractRefernces(self, text):
        print(text)
        #print(len(referenceText))
        print(isinstance(text, string))
        if text is not None:
            referenceText = self.extractReferncesText(text)
            referenceDict = self.extractIndividualReferences(referenceText)
            return referenceDict
        else:
            return {}


    # def extractReferncesText(self, text):
    #     pattern = 'REFERENCES'
    #     if text is not None:
    #         try:
    #             refs_loc = text.index(pattern)
    #             text1 = text[refs_loc:]
    #             return text1
    #         except:
    #             i = 1
    #         try:
    #             pattern1 = pattern.lower()
    #             pattern1 = pattern1[0].upper() + pattern1[1:]
    #             refs_loc = text.index(pattern1)
    #             text1 = text[refs_loc:]
    #             percent_loc  = float(refs_loc)/len(text)
    #             if percent_loc > .80:
    #                 return text1
    #         except:
    #             i = 1

    def extractIndividualReferences(self, refList):
        if refList is not None:
            pattern = '(\[.*?\])'
            text = re.split(pattern, refList)
            key = []
            citation = []
            for i in range(1, len(text)):
                if i % 2 == 0:
                    citation.append(text[i].strip())
                else:
                    key.append(text[i])

            citDict = dict(zip(key, citation))
            return citDict
        else:
            return {}

    # def countUpperCase(self, text):
    #     upperCount = 1
    #     for value in text.split():
    #         if value.istitle():
    #             upperCount += 1
    #     return upperCount

    #
    # def extractRefTitle(self, text):
    #     candList = text.split(". ")
    #     ratioList = []
    #     for cand in candList:
    #         countUp = self.countUpperCase(cand)
    #         ratio = len(cand.split())/countUp
    #         ratioList.append(ratio)
    #     ratioDict = dict(zip(candList, ratioList))
    #     title = max(ratioDict.items(), key=operator.itemgetter(1))[0]
    #     return title

    def cleanSent(self, sent):
        remove = string.punctuation
        remove = remove.replace("-", "")
        pattern = r"[{}]".format(remove)
        sent = re.sub(pattern, " ", sent.strip().lower())
        # removes supurious spaces by breaking sent into array and reforming with only one space
        sent = sent.split()
        return " ".join(sent)

    def processDict(self, dict):
        """ takes in as argument a dictionary and cleans values and keys
        of all punctutation, uppercases. Returns a dict of string k, v """
        cleanDict = {}
        for k , v in dict.items():
            key = self.cleanSent(k)
            value = self.cleanSent(v)
            cleanDict[key] = value
        return cleanDict

    def plotIndexResults(self, list1):
        objects = ("<15", "<100", "<500", "<1000", ">1000" ,  "absent")
        y_pos = np.arange(len(objects))
        print(list1)
        plt.bar(y_pos , list1)
        plt.xticks(y_pos, objects)
        plt.ylabel("Occurences")

        plt.title("Index Location of Target Term")

        plt.show()



class computeTermPDF():
    def __init__(self, allTermArrayCount):
        print(len(allTermArrayCount), "length")
        self.corpus = allTermArrayCount
        self.total = len(list(allTermArrayCount.keys()))
        self.terms = set(list(allTermArrayCount.keys()))
        self.book = {}
        #self.total = self.computeTotal()

    # calculate the prior distribution of terms
    def calculateProbTerm(self):
        for key , value in self.corpus.items():
            self.book[key] = value/self.total

    def computeTotal(self):
        totals = list(self.corpus.values())
        return sum(totals)

    def prob(self, term):
        try:
            return self.book[term]
        except:
            return 0




class pageRankClass():

    def __init__(self, corpus):
        self.graph = nx.Graph()
        # corpus
        self.corpus = corpus
        # pos_tagged corpus
        self.posCorp = []
        # keeps record of neighbours
        self.inoutDict = {}
        # unique vocab in graph
        self.vocab = []
        # term score
        self.score = 0
        # hard cut off point
        self.MAX_ITERATIONS = 50
        #dampening factor
        self.d = 0.85
        # cutoff point for updates
        self.threshold = 0.0001 # convergence threshold
        # holds the final score
        self.textRankDict = {}




    def createPhrasese(self):
        for phrases in self.posCorp:
            bool = False
            #print(phrase)
            phrase = []
            phraseScore = 0
            for word in phrases:
                if word != "_":
                    phrase.append(word)
                    try:
                        phraseScore += self.textRankDict[word]
                    except:
                        print("exception term: ",  word)
                else:
                    if len(phrase) > 1:
                        if bool:
                            print(phrase)
                            bool = False
                        phrase = " ".join(phrase)
                        if phrase not in self.textRankDict.keys():
                            self.textRankDict[phrase] = phraseScore/len(phrase.split())
                        #print(phrase , phraseScore/len(phrase))
                    phrase = []
                    phraseScore = 0

        self.textRankDict = dict(sorted(self.textRankDict.items(), key=lambda x: x[1], reverse = True))




    def computePhraseValue(self, phrase):
        print(phrase)


    def constructGraph(self, testerDoc):
        # iterates over array of array tokens and replaces non pos with _
        self.posCorp = self.extractPosTags(testerDoc)
        # create a a new instance Text without the _
        Text = [[x for x in array if x is not "_"] for array in self.posCorp]


        # method takes array of arrays of tokenised corpus
        graph = self.plotDiGraph([Text])
        # create the inout dict
        # takes construct graph and maps the neighbours of the graph
        self.createInoutDict()

        self.calculatePageRankScore()

        self.textRankDict = dict(zip(self.vocab, self.score))

        self.textRankDict = dict(sorted(self.textRankDict.items(), key=lambda x: x[1], reverse = True))

        return self.textRankDict

    def calculatePageRankScore(self):
        self.score  = np.ones(len(self.vocab), dtype = np.float32)
        for iter in range(self.MAX_ITERATIONS):
            prev_score = np.copy(self.score)

            for i in range(len(self.vocab)):
                summation = 0
                for j in range(len(self.vocab)):
                    if i != j:
                        #print("this be where we have the issue")
                        #print(type(self.vocab))
                        if self.graph.has_edge(self.vocab[i], self.vocab[j]):
                            #print(graph[vocab[i]][vocab[j]]['cousin'])
                            #print(vocab[i], vocab[j])
                            summation += (self.graph[self.vocab[i]][self.vocab[j]]['cousin']/self.inoutDict[self.vocab[j]])*self.score[j]
                #print(" {} : {} ".format(str(vocab[i]), summation))
                self.score[i] = (1 - self.d) + self.d*(summation)

            if np.sum(np.fabs(prev_score - self.score)) <= self.threshold:
                # convergence baby
                print("convergence at {}".format(iter))
                break


    def createInoutDict(self):
        for a in self.graph.nodes():
            #print(a)
            #print(graph[a])
            for k , v in self.graph[a].items():
                if k in self.inoutDict:
                    self.inoutDict[k] += v['cousin']
                else:
                    self.inoutDict[k] = v['cousin']

    def plotDiGraph(self, corpus):
        #g = nx.Graph()
        # methods takes in an array (corpus) of arrays (docs) each containing tokenised sentences (section)
        for doc in corpus:
            for section in doc:
                depth = 4
                while(len(section) > 1):
                    self.graph, section = self.plotArray( section, depth, self.graph)
        # update vocab list
        self.vocab = list(self.graph.nodes())

        #return g

    def plotArray(self, array, depth, g):
        if len(array) < depth + 1:
            depth = len(array)
        targetTerm = array[0]
        for i in range(1, depth):
            g = self.AddGraphConnectionCousin( g, targetTerm, array[i], i)
        return g , array[1:]

    def AddGraphConnectionCousin(self, g, a, b, i):
        g.add_edge(a, b)
        try:
            g[a][b]['cousin'] += 1/i
        except:
            g.add_edge(a, b, cousin = 1)
        return g

    # extract viable terms
    def extractPosTags(self, text):
        desired_tags = ["JJ", "NNS", "NN", "JJS"]
        sentArray = []
        for t in text:
            #t = " ".join(t)
            sent = []
            pos_tag = nltk.pos_tag(t)
            for tag in pos_tag:
                if tag[1] in desired_tags:
                    sent.append(tag[0])
                else:
                    sent.append("_")
            sentArray.append(sent)

        return sentArray
