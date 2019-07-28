import pandas as pd
# commented out is the way to reformat a table

#df = pd.read_csv("results_TextRank.txt")
df = pd.read_csv("results_tfidf_recheck.csv")


print(df.columns)
name = df.columns[0]
print(name)
#
df = df.drop(name, axis = 1)
print(df)
#
#
#
#
# def formatResults(text):
#     text = round(text, 2)
#     print(text)
#     return text
#
df.sort_values(by = ['fscore'],  ascending=False, inplace = True)
#
df.reset_index(inplace = True)
df = df.drop("index", axis = 1)
print(df)

df.to_csv("reorderedTable_tfidf.csv")
# #
#
# print(df.columns)
#
# df.recall = df.recall.apply(formatResults)
# df.precision = df.precision.apply(formatResults)
# df.fscore = df.fscore.apply(formatResults)
#
# print(df.head())
#
# df.to_csv("formatedTable_delimins",sep=' ', header=None,  index = False)






# import pandas as pd
#
# df = pd.read_csv("results_tfidf_test.csv")
#
#
# print(df.head())
# #df.drop(df[df.columns[0]], inplace = True)
# print(df.columns)
#
#
# df2 = pd.DataFrame()
#
# df2['Sentence Deliminators'] = df.Deliminator
# df2['Stopword Removal'] = df.Stopword
# df2['Acronym Expansion'] = df.Acronym_Exp
# df2['Reference Expansion'] = df.Ref_Exp
# df2['Stemming'] = df.Stemming
# df2['precision'] = df.precision
# df2['recall'] = df.recall
# df2['fscore'] =  df.fscore
# print(df2.head())

#
#
#
# def formatResults(text):
#
#     text = round(text, 2)
#     print(text)
#     return text
#
#
# df.recall = df.recall.apply(formatResults)
# df.precision = df.precision.apply(formatResults)
# df.fscore = df.fscore.apply(formatResults)
#
# print(df.head())
#
#df2.to_csv("formatedTable_testing.csv", sep=',', index = False)


    # with open('posAcc.txt', 'w') as f:
    #     f.write("%s\n" % my_list)
