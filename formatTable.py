import pandas as pd

#df = pd.read_csv("results_TextRank.txt")
df = pd.read_csv("tfidf_recheck.csv")

#df.drop(df[df.columns[0]], inplace = True)
print(df.columns)
name = df.columns[0]
print(name)

df = df.drop(name, axis = 1)




def formatResults(text):
    text = round(text, 2)
    print(text)
    return text

df.sort_values(by = ['fscore'],  ascending=False, inplace = True)

df.reset_index(inplace = True)
print(df)


print(df.columns)
#
# df.recall = df.recall.apply(formatResults)
# df.precision = df.precision.apply(formatResults)
# df.fscore = df.fscore.apply(formatResults)
#
# print(df.head())
#
# df.to_csv("formatedTable_delimins",sep=' ', header=None,  index = False)
