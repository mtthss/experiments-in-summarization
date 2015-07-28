import os
import csv

from csv import reader


__author__ = 'matteo'


# initialize paths
path_in = "./evals/rouge_output/"
path_out = "./evals/summary/"

# test collections
test = []

# initialize dictionary
dict = {}
for i in range(1,33):
    dict["2005-"+str(i)] = []
for i in range(1,36):
    dict["2006-"+str(i)] = []
for letter in ["A","B","C","D","E","F","G","H","I","J"]:
    dict["2005-"+letter] = []
for letter in ["A","B","C","D","E","F","G","H","I","J"]:
    dict["2006-"+letter] = []

# read evaluations by rouge
for filename in os.listdir(path_in):

    with open(path_in+filename) as csv:

        r = reader(csv)
        r.next()

        for line in r:

            collection = line[1]

            if len(test)==0 or collection.lower() in test:

                year = "2005" if len(collection)==5 else "2006"
                system = line[2]
                avg_recall = line[3]
                avg_precision = line[4]
                avg_fscore = line[5]

                dict[year+"-"+system].append([float(avg_recall),float(avg_precision), float(avg_fscore)])

# print system evaluations
for key in dict.keys():

    matrix = dict[key]
    if len(matrix)>0:

        recalls = [element[0] for element in matrix]
        precisions = [element[1] for element in matrix]
        fscores = [element[2] for element in matrix]

        mean_avg_recall = sum(recalls)/len(recalls)
        mean_avg_precision = sum(precisions)/len(precisions)
        mean_avg_fscore = sum(fscores)/len(fscores)

        name = key if len(key)==7 else key+" "
        print name, "\t", mean_avg_recall, mean_avg_precision, mean_avg_fscore