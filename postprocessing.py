import os
import csv
import pdb

from csv import reader, writer


__author__ = 'matteo'


# initialize paths
path_in = "./evals/rouge_output/"
path_out = "./evals/summary/"

# test collections
test = []
my_candidates = ["RF-R-MMR", "RF-R"]

# initialize dictionary
dict = {}
for x in my_candidates:
    dict["2005-"+x] = []
    dict["2006-"+x] = []
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
                try:
                    dict[year+"-"+system].append([float(avg_recall),float(avg_precision), float(avg_fscore)])
                except:
                    pdb.set_trace()

# compute scores
list_2005 = []
list_2006 = []

for key in dict.keys():

    year = key.split('-')[0]
    matrix = dict[key]
    if len(matrix)>0:

        recalls = [element[0] for element in matrix]
        precisions = [element[1] for element in matrix]
        fscores = [element[2] for element in matrix]

        mean_avg_recall = sum(recalls)/len(recalls)
        mean_avg_precision = sum(precisions)/len(precisions)
        mean_avg_fscore = sum(fscores)/len(fscores)

        if year == '2005':
            list_2005.append([key, mean_avg_recall, mean_avg_precision, mean_avg_fscore])
        elif year == '2006':
            list_2006.append([key, mean_avg_recall, mean_avg_precision, mean_avg_fscore])
        else:
            print "ERROR!"

        name = key if len(key)==7 else key+" "
        print name, "\t", mean_avg_recall, mean_avg_precision, mean_avg_fscore

# print system evaluations
list_2005.sort(key=lambda x: x[1], reverse=True)
list_2006.sort(key=lambda x: x[1], reverse=True)

csv_2005 = open('./evals/summary/output_2005.csv', 'wb')
csv_2006 = open('./evals/summary/output_2006.csv', 'wb')

write_out_2005 = writer(csv_2005, delimiter = ',')
write_out_2005.writerow(['System', 'MAR', 'MAP', 'MAF'])
write_out_2006 = writer(csv_2006, delimiter = ',')
write_out_2006.writerow(['System', 'MAR', 'MAP', 'MAF'])

for x in list_2005:
    write_out_2005.writerow(x)

for x in list_2006:
    write_out_2006.writerow(x)
