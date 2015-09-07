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
my_candidates = ["LEAD", "RF-R-MMR", "RF-R", "RF-R-MMR-GROUPRNNEMBEDDING", "GBR-MMR-GROUPRNNEMBEDDING",     # multi-lead
                 "RF-R-22-MMR-CNN-0.1","RF-R-22-MMR-CNN-0.2","RF-R-22-MMR-CNN-0.3","RF-R-22-MMR-CNN-0.4","RF-R-22-MMR-CNN-0.5","RF-R-22-MMR-CNN-1","RF-R-22-MMR-CNN-2","RF-R-22-MMR-CNN-3","RF-R-22-MMR-CNN-0.05",
                 "RF-R-22-MMR-GROUPRNNEMBEDDING-0.2", "RF-R-22-MMR-GROUPRNNEMBEDDING-0.1", "RF-R-22-MMR-GROUPRNNEMBEDDING-0.05", "RF-R-22-MMR-GROUPRNNEMBEDDING-0.3", "RF-R-22-MMR-GROUPRNNEMBEDDING-0.4",
                 "RF-R-25-REL","RF-R-24-REL","RF-R-23-REL","RF-R-10-REL","RF-R-9-REL","RF-R-30-REL","RF-R-22-REL","RF-R-23-REL","RF-R-24-REL","RF-R-26-REL",
                 "RF-R-REL", "LINEAR-R-REL", "DECISIONT-REL", "GBR-REL", "KERNELRR-REL",        # relevance
                 "LINEAR-R-MMR-SIMPLERED",  "LINEAR-R-MMR-UNICOSRED",                           # mmr linear regression
                 "RF-R-MMR-SIMPLERED",      "RF-R-MMR-UNICOSRED",                               # mmr random forests
                 "DECISIONT-MMR-SIMPLERED", "DECISIONT-MMR-UNICOSRED"]                          # mmr decision trees

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
count2005 = 0
count2006 = 0
for filename in os.listdir(path_in):

    with open(path_in+filename) as csv:

        r = reader(csv)
        r.next()

        for line in r:

            collection = line[1]

            if len(test)==0 or collection.lower() in test:

                year = "2005" if len(collection)==5 else "2006"

                if year=="2005":
                    count2005 += 1
                elif year=="2006":
                    count2006 += 1
                system = line[2]
                avg_recall = line[3]
                avg_precision = line[4]
                avg_fscore = line[5]
                try:
                    dict[year+"-"+system].append([float(avg_recall),float(avg_precision), float(avg_fscore)])
                except:
                    pdb.set_trace()

print count2005
print count2006

count = 0
list_all = []
for key in dict.keys():

    year = key.split('-')[0]
    matrix = dict[key]
    if len(matrix)>0:

        count += len(matrix)
        recalls = [element[0] for element in matrix]
        precisions = [element[1] for element in matrix]
        fscores = [element[2] for element in matrix]

        mean_avg_recall = sum(recalls)/len(recalls)
        mean_avg_precision = sum(precisions)/len(precisions)
        mean_avg_fscore = sum(fscores)/len(fscores)
        list_all.append([key, mean_avg_recall, mean_avg_precision, mean_avg_fscore])

        name = key if len(key)==7 else key+" "
        print name, "\t", mean_avg_recall, mean_avg_precision, mean_avg_fscore

list_all.sort(key=lambda x: x[1], reverse=True)
csv_all = open('./evals/summary/output_all.csv', 'wb')

write_out_all = writer(csv_all, delimiter = ',')
write_out_all.writerow(['System', 'MAR', 'MAP', 'MAF'])

for x in list_all:
    write_out_all.writerow(x)

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

print len(list_2005)
print len(list_2006)
print len(list_all)