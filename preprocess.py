import os
import nltk

__author__ = 'matteo'


ref = False
rivals = True


# references
if ref:
    year = 2006
    ref_in_path = "./data/references/"+str(year)
    ref_out_path = "./data2/references/"+str(year)+"/"
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    for filename in os.listdir(ref_in_path):

        content = open(ref_in_path+'/'+filename, 'r').read()

        code = filename.split(".")
        new_name = (code[0]+code[3]+"_"+code[4]).lower()
        out_file = open(ref_out_path+new_name, "w")

        for s in sent_detector.tokenize(content):
            out_file.write(s+"\n")

        out_file.close()


# other automatic summaries
if rivals:

    path_in = "./rivals/peers/"
    path_out = "./rivals2/peers/"

    for filename in os.listdir(path_in):

        content = open(path_in+filename, 'r').read()

        code = filename.split(".")
        new_name = (code[0]+code[3]+"_"+code[4]).lower()
        out_file = open(path_out+new_name, "w")

        out_file.write(content)
        out_file.close()

