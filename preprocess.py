import os
import re
import nltk
import random
import cPickle as pk
import xml.etree.ElementTree as ET

from collections import defaultdict


__author__ = 'matteo'


ref = False
rivals = False
LM = True
learnDic = False
random_UKN = True

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def clean(txt, stop=False, stem=False):
    txt = re.sub('"|\'|-|\||\n|<|>|\*|\\\\+', ' ', txt)
    txt = txt.lower()
    txt = ''.join(i for i in txt if not i.isdigit())
    txt = re.sub('\s+', ' ', txt)
    return txt.strip()

test_refs = ["d632i","d426a","D0628A","d324e","D0644H","D0601A","D0630C","D0647B","D0606F",
             "D0637A","D0607G","d407b","d346h","d393f","d350a","d400b","d332h","D0602B","D0623E"]


# references
if ref:

    year = 2005
    ref_in_path = "./data/references/"+str(year)
    ref_out_path = "./data2/references/"+str(year)+"/"
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    for filename in os.listdir(ref_in_path):

        content = open(ref_in_path+'/'+filename, 'r').read()
        code = filename.split(".")

        cond = code[0]+code[3] if year==2006 else (code[0]+code[3]).lower()

        if cond in test_refs:

            new_name = (code[0]+code[3]+"_"+code[4]).lower()
            out_file = open(ref_out_path+new_name, "w")

            for s in sent_detector.tokenize(content):
                out_file.write(s+"\n")

            out_file.close()


# other automatic summaries
if rivals:

    path_in = "./rivals/peers/"
    path_out = "./rivals2/peers/"
    count = 0
    test_refs_low = [x.lower() for x in test_refs]

    for filename in os.listdir(path_in):

        content = open(path_in+filename, 'r').read()
        code = filename.split(".")

        if (code[0]+code[3]).lower() in test_refs_low:

            new_name = (code[0]+code[3]+"_"+code[4]).lower()
            out_file = open(path_out+new_name, "w")

            out_file.write(content)
            out_file.close()


# generate single doc for training language models
if LM:

    path_dict = "./pickled_counts.pkl"

    path_in = "./data/collections/"
    path_out_train = "../neural-language-modelling/duc-train.txt"
    path_out_dev = "../neural-language-modelling/duc-dev.txt"
    path_out_test = "../neural-language-modelling/duc-test.txt"

    out_file_train = open(path_out_train, "w")
    out_file_dev = open(path_out_dev, "w")
    out_file_test = open(path_out_test, "w")

    dict = pk.load(open(path_dict,"r"))

    for year in [2005, 2006]:

        for foldername in os.listdir(path_in+str(year)):

            if foldername in ["duc2005_topics.sgml", "duc2006_topics.sgml"]:
                continue

            count = 0

            for filename in os.listdir(path_in+str(year)+"/"+foldername):

                count += 1

                if count <= 24:

                    root = ET.parse(path_in+str(year)+"/"+foldername+"/"+filename).getroot()
                    node = root.find('TEXT')

                    if node!=None:

                        content = root.find('TEXT').text
                        content = clean(content)
                        new = ""
                        for sent in sent_detector.tokenize(content):
                            if random_UKN:
                                new += ' '.join(["<UKN>" if (random.random()>0.98 and dict[word]>20 and dict[word]<484) else word for word in nltk.tokenize.word_tokenize(sent)])+"\n"
                        content = new

                        if count > 20 and count <=22 and random.random() < 0.7:
                            out_file_dev.write(content)
                        elif count > 22 and count <=24 and random.random() < 0.74:
                            out_file_test.write(content)
                        else:
                            out_file_train.write(content)
                else:
                    continue

    out_file_train.close()
    out_file_dev.close()
    out_file_test.close()


if learnDic:

    dict = defaultdict(int)
    path_in = "./data/collections/"

    for year in [2005, 2006]:

        for foldername in os.listdir(path_in+str(year)):

            if foldername in ["duc2005_topics.sgml", "duc2006_topics.sgml"]:
                continue

            for filename in os.listdir(path_in+str(year)+"/"+foldername):

                root = ET.parse(path_in+str(year)+"/"+foldername+"/"+filename).getroot()
                node = root.find('TEXT')

                if node!=None:

                    content = root.find('TEXT').text

                    for word in nltk.tokenize.word_tokenize(clean(content)):
                        dict[word] += 1

    pk.dump(dict, open("./pickled_counts.pkl","wb"))