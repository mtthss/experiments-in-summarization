import os
import nltk


__author__ = 'matteo'


ref = False
rivals = True
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

        if (code[0]+code[3]).lower() in test_refs:

            new_name = (code[0]+code[3]+"_"+code[4]).lower()
            out_file = open(path_out+new_name, "w")

            out_file.write(content)
            out_file.close()

