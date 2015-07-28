import os


__author__ = 'matteo'


path_in = "./evals/rouge_output"
path_out = "./evals/summary/"

for filename in os.listdir(path_in):

    content = open(path_in+filename, 'r').read()