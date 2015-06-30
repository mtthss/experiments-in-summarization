__author__ = 'matteo'


# return partially applied function
def learn_relscore_function(X_rel, y, algorithm="svr"):

    score = lambda feat, weights: feat*weights

    if(algorithm=="svr"):
        # TODO partially apply weights learned from svr
        pass
    else:
        # TODO partially apply weights learned from svr
        pass

    return score

# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass


