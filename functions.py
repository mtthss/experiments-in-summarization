__author__ = 'matteo'


# return partially applied function
def learn_relscore_function(X_rel, y, algorithm="svr"):

    score = lambda phi, w: sum(phi*w)

    # TODO python interfaces:
    # https://bitbucket.org/wcauchois/pysvmlight
    # https://pypi.python.org/pypi/svmlight

    if(algorithm=="svr"):
        # TODO http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        # TODO partially apply weights learned from svr
        pass
    elif(algorithm=="svm-rank"):
        # TODO http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
        # TODO partially apply weights learned from svm-rank
        pass
    else:
        raise Exception('Invalid algorithm')

    return score

# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass


