import ravop as R

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    equality = y_true() == y_pred()
    accuracy = R.div(R.sum(R.t(equality.tolist()), axis=0), R.t(len(y_true())))
    return accuracy