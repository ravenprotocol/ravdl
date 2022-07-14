import ravop as R

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    equality = R.equal(y_true, y_pred)
    accuracy = R.div(R.sum(equality, axis=0), R.index(R.shape(y_true),indices='[0]'))
    return accuracy