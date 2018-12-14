from sklearn.metrics import f1_score, r2_score, make_scorer
from sklearn import model_selection

def fit_classifier(X, y, classifier, parameters=None):
    """ 
    A pipeline to perform grid search over a set of 
    parameters for a classifier trained on the 
    input data [X, y]. 
    """
    # transform 'f1_score' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(f1_score)

    # create a decision tree regressor object
    classifier = classifier
    
    # create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    # cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)
    # instantiate cross validation sets object
    cv_sets = model_selection.ShuffleSplit(n_splits=10, test_size=0.20, random_state=123)

    # spot check on the first 1000 observations of the training set
    cv_results = model_selection.cross_val_score(classifier, X[:1000], y[:1000], cv=cv_sets, scoring=scoring_fnc)

    if parameters is not None:
        # create a dictionary for the parameter 'max_depth' with a range from 1 to 10
        params = parameters
        # create the grid search cv object --> GridSearchCV()
        grid = model_selection.GridSearchCV(classifier, params, scoring=scoring_fnc, cv=cv_sets)
        # fit the grid search object to the data to compute the optimal model
        grid = grid.fit(X, y)
        # return the optimal model after fitting the data
        return cv_results, grid.best_estimator_

    classifier = classifier.fit(X, y)

    return cv_results, classifier

def fit_regressor(X, y, regressor, parameters=None):
    """ 
    A pipeline to perform grid search over a set of 
    parameters for a regressor trained on the 
    input data [X, y]. 
    """
    # transform 'f1_score' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(r2_score)

    # create a decision tree regressor object
    regressor = regressor
    
    # create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    # cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)
    # instantiate cross validation sets object
    cv_sets = model_selection.ShuffleSplit(n_splits=10, test_size=0.20, random_state=123)

    # spot check on the first 1000 observations of the training set
    cv_results = model_selection.cross_val_score(regressor, X[:1000], y[:1000], cv=cv_sets, scoring=scoring_fnc)

    if parameters is not None:
        # create a dictionary for the parameter 'max_depth' with a range from 1 to 10
        params = parameters
        # create the grid search cv object --> GridSearchCV()
        grid = model_selection.GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
        # fit the grid search object to the data to compute the optimal model
        grid = grid.fit(X, y)
        # return the optimal model after fitting the data
        return cv_results, grid.best_estimator_

    regressor = regressor.fit(X, y)

    return cv_results, regressor
