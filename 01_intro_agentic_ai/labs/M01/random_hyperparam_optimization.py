from sklearn.model_selection import RandomizedSearchCV


def random_hyperparam_optimization(model, param_distributions, X_train, y_train,
                                   cv=5, scoring='accuracy', n_iter=10, random_state=None, n_jobs=-1):
    """
    Perform random hyperparameter optimization for a given model using RandomizedSearchCV.

    Parameters:
    ----------
    model : estimator object
        The machine learning model instance (e.g., RandomForestClassifier) from sklearn.
    
    param_distributions : dict
        Dictionary with parameter names (str) as keys and lists of parameter settings to try as values.
        Each key-value pair defines one parameter and its possible values.
    
    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        The training input samples.
    
    y_train : array-like, shape (n_samples,)
        The target values (class labels) as integers or strings.
    
    cv : int, default=5
        Determines the cross-validation splitting strategy. Specify the number of folds.
    
    scoring : str or callable, default='accuracy'
        A string or a scorer callable object/function with signature `scorer(estimator, X, y)`.
    
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the search. Pass an int for reproducible output.
    
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.

    Returns:
    -------
    best_model : fitted estimator
        The model instance with the best found hyperparameters.

    best_params : dict
        Parameter setting that gave the best results on the hold out data.

    best_score : float
        Mean cross-validated score of the best_estimator.

    Example Usage:
    -------------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> param_distributions = {
    ...     'n_estimators': [10, 50, 100],
    ...     'max_depth': [5, 10, None]
    ... }
    >>> X_train, y_train = load_training_data()  # Replace with your data loading function
    >>> best_model, best_params, best_score = random_hyperparam_optimization(
    ...     model, param_distributions, X_train, y_train, cv=3, n_iter=5)
    >>> print(best_params)
    {'n_estimators': 50, 'max_depth': None}
    >>> print(best_score)
    0.85

    Edge Cases:
    ----------
    - If `param_distributions` is empty, RandomSearchCV will raise a ValueError.
    - Ensure `X_train` and `y_train` have matching dimensions and contain valid data.
    - If `n_iter` is larger than the possible number of parameter combinations, it might raise a warning but will still function.
    - Passing incorrect type for model or param_distributions will raise a TypeError.
    """

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    return best_model, best_params, best_score

import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError


class TestRandomHyperparamOptimization(unittest.TestCase):
    def setUp(self):
        # Create a toy dataset for testing
        self.X_train, self.y_train = make_classification(n_samples=100, n_features=10, random_state=42)

    def test_typical_usage(self):
        model = RandomForestClassifier()
        param_distributions = {'n_estimators': [10, 50], 'max_depth': [5, None]}
        best_model, best_params, best_score = random_hyperparam_optimization(
            model, param_distributions, self.X_train, self.y_train,
            cv=3, scoring='accuracy', n_iter=2, random_state=42
        )
        self.assertIn(best_params['n_estimators'], [10, 50])
        self.assertIn(best_params['max_depth'], [5, None])
        self.assertIsInstance(best_score, float)
        self.assertTrue(best_score >= 0)

    def test_empty_param_distributions(self):
        model = RandomForestClassifier()
        with self.assertRaises(ValueError):
            random_hyperparam_optimization(model, {}, self.X_train, self.y_train)

    def test_invalid_n_iter(self):
        model = RandomForestClassifier()
        param_distributions = {'n_estimators': [10]}
        with self.assertRaises(ValueError):
            random_hyperparam_optimization(
                model, param_distributions, self.X_train, self.y_train, n_iter=0
            )

    def test_invalid_model(self):
        param_distributions = {'n_estimators': [10]}
        with self.assertRaises(TypeError):
            random_hyperparam_optimization(
                None, param_distributions, self.X_train, self.y_train
            )

    def test_invalid_data(self):
        model = RandomForestClassifier()
        param_distributions = {'n_estimators': [10]}
        with self.assertRaises(ValueError):
            random_hyperparam_optimization(model, param_distributions, None, None)

    def test_invalid_param_distributions(self):
        model = RandomForestClassifier()
        with self.assertRaises(ValueError):
            random_hyperparam_optimization(model, {'invalid_param': None}, self.X_train, self.y_train)


if __name__ == '__main__':
    unittest.main()
