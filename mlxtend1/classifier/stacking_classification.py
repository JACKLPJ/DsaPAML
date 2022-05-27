# Stacking classifier

# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import warnings
from scipy import sparse
from sklearn.base import TransformerMixin, clone
#from multiprocessing import Pool
from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from ._base_classification import _BaseStackingClassifier
from DsaPAML import SklearnModels_copy as sm 
#import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed


class StackingClassifier(_BaseXComposition, _BaseStackingClassifier,
                         TransformerMixin):

    """A Stacking classifier for scikit-learn estimators for classification.

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        A list of classifiers.
        Invoking the `fit` method on the `StackingClassifer` will fit clones
        of these original classifiers that will
        be stored in the class attribute
        `self.clfs_` if `use_clones=True` (default) and
        `fit_base_estimators=True` (default).
    meta_classifier : object
        The meta-classifier to be fitted on the ensemble of
        classifiers
    use_probas : bool (default: False)
        If True, trains meta-classifier based on predicted probabilities
        instead of class labels.
    drop_proba_col : string (default: None)
        Drops extra "probability" column in the feature set, because it is
        redundant:
        p(y_c) = 1 - p(y_1) + p(y_2) + ... + p(y_{c-1}).
        This can be useful for meta-classifiers that are sensitive to perfectly
        collinear features.
        If 'last', drops last probability column.
        If 'first', drops first probability column.
        Only relevant if `use_probas=True`.
    average_probas : bool (default: False)
        Averages the probabilities as meta features if `True`.
        Only relevant if `use_probas=True`.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2
    use_features_in_secondary : bool (default: False)
        If True, the meta-classifier will be trained both on the predictions
        of the original classifiers and the original dataset.
        If False, the meta-classifier will be trained only on the predictions
        of the original classifiers.
    store_train_meta_features : bool (default: False)
        If True, the meta-features computed from the training data used
        for fitting the meta-classifier stored in the
        `self.train_meta_features_` array, which can be
        accessed after calling `fit`.
    use_clones : bool (default: True)
        Clones the classifiers for stacking classification if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Hence, if use_clones=True, the original
        input classifiers will remain unmodified upon using the
        StackingClassifier's `fit` method.
        Setting `use_clones=False` is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.
    fit_base_estimators: bool (default: True)
        Refits classifiers in `classifiers` if True; uses references to the
        `classifiers`, otherwise (assumes that the classifiers were
        already fit).
        Note: fit_base_estimators=False will enforce use_clones to be False,
        and is incompatible to most scikit-learn wrappers!
        For instance, if any form of cross-validation is performed
        this would require the re-fitting classifiers to training folds, which
        would raise a NotFitterError if fit_base_estimators=False.
        (New in mlxtend v0.6.)

    Attributes
    ----------
    clfs_ : list, shape=[n_classifiers]
        Fitted classifiers (clones of the original classifiers)
    meta_clf_ : estimator
        Fitted meta-classifier (clone of the original meta-estimator)
    train_meta_features : numpy array, shape = [n_samples, n_classifiers]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and n_classifiers is the number of classfiers.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
    """

    def __init__(self, classifiers, meta_classifier, #preprocessing,
                 use_probas=False, drop_proba_col=None,
                 average_probas=False, verbose=0,
                 use_features_in_secondary=False,
                 store_train_meta_features=False,
                 use_clones=True, fit_base_estimators=True):
      #  multiprocessing.set_start_method('forkserver',force=True)
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas
       # self.preprocessing=preprocessing
        allowed = {None, 'first', 'last'}
        if drop_proba_col not in allowed:
            raise ValueError('`drop_proba_col` must be in %s. Got %s'
                             % (allowed, drop_proba_col))
        self.drop_proba_col = drop_proba_col

        self.average_probas = average_probas
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.use_clones = use_clones
        self.fit_base_estimators = fit_base_estimators

    @property
    def named_classifiers(self):
        return _name_estimators(self.classifiers)

    def fit(self, X, y, sample_weight=None):
        """ Fit ensemble classifers and the meta-classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        if not self.fit_base_estimators:
            warnings.warn("fit_base_estimators=False "
                          "enforces use_clones to be `False`")
            self.use_clones = False

        if self.use_clones:
            self.clfs_ = clone(self.classifiers)
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.clfs_ = self.classifiers
            self.meta_clf_ = self.meta_classifier

        if self.fit_base_estimators:
            if self.verbose > 0:
                print("Fitting %d classifiers..." % (len(self.classifiers)))
                
            for a in range(len(self.clfs_)):
                if sample_weight is None:
                    if self.preprocessing:
                        try:
                            self.clfs_[a][0] = self.clfs_[a][0].fit(sm.Preprocessing(X, self.clfs_[a][1]), y)
                        except:
                            self.clfs_.remove(self.clfs_[a])
                            continue
                        
                    else:
                        try:
                            self.clfs_[a][0] = self.clfs_[a][0].fit(X, y)
                        except:
                            self.clfs_.remove(self.clfs_[a])
                            continue
                        

                else:
                    try:
                        self.clfs_[a][0] = self.clfs_[a][0].fit(X, y, sample_weight)
                    except:
                        self.clfs_.remove(self.clfs_[a])
                        continue
                    
        meta_features = self.predict_meta_features(X)

        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features

        if not self.use_features_in_secondary:
            pass
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))

        if sample_weight is None:
            meta_features=meta_features[:, ~np.isnan(meta_features).any(axis=0)]
            self.meta_clf_.fit(meta_features, y)
        else:
            meta_features=meta_features[:, ~np.isnan(meta_features).any(axis=0)]
            self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

        return self

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        return self._get_params('named_classifiers', deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params('classifiers', 'named_classifiers', **params)
        return self

    def predict_meta_features(self, X):
        """ Get meta-features of test-data.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        meta-features : numpy array, shape = [n_samples, n_classifiers]
            Returns the meta-features for test data.

        """
        check_is_fitted(self, 'clfs_')
        
        if self.use_probas:
            executor = ThreadPoolExecutor()
            all_results = []
    
            for clf in self.clfs_:
                try:
                    all_results.append(executor.submit(clf[0].predict_proba, X))
                except:
                    continue
    
            if self.drop_proba_col == 'last':
                All_res = []
                for sub_res in as_completed(all_results):
                    try:
                        All_res.append(sub_res.result()[:, :-1])
                    except:
                        continue
                
                probas = np.asarray(All_res)
                
            elif self.drop_proba_col == 'first':
                All_res = []
                for sub_res in as_completed(all_results):
                    try:
                        All_res.append(sub_res.result()[:, 1:])
                    except:
                        continue
                
                probas = np.asarray(All_res)
                
               
            else:
                All_res = []
                for future in as_completed(all_results):
                    try:
                        a = future.result()
                        All_res.append(a)
                    except:
                        continue
    
                probas = np.asarray(All_res)
            if self.average_probas:
                vals = np.average(probas, axis=0)
            else:
                vals = np.concatenate(probas, axis=1)
        else:
            All_res = []
            for clf in self.clfs_:
                try:
                    All_res.append(clf[0].predict(X))
                except:
                    continue
            vals = np.column_stack(All_res)
              
        return vals