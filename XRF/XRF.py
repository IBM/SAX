from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import numpy as np
import threading
from joblib import Parallel
import random
import math
from deap import base
from deap import algorithms
from deap import creator
from deap import tools
import multiprocessing


def _accumulate_prediction(predict, weight, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += weight * prediction
        else:
            for i in range(len(out)):
                out[i] += weight * prediction[i]


class XRandomForestClassifier(RandomForestClassifier):
    def __init__(self,
                 n_estimators=100,
                 *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0,
                 max_samples=None,
                 xai_weight=0.1,  # From here on - parameters specific to this class
                 num_generations=40,
                 feature_preferences=None,
                 mutation_probability=0.4,
                 mating_probability=0.3,
                 normalize_weights=None,
                 target_function = 'Cosine',
                 performance_metric='accuracy',
                 deap_parallelize=True,
                 ext_verbose = 0):

        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         class_weight=class_weight,
                         ccp_alpha=ccp_alpha,
                         max_samples=max_samples)

        self.xai_weight = xai_weight
        self._e_weight = np.ones(n_estimators).astype(float) / n_estimators
        self.feature_preferences = feature_preferences
        self.num_generations = num_generations
        self.mutation_probability = mutation_probability
        self.mating_probability = mating_probability
        self.normalize_weights = normalize_weights
        self.target_function = target_function
        self.performance_metric = performance_metric
        self.ext_verbose = ext_verbose
        self.deap_parallelize = deap_parallelize

        return
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return super().get_params(deep)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_accumulate_prediction)(e.predict_proba, self._e_weight[i], X, all_proba, lock)
            for (i, e) in enumerate(self.estimators_)
        )

        for proba in all_proba:
            proba /= sum(self._e_weight)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def xai_score(self):
        if self.feature_preferences is None:
            scr = 0
        elif len(self.feature_preferences) != len(self.feature_importances_):
            scr = 0
        else:
            if self.target_function == 'Cosine':
                scr = (cosine_similarity(np.array(self.feature_preferences).reshape(1, -1), \
                                         np.array(self.feature_importances_).reshape(1, -1))[0][0] + 1) / 2
            elif self.target_function == 'CrossEntropy':
                scr = -np.log2(np.array(self.feature_importances_))*np.array(self.feature_preferences)
                scr = scr.sum()
                # Transform the CE distance to a similarity measure (s = 1/(d+1)):
                scr = 1 / (scr + 1)
            if 2 <= self.ext_verbose:
                print(f'xai_score={scr} | feature_importances_={self.feature_importances_}')
        return scr

    def score(self, X, y, sample_weight=None):
        if 'f1_score' == self.performance_metric:
            return f1_score(y, self.predict(X), sample_weight=sample_weight, average='macro')
        else:  # default ('accuracy')
            return super().score(X, y, sample_weight)

    def objective_function(self, e_weights):
        if self.normalize_weights == 'Softmax':
            e_weights = [math.e**w for w in e_weights]
            e_weights /= np.sum(e_weights)
        if self.normalize_weights == 'Abs':
            e_weights = [abs(w) for w in e_weights]
            e_weights /= np.sum(e_weights)
        self._e_weight = e_weights
        score = self.score(self.X, self.y) + self.xai_weight * self.xai_score()
        return [score]

    def __mutate(self, mean, sigma):
        return lambda individual, indpb: tools.mutGaussian(individual, mean, sigma, indpb)

    def __weights_search(self, X, y):
        self.X = X
        self.y = y
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        tbx = base.Toolbox()
        INDIVIDUAL_SIZE = self.n_estimators

        random.seed(self.random_state)
        tbx.register("attr_int", random.uniform, 0, 1)
        tbx.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     tbx.attr_int,
                     n=INDIVIDUAL_SIZE)
        tbx.register("population", tools.initRepeat, list, tbx.individual)
        tbx.register("evaluate", self.objective_function)
        tbx.register("mate", tools.cxOnePoint)
        tbx.register("mutate", self.__mutate(0.5, 0.05), indpb=0.01)
        tbx.register("select", tools.selTournament, tournsize=5)

        pop = tbx.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        if self.deap_parallelize:
            pool = multiprocessing.Pool()
            tbx.register("map", pool.map)
        pop, log = algorithms.eaSimple(pop, 
                                       tbx, 
                                       cxpb=self.mating_probability, 
                                       mutpb=self.mutation_probability, 
                                       ngen=self.num_generations,
                                       stats=stats, 
                                       halloffame=hof, 
                                       verbose=self.ext_verbose)
        if self.deap_parallelize:
            pool.close()
        del self.X
        del self.y
        weights = hof.items[0]
        if self.normalize_weights == 'Softmax':
            weights = [math.e**w for w in weights]
            weights /= np.sum(weights)
        elif self.normalize_weights == 'Abs':
            weights = [abs(w) for w in weights]
            weights /= np.sum(weights)
        if 2 <= self.ext_verbose:
            print(weights)
        return weights

    def fit(self, X, y, sample_weight=None) -> 'XRandomForestClassifier':
        _ = super().fit(X, y, sample_weight)
        self._e_weight = np.ones(self.n_estimators).astype(float) / self.n_estimators
        self._e_weight /= sum(self._e_weight)
        max_target = self.score(X, y) + self.xai_weight * self.xai_score()
        if 2 <= self.ext_verbose:
            print(f'Score for initial weights ({self._e_weight}): {max_target}')
        if 0 < self.num_generations:
            self._e_weight = self.__weights_search(X, y)
        return self

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(
            n_jobs=self.n_jobs, **_joblib_parallel_args(prefer="threads")
        )(
            delayed(getattr)(tree, "feature_importances_")
            for tree in self.estimators_
            if tree.tree_.node_count > 1
        )

        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.average(all_importances, weights=self._e_weight, axis=0)
        return all_importances / np.sum(all_importances)
