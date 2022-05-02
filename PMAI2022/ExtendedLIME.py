from typing import Tuple, List, Dict

import numpy as np
from lime import explanation
from lime.lime_tabular import LimeTabularExplainer, TableDomainMapper
from sklearn.metrics import pairwise_distances


class ExtendedLime(LimeTabularExplainer):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None,
                 nan_filler=-1,
                 bpr=None):
        super(ExtendedLime, self).__init__(training_data,
                                           mode=mode,
                                           training_labels=training_labels,
                                           feature_names=feature_names,
                                           categorical_features=categorical_features,
                                           categorical_names=categorical_names,
                                           kernel_width=kernel_width,
                                           kernel=kernel,
                                           verbose=verbose,
                                           class_names=class_names,
                                           feature_selection=feature_selection,
                                           discretize_continuous=discretize_continuous,
                                           discretizer=discretizer,
                                           sample_around_instance=sample_around_instance,
                                           random_state=random_state,
                                           training_data_stats=training_data_stats)

        # Set the defaults to the values for categorical features :
        self.scaler.mean_ = np.zeros(*self.scaler.mean_.shape)
        self.scaler.scale_ = np.ones(*self.scaler.scale_.shape)

        # Compute the values for numeric features :
        for i in range(self.scaler.scale_.shape[0]):
            if i not in self.categorical_features:
                self.scaler.mean_[i] = np.mean(training_data[:, i][training_data[:, i] != nan_filler])
                self.scaler.scale_[i] = np.std(training_data[:, i][training_data[:, i] != nan_filler])

        self.__random_seed = random_state
        self.__nan_filler = nan_filler
        # Initialize the BPR member :
        self.__bpr = bpr
        return

    def __generate_neighborhood(self,
                                data_row,
                                num_samples: int = 5000,
                                edit_samples: bool = False,
                                prune_samples: bool = False) -> Tuple[List[List[float]], List[List[float]]]:
        np.random.seed(self.__random_seed)
        dataset, dataset_inverse = self._LimeTabularExplainer__data_inverse(data_row,
                                                                            num_samples,
                                                                            sampling_method='gaussian')

        if edit_samples:
            # Do not perturbate features that are NaN (of NaN filler) - keep them NaNs also in the perturbations :
            for i in range(len(data_row)):
                if data_row[i] == self.__nan_filler:
                    dataset[:, i] = self.__nan_filler
                    dataset_inverse[:, i] = self.__nan_filler
            dataset_inverse = np.array([self.__bpr.fix_sample(v) for v in dataset_inverse])

        if prune_samples:
            valid_instances = [self.__bpr.check_validity(v) for v in dataset_inverse]
            dataset = dataset[np.where(valid_instances)[0]]
            dataset_inverse = dataset_inverse[np.where(valid_instances)[0], :]

        return dataset, dataset_inverse

    def __get_feature_importance(self, dataset: List[List[float]], dataset_inverse: List[List[float]], \
                                 predict_fn: callable, top_labels: int = 0) -> Dict[int, float]:
        scaled_data = (dataset - self.scaler.mean_) / self.scaler.scale_

        distances = pairwise_distances(scaled_data,
                                       scaled_data[0].reshape(1, -1),
                                       metric='euclidean').ravel()

        yss = predict_fn(dataset_inverse)
        np.random.seed(self.__random_seed)
        explnation_const_skill = self.base.explain_instance_with_data(neighborhood_data=scaled_data,
                                                                      neighborhood_labels=yss,
                                                                      distances=distances,
                                                                      label=1,
                                                                      num_features=5)

        feature_importance = {num: importance for (num, importance) in explnation_const_skill[1]}

        domain_mapper = TableDomainMapper(self.feature_names,
                                          feature_importance.values,
                                          scaled_data[0],
                                          categorical_features=self.categorical_features,
                                          discretized_feature_names=None,
                                          feature_indexes=feature_importance.keys)

        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)

        ret_exp.intercept[1] = explnation_const_skill[0]
        ret_exp.local_exp[1] = explnation_const_skill[1]
        ret_exp.score[1] = explnation_const_skill[2]
        ret_exp.local_pred[1] = explnation_const_skill[3]
        ret_exp.predict_proba = yss[0]

        return ret_exp

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         num_samples=5000,
                         bpr_edit: bool = False,
                         bpr_prune: bool = False):

        dataset, dataset_inverse = self.__generate_neighborhood(data_row, num_samples, bpr_edit, bpr_prune)
        return self.__get_feature_importance(dataset, dataset_inverse, predict_fn)
