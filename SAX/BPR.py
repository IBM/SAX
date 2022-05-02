import numpy as np
from typing import List


class BusinessProcessRules:
    def __init__(self, features_names: List[str], nan_filler: int = -1) -> None:
        self.__feature_mapping = {features_names[i]: i for i in range(len(features_names))}
        self.__nan_filler = nan_filler

        # Preset conditions' values and relevant features' names : 
        self.__amount_mean = 1000
        self.__credit_score_mean = 700
        self.__risk_mean = 0.6
        self.__amount_feature = 'amount'
        self.__credit_score_feature = 'credit_score'
        self.__risk_feature = 'risk'
        self.__agent_feature = 'is_skilled'
        self.__assessment_feature = 'is_credit'
        return

    def __check_amount_matches_credit_check_or_risk(self, instance: List[float]) -> bool:
        checks = []
        if instance[self.__feature_mapping[self.__amount_feature]] >= self.__amount_mean:
            checks.append(instance[self.__feature_mapping[self.__assessment_feature]] == 1)  # BPR3
            checks.append((instance[self.__feature_mapping[self.__assessment_feature]] == 1) &
                          (instance[self.__feature_mapping[self.__credit_score_feature]] >= 0))  # BPR6
        else:
            checks.append(instance[self.__feature_mapping[self.__assessment_feature]] == 0)  # BPR4
            checks.append((instance[self.__feature_mapping[self.__assessment_feature]] == 0) &
                          (instance[self.__feature_mapping[self.__risk_feature]] >= 0))  # BPR7
        checks.append((instance[self.__feature_mapping[self.__credit_score_feature]] == self.__nan_filler) ^
                      (instance[self.__feature_mapping[self.__risk_feature]] == self.__nan_filler))  # BPR5
        return np.all(checks)

    def __check_agent_matches_credit_check_or_risk(self, instance: List[float]) -> bool:
        checks = []
        if 0 <= instance[self.__feature_mapping[self.__credit_score_feature]]:
            if instance[self.__feature_mapping[self.__credit_score_feature]] < self.__credit_score_mean:
                checks.append(instance[self.__feature_mapping[self.__agent_feature]] == 1)  # BPR8
            else:
                checks.append(instance[self.__feature_mapping[self.__agent_feature]] == 0)  # BPR9
            # print(f'agent_matches_credit_check_or_risk - credit_score : {checks}')
        elif 0 <= instance[self.__feature_mapping[self.__risk_feature]]:
            if instance[self.__feature_mapping[self.__risk_feature]] < self.__risk_mean:
                checks.append(instance[self.__feature_mapping[self.__agent_feature]] == 0)  # BPR10
            else:
                checks.append(instance[self.__feature_mapping[self.__agent_feature]] == 1)  # BPR11
            # print(f'agent_matches_credit_check_or_risk - risk : {checks}')
        return np.all(checks)

    def check_validity(self, instance: List[float]) -> bool:
        return self.__check_amount_matches_credit_check_or_risk(instance) & \
               self.__check_agent_matches_credit_check_or_risk(instance)

    def fix_sample(self, instance: List[float]) -> List[float]:
        new_inst = instance.copy()
        # Gateway I - amount :
        if new_inst[self.__feature_mapping[self.__amount_feature]] >= self.__amount_mean:
            new_inst[self.__feature_mapping[self.__assessment_feature]] = 1
            # Replace risk score with the NaN filler :
            new_inst[self.__feature_mapping[self.__risk_feature]] = self.__nan_filler
            # Gateway I.I - credit score :
            if new_inst[self.__feature_mapping[self.__credit_score_feature]] < self.__credit_score_mean:
                new_inst[self.__feature_mapping[self.__agent_feature]] = 1
            else:
                new_inst[self.__feature_mapping[self.__agent_feature]] = 0
        else:
            new_inst[self.__feature_mapping[self.__assessment_feature]] = 0
            # Replace credit score with the NaN filler :
            new_inst[self.__feature_mapping[self.__credit_score_feature]] = self.__nan_filler
            # Gateway II.I - risk score :
            if new_inst[self.__feature_mapping[self.__risk_feature]] < self.__risk_mean:
                new_inst[self.__feature_mapping[self.__agent_feature]] = 0
            else:
                new_inst[self.__feature_mapping[self.__agent_feature]] = 1
        return new_inst
