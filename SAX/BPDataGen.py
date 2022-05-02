import numpy as np
import pandas as pd
from scipy.stats import norm


class BPDataGen():
    def __init__(self, 
                 num_samples: int=1000,
                 amount_threshold: float=1000, 
                 amount_sd: float=200,
                 credit_threshold: float=700,
                 cerdit_sd: float=200,
                 risk_threshold: float=0.6,
                 risk_sd = 0.15,
                 skilled_agent_reject_rate = 0.95,
                 novice_agent_reject_rate = 0.01,
                 random_seed: int=3) -> None:
        self.__num_samples = num_samples
        self.__amount_threshold = amount_threshold
        self.__amount_sd = amount_sd
        self.__credit_threshold = credit_threshold
        self.__cerdit_sd = cerdit_sd
        self.__risk_threshold = risk_threshold
        self.__risk_sd = risk_sd
        self.__random_seed = random_seed
        self.__skilled_agent_percentile = round(skilled_agent_reject_rate * 100)
        self.__novice_agent_percentile = round(novice_agent_reject_rate * 100)
        self.__data = None
        
        # Defining feature names :
        self.__done_receive_loan_feature = 'done_receive_loan'
        self.__done_verify_amount_feature = 'done_verify_amount'
        self.__done_credit_check_feature = 'done_credit_check'
        self.__done_risk_assessment_feature = 'done_risk_assessment'
        self.__amount_feature = 'amount'
        self.__risk_feature = 'risk'
        self.__credit_score_feature = 'credit_score'
        self.__done_skilled_agent_feature = 'done_skilled_agent'
        self.__done_novice_agent_feature = 'done_novice_agent'
        self.__label = 'done_accept'
        self.__norm_score_feature = 'total_score'
        self.__agent_feature = 'is_skilled'
        self.__assessment_feature = 'is_credit'

        return
    
    def __handle_boolean_indicators(self, 
               data: pd.DataFrame,
               inplace: bool=False) -> pd.DataFrame:
        if not inplace:
            data = data.copy()
        
        data[self.__done_receive_loan_feature] = True
        data[self.__done_verify_amount_feature] = True

        # do credit check if amount >= 1000 else do risk assessment
        data[self.__done_credit_check_feature] = np.where(data[self.__amount_feature] >= self.__amount_threshold, 
                                                          True, False)
        data[self.__done_risk_assessment_feature] = np.where(data[self.__amount_feature] < self.__amount_threshold, 
                                                             True, False)

        # if done credit check then keep credit score, otherwise change to none
        data[self.__credit_score_feature] = np.where(data[self.__done_credit_check_feature] == False, np.nan, 
                                                     data[self.__credit_score_feature])

        # if done risk assessment then keep risk score, otherwise change to none
        data[self.__risk_feature] = np.where(data[self.__done_risk_assessment_feature] == False, np.nan, 
                                             data[self.__risk_feature])

        data[self.__done_skilled_agent_feature] = False
        # if done_credit_check == True & score < 700 then review by skilled agent
        data[self.__done_skilled_agent_feature] = np.where((data[self.__done_credit_check_feature] == True) & 
                                                           (data[self.__credit_score_feature] < self.__credit_threshold), 
                                                           True, data[self.__done_skilled_agent_feature])
        # if done_risk_assessment == True & risk >= 0.6 also review by skilled agent
        data[self.__done_skilled_agent_feature] = np.where((data[self.__done_risk_assessment_feature] == True) & 
                                                           (data[self.__risk_feature] >= self.__risk_threshold), 
                                                           True, data[self.__done_skilled_agent_feature])

        data[self.__done_novice_agent_feature] = False
        # if done_credit_check == True & score >= 700 then review by novice agent
        data[self.__done_novice_agent_feature] = np.where((data[self.__done_credit_check_feature] == True) & 
                                                          (data[self.__credit_score_feature] >= self.__credit_threshold), 
                                                          True, data[self.__done_novice_agent_feature])
        # if done_risk_assessment == True & risk < 0.6 also review by novice agent
        data[self.__done_novice_agent_feature] = np.where((data[self.__done_risk_assessment_feature] == True) & 
                                                          (data[self.__risk_feature] < self.__risk_threshold), 
                                                          True, data[self.__done_novice_agent_feature])
        if inplace:
            return None
        else:
            return data
    
    def __compute_total_score_for_row(self, row, min_credit_score, max_credit_score, min_risk, max_risk) -> float:
        total_score = 0.
        if (True == row[self.__done_credit_check_feature]) & (row[self.__credit_score_feature] < self.__credit_threshold):
            total_score = (row[self.__credit_score_feature] - min_credit_score) / (self.__credit_threshold - min_credit_score)
        elif (True == row[self.__done_credit_check_feature]) & (row[self.__credit_score_feature] >= self.__credit_threshold):
            total_score = (row[self.__credit_score_feature] - self.__credit_threshold) / (max_credit_score - self.__credit_threshold)
        elif (True == row[self.__done_risk_assessment_feature]) & (row[self.__risk_feature] >= self.__risk_threshold):
            total_score = 1 - ((row[self.__risk_feature] - self.__risk_threshold) / (max_risk - self.__risk_threshold))
        elif (True == row[self.__done_risk_assessment_feature]) & (row[self.__risk_feature] < self.__risk_threshold):
            total_score = 1 - (row[self.__risk_feature] - min_risk) / (self.__risk_threshold - min_risk)
        return total_score
    
    def __compute_total_score(self, data: pd.DataFrame, inplace: bool=False) -> pd.DataFrame:
        if not inplace:
            data = data.copy()
        
        min_risk = np.min(data[self.__risk_feature])
        max_risk = np.max(data[self.__risk_feature])
        min_credit_score = np.min(data[self.__credit_score_feature])
        max_credit_score = np.max(data[self.__credit_score_feature])

        data[self.__norm_score_feature] = data.apply(lambda x: self.__compute_total_score_for_row(x, 
                                                                                                  min_credit_score, 
                                                                                                  max_credit_score, 
                                                                                                  min_risk, 
                                                                                                  max_risk), 
                                                     axis=1)

        if inplace:
            return None
        else:
            return data
    
    def __add_labels(self, data: pd.DataFrame, inplace: bool=False) -> pd.DataFrame:
        if not inplace:
            data = data.copy()
        
        novice_loans = data[data[self.__done_novice_agent_feature] == True]
        skilled_loans = data[data[self.__done_skilled_agent_feature] == True]
        skilled_acceptance_threshold = np.percentile(skilled_loans[self.__norm_score_feature], self.__skilled_agent_percentile)
        novice_acceptance_threshold = np.percentile(novice_loans[self.__norm_score_feature], self.__novice_agent_percentile)
        
        data[self.__label] = False
        # accept if reviewed by skilled & score > skilled threshold
        data[self.__label] = np.where((data[self.__done_skilled_agent_feature] == True) & 
                                      (data[self.__norm_score_feature] > skilled_acceptance_threshold), 
                                      True, data[self.__label])
        # accept if reviewed by novice & score > novice threshold
        data[self.__label] = np.where((data[self.__done_novice_agent_feature]==True) & 
                                      (data[self.__norm_score_feature] > novice_acceptance_threshold), 
                                      True, data[self.__label])

        if inplace:
            return None
        else:
            return data
    
    def __clean_data(self, data: pd.DataFrame, inplace: bool=False) -> pd.DataFrame:
        if not inplace:
            data = data.copy()
        
        data.rename(columns={self.__done_credit_check_feature: self.__assessment_feature, 
                             self.__done_skilled_agent_feature: self.__agent_feature}, inplace=True)
        data.drop(columns=[self.__done_receive_loan_feature, self.__done_verify_amount_feature, 
                           self.__done_risk_assessment_feature, self.__done_novice_agent_feature, 
                           self.__norm_score_feature], inplace=True)

        if inplace:
            return None
        else:
            return data
    
    def gen_data(self):
        # Generating random amounts, credit scores, and risk scores :
        np.random.seed(seed=self.__random_seed)
        amounts = norm.rvs(self.__amount_threshold, self.__amount_sd, self.__num_samples)
        credit_scores = norm.rvs(self.__credit_threshold, self.__cerdit_sd, self.__num_samples)
        risks = norm.rvs(self.__risk_threshold, self.__risk_sd, self.__num_samples)
        data = np.hstack((amounts.reshape(-1, 1), credit_scores.reshape(-1, 1), risks.reshape(-1, 1)))
        data = pd.DataFrame(data, columns=[self.__amount_feature, self.__credit_score_feature, self.__risk_feature])
        # Updating the data based on the business process rules :
        self.__handle_boolean_indicators(data, inplace=True)
        self.__compute_total_score(data, inplace=True)
        self.__add_labels(data, inplace=True)
        self.__clean_data(data, inplace=True)
        self.__data = data
        return
    
    def get_data(self):
        return self.__data
    
    def save_data_to_file(self, output_filename: str):
        self.__data.to_csv(output_filename, index=False)
