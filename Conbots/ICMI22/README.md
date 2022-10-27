# XGboost classifier for engagement in motor learning

## Paper abstract
`
Teaching motor skills such as playing music, handwriting, and driving, can greatly benefit from recently developed technologies such as wearable gloves for haptic feedback or robotic sensorimotor exoskeletons for the mediation of effective human-human and robot-human physical interactions. At the heart of such teacher-learner interactions still stands the critical role of the ongoing feedback a teacher can get about the student’s engagement status during the learning and practice sessions. Particularly for motor learning, such feedback is key in guiding the teacher on how to control and adapt the intensity of the physical interaction to best adapt it to the gradually evolving performance of the learner. In this paper, our focus is on the development of a near real-time machine-learning model that can acquire its input from a set of readily available noninvasive, privacy-preserving, body-worn sensors, for the benefit of tracking the engagement of the learner in the motor task. Such instrumentation is a first step toward the development of an exoskeleton control system that will use such feedback for personally adapted facilitation of motor learning sessions. We used the specific case of violin playing as a target domain in which data was empirically acquired, the latent construct of engagement in motor learning was carefully developed for data labeling, and a machine-learning model was rigorously trained and validated.
`



## The code
The code contains two functions: predict_from_existing_model, and train_new_model


### The predict_from_existing_model function
The functions upload a pre-trained XGboost classifer and use it to predict the given test set.

Input
  1.  pre-trained model 
  2.  numpy matrix of the test data. The rows of the matrix are the samples while the columns represents the sensors features
        The last column is the lable - the level of EML
        The features are (ordered) based on the specific model that is used
 
For the global model with all sensors (pre_trained_XGboost_model.json) the features are: 
```
IMU_B_1_jerk_rmssd , GYR_B_2_jerk_median , imu_b_jerk_magnitude_kurt , GYR_B_1_kurt , IMU_A_0_jerk_rmssd , gyr_a_jerk_magnitude_mean , IMU_A_0_rmssd , IMU_B_2_var , GYR_B_0_var , IMU_A_2_skew , GYR_B_1_jerk_var , imu_a_jerk_magnitude_rmssd , hrv_sdsd , IMU_B_1_jerk_kurt , GYR_B_2_jerk_kurt , hrv_total_power , GYR_A_2_jerk_skew , IMU_A_1_mean , IMU_B_1_skew , gyr_a_magnitude_median , IMU_A_1_jerk_median , IMU_A_1_jerk_skew , imu_a_jerk_magnitude_kurt , gyr_b_magnitude_var , IMU_B_0_kurt , imu_b_jerk_magnitude_skew , imu_a_magnitude_kurt , IMU_B_0_jerk_mean , GYR_A_2_jerk_median , GYR_B_2_jerk_skew , imu_b_magnitude_median , IMU_A_2_jerk_median , GYR_B_0_median , GYR_B_1_var , gsr_SHF_mean , IMU_A_1_iqr , gyr_b_jerk_magnitude_median , hrv_hf , IMU_B_1_jerk_mean , GYR_A_0_jerk_rmssd , gyr_a_jerk_magnitude_iqr , IMU_A_2_jerk_rmssd , IMU_B_1_kurt , GYR_B_2_rmssd , GYR_A_1_rmssd , GYR_A_2_mean , gyr_a_magnitude_kurt , IMU_B_0_jerk_median , GYR_A_1_iqr , IMU_B_2_jerk_rmssd , IMU_A_2_rmssd , gsr_SHS_var , IMU_B_0_jerk_var , gyr_b_magnitude_median , GYR_A_0_skew , IMU_A_1_jerk_rmssd , GYR_A_0_median , GYR_B_2_jerk_mean , imu_a_jerk_magnitude_var , IMU_A_2_jerk_iqr , IMU_B_2_rmssd , GYR_A_1_var , GYR_B_0_skew , gyr_a_jerk_magnitude_median , GYR_B_1_jerk_mean , IMU_A_1_jerk_iqr , gsr_SHA_mean , GYR_A_2_jerk_var , IMU_B_0_median , IMU_A_2_median , GYR_B_0_kurt , gyr_a_magnitude_skew , IMU_A_1_jerk_mean , imu_b_magnitude_kurt , GYR_A_1_jerk_var , IMU_B_0_skew , GYR_A_0_jerk_skew , GYR_A_1_jerk_skew , GYR_A_0_iqr , gyr_b_jerk_magnitude_mean , imu_a_magnitude_mean , GYR_A_0_jerk_mean , GYR_A_1_jerk_iqr , IMU_B_2_jerk_skew , GYR_B_0_rmssd , hrv_nni_50 , GYR_A_1_mean , imu_b_jerk_magnitude_var , GYR_B_1_jerk_kurt , imu_a_magnitude_rmssd , GYR_A_1_jerk_kurt , IMU_A_2_jerk_kurt , imu_a_magnitude_skew , GYR_B_2_mean , imu_b_magnitude_iqr , GYR_A_1_jerk_median , GYR_A_0_rmssd , IMU_A_2_jerk_skew , imu_b_jerk_magnitude_mean , gyr_b_jerk_magnitude_iqr , gyr_b_magnitude_mean , GYR_B_1_jerk_skew , hrv_pnni_50 , imu_a_jerk_magnitude_median , GYR_B_0_jerk_var , IMU_B_2_jerk_var , GYR_A_2_median , gyr_b_magnitude_kurt , hrv_sd1 , IMU_B_2_jerk_median , GYR_A_0_var , IMU_B_2_skew , imu_b_magnitude_rmssd , GYR_B_1_skew , GYR_B_1_median , GYR_B_1_jerk_rmssd , IMU_B_1_var , gyr_a_magnitude_var , GYR_A_1_kurt , GYR_B_0_jerk_kurt , imu_a_jerk_magnitude_skew , imu_a_magnitude_median , gyr_a_magnitude_mean , GYR_B_2_jerk_var , gyr_b_magnitude_rmssd , IMU_A_1_kurt , IMU_A_2_mean , IMU_A_2_iqr , hrv_range_nni , gsr_SHA_ae_2 , GYR_B_2_var , gyr_a_magnitude_rmssd , gyr_a_magnitude_iqr , hrv_mean_nni , IMU_B_1_jerk_skew , hrv_vlf , gsr_SHF_ae_3 , IMU_B_0_jerk_rmssd , IMU_B_1_iqr , IMU_A_0_kurt , IMU_A_2_var , gsr_SHA_ae_3 , IMU_A_2_kurt , IMU_B_2_jerk_mean , gsr_SHF_var , gsr_SHS_mean , IMU_B_2_iqr , GYR_B_2_jerk_rmssd , IMU_A_0_skew , imu_a_magnitude_var , IMU_A_1_skew , GYR_A_2_jerk_iqr , imu_b_jerk_magnitude_iqr , hrv_lf , IMU_B_2_median , IMU_B_0_var , GYR_A_2_jerk_rmssd , gsr_SHS_ae_3 , IMU_A_0_jerk_kurt , IMU_B_1_jerk_iqr , hrv_pnni_20 , GYR_B_0_mean , gyr_a_jerk_magnitude_kurt , IMU_A_0_var , GYR_B_0_iqr , GYR_A_0_jerk_kurt , GYR_A_2_rmssd , imu_a_jerk_magnitude_iqr , IMU_A_1_jerk_var , GYR_A_2_var , imu_b_magnitude_mean , GYR_B_2_kurt , GYR_B_0_jerk_median , IMU_B_1_rmssd , GYR_B_1_jerk_iqr , imu_b_jerk_magnitude_rmssd , IMU_B_0_rmssd , GYR_B_1_iqr , GYR_A_2_iqr , gsr_SHS_ae_2 , IMU_B_1_median , GYR_B_2_skew , GYR_B_1_mean , IMU_B_2_jerk_iqr , IMU_A_0_jerk_iqr , GYR_B_2_median , IMU_B_1_mean , GYR_B_1_rmssd , gyr_b_magnitude_skew , IMU_B_0_mean , gsr_SHA_ae_0 , GYR_B_1_jerk_median , IMU_B_2_jerk_kurt , gyr_b_jerk_magnitude_skew , gyr_a_jerk_magnitude_skew , IMU_A_0_mean , IMU_A_0_jerk_var , IMU_A_0_iqr , GYR_B_0_jerk_iqr , IMU_B_2_mean , IMU_A_1_jerk_kurt , IMU_A_1_var , gsr_SHF_ae_1 , gyr_b_magnitude_iqr , IMU_B_0_jerk_kurt , GYR_A_1_jerk_mean , GYR_A_1_median , IMU_B_0_jerk_skew , GYR_B_0_jerk_mean , GYR_B_2_jerk_iqr , IMU_A_0_median , IMU_B_0_iqr , IMU_B_0_jerk_iqr , gsr_SHA_ae_1 , IMU_B_1_jerk_median , gyr_b_jerk_magnitude_var , GYR_A_2_skew , gyr_a_jerk_magnitude_rmssd , imu_a_jerk_magnitude_mean , gyr_a_jerk_magnitude_var , imu_b_jerk_magnitude_median , GYR_A_0_jerk_var , IMU_A_1_rmssd , GYR_A_2_jerk_kurt , GYR_A_0_jerk_median , imu_b_magnitude_var , gyr_b_jerk_magnitude_rmssd , imu_b_magnitude_skew , hrv_sdnn , gsr_SHF_ae_2 , IMU_A_0_jerk_mean , GYR_A_0_kurt , gsr_SHS_ae_0 , IMU_B_1_jerk_var , GYR_A_2_kurt , gsr_SHA_var , IMU_A_0_jerk_median , IMU_A_1_median , gsr_SHF_ae_0 , GYR_B_0_jerk_skew , IMU_A_0_jerk_skew , imu_a_magnitude_iqr , GYR_A_1_jerk_rmssd , IMU_A_2_jerk_var , IMU_B_2_kurt , hrv_median_nni , GYR_A_0_jerk_iqr , hrv_rmssd , GYR_A_0_mean , gsr_SHS_ae_1 , GYR_A_2_jerk_mean , GYR_B_0_jerk_rmssd , hrv_nni_20 , GYR_B_2_iqr , GYR_A_1_skew , gyr_b_jerk_magnitude_kurt , IMU_A_2_jerk_mean , discomfort_bins, nirvana_bins
```


For the motion model (GYR+IMU),  the features are: 

```
'IMU_B_1_jerk_rmssd' 'GYR_B_2_jerk_median' 'imu_b_jerk_magnitude_kurt'
 'GYR_B_1_kurt' 'IMU_A_0_jerk_rmssd' 'gyr_a_jerk_magnitude_mean'
 'IMU_A_0_rmssd' 'IMU_B_2_var' 'GYR_B_0_var' 'IMU_A_2_skew'
 'GYR_B_1_jerk_var' 'imu_a_jerk_magnitude_rmssd' 'IMU_B_1_jerk_kurt'
 'GYR_B_2_jerk_kurt' 'GYR_A_2_jerk_skew' 'IMU_A_1_mean' 'IMU_B_1_skew'
 'gyr_a_magnitude_median' 'IMU_A_1_jerk_median' 'IMU_A_1_jerk_skew'
 'imu_a_jerk_magnitude_kurt' 'gyr_b_magnitude_var' 'IMU_B_0_kurt'
 'imu_b_jerk_magnitude_skew' 'imu_a_magnitude_kurt' 'IMU_B_0_jerk_mean'
 'GYR_A_2_jerk_median' 'GYR_B_2_jerk_skew' 'imu_b_magnitude_median'
 'IMU_A_2_jerk_median' 'GYR_B_0_median' 'GYR_B_1_var' 'IMU_A_1_iqr'
 'gyr_b_jerk_magnitude_median' 'IMU_B_1_jerk_mean' 'GYR_A_0_jerk_rmssd'
 'gyr_a_jerk_magnitude_iqr' 'IMU_A_2_jerk_rmssd' 'IMU_B_1_kurt'
 'GYR_B_2_rmssd' 'GYR_A_1_rmssd' 'GYR_A_2_mean' 'gyr_a_magnitude_kurt'
 'IMU_B_0_jerk_median' 'GYR_A_1_iqr' 'IMU_B_2_jerk_rmssd' 'IMU_A_2_rmssd'
 'IMU_B_0_jerk_var' 'gyr_b_magnitude_median' 'GYR_A_0_skew'
 'IMU_A_1_jerk_rmssd' 'GYR_A_0_median' 'GYR_B_2_jerk_mean'
 'imu_a_jerk_magnitude_var' 'IMU_A_2_jerk_iqr' 'IMU_B_2_rmssd'
 'GYR_A_1_var' 'GYR_B_0_skew' 'gyr_a_jerk_magnitude_median'
 'GYR_B_1_jerk_mean' 'IMU_A_1_jerk_iqr' 'GYR_A_2_jerk_var'
 'IMU_B_0_median' 'IMU_A_2_median' 'GYR_B_0_kurt' 'gyr_a_magnitude_skew'
 'IMU_A_1_jerk_mean' 'imu_b_magnitude_kurt' 'GYR_A_1_jerk_var'
 'IMU_B_0_skew' 'GYR_A_0_jerk_skew' 'GYR_A_1_jerk_skew' 'GYR_A_0_iqr'
 'gyr_b_jerk_magnitude_mean' 'imu_a_magnitude_mean' 'GYR_A_0_jerk_mean'
 'GYR_A_1_jerk_iqr' 'IMU_B_2_jerk_skew' 'GYR_B_0_rmssd' 'GYR_A_1_mean'
 'imu_b_jerk_magnitude_var' 'GYR_B_1_jerk_kurt' 'imu_a_magnitude_rmssd'
 'GYR_A_1_jerk_kurt' 'IMU_A_2_jerk_kurt' 'imu_a_magnitude_skew'
 'GYR_B_2_mean' 'imu_b_magnitude_iqr' 'GYR_A_1_jerk_median'
 'GYR_A_0_rmssd' 'IMU_A_2_jerk_skew' 'imu_b_jerk_magnitude_mean'
 'gyr_b_jerk_magnitude_iqr' 'gyr_b_magnitude_mean' 'GYR_B_1_jerk_skew'
 'imu_a_jerk_magnitude_median' 'GYR_B_0_jerk_var' 'IMU_B_2_jerk_var'
 'GYR_A_2_median' 'gyr_b_magnitude_kurt' 'IMU_B_2_jerk_median'
 'GYR_A_0_var' 'IMU_B_2_skew' 'imu_b_magnitude_rmssd' 'GYR_B_1_skew'
 'GYR_B_1_median' 'GYR_B_1_jerk_rmssd' 'IMU_B_1_var' 'gyr_a_magnitude_var'
 'GYR_A_1_kurt' 'GYR_B_0_jerk_kurt' 'imu_a_jerk_magnitude_skew'
 'imu_a_magnitude_median' 'gyr_a_magnitude_mean' 'GYR_B_2_jerk_var'
 'gyr_b_magnitude_rmssd' 'IMU_A_1_kurt' 'IMU_A_2_mean' 'IMU_A_2_iqr'
 'GYR_B_2_var' 'gyr_a_magnitude_rmssd' 'gyr_a_magnitude_iqr'
 'IMU_B_1_jerk_skew' 'IMU_B_0_jerk_rmssd' 'IMU_B_1_iqr' 'IMU_A_0_kurt'
 'IMU_A_2_var' 'IMU_A_2_kurt' 'IMU_B_2_jerk_mean' 'IMU_B_2_iqr'
 'GYR_B_2_jerk_rmssd' 'IMU_A_0_skew' 'imu_a_magnitude_var' 'IMU_A_1_skew'
 'GYR_A_2_jerk_iqr' 'imu_b_jerk_magnitude_iqr' 'IMU_B_2_median'
 'IMU_B_0_var' 'GYR_A_2_jerk_rmssd' 'IMU_A_0_jerk_kurt' 'IMU_B_1_jerk_iqr'
 'GYR_B_0_mean' 'gyr_a_jerk_magnitude_kurt' 'IMU_A_0_var' 'GYR_B_0_iqr'
 'GYR_A_0_jerk_kurt' 'GYR_A_2_rmssd' 'imu_a_jerk_magnitude_iqr'
 'IMU_A_1_jerk_var' 'GYR_A_2_var' 'imu_b_magnitude_mean' 'GYR_B_2_kurt'
 'GYR_B_0_jerk_median' 'IMU_B_1_rmssd' 'GYR_B_1_jerk_iqr'
 'imu_b_jerk_magnitude_rmssd' 'IMU_B_0_rmssd' 'GYR_B_1_iqr' 'GYR_A_2_iqr'
 'IMU_B_1_median' 'GYR_B_2_skew' 'GYR_B_1_mean' 'IMU_B_2_jerk_iqr'
 'IMU_A_0_jerk_iqr' 'GYR_B_2_median' 'IMU_B_1_mean' 'GYR_B_1_rmssd'
 'gyr_b_magnitude_skew' 'IMU_B_0_mean' 'GYR_B_1_jerk_median'
 'IMU_B_2_jerk_kurt' 'gyr_b_jerk_magnitude_skew'
 'gyr_a_jerk_magnitude_skew' 'IMU_A_0_mean' 'IMU_A_0_jerk_var'
 'IMU_A_0_iqr' 'GYR_B_0_jerk_iqr' 'IMU_B_2_mean' 'IMU_A_1_jerk_kurt'
 'IMU_A_1_var' 'gyr_b_magnitude_iqr' 'IMU_B_0_jerk_kurt'
 'GYR_A_1_jerk_mean' 'GYR_A_1_median' 'IMU_B_0_jerk_skew'
 'GYR_B_0_jerk_mean' 'GYR_B_2_jerk_iqr' 'IMU_A_0_median' 'IMU_B_0_iqr'
 'IMU_B_0_jerk_iqr' 'IMU_B_1_jerk_median' 'gyr_b_jerk_magnitude_var'
 'GYR_A_2_skew' 'gyr_a_jerk_magnitude_rmssd' 'imu_a_jerk_magnitude_mean'
 'gyr_a_jerk_magnitude_var' 'imu_b_jerk_magnitude_median'
 'GYR_A_0_jerk_var' 'IMU_A_1_rmssd' 'GYR_A_2_jerk_kurt'
 'GYR_A_0_jerk_median' 'imu_b_magnitude_var' 'gyr_b_jerk_magnitude_rmssd'
 'imu_b_magnitude_skew' 'IMU_A_0_jerk_mean' 'GYR_A_0_kurt'
 'IMU_B_1_jerk_var' 'GYR_A_2_kurt' 'IMU_A_0_jerk_median' 'IMU_A_1_median'
 'GYR_B_0_jerk_skew' 'IMU_A_0_jerk_skew' 'imu_a_magnitude_iqr'
 'GYR_A_1_jerk_rmssd' 'IMU_A_2_jerk_var' 'IMU_B_2_kurt' 'GYR_A_0_jerk_iqr'
 'GYR_A_0_mean' 'GYR_A_2_jerk_mean' 'GYR_B_0_jerk_rmssd' 'GYR_B_2_iqr'
 'GYR_A_1_skew' 'gyr_b_jerk_magnitude_kurt' 'IMU_A_2_jerk_mean'
```



For the biometric model (HRV+GSR) the features are: 

```
'hrv_sdsd' 'hrv_total_power' 'gsr_SHF_mean' 'hrv_hf' 'gsr_SHS_var'
 'gsr_SHA_mean' 'hrv_nni_50' 'hrv_pnni_50' 'hrv_sd1' 'hrv_range_nni'
 'gsr_SHA_ae_2' 'hrv_mean_nni' 'hrv_vlf' 'gsr_SHF_ae_3' 'gsr_SHA_ae_3'
 'gsr_SHF_var' 'gsr_SHS_mean' 'hrv_lf' 'gsr_SHS_ae_3' 'hrv_pnni_20'
 'gsr_SHS_ae_2' 'gsr_SHA_ae_0' 'gsr_SHF_ae_1' 'gsr_SHA_ae_1' 'hrv_sdnn'
 'gsr_SHF_ae_2' 'hrv_sd2' 'gsr_SHS_ae_0' 'gsr_SHA_var' 'gsr_SHF_ae_0'
 'hrv_median_nni' 'hrv_rmssd' 'gsr_SHS_ae_1' 'hrv_nni_20'
```


For the wrist motion model (GYR+IMU on wrist, jerk features only) the features are: 

```
'IMU_A_0_jerk_rmssd' 'gyr_a_jerk_magnitude_mean' 'GYR_A_2_jerk_skew'
 'IMU_A_1_jerk_median' 'IMU_A_1_jerk_skew' 'GYR_A_2_jerk_median'
 'IMU_A_2_jerk_median' 'GYR_A_0_jerk_rmssd' 'gyr_a_jerk_magnitude_iqr'
 'IMU_A_2_jerk_rmssd' 'IMU_A_1_jerk_rmssd' 'IMU_A_2_jerk_iqr'
 'gyr_a_jerk_magnitude_median' 'GYR_A_2_jerk_var' 'IMU_A_1_jerk_mean'
 'GYR_A_1_jerk_var' 'GYR_A_0_jerk_skew' 'GYR_A_1_jerk_skew'
 'GYR_A_0_jerk_mean' 'GYR_A_1_jerk_iqr' 'GYR_A_1_jerk_kurt'
 'IMU_A_2_jerk_kurt' 'GYR_A_1_jerk_median' 'IMU_A_2_jerk_skew'
 'GYR_A_2_jerk_iqr' 'GYR_A_2_jerk_rmssd' 'IMU_A_0_jerk_kurt'
 'gyr_a_jerk_magnitude_kurt' 'GYR_A_0_jerk_kurt' 'IMU_A_1_jerk_var'
 'IMU_A_0_jerk_iqr' 'gyr_a_jerk_magnitude_skew' 'IMU_A_0_jerk_var'
 'IMU_A_1_jerk_kurt' 'GYR_A_1_jerk_mean' 'gyr_a_jerk_magnitude_rmssd'
 'gyr_a_jerk_magnitude_var' 'GYR_A_0_jerk_var' 'GYR_A_2_jerk_kurt'
 'GYR_A_0_jerk_median' 'IMU_A_0_jerk_mean' 'IMU_A_0_jerk_median'
 'IMU_A_0_jerk_skew' 'GYR_A_1_jerk_rmssd' 'IMU_A_2_jerk_var'
 'GYR_A_0_jerk_iqr' 'GYR_A_2_jerk_mean' 'IMU_A_2_jerk_mean'
```







### The train_new_model function
This is the complete pipline process
Input:
  1. csv data file
  2. test set percentage (between 0 and 1)



    
## Software Requirements
```
numpy
sklearn
xgboost
```


## How to cite
Please consider citing our paper if you use code or ideas from this project:



