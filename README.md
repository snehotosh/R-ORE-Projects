This contains code for ORE for doing classification problem like churn (in this case) using:
1) UC1 - Employee Attrition
2) UC2 - Employee Performance
3) UC3 - Employee Profiling
4) UC4 - Payroll Cost
5) UC5 - Renege in Recruitment (Coming Soon)

Data Preparation
------------------
AA_SCORE_DATA_PREP_PKG - PLSQL for scoring data    (acting as main library)
AA_SCORE_DATA_RUN_PRC - PLSQL for scoring data (Calling API)            
AA_TRAINING_DATA_PREP_PKG - PLSQL for training data (acting as main library)
AA_TRAIN_DATA_RUN_PRC - PLSQL for training data    (Calling API)

ORE Model triggering
--------------------
AA_USECASE_ALGO_LIB_PKG    - PLSQL library package for model training
AA_USECASE_ALGO_RUN_PKG    - PLSQL run package for model training

ORE codes stored in Oracle DB
-----------------------------
CLASSRFCHURN - Whole ML workflow for Employee Performance
CLASSRFPERF - Whole ML workflow for Employee Performance   
CLUSTPROFILE - Clustering on employee data.Used PAM with Silhoutte co-eff to determine cluster.   
MODELPERFPRED - Code written to handle on employee performance
MODELRFPRED - Code written to handle Random Forest prediction on Churn
PAYROLLTS -  Time series code to analyse the payroll forecasting
