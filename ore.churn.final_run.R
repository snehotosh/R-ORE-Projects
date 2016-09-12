
# /u01/app/oracle/product/12.1.0.2/dbhome_1/R/library
setwd('/u01/rworkspace')


########################################################################################
############################     Connecting to ORE             #########################
########################################################################################

library(ORE)
ore.connect("rquser",service_name="orcl",host="localhost",password="welcome1", all=TRUE)
ore.ls()

########################################################################################
################     Importing all necessary libraries         #########################
########################################################################################
library(caTools)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ROCR)
library(ggplot2)
library(reshape2)
library(DMwR)
library(e1071)
library(gbm)

########################################################################################
#################################      PARAMETER         ###############################
########################################################################################

P_HIST_BP_COLS <- c('TOTAL_DAYCALLS','TOTAL_EVECALLS','TOTAL_INTLCALLS','NO_CUST_SRV_CALLS')

P_FORMULA <- formula(ISCHURN ~ INT_PLAN + VOICE_MAILPLAN + NO_VMAIL_MSG + TOTAL_DAYMINUTES + TOTAL_DAYCALLS + TOTAL_DAYCHARGE  + 
                       TOTAL_EVEMINUTES + TOTAL_EVECALLS + TOTAL_EVECHARGE  + TOTAL_NIGHTMINUTES + TOTAL_NIGHTCALLS + TOTAL_NIGHTCHARGE  + TOTAL_INTLMINUTES  + 
                       TOTAL_INTLCALLS  + TOTAL_INTLCHARGE + NO_CUST_SRV_CALLS)

P_XCOLS <- c('INT_PLAN','VOICE_MAILPLAN','TOTAL_DAYMINUTES','TOTAL_DAYCALLS','TOTAL_DAYCHARGE', 
             'TOTAL_EVEMINUTES','TOTAL_EVECALLS','TOTAL_EVECHARGE','TOTAL_NIGHTMINUTES','TOTAL_NIGHTCALLS',
             'TOTAL_NIGHTCHARGE','TOTAL_INTLMINUTES','TOTAL_INTLCALLS','TOTAL_INTLCHARGE','NO_CUST_SRV_CALLS')

resVar <- 'ISCHURN'

destFolder <- '/u01/rworkspace/ssiout'

########################################################################################
#################################      DESCRIPTIVE STAT  ###############################
########################################################################################

# --- Run Decriptive Statistic
source("/u01/rworkspace/ssiinitiative/ore.churn.desc.stat_lib.R")

# --- Run Descriptive Stats (TESTED FINE)
skewness_kurtosis_DF <- ore.doEval( FUN.NAME = "ore.churn.desc_stat",FUN.OWNER = "RQUSER",
            dataset = CHURN,
            Y = resVar,
            imputeType = 'knn',
            displaycols = P_HIST_BP_COLS,
            destLoc = destFolder,
            boxplot_halign = FALSE,
            ore.connect=TRUE
            )

# Display
skewness_kurtosis_DF

########################################################################################
#################################      MODEL RUNNING     ###############################
########################################################################################

# --- Run Decision Tree
source("/u01/rworkspace/ssiinitiative/ore.churn.mdl.DT_lib.R")

mod.DT.details <- ore.doEval(FUN.NAME = "ore.churn.DT",
                        FUN.OWNER = "RQUSER",
                        p_dataframe = CHURN,
                        pY = resVar,
                        p_imputetype = 'knn',
                        p_spltratio = 0.70,
                        p_formula = P_FORMULA,
                        p_storageDir = destFolder,
                        p_topn = 20,
                        ds.name = 'dsDTstore',
                        ore.connect=TRUE)

print(mod.DT.details)

# --- Run Random Forest
source("/u01/rworkspace/ssiinitiative/ore.churn.mdl.RF_lib.R")

mod.RF.details <- ore.doEval(FUN.NAME = "ore.churn.RF",
                             FUN.OWNER = "RQUSER",
                             p_dataframe = CHURN,
                             pY = resVar,
                             p_imputetype = 'knn',
                             p_spltratio = 0.70,
                             p_formula = P_FORMULA,
                             pntree = 1001,
                             p_topn = 20,
                             p_storageDir = destFolder,
                             ds.name = 'dsRFstore',
                             ore.connect=TRUE
                            )

mod.RF.details

# --- Run GBM (Gradient Boosting)
source("/u01/rworkspace/ssiinitiative/ore.churn.mdl.GBM_lib.R")

mod.GBM.details <- ore.doEval(FUN.NAME = "ore.churn.GBM",
                              FUN.OWNER = "RQUSER",
                              p_dataframe = CHURN,
                              pY = resVar,
                              p_imputetype = 'knn',
                              p_spltratio = 0.70,
                              p_formula = P_FORMULA,
                              p_tree=101,
                              p_shrinkage=0.01,
                              p_bagfrac=0.5,
                              p_train_frac=1,
                              p_cv=0,
                              p_verbose=FALSE,
                              p_dist = 'bernoulli',
                              p_idepth = 10,
                              p_minobs = 20,
                              storageDir = destFolder,
                              ds.name = 'dsGBMstore',
                              ore.connect = TRUE
                              )

print(mod.GBM.details)

class(mod.GBM.details)

# --- Model Comparison
df <- data.frame(rownames(ore.pull(mod.DT.details)),ore.pull(mod.DT.details),ore.pull(mod.RF.details),ore.pull(mod.GBM.details))
colnames(df) <- c('Metrics','DecisionTree','RandomForest','GradientBoostingTree')
dfcomp <- data.frame(df[1],df[2],df[3],df[4],row.names = NULL)
print(dfcomp)

# --- Checking all models are saved in ore datastore

ore.datastore(name = 'dsDTstore')
ore.datastore(name = 'dsRFstore')
ore.datastore(name = 'dsGBMstore')

# --- Checking all scripts are saved in the ore script repo

ore.scriptList(pattern = "ore.churn.")

# --- Disconnect
ore.disconnect()



