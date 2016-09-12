#################################################################
## Churn Prediction Library for BIW Offering
## This is a final version of RandomForest
## Done by Snehotosh
#################################################################

if(nrow(ore.scriptList(name = "ore.churn.RF")) > 0)
{
  ore.scriptDrop("ore.churn.RF")
}

ore.scriptCreate("ore.churn.RF",
                 function(p_dataframe,
                          pY,
                          p_imputetype = 'knn',
                          p_spltratio = 0.70,
                          p_formula,
                          pntree = 1001,
                          p_topn = 20,
                          p_storageDir,
                          ds.name = 'dsRFstore'
                 )
                 {
                   
                   #-------------------------------------------------------------------------
                   ## 0.Data splitting into train and validation Set
                   #-------------------------------------------------------------------------
                   data.split <- function(df,resVal,seedvalue,spltratio)
                   {
                     library(caTools)
                     set.seed(100)
                     split <- sample.split(Y = df[ ,resVal],SplitRatio = spltratio)				  
                   }     
                   
                   #-------------------------------------------------------------------------
                   ## 1.Building Random Forest Model
                   #-------------------------------------------------------------------------
                   mdl.training.RF <- function(p_trainx,p_trainy,p_ntree){
                     # Finding optimum mtry
                     library(randomForest)
                     trf <- tuneRF(x = p_trainx ,y = p_trainy, mtryStart=2, ntreeTry=1501, stepFactor=2, improve=0.05, trace=T, plot=FALSE)
                     opti_mtry <- trf[which.min(trf[ ,'OOBError'])]
                     
                     # Building Model
                     mdl <- randomForest(x = p_trainx,y = p_trainy,ntree=p_ntree,mtry=opti_mtry)  
                     return(mdl)
                   }		
                   
                   #-------------------------------------------------------------------------
                   ## 2.RandomForest variable Importance
                   #-------------------------------------------------------------------------
                   mdl.rf.var.importance <- function(modelfit,title,barcolor='grey',topn,storageDir,pdfname,...){  
                     #pdf(paste0(storageDir,"/",pdfname,".pdf"))
                     library(ggplot2)
                     rf_imp <- data.frame(variable = rownames(modelfit$importance),MeanDecreaseGini=modelfit$importance,row.names = NULL)
                     rf_imp$variable <- factor(rf_imp$variable,levels = rf_imp[order(rf_imp$MeanDecreaseGini),"variable"])      
                     ret_df <- data.frame(head(rf_imp[order(-rf_imp$MeanDecreaseGini),],topn),row.names = NULL)
                     
                     gg <- ggplot(data = ret_df,aes(x=variable,y=MeanDecreaseGini)) 
                     varImpchart <- gg + geom_bar(stat='identity',fill=barcolor) + coord_flip() + ggtitle(title)
                     print(varImpchart)
                     #dev.off()
                     #ggsave(filename = varImpchart,device = paste0(storageDir,'/',pdfname,'.pdf'))
                     return(ret_df)
                   }
                   
                   # -------------------------------------------------------------------------
                   ## 3.Model Decision Tree prediction
                   # -------------------------------------------------------------------------
                   mdl.pred.RF <- function(model,validationset,predtype='class',...)
                   {
                     # predicting on the validation Set
                     predtree <- predict(object = model,newdata = validationset,type = predtype)  
                     return(predtree)
                   }
                   
                   # ------------------------------------------------------------------------------
                   ## 4.Plot ROC
                   # ------------------------------------------------------------------------------
                   
                   plotROC <- function(title,outcol,predcol,areaCol='blue') {
                     pred <- prediction(predcol,outcol)
                     perf <- performance(pred,'tpr','fpr')
                     auc <- as.numeric(performance(pred,'auc')@y.values)
                     dframe <- data.frame(
                       FPR=perf@x.values[[1]],
                       TPR=perf@y.values[[1]])
                     plot=ggplot() +
                       geom_ribbon(data=dframe,aes(x=FPR,ymax=TPR,ymin=0), fill=areaCol,alpha=0.3) +
                       geom_point(data=dframe,aes(x=FPR,y=TPR)) + geom_line(aes(x=c(0,1),y=c(0,1))) + coord_fixed() +
                       ggtitle(paste(title,'\nAUC:',format(auc,digits=3)))
                     
                     plot
                     #list(df=dframe,plot=plot)
                   }
                   
                   # ------------------------------------------------------------------------------
                   ## 5.Get AUC
                   # ------------------------------------------------------------------------------
                   
                   getAUC <- function(title,outcol,predcol,areaCol='blue') {
                     pred <- prediction(predcol,outcol)
                     perf <- performance(pred,'tpr','fpr')
                     auc <- as.numeric(performance(pred,'auc')@y.values)
                     auc
                   }
                   
                   # ------------------------------------------------------------------------------
                   ## 6.Gain Chart
                   # ------------------------------------------------------------------------------
                   
                   areaCalc <- function(x,y) {
                     # append extra points to get rid of degenerate cases
                     x <- c(0,x,1)
                     y <- c(0,y,1)
                     n <- length(x)
                     sum(0.5*(y[-1]+y[-n])*(x[-1]-x[-n]))
                   }
                   
                   gainCurve = function(truthcol, predcol, title) {
                     library(reshape2)
                     truthcol <- as.numeric(truthcol)
                     # data frame of pred and truth, sorted in order of the predictions
                     d = data.frame(predcol=predcol,truthcol=truthcol)
                     predord = order(d[['predcol']], decreasing=TRUE) # reorder, with highest first
                     wizard = order(d[['truthcol']], decreasing=TRUE)
                     npop = dim(d)[1]
                     
                     # data frame the cumulative prediction/truth as a function
                     # of the fraction of the population we're considering, highest first
                     results = data.frame(pctpop= (1:npop)/npop,
                                          model = cumsum(d[predord,'truthcol'])/sum(d[['truthcol']]),
                                          wizard = cumsum(d[wizard, 'truthcol'])/sum(d[['truthcol']]))
                     
                     # calculate the areas under each curve
                     # gini score is 2* (area - 0.5)
                     idealArea = areaCalc(results$pctpop,results$wizard) - 0.5
                     modelArea = areaCalc(results$pctpop,results$model) - 0.5
                     giniScore = modelArea/idealArea # actually, normalized gini score
                     
                     # melt the frame into the tall form, for plotting
                     results = melt(results, id.vars="pctpop", measure.vars=c("model", "wizard"),
                                    variable.name="sort_criterion", value.name="pct_outcome")
                     
                     gplot = ggplot(data=results, aes(x=pctpop, y=pct_outcome, color=sort_criterion)) + 
                       geom_point() + geom_line() + 
                       geom_abline(color="gray") +
                       ggtitle(paste("Gain curve,", title, '\n', 
                                     'relative Gini score', format(giniScore,digits=2))) +
                       xlab("% items in score order") + ylab("% total category") +
                       scale_x_continuous(breaks=seq(0,1,0.1)) +
                       scale_y_continuous(breaks=seq(0,1,0.1)) +
                       scale_color_manual(values=c('model'='darkblue', 'wizard'='darkgreen'))
                     
                     gplot
                   }
                   
                   # ------------------------------------------------------------------------------
                   ## 7.Model Metrics Display
                   # ------------------------------------------------------------------------------
                   
                   mdl.Metric.display <- function(mdlpred,dataset,Y,storageDir,pdfname,...)
                   {
                     ## This code is written by Snehotosh Banerjee
                     ## Dated 4th Feb 2016
                     ## The function is calculating all required model performance metric
                     
                     ## Model Accuracy Measures
                     # 1.Accuracy = (TP+TN)/(TP+FN+FP+TN) (1)
                     # 2.Mis-classification Rate (Error Rate) = 1 - Accuracy
                     # 3.FP rate = FP/(TN+FP) (2)
                     # 4.TP rate = Recall = Sensitivity = TP/(TP+FN) (3)
                     # 5.Specificity
                     # 6.Precision = TP/(TP+FP) (4)
                     # 7.F - value = ((1+ß^2)*Recall*Precision) /(ß^2*Recall + Precision)
                     # 8.ROC Curve - AUC score
                     # 9.Cohen's Kappa
                     # 10.Null Error Rate (Accuracy Paradox)
                     # 11.Gini Coeff = 2*AUC - 1
                     
                     # Bulding Confusion Matrix      
                     confusionMatrix <- table(Actual = dataset[ ,Y],Pred = mdlpred)      
                     print('The Confusion Matrix')
                     cat('\n')
                     print(confusionMatrix)
                     cat('\n')
                     
                     # Calculating Accuracy
                     nlev <- nlevels(dataset[ ,Y])
                     
                     # Getting the NULL hypothesis level name
                     charlev <- levels(dataset[ ,Y])[1]
                     
                     # Basic measures calculation from CM
                     TP <- confusionMatrix[1,1]
                     TN <- confusionMatrix[nlev,nlev]
                     FN <- confusionMatrix[1,nlev]
                     FP <- confusionMatrix[nlev,1]
                     TOT <- sum(confusionMatrix)      
                     
                     # Measures
                     Accuracy <- round(as.numeric((TP + TN)/TOT),3)
                     ErrorRate <- 1 - Accuracy
                     FPR <- round(as.numeric(FP/(TN+FP)),3)
                     Recall <- round(as.numeric(TP/(TP+FN)),3)
                     Specificity <- round(as.numeric(TN/(TN + FP)),3)
                     Precision <- round(as.numeric(TP/(TP+FP)),3)
                     Fvalue <- round(as.numeric(2*Recall*Precision/(Recall + Precision)),3)
                     
                     # Writing to PDF
                     #pdf(paste0(storageDir,"/",pdfname,".pdf"))
                     
                     # Get AUC 
                     library(ROCR)
                     AUC <- getAUC(title = 'ROC',outcol = ifelse(dataset[,Y] == '1',1,0),predcol = ifelse(mdlpred == '1',1,0))
                     GiniCoeff <- round(2*as.numeric(AUC) - 1,3)
                     
                     # Plot ROC
                     library(ROCR)
                     print(plotROC(title = 'ROC Chart',outcol = ifelse(dataset[,Y] == '1',1,0),predcol = ifelse(mdlpred == '1',1,0),areaCol = 'red'))
                     
                     # Get the Gain Chart
                     print(gainCurve(truthcol = ifelse(dataset[,Y] == '1',1,0) ,predcol = ifelse(mdlpred == '1',1,0),title = 'Gain Chart'))
                     
                     perf_metric <- rbind(Accuracy,ErrorRate,FPR,Recall,Specificity,Precision,Fvalue,AUC,GiniCoeff)
                     colnames(perf_metric) <- 'Perf Metrics'
                     perf_metric <- as.data.frame(perf_metric)   
                     
                     #dev.off()
                     return(perf_metric)
                   }
                   
                   # -------------------------------------------------------------------
                   ## 8.Data Imputation
                   #--------------------------------------------------------------------
                   
                   impute.data <- function(data,imputetype='mean',...)
                   {
                     ## This function imputes NA by mean or median values
                     if(imputetype == 'mean'){
                       for (i in which(sapply(data, is.numeric))) {
                         data[is.na(data[, i]), i] <- mean(data[, i],  na.rm = TRUE)
                       }
                     } else if(imputetype == 'median') {
                       for (i in which(sapply(data, is.numeric))) {
                         data[is.na(data[, i]), i] <- median(data[, i],  na.rm = TRUE)
                       }
                     } else if(imputetype == 'knn'){
                       library(DMwR)      
                       data <- knnImputation(data = data)
                     }else{
                       stop("wrong imputation type.Only mean,median and knn is supported")
                     }
                     
                     return(data)
                   }
                   
                   # -------------------------------------------------------------------
                   ## 9.Converting formula to vectors of string
                   #--------------------------------------------------------------------	
                   
                   x_variables <- function(form,Y)
                   {
                     aa <- gsub(pattern =" ",replacement="",x=paste0(format(form), collapse = ""))
                     bb <- gsub(pattern = "[+~]",replacement=",",x=aa)
                     cc <- unlist(strsplit(x = bb,split = "[,]"))
                     xcols <- cc[!cc %in% c(Y)]			  
                     return(xcols)
                   }
                   
                   ##########################################################################
                   ################### Random Forest Model ##################################
                   ################### THE MAIN FUNCTION ####################################
                   
                   dataset <- ore.pull(p_dataframe)
                   
                   ## Data splitting into training and validation set
                   splt <- data.split(df = dataset,resVal = pY,seedvalue = 1000,spltratio = p_spltratio)
                   
                   # Creation of training and validation set
                   training <- subset(x = dataset,splt == TRUE)
                   validation <- subset(x = dataset,splt == FALSE)
                   
                   # Imputing the NA on main dataset
                   training_imputed <- impute.data(data = training,imputetype = p_imputetype)
                   
                   
                   print('#################################################')
                   print('***1.Model Building -  RandomForest...')
                   print('#################################################')
                   
                   p_trainx <- training_imputed[ ,x_variables(p_formula,pY)]
                   p_trainy <- training_imputed[,pY]
                   
                   print('***2.Training the RF Model')
                   mdlrf <- mdl.training.RF(p_trainx = p_trainx,p_trainy = p_trainy,p_ntree = pntree)
                   
                   cat('\n')
                   print('***3.Random Forest Model Summary')
                   summary(mdlrf)
                   
                   print('***4.Random Forest Variable Importance')
                   mdlRF_Imp <- mdl.rf.var.importance(modelfit = mdlrf,title = 'RF variable Importance',topn = p_topn,barcolor = 'brown',storageDir = p_storageDir,pdfname ='5.RF_VariableImp')
                   print(mdlRF_Imp)
                   cat('\n')
                   
                   print('####################################################')
                   print('***5.COMPLETED - Model Building -  Random Forest...')
                   print('####################################################')
                   
                   print('***6.Predicting the Model on Validation set')
                   predRF_raw <- mdl.pred.RF(model = mdlrf,validationset = validation,predtype = 'class')
                   predRF_prob <- mdl.pred.RF(model = mdlrf,validationset = validation,predtype = 'prob')
                   
                   pref_metricRF <- mdl.Metric.display(mdlpred = predRF_raw,dataset = validation,Y = pY,storageDir = p_storageDir,pdfname = '6.RF Metric')
                   
                   print('Saving the RF Model')
                   save(mdlrf,file = paste0(p_storageDir,'/','trainedModRF.rda'))
                   
                   if (nrow(ore.datastore(name=ds.name)) > 0 ) 
                   {
                     ore.delete(name = ds.name)
                   }
                   ore.save(mdlrf,name = ds.name,append = TRUE)
                   
                   return(pref_metricRF)   
                 })