#################################################################
## Gradient Boosting
## This is a final version
#################################################################

if(nrow(ore.scriptList(name = "ore.churn.GBM")) > 0)
{
  ore.scriptDrop("ore.churn.GBM")
}

ore.scriptCreate("ore.churn.GBM",
                 function(p_dataframe,
                          pY,
                          p_imputetype = 'knn',
                          p_spltratio = 0.70,
                          p_formula,
                          p_tree=6001,
                          p_shrinkage=0.01,
                          p_bagfrac=0.5,
                          p_train_frac=1,
                          p_cv=3,
                          p_verbose=FALSE,
                          p_dist = 'bernoulli',
                          p_idepth = 10,
                          p_minobs = 20,
                          storageDir,
                          ds.name = 'dsGBMstore')                             
                 {    
                   #pdf(paste0(storageDir,"/",pdfname,".pdf"))
                   
                   # Assigning the parameters pertaining to Input training and Validation Dataframe
                   
                   #-------------------------------------------------------------------------
                   ## 0.Data splitting into train and validation Set
                   #-------------------------------------------------------------------------
                   data.split <- function(df,resVal,seedvalue,spltratio)
                   {
                     library(caTools)
                     set.seed(100)
                     split <- sample.split(Y = df[ ,resVal],SplitRatio = spltratio)				  
                   }	
                   
                   # -------------------------------------------------------------------
                   ## 1.Data Imputation
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
                   
                   # ------------------------------------------------------------------------------
                   ## 6.Plot ROC
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
                   ## 7.Get AUC
                   # ------------------------------------------------------------------------------
                   
                   getAUC <- function(title,outcol,predcol,areaCol='blue') {
                     pred <- prediction(predcol,outcol)
                     perf <- performance(pred,'tpr','fpr')
                     auc <- as.numeric(performance(pred,'auc')@y.values)
                     auc
                   }
                   
                   # ------------------------------------------------------------------------------
                   ## 8.Gain Chart
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
                   
                   #########################################################    
                   ####			            MAIN METHOD				              ####
                   #########################################################
                   
                   dataset <- ore.pull(p_dataframe)
                   
                   ## Data splitting into training and validation set
                   splt <- data.split(df = dataset,resVal = pY,seedvalue = 1000,spltratio = p_spltratio)
                   
                   # Creation of training and validation set
                   training <- subset(x = dataset,splt == TRUE)
                   validation <- subset(x = dataset,splt == FALSE)
                   
                   # Imputing the NA on main dataset
                   training_imputed <- impute.data(data = training,imputetype = p_imputetype)                   
                   
                   # Assignment to prevent rest of the code change
                   trainingSet <- training  
                   validationSet <- validation      
                   
                   # Converting reponse variable to number
                   trainingSet[ ,pY] <- ifelse(trainingSet[,pY] == '1',1,0)
                   validationSet[,pY] <- ifelse(validationSet[,pY] == '1',1,0)
                   
                   # Loading GBM
                   library(gbm)
                   
                   # Building Model
                   modGBM <- gbm(formula = p_formula,
                                 data = trainingSet,                         
                                 distribution= p_dist, 
                                 n.trees=p_tree,
                                 shrinkage=p_shrinkage, 
                                 interaction.depth=p_idepth,
                                 bag.fraction = p_bagfrac, 
                                 train.fraction = p_train_frac,
                                 n.minobsinnode = p_minobs,
                                 cv.folds = p_cv,          
                                 verbose=p_verbose)
                   
                   # shows error in training and cv sets
                   if(p_cv > 1){
                     best.iter.cv <- gbm.perf(modGBM, method="cv")
                     print(paste('The Best no. of trees based on CV:',best.iter.cv))
                     summary(modGBM,n.trees=best.iter.cv) # based on the estimated best number of trees
                   }
                   
                   best.iter.oob <- gbm.perf(modGBM, method="OOB")
                   print(paste('The Best no. of trees based on OOB:',best.iter.oob))
                   
                   # plot the performance # plot variable influence            
                   summary(modGBM,n.trees=best.iter.oob) # based on the estimated best number of trees
                   
                   # look at the effect of each variable.Does it make sense?
                   for(i in 1:length(modGBM$var.names)){
                     plot(x = modGBM,i.var = i,n.trees = best.iter.oob,type = "response",main = "The chart")
                   }
                   
                   # prediction returning 1 or 0
                   validationSet$predGBM <- ifelse(predict.gbm(object = modGBM,newdata = validationSet[,-which(names(validationSet) == pY)],
                                                               n.trees = ifelse(p_cv > 0,best.iter.cv,best.iter.oob),
                                                               type = 'response')> 0.50,1,0)
                   
                   # prediction returning probability
                   validationSet$predGBMProb <- predict.gbm(object = modGBM,newdata = validationSet[,-which(names(validationSet) == pY)],
                                                            n.trees = ifelse(p_cv > 0,best.iter.cv,best.iter.oob),
                                                            type = 'response') 
                   
                   # Drawing Label Separation Density Curve
                   library(ggplot2)
                   validationSet$Label <- ifelse(validationSet[,pY]==1,TRUE,FALSE)
                   
                   str(validationSet)
                   
                   print(ggplot(data=validationSet,aes_string(x='predGBM',color='Label')) + geom_density() + ggtitle(label = 'Label Separation Density Curve'))
                   
                   # Derive confusion Matrix
                   confusionMatrix <- table(actual=relevel(factor(validationSet[,pY]),2),pred=relevel(factor(validationSet[,'predGBM']),2))
                   print('The Confusion Matrix:')
                   print(confusionMatrix)
                   
                   # Basic measures calculation from CM
                   
                   TP <- confusionMatrix[1,1]
                   TN <- confusionMatrix[2,2]
                   FN <- confusionMatrix[1,2]
                   FP <- confusionMatrix[2,1]
                   
                   TOT <- sum(confusionMatrix)
                   
                   # Measures
                   Accuracy <- round(as.numeric((TP + TN)/TOT),3)
                   ErrorRate <- 1 - Accuracy
                   FPR <- round(as.numeric(FP/(TN+FP)),3)
                   Recall <- round(as.numeric(TP/(TP+FN)),3)
                   Specificity <- round(as.numeric(TN/(TN + FP)),3)
                   Precision <- round(as.numeric(TP/(TP+FP)),3)
                   Fvalue <- round(as.numeric(2*Recall*Precision/(Recall + Precision)),3)
                   
                   cat('\n')
                   # print roc area
                   AUC <- gbm.roc.area(obs = as.numeric(validationSet[,pY]),pred = as.numeric(validationSet[,'predGBM']))
                   cat('\n')
                   print(paste('The Area Under The Curve(AUC):',AUC))
                   print(paste('The Gini Score:',2*AUC - 1))
                   
                   GiniCoeff <- 2*AUC - 1
                   
                   perf_metric <- rbind(Accuracy,ErrorRate,FPR,Recall,Specificity,Precision,Fvalue,AUC,GiniCoeff)
                   colnames(perf_metric) <- 'Perf Metrics'
                   perf_metric <- as.data.frame(perf_metric) 
                   
                   library(ROCR)
                   print(plotROC(title = 'ROC Chart',outcol = as.numeric(validationSet[,pY]),predcol = as.numeric(validationSet[,'predGBMProb']),areaCol = 'red'))
                   
                   # Gain Chart
                   print(gainCurve(truthcol = as.numeric(validationSet[,pY]),predcol = as.numeric(validationSet[,'predGBM']),title = 'Gain Chart'))
                   
                   
                   # Saving the Model
                   save(modGBM,file = paste0(storageDir,'/','trainedModGBM.rda'))
                   
                   if (nrow(ore.datastore(name=ds.name)) > 0 ) 
                   {
                     ore.delete(name = ds.name)
                   }
                   ore.save(modGBM,name = ds.name,append = TRUE)
                   #sink()
                   
                   return(perf_metric)
                 })
