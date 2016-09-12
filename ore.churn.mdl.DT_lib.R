#################################################################
## Churn Prediction Library
## This is a final version of Decision Tree
#################################################################
if(nrow(ore.scriptList(name = "ore.churn.DT")) > 0)
{
  ore.scriptDrop("ore.churn.DT")
}

ore.scriptCreate("ore.churn.DT",
                 function(p_dataframe, 
                          pY,
                          p_imputetype = 'knn',
                          p_spltratio = 0.70,
                          p_formula,
                          p_storageDir,
                          p_topn = 20,
                          ds.name = 'dsDTstore',...)
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
                   ## 1. The function actually building the model
                   #-------------------------------------------------------------------------
                   mdl.training.DT <- function(trainingset,formulae,storageDir,topN,...)
                   {
                     # Loading the rpart library
                     library(rpart)
                     library(rpart.plot)      
                     
                     # Building Decision Tree Model
                     treemdl <- rpart(formula = formulae,data = trainingset,method = "class",control = rpart.control(minsplit = 2, minbucket = 1))
                     
                     # Plotting the Original Tree without Pruning
                     invisible(prp(treemdl, extra=2, uniform=T, branch=1, yesno=T, border.col=0, xsep="/",box.col=c("pink", "palegreen3")[treemdl$frame$yval],nn=TRUE,ni=TRUE,main="Original Tree"))
                     
                     # Plotting the complexity parameter - cp
                     plotcp(treemdl)    
                     
                     # Getting the cp dataframe
                     cpdf <- data.frame(printcp(treemdl))
                     
                     # Finding the optimum cp against min xerror
                     opticp <- treemdl$cptable[which.min(treemdl$cptable[,"xerror"]),"CP"]
                     print(opticp)
                     
                     # Tree Pruning
                     model.tree.fit <- prune(tree = treemdl,cp = opticp)
                     
                     # Ploting the Tree
                     # plot(model.tree.fit,uniform = TRUE,main = "Pruned Tree based on min CP")
                     # text(model.tree.fit, use.n = TRUE, cex = 0.75)
                     
                     # Checking the variable importance
                     mdl.dt.var.importance(modelfit = model.tree.fit,title = "DT Variable Importance",barcolor = "brown",topn = topN)
                     
                     ex_desc <- c('0 - No extra information (the default)',
                                  '1 - Display the number of obs that fall in the node',
                                  '2 - Class models: display the classification rate at the node',
                                  '3 - Class models: misclassification rate at the node',
                                  '4 - Class models: probability per class of obs in the node',
                                  '5 - Class models: like 4 but do not display the fitted class.',
                                  '6 - Class models: the prob of the second class only.',
                                  '7 - Class models: like 6 but do not display the fitted class.',
                                  '8 - Class models: the prob of the fitted class.',
                                  '9 - Class models: the prob times the fraction of obs in the node')    
                     
                     # Plotting Tree
                     plot.new()
                     text(0, .5, "Plotting Tree based on various display and details", pos=4, offset=1,font = 2,col = "red")
                     
                     for(i in c(1:9)){  
                       invisible(prp(model.tree.fit, extra=i, uniform=T, branch=1, yesno=T, border.col=0, xsep="/",box.col=c("pink", "palegreen3")[model.tree.fit$frame$yval],nn=TRUE,ni=TRUE,main=ex_desc[i]))
                       #print(pp)
                     }    
                     
                     # Plotting Tree based on node complexity
                     plot.new()
                     text(0, .5, "Plotting Tree based on node complexity", pos=4, offset=1,font = 2,col = "red")
                     
                     plot.tree.complex(modfit = model.tree.fit)    
                     #dev.off()
                     
                     return(model.tree.fit)
                   }
                   
                   #-------------------------------------------------------------------------
                   ## 2.Tree Pruning based on node complexity	
                   #-------------------------------------------------------------------------
                   plot.tree.complex <- function(modfit,...)
                   {           
                     complexities <- sort(unique(modfit$frame$complexity)) # a vector of complexity values
                     for(complexity in complexities) {
                       cols <- ifelse(modfit$frame$complexity >= complexity, 1, "grey")
                       dev.hold() # hold screen output to prevent flashing
                       invisible(prp(modfit,extra=2, uniform=T, branch=1, yesno=T, col=cols, branch.col=cols, split.col=cols,box.col=c("pink", "palegreen3")[modfit$frame$yval],nn=TRUE,ni=TRUE,main=paste("Tree Pruning on Node Complexity:",complexity)))
                       dev.flush()
                       Sys.sleep(1) # pause for one second    
                     }         
                   }
                   
                   #-------------------------------------------------------------------------
                   ## 3.Model Metrics Display
                   #-------------------------------------------------------------------------
                   mdl.Metric.display <- function(mdlpred,dataset,Y,storageDir,...)
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
                     
                     return(perf_metric)
                   }	
                   
                   #-------------------------------------------------------------------------
                   ## 4.Decision Tree variable Importance
                   #-------------------------------------------------------------------------
                   mdl.dt.var.importance <- function(modelfit,title,barcolor,topn,...){           
                     library(ggplot2)
                     
                     varimp_dt <- data.frame(varimp = modelfit['variable.importance'][[1]])
                     varimp_dt_df <- data.frame(variable = rownames(varimp_dt),varimp = varimp_dt[,'varimp'])  
                     varimp_dt_df$variable <- factor(varimp_dt_df[,'variable'],levels = varimp_dt_df[order(varimp_dt_df[,'varimp']),"variable"])  
                     
                     ## Printing the plot to PDF         
                     ret_df <- data.frame(head(varimp_dt_df[order(-varimp_dt_df$varimp),],topn),row.names = NULL)
                     
                     # creating Chart
                     gg <- ggplot(data = ret_df,aes(x=variable,y=varimp)) + geom_bar(stat='identity',fill=barcolor) + coord_flip() + ggtitle(title)
                     print(gg)
                     return(ret_df)
                   }
                   # -------------------------------------------------------------------------
                   ## 5.Model Decision Tree prediction
                   # -------------------------------------------------------------------------
                   mdl.pred.DT <- function(model,validationset,predtype='class',...)
                   {
                     # predicting on the validation Set
                     predtree <- predict(object = model,newdata = validationset,type = predtype)  
                     return(predtree)
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
                   
                   # -------------------------------------------------------------------
                   ## 10.Data Imputation
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
                   
                   ######################################################
                   ## THE MAIN FUNCTION ##
                   ## This is the Entry point
                   ######################################################                  	    
                   
                   dataset <- ore.pull(p_dataframe)
                   
                   ## Data splitting into training and validation set
                   splt <- data.split(df = dataset,resVal = pY,seedvalue = 1000,spltratio = p_spltratio)
                   
                   # Creation of training and validation set
                   training <- subset(x = dataset,splt == TRUE)
                   validation <- subset(x = dataset,splt == FALSE)
                   
                   # Imputing the NA on main dataset
                   training_imputed <- impute.data(data = training,imputetype = p_imputetype)
                   
                   print('#################################################')
                   print('***1.Model Building -  Decision Tree...')
                   print('#################################################')
                   
                   modDT <- mdl.training.DT(trainingset = training,formulae = p_formula,storageDir = p_storageDir,topN = p_topn)
                   cat('\n')
                   
                   print('***2.Model Decision Tree prediction...')
                   cat('\n')
                   predtree <- mdl.pred.DT(model = modDT,validationset = validation)
                   predtree_prob <- mdl.pred.DT(model = modDT,validationset = validation,predtype = 'prob')
                   
                   print('***3.Model Metrics...')
                   cat('\n')
                   mdl_metric_DT <- mdl.Metric.display(mdlpred = predtree,dataset = validation,Y = pY,storageDir = p_storageDir,...)
                   
                   cat('\n')
                   print('#################################################')
                   print('***.COMPLETED - Model Building -  Decision Tree...')
                   print('#################################################')
                   
                   print('Saving the DT Model')
                   save(modDT,file = paste0(p_storageDir,'/','trainedModDT.rda'))
                   
                   if (nrow(ore.datastore(name=ds.name)) > 0 ) 
                   {
                     ore.delete(name = ds.name)
                   }
                   ore.save(modDT,name = ds.name,append = TRUE)
                   
                   #return(mdl_metric_DT) 
                   mdl_metric_DT
                   
                 })
