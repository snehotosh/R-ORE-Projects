#################################################################
## Churn Prediction Library
## This is a final version of Decision Tree
## Done by Snehotosh
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
                   mdl.training.DT <- function(trainingset,formulae,storageDir,topN)
                   {
                     # Loading the rpart library
                     library(rpart)
                     library(rpart.plot)      
                     
                     # Building Decision Tree Model
                     treemdl <- rpart(formula = formulae,data = trainingset,method = "class",control = rpart.control(minsplit = 2, minbucket = 1))
                     
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
                     prp(model.tree.fit,uniform = TRUE,main = paste("Pruned Tree based on min CP:", toString(opticp)))
                     #text(model.tree.fit, use.n = TRUE, cex = 0.75)
                     
                     # Checking the variable importance
                     mdl.dt.var.importance(modelfit = model.tree.fit,title = "DT Variable Importance",barcolor = "brown",topn = topN)                     
                     
                     return(model.tree.fit)
                   }     
                   
                   #-------------------------------------------------------------------------
                   ## 3.Model Metrics Display
                   #-------------------------------------------------------------------------
                   metricROCR <- function(model,testdata,pY,ds.name)
                   {
                     library(ROCR)
                     library(ggplot2)
                     
                     # Predict
                     predDT <- predict(object = model,newdata = testdata,type = "class")
                     predDTprob <- predict(object = model,newdata = testdata,type = "prob")
                     
                     # Creating predicted probability and actual label predicted dataframe
                     pred.df <- data.frame(predicted=as.double(predDTprob[,1]),actual=as.numeric(ifelse(testdata[,pY]=="Y",1,0)))
                     pred.df <- pred.df[order(pred.df$predicted, decreasing=TRUE), ]
                     
                     # Calculate ROCR Prediction
                     pred.rocr <- prediction(pred.df$predicted, pred.df$actual)
                     
                     # Stats
                     roc.perf <- performance(pred.rocr, measure = "tpr", x.measure = "fpr")
                     tpr.perf <- performance(pred.rocr, measure = "tpr")
                     fpr.perf <- performance(pred.rocr, measure = "fpr")
                     fnr.perf <- performance(pred.rocr, measure = "fnr")
                     tnr.perf <- performance(pred.rocr, measure = "tnr")
                     recall.perf <- performance(pred.rocr, measure = "prec", x.measure = "rec")
                     sensspec.perf <- performance(pred.rocr, measure = "sens", x.measure = "spec")
                     lift.perf <- performance(pred.rocr, measure = "lift", x.measure = "rpp")
                     auc.perf <- performance(pred.rocr, measure = "auc")
                     accuracy.perf <- performance(pred.rocr, measure = "acc")
                     err.perf <- performance(pred.rocr, measure = "err")
                     calibration.perf <- performance(pred.rocr, measure = "cal")
                     pcmiss.perf <- performance(pred.rocr,"pcmiss","lift")
                     prbe.perf <- performance(pred.rocr, "prbe")
                     
                     ## Scores ##
                     
                     # AUC Score
                     auc_score <- auc.perf@y.values[[1]]
                     
                     # Precision/Recall breakeven Score
                     prbe.score <- prbe.perf@x.values[[1]]
                     
                     # Accuracy Rate Score
                     acc_rate <- max(accuracy.perf@y.values[[1]])
                     
                     # Accuracy Rate gt 50% Score
                     acc_roc_gt_50 <- accuracy.perf@y.values[[1]][max(accuracy.perf@x.values[[1]] > 0.5)]
                     
                     # Error Rate Score
                     error_rate <- min(err.perf@y.values[[1]])
                     
                     ## Various Metric Plots ##
                     # pos/neg densities
                     ggplot(data=pred.df,aes(x=predicted)) + geom_density(aes(fill=factor(actual)), size=1, alpha=.3) +
                       scale_x_continuous("Predicted", breaks=(0:4)/4, limits=c(0,1), labels=sprintf("%d%%", (0:4)*25)) +
                       scale_y_sqrt("Density") + scale_fill_manual(values = c("red","blue")) + ggtitle(label = "Label Separation Density Curve")
                     
                     # Draw ROC curve
                     plot(roc.perf, main="ROC with Convex Hull", colorize=TRUE,print.cutoffs.at = seq(0.1, 0.9, 0.1), lwd = 2)
                     ch = performance(pred.rocr , "rch")
                     plot(ch, add = TRUE, lty = 2)
                     
                     # Recall-Precision Plot
                     plot(recall.perf,colorize = T,print.cutoffs.at = seq(0.1, 0.9, 0.1), lwd = 2,main = "Recall-Precision Plot")
                     
                     # Sensitivity-Specificity Plot
                     plot(sensspec.perf,colorize = T,print.cutoffs.at = seq(0.1, 0.9, 0.1), lwd = 2,main = "Sensitivity vs Specificity")
                     
                     # Lift Plot
                     plot(lift.perf,colorize = T,print.cutoffs.at = seq(0.1, 0.9, 0.1), lwd = 2,main = "Lift Plot")
                     
                     # Accuracy - Boxplot (Spread)
                     plot(accuracy.perf, avg= "vertical", spread.estimate="boxplot", show.spread.at= seq(0.1, 1.0, by=0.1),main = "Accuracy - Boxplot (Spread)")
                     
                     # Accuracy vs Cutoff
                     # Get the cutoff for the best accuracy
                     bestAccInd <- which.max(accuracy.perf@"y.values"[[1]])
                     bestMsg <- paste("best accuracy=", accuracy.perf@"y.values"[[1]][bestAccInd],"at cutoff=", round(accuracy.perf@"x.values"[[1]][bestAccInd], 4))
                     plot(accuracy.perf, sub=bestMsg,main = "Accuracy vs Cutoff")
                     
                     # TPR vs Cutoff
                     plot(tpr.perf,main = "TPR vs Cutoff")
                     
                     # TNR vs Cutoff
                     plot(tnr.perf,main = "TNR vs Cutoff")
                     
                     # FPR vs Cutoff
                     plot(fpr.perf,main = "FPR vs Cutoff")
                     
                     # FNR vs Cutoff
                     plot(fnr.perf,main = "FNR vs Cutoff")
                     
                     # Prediction-conditioned miss 
                     plot(pcmiss.perf, colorize=T, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(1.2,1.2), avg="threshold", lwd=3)
                     
                     # Confusion Matrix
                     confusionMatrix <- table(predDT ,testdata[,pY])
                     confusionMatrix
                     
                     ore.save(confusionMatrix,name = ds.name,append = TRUE)
                     
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
                     
                     df <- rbind(auc_score = auc_score,prbe_score = prbe.score,acc_rocr = acc_rate,accuracy = Accuracy,acc_rocr_gt50 = acc_roc_gt_50,error_rate_rocr = error_rate,error_rate = ErrorRate,fpr = FPR,recall = Recall,specificity = Specificity,precision = Precision,fval = Fvalue)
                     #colnames(df) <- "scores"
                     perf_metric <- data.frame(name=rownames(df),score=df,row.names = NULL)   
                     perf_metric
                   }                
                   
                   #-------------------------------------------------------------------------
                   ## 4.Decision Tree variable Importance
                   #-------------------------------------------------------------------------
                   mdl.dt.var.importance <- function(modelfit,title,barcolor,topn){           
                     library(ggplot2)
                     
                     varimp_dt <- modelfit["variable.importance"][[1]]
                     varimp_dt_df <- data.frame(variable = names(varimp_dt),varimp = varimp_dt)  
                     varimp_dt_df$variable <- factor(varimp_dt_df[,"variable"],levels = varimp_dt_df[order(varimp_dt_df[,"varimp"]),"variable"])  
                     
                     ## Printing the plot to PDF         
                     ret_df <- data.frame(head(varimp_dt_df[order(-varimp_dt_df$varimp),],topn),row.names = NULL)
                     
                     # creating Chart
                     gg <- ggplot(data = ret_df,aes(x=variable,y=varimp)) + geom_bar(stat="identity",fill=barcolor) + coord_flip() + ggtitle(title)
                     print(gg)
                     return(ret_df)
                   }
                   # -------------------------------------------------------------------------
                   ## 5.Model Decision Tree prediction
                   # -------------------------------------------------------------------------
                   mdl.pred.DT <- function(model,validationset,predtype="class")
                   {
                     # predicting on the validation Set
                     predtree <- predict(object = model,newdata = validationset,type = predtype)  
                     return(predtree)
                   }    
                   
                   # -------------------------------------------------------------------
                   ## 10.Data Imputation
                   #--------------------------------------------------------------------
                   
                   impute.data <- function(data,imputetype="mean")
                   {
                     ## This function imputes NA by mean or median values
                     if(imputetype == "mean"){
                       for (i in which(sapply(data, is.numeric))) {
                         data[is.na(data[, i]), i] <- mean(data[, i],  na.rm = TRUE)
                       }
                     } else if(imputetype == "median") {
                       for (i in which(sapply(data, is.numeric))) {
                         data[is.na(data[, i]), i] <- median(data[, i],  na.rm = TRUE)
                       }
                     } else if(imputetype == "knn"){
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
                   
                   # Pulling dataset into ORE transparency Layere
                   dataset <- ore.pull(p_dataframe)
                   
                   # Releveling
                   dataset[ ,pY] <- relevel(dataset[ ,pY],'Y')
                   
                   ## Data splitting into training and validation set
                   splt <- data.split(df = dataset,resVal = pY,seedvalue = 1000,spltratio = p_spltratio)
                   
                   # Creation of training and validation set
                   training <- subset(x = dataset,splt == TRUE)
                   validation <- subset(x = dataset,splt == FALSE)
                   
                   # Imputing the NA on main dataset
                   training_imputed <- impute.data(data = training,imputetype = p_imputetype)
                   
                   print("#################################################")
                   print("***1.Model Building -  Decision Tree...")
                   print("#################################################")
                   
                   modDT <- mdl.training.DT(trainingset = training_imputed,formulae = p_formula,storageDir = p_storageDir,topN = p_topn)
                   
                   print("#################################################")
                   print("***.COMPLETED - Model Building -  Decision Tree...")
                   print("#################################################")
                   
                   print("Saving the DT Model")
                   save(modDT,file = paste0(p_storageDir,"/","trainedModDT.rda"))
                   
                   if (nrow(ore.datastore(name=ds.name)) > 0 ) 
                   {
                     ore.delete(name = ds.name)
                   }
                   ore.save(modDT,name = ds.name,append = TRUE)   
                   
                   print("***3.Model Metrics...")
                   mdl_metric_DT <- metricROCR(model = modDT,testdata = validation,pY = pY,ds.name = ds.name)
                   
                   # Returning the Metrics                   
                   mdl_metric_DT
                 })
