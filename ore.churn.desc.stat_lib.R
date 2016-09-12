#################################################################
## Churn Prediction Library for BIW Offering
## This is a final version
## Done by Snehotosh
#################################################################

if(nrow(ore.scriptList(name = "ore.churn.desc_stat")) > 0)
{
  ore.scriptDrop("ore.churn.desc_stat")
}

ore.scriptCreate("ore.churn.desc_stat", 
                 function(dataset,Y,imputeType,displaycols,destLoc,boxplot_halign,...)
                 {
                   ## 1.Converting to R dataframe
                   dataset <- ore.pull(dataset)
                   

                   ## 2.Finding NULL columns
                   na.count.check <- function(df)
                   {
                     na_count <-data.frame(nullCount=sapply(df, function(y) sum(length(which(is.na(y))))))
                     na_count_df <- data.frame(variable = rownames(na_count),nullcount=na_count[1],row.names = NULL)
                     print(na_count_df)
                     return(na_count_df)
                   }
                   
                   ## 3.Histogram Plotting
                   hist.vis <- function(dframe,cols,displayMatrix,storageDir,...)
                   {
                     dframe1 <- dframe[ ,cols]
                     colnames1 <- dimnames(dframe1)[[2]]           
                     
                     #par(mfrow = displayMatrix)  
                     print("Histogram for chosen Columns...")
                     ## Printing to PDF
                     for (i in 1:length(cols)) {
                       h <- hist(dframe1[,i], main=colnames1[i], col="gray", border="white",xlab = colnames1[i],probability=TRUE)
                       d <- density(dframe1[,i])
                       lines(d, col="red")  
                     }          
                   }
                   
                   ## 4.Boxplot Plotting
                   boxplot.vis <- function(dframe,idpos = -1,option = 1,halign = TRUE,talign = 1,raincolor = 10,title,storageDir,...)
                   {
                     ## This function returns Boxplot for both normal and ggplot ploting.         
                     if(option == 1)
                     {
                       boxplot(dframe[, c(idpos)], horizontal = halign, main = title,las = talign,col = rainbow(raincolor))
                     } else{
                       library(ggplot2)
                       library(reshape2)
                       
                       Df <- melt(dframe)
                       print('Using ggplot...')
                       print(ggplot(data = Df, aes(variable, value, color = factor(variable))) + geom_boxplot())
                     }        
                   }
            
                   ## 5.Impute Data
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
                   
                   ## 6.Checking Skewness,normality and Kurtosis
                   check_datadist <- function(df,...)
                   {  
                     ## This function gives data distribution statistics
                     # Skewness
                     library(e1071)
                     sk <- skewness(df,na.rm = TRUE)
                     kur <- kurtosis(df,na.rm = TRUE)
                     normalityTest <- data.frame(skewness = sk,kurtosis = kur)
                   }     
                   
                   ########################################################################################
                   # ---- Descriptive Statistics -----
                   # The MAIN Function
                   ########################################################################################  
                   
                   P_DESTINATION_LOC <- destLoc
                   P_BOXPLOT_ALIGN <- boxplot_halign
                   
                   print(paste('The dataset contains',nrow(dataset),'records and',ncol(dataset),'features')) 
                   print(paste('The % of Yes(1)/No(0) in the dataset:',prop.table(table(dataset[,Y]))[1]*100,'/',prop.table(table(dataset[,Y]))[2]*100))
                   
                   print('***1.Data Summary...')
                   summary(dataset)
                   cat('\n')
                   
                   print('***2.Checking for NULL columns...')
                   na.count.check(dataset)
                   cat('\n')
                   
                   print('**3.Data imputation...')
                   cat('\n')
                   dataset_imputed <- impute.data(data = dataset,imputetype = imputeType)
                   
                   print('***4.Re-Checking for NULL columns...')
                   na.count.check(dataset_imputed)
                   cat('\n')
                   
                   print('***5.Data Visualization and Data Distribution...')
                   print('\n')
                   print('### 5.1. Drawing Histograms')
                   cat('\n')
                   # Choose columns
                   n <- length(displaycols)
                   print(hist.vis(dframe = dataset_imputed,cols = displaycols,displayMatrix = c(n/2,2),storageDir = P_DESTINATION_LOC))
                   print('### 5.2. Drawing Boxplot to see spread')
                   cat('\n')
                   
                   # Scaling 
                   print(boxplot.vis(dframe = scale(dataset_imputed[,displaycols]),title = "Quantiles",storageDir = P_DESTINATION_LOC,halign = P_BOXPLOT_ALIGN))
                   
                   print('### 5.3. Checking Skewness,normality and Kurtosis')
                   d1<- data.frame(x = c("skewness","kurtosis"),sapply(X = dataset_imputed[,displaycols],FUN = check_datadist),row.names = NULL)
                   #print(d1)         
                 })