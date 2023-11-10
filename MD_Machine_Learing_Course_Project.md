Machine Learning Course Project
================
Michael G Harpole
2023-11-09

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

### Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://groupware.les.inf.puc-rio.br/har> . If you use the document you
create for this class for any purpose please cite them as they have been
very generous in allowing their data to be used for this kind of
assignment.

## Read in Data and Clean up

Variables with greater then 80% not available data were removed leaving
60 variables to observe. Removed first 7 variables because they are not
related to the exercise.The training set was split into a 70/30 split.
The first 70% is used for training the machine learning models and the
other 30% is for cross validation. The cross validation error will be
used as an estimate of the out of sample error.

``` r
QUIZ_Set <- read_csv("pml-testing.csv")
```

    ## New names:
    ## Rows: 20 Columns: 160
    ## ── Column specification
    ## ──────────────────────────────────────────────────────── Delimiter: "," chr
    ## (3): user_name, cvtd_timestamp, new_window dbl (57): ...1,
    ## raw_timestamp_part_1, raw_timestamp_part_2, num_window, rol... lgl (100):
    ## kurtosis_roll_belt, kurtosis_picth_belt, kurtosis_yaw_belt, skewn...
    ## ℹ Use `spec()` to retrieve the full column specification for this data. ℹ
    ## Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## • `` -> `...1`

``` r
training_data <- read_csv("pml-training.csv",na=c("#DIV/0!","NA"))
```

    ## New names:
    ## Rows: 19622 Columns: 160
    ## ── Column specification
    ## ──────────────────────────────────────────────────────── Delimiter: "," chr
    ## (4): user_name, cvtd_timestamp, new_window, classe dbl (150): ...1,
    ## raw_timestamp_part_1, raw_timestamp_part_2, num_window, rol... lgl (6):
    ## kurtosis_yaw_belt, skewness_yaw_belt, kurtosis_yaw_dumbbell, skew...
    ## ℹ Use `spec()` to retrieve the full column specification for this data. ℹ
    ## Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## • `` -> `...1`

``` r
# glimpse(training_data)
training_data <- training_data %>% select(which(colMeans(is.na(.)) <0.20)) %>% select(-seq(7))
training_data_part <- createDataPartition(training_data$classe, p=0.70, list=FALSE)
training_set <- training_data %>% slice(training_data_part)
testing_set <- training_data %>% slice(-training_data_part)
```

## Boosting Model

The boosting algorithms utilize a bootstrapping method based on a set of
weak classifiers to generate stronger classifiers.

### Generate model

``` r
BoostFit <- train(classe ~ ., method="gbm",data=training_set,verbose=FALSE)
print(BoostFit)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7493362  0.6819976
    ##   1                  100      0.8184096  0.7700877
    ##   1                  150      0.8494546  0.8094285
    ##   2                   50      0.8546679  0.8158235
    ##   2                  100      0.9051139  0.8798643
    ##   2                  150      0.9291042  0.9102551
    ##   3                   50      0.8954581  0.8675843
    ##   3                  100      0.9391916  0.9230325
    ##   3                  150      0.9570946  0.9456999
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 150, interaction.depth =
    ##  3, shrinkage = 0.1 and n.minobsinnode = 10.

### Check accuracy of Boosting model

``` r
predictGBM <- predict(BoostFit, newdata=testing_set)
confusion_matrix_GBM <- confusionMatrix(predictGBM, as.factor(testing_set$classe))
confusion_matrix_GBM
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1641   27    0    0    2
    ##          B   18 1084   37    2   13
    ##          C    6   24  979   28   10
    ##          D    8    2    9  929   17
    ##          E    1    2    1    5 1040
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.964           
    ##                  95% CI : (0.9589, 0.9686)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9544          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.038e-07       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9803   0.9517   0.9542   0.9637   0.9612
    ## Specificity            0.9931   0.9853   0.9860   0.9927   0.9981
    ## Pos Pred Value         0.9826   0.9393   0.9351   0.9627   0.9914
    ## Neg Pred Value         0.9922   0.9884   0.9903   0.9929   0.9913
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2788   0.1842   0.1664   0.1579   0.1767
    ## Detection Prevalence   0.2838   0.1961   0.1779   0.1640   0.1782
    ## Balanced Accuracy      0.9867   0.9685   0.9701   0.9782   0.9797

The general boosting model has an accuracy of 96.4% on the testing set
so the out of sample error estimate is 3.6%.

## Random Forest

A random forest model is generating by bootstrapping the samples and
generating multiple trees based of the bootstrapped samples. Multiple
random forests are generated and then the one with the highest accuracy
is chosen.

### Generate Model

``` r
TreeFit <- train(classe ~ .,data=training_set,method="rf",prox=TRUE)
TreeFit
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9883018  0.9851993
    ##   27    0.9890880  0.9861948
    ##   52    0.9811216  0.9761158
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

### Check Accuracy of random forest model

``` r
predict_RF <- predict(TreeFit, newdata=testing_set)
confusion_matrix_RF <- confusionMatrix(predict_RF, as.factor(testing_set$classe))
confusion_matrix_RF
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671   10    0    0    0
    ##          B    3 1127   12    0    0
    ##          C    0    2 1009    4    4
    ##          D    0    0    5  960    3
    ##          E    0    0    0    0 1075
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9927          
    ##                  95% CI : (0.9902, 0.9947)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9908          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9895   0.9834   0.9959   0.9935
    ## Specificity            0.9976   0.9968   0.9979   0.9984   1.0000
    ## Pos Pred Value         0.9941   0.9869   0.9902   0.9917   1.0000
    ## Neg Pred Value         0.9993   0.9975   0.9965   0.9992   0.9985
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1915   0.1715   0.1631   0.1827
    ## Detection Prevalence   0.2856   0.1941   0.1732   0.1645   0.1827
    ## Balanced Accuracy      0.9979   0.9932   0.9907   0.9971   0.9968

The accuracy of the random forest model is 99.27% on the testing set so
the out of sample error estimate is 0.73%.

## Quiz Data

The Random forest model is more accurate so it will be utilized to
calculate the quiz questions

``` r
quiz_RF_Predict <- predict(TreeFit, newdata=QUIZ_Set)
quiz_RF_Predict
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

I scored 20 out of 20 in the quiz predictions.

``` r
stopCluster(cl)
sessionInfo()
```

    ## R version 4.3.1 (2023-06-16 ucrt)
    ## Platform: x86_64-w64-mingw32/x64 (64-bit)
    ## Running under: Windows 10 x64 (build 19045)
    ## 
    ## Matrix products: default
    ## 
    ## 
    ## locale:
    ## [1] LC_COLLATE=English_United States.utf8 
    ## [2] LC_CTYPE=English_United States.utf8   
    ## [3] LC_MONETARY=English_United States.utf8
    ## [4] LC_NUMERIC=C                          
    ## [5] LC_TIME=English_United States.utf8    
    ## 
    ## time zone: America/New_York
    ## tzcode source: internal
    ## 
    ## attached base packages:
    ## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
    ## [8] base     
    ## 
    ## other attached packages:
    ##  [1] doParallel_1.0.17    iterators_1.0.14     foreach_1.5.2       
    ##  [4] randomForest_4.7-1.1 caret_6.0-94         lattice_0.21-8      
    ##  [7] lubridate_1.9.2      forcats_1.0.0        stringr_1.5.0       
    ## [10] dplyr_1.1.2          purrr_1.0.2          readr_2.1.4         
    ## [13] tidyr_1.3.0          tibble_3.2.1         ggplot2_3.4.3       
    ## [16] tidyverse_2.0.0     
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.2.0     timeDate_4022.108    fastmap_1.1.1       
    ##  [4] pROC_1.18.4          digest_0.6.33        rpart_4.1.19        
    ##  [7] timechange_0.2.0     lifecycle_1.0.3      survival_3.5-7      
    ## [10] magrittr_2.0.3       compiler_4.3.1       rlang_1.1.1         
    ## [13] tools_4.3.1          utf8_1.2.3           yaml_2.3.7          
    ## [16] data.table_1.14.8    knitr_1.43           bit_4.0.5           
    ## [19] plyr_1.8.8           withr_2.5.0          nnet_7.3-19         
    ## [22] grid_4.3.1           stats4_4.3.1         fansi_1.0.4         
    ## [25] e1071_1.7-13         colorspace_2.1-0     future_1.33.0       
    ## [28] globals_0.16.2       scales_1.2.1         MASS_7.3-60         
    ## [31] cli_3.6.1            rmarkdown_2.24       crayon_1.5.2        
    ## [34] generics_0.1.3       rstudioapi_0.15.0    future.apply_1.11.0 
    ## [37] reshape2_1.4.4       tzdb_0.4.0           proxy_0.4-27        
    ## [40] splines_4.3.1        vctrs_0.6.3          hardhat_1.3.0       
    ## [43] Matrix_1.6-1         hms_1.1.3            bit64_4.0.5         
    ## [46] listenv_0.9.0        gower_1.0.1          recipes_1.0.7       
    ## [49] glue_1.6.2           parallelly_1.36.0    codetools_0.2-19    
    ## [52] stringi_1.7.12       gtable_0.3.4         munsell_0.5.0       
    ## [55] pillar_1.9.0         htmltools_0.5.6      ipred_0.9-14        
    ## [58] lava_1.7.2.1         gbm_2.1.8.1          R6_2.5.1            
    ## [61] vroom_1.6.3          evaluate_0.21        class_7.3-22        
    ## [64] Rcpp_1.0.11          nlme_3.1-162         prodlim_2023.03.31  
    ## [67] xfun_0.40            pkgconfig_2.0.3      ModelMetrics_1.2.2.2
