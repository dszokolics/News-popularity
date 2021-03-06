# News-popularity

This repo contains the models I built for predicting news popularity (https://www.kaggle.com/c/ceuba2020). The repo's structure is the following:

- data/: contains the input data

- EDA/: it holds the HTML output of the explaratory profiling

- model_params/: contains the best hyperparams for the models used for the ensemble

- models/: the scripts used for hyperparameter tuninig

- utils/: preprocess script

- ensemble_model.py: This script reads the hyperparams & trains the base learners. After that, it does the hyperparameter tuning & training the meta-learner.

- exploration.py: quick data exploration and creation of the pandas-profiling output.

## Analytics process

The process is presented in a linear order, which was of course, not the case, but it would be harder to interpred in a realistic way.

### 1. Exploratory Data Analysis

I run a profiling (from exploration.py), which created the report in the EDA folder. Based on this, the data is quite clean. There are some linear dependencies and non-normal distributions, but in general, not much cleaning is needed.

### 2. Data preparation

I did three types of data preparation. 

- Basic cleaning (utils/preprocess.py/preprocess with pca=False): I dropped the weekend dummy because of strong linear dependency, logarithmized some columns, and finally, scaled all the independent variables.

- Cleaning and PCA (utils/preprocess.py/preprocess with pca=True): In addition to the steps in the previous case, I grouped highly correlated variables. I identified groups which are highly correlated, dropped them, and used their PCAs instead.

- Preparation for linear models (utils/preprocess.py/preprocess_lm): In this setup, I dropped some other columns because of linear dependency. Also, I created additional features, such as the square and log of all the continous variables, and I added some interactions with the most important features. All the variables are normalized.

The preparation process involves the splitting of the data into train, test and validation sets. I did the split prior to the cleaning, and did the same cleaning steps on all of them. Each step was executed based on the train data.

The first two dataset was used for every model types - I trained (at least) two models from each type for the two dataset.

### 3. Baseline model

At first, I run H2O's autoML (with 4 minutes runtime). It's great for estimating the potential in the data, and it also gives hints about the most promising model types.

The autoML had an AUC (area under the ROC curve) of .71 locally, and had a slightly worse, .702 performance on the public test set on Kaggle. The best models were all GBMs, so later I focused on Boosted Trees.

### 4. Base learners

I trained multiple models to find the best performant one. At last, it turned out that none of them can beat an ensemble of them. I trained all the models (except Linear Regression) on the PCA and non-PCA dataset. The models were the following:

- Generalized Linear Model (H2O) x1

- Random Forest (sklearn) x2

- Gradient Boosting Machine (H2O) x3

- XGBoost x2

- Deep Learning (H2O) x3

I trained two GBMs on the non-PCA dataset, and two DL on the PCA dataset, that's why they have three final models. I was focusing on The tree-based models, hence for simplicity, I chose H2O over Keras as a library for deep learning.

I used hyperopt to find the best hyperparameters for the models (scripts in the model directory). This module uses an algorithm called Tree-structured Parzen Estimator Approach, which is more efficient in finding the best parameters than grid search or random search. For the training, I had to define the parameter space, where the search is conducted. After the training I checked if the parameters are not at the edge of the parameter space, and also checked the cross validation (within the train set) and out of sample performance. I exported the dictionary of the best parameters to the model_params folder as yaml files.

I used three datasets for training: train, validation and the holdout set. The validation set was used for early stopping, and evaluating model performance during the hyperparameter optimization. After finding the best parameters, I checked the model performance on the test set and on the train set with cross validation, and if it wasn't far from the validation performance, I accepted the model. Actually, using cross-validation performance either for early stopping or hyperparameter optimization could be more beneficial, but also much slower. In my setup, I think I slightly overfitted the validation set.

Although the final evaluation is based on the AUC metric, I used logloss for hyperparameter search (and for early stopping where applicable). This metric seemed less noisy than AUC, so I expected that it will reduce overfitting.

Exact performances will be reported in the next section.

#### Generalized Linear Model

I optimized two parameters of GLM: lambda for the regularization strength, and alpha for the distribution between L1 and L2. I used the dataset prepared for the linear case. In order to save time and avoid overfitting, I applied early stopping.

Performance of the holdout set (ROC): 0.672

#### Random Forest

The three main parameters I used for optimization are the maximum depth, the minimum samples required for split and the sample size. The two resulting parameter sets are quite similar, and later it turned out, that their results are highly correlated. Although I was sceptical about using both in the same ensemle, it performed worse if I dropped one of them.

I tried recursive feature elimination as a preparatory step for this model, but it didn't help, so I kept all my features.

Performance of the holdout set (ROC):

- With PCA: 0.708

- Without PCA: 0.708

#### Gradient Boosting Machine

I managed to train three GBMs with different hyperparameter sets, and reasonable performance, so I kept all of them. I used early stopping for training (4-8 rounds and 10^{-5}-10^{-6} stopping tolerance). I started using the number of trees as a hyperparameter, but later on I realized, that it's doesn't make sense with early stopping, so I set it to 500. One of my models was trained before this change, so it has a limited number of trees along with higher learning rate.

It is interesting to see the tradeoffs between the different parameters in the three models. The max_depth and the min_rows (minimum samples needed before splitting) clearly moves together, as they control for the complexity of one tree. Also, the sample rate and the column sample rate seems to correlate negatively, which is also intuitive, as they are both aiming at decorrelating the trees.

![](https://github.com/dszokolics/News-popularity/blob/master/gbm_params.png)

Performance of the holdout set (ROC):

- With PCA: 0.710

- Without PCA (learning rate of 0.3): 0.711

- Without PCA (learning rate of 0.03): 0.708

#### XGBoost

I expected the best performance from XGBoost, based on the autoML results. Actually I wasn't able to determine whether the H2O GLM or this had the better performance, because they are close, and their performances have significant variance (between .69 and .73).

I used the hyperparameter gamma (minimum loss reduction needed to split) instead of the minimum samples needed to split, and I added the two regularization parameters, alpha and lambda. The model which uses PCA for preprocessing uses higher max depth, and has higher alpha, lambda and gamma parameters. For me, it seems that the non-PCA model has a bit higher complexity based on these.

Performance of the holdout set (ROC):

- With PCA: 0.707

- Without PCA: 0.708

#### Deep Learning

I wasn't expecting a great performance from deep learning, as the dataset is not that large, and the automl showed superior performance for the tree-based methods. After submitting my final results, I realized that I made the same mistake, as I did for the GBM: I used the number of epochs as a parameter for optimization, and applied early stopping as well. In the last uptade, I fixed the number of epochs at 25.

Performance of the holdout set (ROC):

- With PCA (without rate decay): 0.690

- With PCA (with rate decay): 0.683

- Without PCA: 0.682

### Meta learner

I tried multiple models as meta learner, but I ended up using a penalized linear regression. As the inputs are already probabilities, it is plausible to expect linear relationship with the outcome probabilities. Also, the evaluation metric is ROC, which doesn't take exact values into account, only relative values. In this sense, the link function of the GLM doesn't really matter.

I used the out-of-sample results from the cross validation for features. Altogether I had 11 base learners. The parameters of the final meta learner is in the meta.yaml. I haven't normalized my values before the regression, because they were already really close to eachother in terms of mean and standard deviation.

I also tried to use the first component of the PCA based on the base learner predictions supposing that it's the most conservative approach, and I saw that my model overfits a bit. Actually it had a nice performance, but still worse than the linear regression.

Performance: 0.712

During the training I was focusing on my local cross validation and holdout set performances instead of the Kaggle public leaderboard. As I experienced, the results on Kaggle were around 1% below my cross validated average, and it was usually below the holdout set performance. In the end, I tried multiple seeds for the initial train/test split, and I ended up with results which were similar to my cross validated results, so I concluded that my model doesn't overfit.
