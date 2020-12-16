# -*- coding: utf-8 -*-
"""Copy of classification test for MA710.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I7ZRDUPVbQRPQqBA3ufRBG6J45Dyct9J
"""

from pycaret.utils import enable_colab
enable_colab()

!pip install pycaret

import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

## load dataset from google drive

from google.colab import drive
drive.mount('/content/drive')

dataset=pd.read_csv("/content/drive/MyDrive/lending club models/lc_cleaned_combined.csv") # dataset location on Drive

dataset.head()

!pip install pandas-profiling

from pandas_profiling import ProfileReport
profile = ProfileReport(dataset, title='Pandas Profiling Report')

profile

profile.to_file('lc_data_profile.html')

#check the shape of data
dataset.shape

data = dataset.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True,inplace=True)
data_unseen.reset_index(drop=True,inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

data.columns

#for classification purpose
from pycaret.classification import *
#?setup

exp_clf101 = setup(data = data, target = 'loan_status', session_id=123,train_size= 0.8, 
                   numeric_features=['pub_rec']
                  ,ignore_features=['Unnamed: 0', 'id', 'member_id',  'funded_amnt',
       'funded_amnt_inv',  'installment', 
       'sub_grade', 'emp_title', 
       'verification_status', 'issue_d', 'pymnt_plan', 'desc',
       'title',   'delinq_2yrs',
        'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 
       'revol_util', 'initial_list_status', 'out_prncp',
       'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
       'next_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med',
       'mths_since_last_major_derog', 'policy_code', 'application_type',
       'annual_inc_joint', 'dti_joint', 'verification_status_joint',
       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'url',
       'open_il_6m']) 
##exclude unimporatant predictors

"""# 7.0 Comparing All Models

Comparing all models to evaluate performance is the recommended starting point for modeling once the setup is completed (unless you exactly know what kind of model you need, which is often not the case). This function trains all models in the model library and scores them using stratified cross validation for metric evaluation. The output prints a score grid that shows average Accuracy, AUC, Recall, Precision, F1 and Kappa accross the folds (10 by default) of all the available models in the model library.
"""

?compare_models
#ignore xgboost model for an error issue with column name

# k=5 Cross-validation for speedy output for now, ignore 'xgboost' model becuase we have "["or "]" or "," in emp_year

compare_models(exclude=["xgboost"])

"""# 8.0 Create a Model"""

rf=create_model('rf')

treenet_gb=create_model('lr')

"""### 8.1 confirm Classifier based on last session"""

#apparently logistic regression is good enough
lr=create_model('lr')

#random forest
rf=create_model('rf')
#SVM

?create_model

#trained model object is stored in the variable 'dt'. 
print(rf)
print(lr)

"""### 8.3 Random Forest Classifier"""

rf = create_model('rf')

rf = create_model('lr')

"""# 9.0 Tune a Model

### 9.1 Tune model
"""

tuned_lr=tune_model(lr,optimize="Accuracy")

tuned_rf=tune_model(rf,optimize="Accuracy")

#tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_rf)
print(tuned_lr)

"""# 10.0 Plot a Model

Before model finalization, the `plot_model()` function can be used to analyze the performance across different aspects such as AUC, confusion_matrix, decision boundary etc. This function takes a trained model object and returns a plot based on the test / hold-out set. 

There are 15 different plots available, please see the `plot_model()` docstring for the list of available plots.

### 10.1 AUC Plot
"""

plot_model(tuned_lr)

plot_model(tuned_rf)

evaluate_model(tuned_lr)

evaluate_model(tuned_rf)

"""# 11.0 Predict on test / hold-out Sample

Before finalizing the model, it is advisable to perform one final check by predicting the test/hold-out set and reviewing the evaluation metrics. If you look at the information grid in Section 6 above, you will see that 30% (6,841 samples) of the data has been separated out as test/hold-out sample. All of the evaluation metrics we have seen above are cross validated results based on the training set (70%) only. Now, using our final trained model stored in the `tuned_lr` variable we will predict against the hold-out sample and evaluate the metrics to see if they are materially different than the CV results.
"""

predict_model(tuned_lr)
predict_model(tuned_rf)

"""# 12.0 Finalize Model for Deployment"""

final_lr = finalize_model(tuned_lr)
final_rf = finalize_model(tuned_rf)

#Final Random Forest model parameters for deployment
print(final_lr)
print(final_rf)

plot_model(final_lr)

"""# 13.0 Predict on unseen data"""

unseen_predictions = predict_model(final_lr, data=data_unseen)
unseen_predictions.head()

"""The `Label` and `Score` columns are added onto the `data_unseen` set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background.

# 14.0 Saving the model
"""

save_model(final_lr,'Final Logistic Classification Model')

save_model(final_rf,'Final random forest Model')

"""(TIP : It's always good to use date in the filename when saving models, it's good for version control.)

# 15.0 Loading the saved model
"""

saved_final_lr = load_model('Final logistic classification- 1 cleaned data')

new_prediction = predict_model(saved_final_lr, data=data_unseen)
new_prediction.head

dfp=pd.DataFrame(new_prediction)
dfp.to_csv('prediction-Logistic Regression-1st clean data.csv')

"""Notice that the results of `unseen_predictions` and `new_prediction` are identical."""