# lendingclub-sklearn-dash
This is a cool project for utilizing Dash and Plotly as an interactive tool for investor data exploration and lender approval prediction of Lending Club.
The original dataset comes from Kaggle https://www.kaggle.com/husainsb/lendingclub-issued-loans and has 690.95 MB data(train+test) from 2007 to 20017.
Dataset has been cleaned in Jupyter notebook and columns have been dropped based on feature selection done in pycaret.
Logistic Regression and Random Forest have been determined good regressors in pycaret and they are re-trained with sklearn to allow Heroku deployment with around 2% accuracy drop in both 
Logistic Regression and Random Forest with 50 trees(down from 100 trees for Heroku performance). Despite this accuracy drop, it still achieves 78% prediction accuracy on testing data with a 15% out of 100% split.
For successful Heroku deployment, a paid option is selected for one month due to RAM constraints causing the entire app crash at starting up or with user interaction on the web page.
Despite careful scrutiny on code simplification to make it more efficient to run on Heroku servers, the app would constantly max out 1GB RAM and cause performance concerns. 
