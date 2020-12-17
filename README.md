# lendingclub-sklearn-dash
This is a cool project for utilizing Dash and Plotly as an interactive tool for data exploration and an alternative tool to give loan approval odds prediction of Lending Club.
Link to the Heroku app is: https://lendingdash.herokuapp.com/

This app consists of two pages:

Investor EDA Page: https://lendingdash.herokuapp.com/apps/page1

Prediction Page: https://lendingdash.herokuapp.com/apps/page2

The original dataset comes from Kaggle https://www.kaggle.com/husainsb/lendingclub-issued-loans and has 690.95 MB data(train+test) from 2007 to 2017.
Dataset has been cleaned in Jupyter notebook and columns have been dropped based on feature selection done in pycaret.

Logistic Regression and Random Forest have been determined good regressors in pycaret and they are re-trained with sklearn to allow Heroku deployment with around 2% accuracy drop in both 
Logistic Regression and Random Forest with 50 trees(down from 100 trees for Heroku performance). Despite this accuracy drop, it still achieves 78% prediction accuracy on testing data with a 15% out of 100% split.

For successful Heroku deployment, a paid option is selected for one month due to RAM constraints causing the entire app crash at starting up or with user interaction on the web page. This perhaps is caused by dash core component datatable fetching data from dataframe while processing multiple aggregation functions and my dataset is relatively large. An optimal solution is to mannually make these aggregated dataframes into .csv and fetch data from. However, this defeats the idea of utilizing Dash + Plotly as an easy-to-use data exploration tool while staying interactive.

Despite careful scrutiny on code simplification to make it more efficient to run on Heroku servers, the app would constantly max out 4GB RAM and cause performance concerns. 
Link to the Heroku app is: https://lendingdash.herokuapp.com/
Many components have been implemented using dbc(dash bootstrap component) and dcc(dash core component). Links to these components can be found here: https://dash-bootstrap-components.opensource.faculty.ai/ and https://dash.plotly.com/dash-core-components.


PS: when you are selecting from a dropdown, the app will update the graph y-axis, it's just super slow on Heroku. Yes, every dropdown works and just takes time! And yes, many fields have hover tooltips such as in datatables where it can show descriptions for columns. Yes, these dash datatables work like normal spreadsheet, however, the editing feauture has been disabled while normal sorting feature is still available.
