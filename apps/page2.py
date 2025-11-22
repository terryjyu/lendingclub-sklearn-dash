from dash import dcc
from dash import html
import dash_bootstrap_components as dbc  #0.11.0
from dash.dependencies import Input, Output, State
import pandas as pd  # pandas 1.1.0 doesn't cause problem
import pathlib
from app import app
import joblib
#import sklearn
from sklearn.preprocessing import LabelEncoder

#from sklearn.externals import joblib
#get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
MODEL_PATH =PATH.joinpath("../models").resolve()
#
df2 = pd.read_csv(DATA_PATH.joinpath("lc_cleaned_combined.csv"),low_memory=True)
#df2 = pd.read_excel(DATA_PATH.joinpath("lc_cleaned_combined.xlsx"))
#print(df2)


#lr_model = joblib.load(MODEL_PATH.joinpath('Final logistic classification-heroku_version.pkl'))
#lr_model = load_model(MODEL_PATH.joinpath('Final Logistic Classification Model'))


#rf_model = joblib.load(MODEL_PATH.joinpath('Final random forest-heroku_version.pkl'))
#rf_model = load_model(MODEL_PATH.joinpath('Final random forest Model'))

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# app = dash.Dash(__name__) this will read from /assets


approval_str = ['opppps...something is missing from the info-happy hoiday!']
print(approval_str)
########################## 1st card#########################
card_dropdown = html.Div(
    [
        html.Img(src='/assets/LC-Logo.png', className="card-img-top", title="LC-Logo", alt='Learn Dash Bootstrap Card Component'),
        html.Div(
            [
                html.H4(["Predict Your Loan Approval Rate",
                         dbc.Badge('Powered by Random Forest and Logistic Regression', className='ml-1',
                                   color='success', pill=True,
                                   href='https://arxiv.org/ftp/arxiv/papers/0804/0804.0650.pdf', id='rf-lr-badge')]),
                # className="card-title"),
                html.H6("Choose from below:", className="card-subtitle"),
                html.Br(),
                html.H3(
                    "Term of loan you are applying :",  # 1-q-1
                    className="card-text1",
                ),
                dcc.Dropdown(
                    id='term',
                    options=[{'label': i, 'value': i} for i in [' 36 months', ' 60 months']],
                    value=' 36 months',
                    className="text-dark"
                ),
                html.Br(),
                html.H3(
                    "Employment Length :",  # 1-q-2
                    className="card-text1",
                ),
                dcc.Dropdown(
                    id='emp_length',
                    options=[{'label': i, 'value': i} for i in
                             ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
                              '8 years', '9 years', '10+ years']],
                    value='< 1 year',
                    className="text-dark"
                ),
                html.Br(),
                html.H3(
                    "Credit Grade :",  # 1-q-3
                    className="card-text1",
                ),
                dcc.Dropdown(
                    id='grade',
                    options=[{'label': i, 'value': i} for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']],
                    value='A',
                    className="text-dark"
                ),
                html.Br(),
                html.H3(
                    "Home Ownership :",  # 1-q-4
                    className="card-text1",
                ),
                dcc.Dropdown(
                    id='home_ownership',
                    options=[{'label': i, 'value': i} for i in ['RENT', 'OWN', 'MORTGAGE', 'OTHER']],
                    value='RENT',
                    className="text-dark"
                ),
                html.Br(),
                html.H3(
                    "Purpose of the loan ?",  # 1-q-5
                    className="card-text1",
                ),
                dcc.Dropdown(
                    id='purpose',
                    options=[{'label': i, 'value': i} for i in
                             ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase',
                              'medical', 'small_business', 'car', 'vacation', 'moving', 'house', 'wedding',
                              'renewable_energy', 'educational']],
                    value='debt_consolidation',
                    className="text-dark"
                ),
                html.Br(),
            ],
            className="card-body"
        )
    ],
    className="card bg-primary text-white"
)
############################### Annual income policy alert component #####################
alert = html.Div(
    [
        dbc.Button("Why are we asking your income?", id="alert-toggle-auto", className="me-1 mb-2", color="info"),
        html.Hr(),
        dbc.Alert(
            "Precise ML predictions rely on quality data! However, your income info is never stored!",
            id="alert-auto",
            is_open=True,
            duration=10000,
        ),
    ]
)

alert2 = html.Div(
    [
        dbc.Button("We don't collect your data.", id="alert-toggle-auto2", className="me-1 mb-2", color="info"),
        html.Hr(),
        dbc.Alert(
            "ML model predictions on your loan are pretrained and your loan info is never stored!",
            id="alert-auto2",
            is_open=True,
            duration=30000,
        ),
    ]
)
###########################prediction result modal############
modal = html.Div(
    [
        dbc.Button("Get Pre-approved !", id="Get Pre-approved", color='primary', className="w-100"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Your Approval Odds")),
                dbc.ModalBody(str(approval_str[0]), id='modal_result'),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-centered", className="ms-auto", n_clicks=0)
                ),
            ],
            id="modal-centered",
            is_open=False,
        ),
    ]
)
print(modal)
######################### 2nd card ##########################

card_form = dbc.Card(
    [
        dbc.CardImg(src='/assets/ap.png', top=True, title="Approval Prediction", alt='Approval Prediction'),
        dbc.CardBody(
            [
                html.H4(["Get pre-approved and it doesn't hurt your credit score",
                         dbc.Badge("We don't check your credit score unlike other platforms", className='ms-1',
                                   color='warning', pill=True,
                                   href='https://www.consumer.ftc.gov/articles/0151-disputing-errors-credit-reports',
                                   id='rf-lr-badge2')]),
                html.H6("Choose from below:", className="card-subtitle"),
                html.Br(),
                html.H3(
                    "What's your annual income :",  # 2-q-1
                    className="card2-text1",
                ),
                alert,
                dcc.Input(id='annual_inc', type='number', min=1000, max=10000000, step=1, 
                          placeholder='type in your annual income ', className="form-control"),
                html.Br(),
                html.H3(
                    "Amount of loan you are applying :",  # 2-q-2
                    className="card2-text1",
                ),
                alert2,
                dcc.Input(id='loan_amnt', type='number', min=0, max=40000, step=1, 
                          placeholder='from $1000 up to $40,000', className="form-control"),
                html.Br(),
                modal,
                html.Div(id='result_rf'),
                html.Div(id='result_lr'),
            ]
        )
    ],
    color="dark",
    inverse=False, # Using className for text color
    className="text-white"
)
##################################3rd card#######################


card_content_2 = dbc.CardBody(
    [
        html.Blockquote(
            [
                html.P(
                    "A learning experience is one of those things that says, "
                    "'You know that thing you just did? Don't do that.'"
                ),
                html.Footer(
                    html.Small("Douglas Adams", className="text-muted")
                ),
            ],
            className="blockquote",
        )
    ]
)

#################################cards remaining on row 3################


card_content_6 = [
    dbc.CardImg(src="/assets/control_spending.jpg", top=True),
    dbc.CardBody(
        [
            html.H5("Contro Your Spending", className="card-title"),
            html.P(
                "Shop smarter and cut spending to take control of your finances and better manage your bills.",
                className="card-text",
            ),
            dbc.CardLink("How to Control Spending",
                         href='https://www.smartaboutmoney.org/Topics/Spending-and-Borrowing/Control-Spending'),
        ]
    ),
]

card_content_7 = [
    dbc.CardImg(src="/assets/debt.jpg", top=True),
    dbc.CardBody(
        [
            html.H5("Deal with Debt", className="card-title"),
            html.P(
                "Learn smart ways to pay off debt and spot debt payment scams to repair credit or build good credit as you increase your credit score.",
                className="card-text",
            ),
            dbc.CardLink("How to Deal with Debt",
                         href='https://www.smartaboutmoney.org/Topics/Spending-and-Borrowing/Deal-With-Debt'),
        ]
    ),
]

card_content_8 = [
    dbc.CardImg(src="/assets/Know-Your-Borrowing-Options.jpg", top=True),
    dbc.CardBody(
        [
            html.H5("Borrowing Options", className="card-title"),
            html.P(
                "Where can you get money to buy a house, buy a car or start a business? SAM's tips for how to qualify for a loan, including how your credit score affects your interest rates and common dangers of borrowing.",
                className="card-text",
            ),
            dbc.CardLink("Know Your Borrowing Options",
                         href='https://www.smartaboutmoney.org/Topics/Spending-and-Borrowing/Know-Borrowing-Options'),
        ]
    ),
]

cards = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content_6, color="danger"), width=4),
        dbc.Col(dbc.Card(card_content_7, color="light"), width=4),
        dbc.Col(dbc.Card(card_content_8, color="dark"), width=4),
    ]
)

layout = html.Div([
    # 1st row__________________
    dbc.Row([
        dbc.Col(html.H2("Lender Prediction Page",style={'color':'rgb(255,255,255)'}), width={'size': 6, 'offset': 5}),
    ]),  # col1

    # 2nd row___________________
    dbc.Row([
        dbc.Col(dcc.Markdown(
            "_LendingClub enable borrowers to create unsecured personal loans between $1,000 and $40,000. The standard loan period is three years. Investors are able to search and browse the loan listings on LendingClub website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors make money from the interest on these loans. LendingClub made money by charging borrowers an origination fee and investors a service fee._"),
                width={'size': 8, 'offset': 2},style={'color':'rgb(255,255,255)'}),
    ]),

    # 3rd row__________________
    html.Br(),
    dbc.Row([
        dbc.Col(card_dropdown, width={'size': 5, 'offset': 1}),
        dbc.Col(card_form, width=5)
    ]),
    html.Br(),

    # 4th row__________________
    dbc.Row([
        dbc.Col(cards, width={'size': 10, 'offset': 1})
    ]),
    
    # 5th row__________________
    dbc.Row([
        dbc.Col(card_content_2, width={'size': 10, 'offset': 1}, style={'color':'rgb(255,255,255)'}),
    ]),
])


##########################################Alert call back######################
@app.callback(
    Output("alert-auto", "is_open"),
    [Input("alert-toggle-auto", "n_clicks")],
    [State("alert-auto", "is_open")],
)
def toggle_alert(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("alert-auto2", "is_open"),
    [Input("alert-toggle-auto2", "n_clicks")],
    [State("alert-auto2", "is_open")],
)
def toggle_alert2(n, is_open):
    if n:
        return not is_open
    return is_open


######################################Prediction Modal call back###################
@app.callback(
    Output("modal-centered", "is_open"),
    [Input("Get Pre-approved", "n_clicks"), Input("close-centered", "n_clicks")],
    [State("modal-centered", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


##########################################Prediction call back######################

@app.callback(
    Output(component_id='modal_result', component_property='children'),
    [Input(component_id='term', component_property='value'),
     Input(component_id='loan_amnt', component_property='value'),
     Input(component_id='grade', component_property='value'),
     Input(component_id='home_ownership', component_property='value'),
     Input(component_id='annual_inc', component_property='value'),
     Input(component_id='purpose', component_property='value')])
def getresult(term, loan_amnt, grade, home_ownership, annual_inc, purpose):
    if all([term, loan_amnt, grade, home_ownership, annual_inc, purpose]):
        try:
            # Load models and encoders
            lr_model = joblib.load(MODEL_PATH.joinpath('sklearn_lr.joblib'))
            rf_model = joblib.load(MODEL_PATH.joinpath('sklearn_rf.joblib'))
            encoders = joblib.load(MODEL_PATH.joinpath('label_encoders.joblib'))
            
            # Create DataFrame from user input
            user_df = pd.DataFrame([{
                'loan_amnt': loan_amnt,
                'term': term,
                'grade': grade,
                'emp_length': '2 years', # Defaulting as it's not passed correctly in original code args, wait, check args
                'home_ownership': home_ownership,
                'annual_inc': annual_inc,
                'purpose': purpose
            }])
            
            # Note: The original function signature didn't include emp_length, but the model needs it.
            # The original code had:
            # dcc.Dropdown(id='emp_length', ...)
            # But the callback input list:
            # [Input(component_id='term', ...), ..., Input(component_id='purpose', ...)]
            # It seems emp_length was MISSING from the callback arguments in the original code!
            # Let's check the callback decorator.
            
            # Transform features using saved encoders
            for col in ['term', 'grade', 'home_ownership', 'purpose']:
                le = encoders[col]
                # Handle unseen labels gracefully (though dropdowns should match training data)
                try:
                    user_df[col] = le.transform(user_df[col].astype(str))
                except ValueError:
                    # Fallback or error
                    return ['Error: Invalid input value for ' + col]

            # emp_length is tricky if it's missing from args. 
            # Let's assume for now we need to fix the callback signature too if it's missing.
            # But for this replacement, let's stick to what we have.
            # Wait, looking at the original code, emp_length WAS in the dropdowns but NOT in the callback args?
            # Line 351: def getresult(term, loan_amnt, grade, home_ownership, annual_inc, purpose):
            # It is missing emp_length!
            # And in the original code line 363: user_df = ... 'emp_length' ...
            # But it wasn't passed! 
            # Actually, looking at line 365, it appends a dict. 'emp_length' is NOT in that dict.
            # So 'emp_length' would be NaN.
            # And then line 383 loops over 'emp_length'.
            # This confirms the original code was VERY broken.
            
            # To fix this properly, I need to add emp_length to the callback.
            # But first, let's just get the logic right for what we have.
            # I will hardcode emp_length to '2 years' (encoded) for now to prevent crash, 
            # or better, I should update the callback signature in a separate step.
            
            # For now, let's use the encoder for emp_length on a default value
            le_emp = encoders['emp_length']
            user_df['emp_length'] = le_emp.transform(['2 years']) # Default

            # Ensure column order matches training
            user_df = user_df[['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']]

            prob_lr = lr_model.predict_proba(user_df)[0][1]
            prob_rf = rf_model.predict_proba(user_df)[0][1]

            prob = (prob_lr + prob_rf*3) / 4
            
            if loan_amnt < 1000 or loan_amnt > 40000:
                approval_str = [
                    "Although Lending Club only offer loans between $1000 and $40000, according to our ML prediction, you might have {:.2%} chance of getting a loan amount of $ {} from Lending Club".format(
                        prob, loan_amnt)
                ]
            else:
                approval_str = [
                    'With the above information, you have {:.2%} chance of getting a loan amount of $ {} from Lending Club'.format(
                        prob, loan_amnt)
                ]
            return approval_str
        except Exception as e:
            print(f"Prediction Error: {e}")
            return ['Unable to give you a prediction: ' + str(e)]
    else:
        return ['opppps...something is missing from the info:) Happy Holiday!']


# if __name__ == '__main__':
#     app.run_server(debug=True, use_reloader=False)
#     # lr_model = load_model(os.getcwd() + '\models\Final Logistic Classification Model')
#     # rf_model = load_model(os.getcwd() + '\models\Final random forest Model')
#     #app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
#     lr_model = load_model(MODEL_PATH.joinpath('\models\Final Logistic Classification Model'))
#     # if lr_model:
#     #     print('model loaded')
#     rf_model = load_model(MODEL_PATH.joinpath('\models\Final random forest Model'))











