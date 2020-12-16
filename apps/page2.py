import dash_core_components as dcc
import dash_html_components as html
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
card_dropdown = dbc.Card(
    [
        dbc.CardImg(src='/assets/LC-Logo.png', top=True, bottom=False,
                    title="LC-Logo", alt='Learn Dash Bootstrap Card Component'),
        dbc.CardBody(
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
                    className="card1-text1",
                ),
                dcc.Dropdown(id='term',  # 1-a-1
                             options=[{'label': "36 Months", 'value': '36 Months'},
                                      {'label': "60 Months", 'value': '60 Months'},
                                      ],
                             value='36 Months', clearable=False, style={"color": "#000000"}),
                html.Br(),
                html.H3(
                    "Your years of employeement :",  # 1-q-2
                    className="card1-text2",
                ),
                dcc.Dropdown(id='emp_length',  # a-2
                             options=[{'label': x, 'value': x} for x in df2.emp_length.dropna().unique()],
                             value='2 years', clearable=False, style={"color": "#000000"}),
                html.Br(),
                html.H3(
                    "Your estimated credit grade:",  # 1-q-3
                    className="card1-text3",
                ),
                dcc.Dropdown(id='grade',  # 1-a-3
                             options=[{'label': "A FICO>770", 'value': 'A'},
                                      {'label': "B 663<FICO<770", 'value': 'B'},
                                      {'label': "C FICO<663", 'value': 'C'},
                                      ], value='A', clearable=False, style={"color": "#000000"}),

                html.Br(),
                html.H3(
                    "Do you own a house ? ",  # 1-q-4
                    className="card1-text4",
                ),
                dcc.Dropdown(id='home_ownership',  # 1-a-4
                             options=[{'label': x, 'value': x} for x in df2.home_ownership.dropna().unique()],
                             value='A', clearable=False, style={"color": "#000000"}),

                html.Br(),
                html.H3(
                    "Purpose of the loan ? ",  # 1-q-5
                    className="card1-text5",
                ),
                dcc.Dropdown(id='purpose',  # 1-a-5
                             options=[{'label': x, 'value': x} for x in df2.purpose.dropna().unique()], clearable=False,
                             style={"color": "#000000"}),

            ]
        ),
    ], color='dark',  # https://bootswatch.com/default/ for more card colors
    inverse=True,  ## change color of text (black or white)
    outline=False,
)  # True = remove the block colors from the background and header#

############################### Annual income policy alert component #####################
alert = html.Div(
    [
        dbc.Button("Why are we asking your income?", id="alert-toggle-auto", className="mr-1"),
        html.Hr(),
        dbc.Alert(
            "Precise ML preditions rely on quality data! However, your income info is never stored!",
            id="alert-auto",
            is_open=True,
            duration=4000,
        ),
    ]
)

alert2 = html.Div(
    [
        dbc.Button("We don't collect your data.", id="alert-toggle-auto2", className="mr-1"),
        html.Hr(),
        dbc.Alert(
            "ML predictions are pretrained and your loan info is never stored!",
            id="alert-auto2",
            is_open=True,
            duration=4000,
        ),
    ]
)
###########################prediction result modal############
modal = html.Div(
    [
        dbc.Button("Get Pre-approved", id="Get Pre-approved", color='primary', block=True, ),
        dbc.Modal(
            [
                dbc.ModalHeader("Your Approving Odds",style={'color':'white'}),
                dbc.ModalBody(str(approval_str[0]), id='modal_result',style={'color':"white"}),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close-centered", className="ml-auto"
                    )
                ),
            ],
            id="modal-centered",
            centered=True,
        ),
    ]
)
print(modal)
######################### 2nd card ##########################
card_form = dbc.Card(
    [
        dbc.CardImg(src='/assets/ap.png', top=True, bottom=False,
                    title="LC-Logo", alt='Card 2 image'),
        dbc.CardBody(
            [
                html.H4(["Get pre-approved and it doesn't hurt your credit score",
                         dbc.Badge("We don't check your credit score unlike other platforms", className='ml-1',
                                   color='warning', pill=True,
                                   href='https://www.consumer.ftc.gov/articles/0151-disputing-errors-credit-reports',
                                   id='rf-lr-badge2')]),  # className="card-title"),
                html.H6("Choose from below:", className="card-subtitle"),
                html.Br(),
                html.H3(
                    "What's your annual income :",  # 2-q-1
                    className="card2-text1",
                ),

                alert,
                dbc.Input(id='annual_inc', type='number', min=1000, max=10000000, step=1,
                          placeholder='type in your annual income '),

                html.Br(),
                html.H3(
                    "Amount of loan you are applying:",  # 2-q-1
                    className="card2-text3",
                ),

                alert2,
                dbc.Input(id='loan_amnt', type='number', min=0, max=40000, step=1, placeholder='up to $40,000'),
                html.Br(),
                # dbc.Button('Get Pre-approved',color='primary',block=True,id='Get Pre-approved'),
                modal,
                html.Div(id='result_rf'),  # result row
                html.Div(id='result_lr'),  # result row

            ]
        ),
    ], color='dark',  # https://bootswatch.com/default/ for more card colors
    inverse=True,  ## change color of text (black or white)
    outline=False,
)  # True = remove the block colors from the background and header#

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

cards = dbc.CardColumns(
    [

        dbc.Card(card_content_6, color="danger", inverse=True),
        dbc.Card(card_content_7, color="light"),
        dbc.Card(card_content_8, color="dark", inverse=True),
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
            "_LendingClub enabled borrowers to create unsecured personal loans between $1,000 and $40,000. The standard loan period was three years. Investors were able to search and browse the loan listings on LendingClub website and select loans that they wanted to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors made money from the interest on these loans. LendingClub made money by charging borrowers an origination fee and investors a service fee._"),
                width={'size': 8, 'offset': 2},style={'color':'rgb(255,255,255)'}),
    ]),
    # 3rd row__________________
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(card_dropdown, width={'size': 5, 'offset': 1}),  # card col1
        dbc.Col(card_form, width={'size':5,'offset':0})    # card col2


    ]),#, justify='left'
    dbc.Row([
        dbc.Col(dbc.Col(cards),     # three cards -hyperlink image cards
                )
    ]),
    dbc.Row([
        # dbc.Col(card_content_4)
        dbc.Col(card_content_2,width={'size': 10, 'offset': 1},style={'color':'rgb(255,255,255)'}),

    ]),#, justify='left'
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
        lr_model = joblib.load(MODEL_PATH.joinpath('sklearn_lr.joblib'))
        if lr_model:
            print('lr_model loaded')
        rf_model = joblib.load(MODEL_PATH.joinpath('sklearn_rf.joblib'))
        if rf_model:
            print('rf_model loaded')
        #print([term, loan_amnt, grade, home_ownership, annual_inc, purpose])
        # if term is not None and term is not '':
        try:
            # user_input=  #'int_rate',
            user_df = pd.DataFrame(columns=['loan_amnt', 'term', 'grade',
                                             'emp_length', 'home_ownership', 'annual_inc',
                                             'purpose', ]).append({'loan_amnt': loan_amnt,
                                                                   'term': term,
                                                                'grade': grade,
                                                                'home_ownership': home_ownership,
                                                                'annual_inc': annual_inc,
                                                                'purpose': purpose
                                                                }, ignore_index=True)
            # user_df = pd.DataFrame(columns=df2.columns).append({'term': term,
            #                                                     'loan_amnt': loan_amnt,
            #                                                     'grade': grade,
            #                                                     'home_ownership': home_ownership,
            #                                                     'annual_inc': annual_inc,
            #                                                     'purpose': purpose
            #                                                     }, ignore_index=True)
            ### label encode the categorical values and convert them to numbers

            print(user_df)
            le=LabelEncoder()
            for i in ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']:
                le.fit(user_df[i].astype(str))
                user_df[i] = le.transform(user_df[i].astype(str))
            print(user_df)

            prob_lr = lr_model.predict_proba(user_df)[0][1]
            print(prob_lr)
            prob_rf = rf_model.predict_proba(user_df)[0][1]
            #prob_rf = 0.80
            print(prob_rf)
            #prob_lr = predict_model(lr_model, data=user_df).Score[0]
            #prob_rf = predict_model(rf_model, data=user_df).Score[0]

            prob = (prob_lr + prob_rf*3) / 4
            # print(prob)
            # print('With the above information, you have {} chance of getting a loan amount of ${:,.2f} from Lending Club'.format(prob, loan_amnt))
            approval_str = [
                'With the above information, you have {:.2%} chance of getting a loan amount of $ {} from Lending Club'.format(
                    prob, loan_amnt)]
            return approval_str
        except ValueError:
            approval_str = ['Unable to give you a prediction']
            return approval_str
    else:
        approval_str = ['opppps...something is missing from the info-happy hoiday!']
        return approval_str


# if __name__ == '__main__':
#     app.run_server(debug=True, use_reloader=False)
#     # lr_model = load_model(os.getcwd() + '\models\Final Logistic Classification Model')
#     # rf_model = load_model(os.getcwd() + '\models\Final random forest Model')
#     #app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
#     lr_model = load_model(MODEL_PATH.joinpath('\models\Final Logistic Classification Model'))
#     # if lr_model:
#     #     print('model loaded')
#     rf_model = load_model(MODEL_PATH.joinpath('\models\Final random forest Model'))











