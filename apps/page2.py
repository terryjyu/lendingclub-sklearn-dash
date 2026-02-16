from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd
import pathlib
from app import app
import joblib
from sklearn.preprocessing import LabelEncoder

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
MODEL_PATH = PATH.joinpath("../models").resolve()

df2 = pd.read_csv(DATA_PATH.joinpath("lc_cleaned_combined.csv"), low_memory=True)

# Load models and encoders globally to prevent reloading on every callback
try:
    lr_model = joblib.load(MODEL_PATH.joinpath('sklearn_lr.joblib'))
    rf_model = joblib.load(MODEL_PATH.joinpath('sklearn_rf.joblib'))
    encoders = joblib.load(MODEL_PATH.joinpath('label_encoders.joblib'))
except Exception as e:
    print(f"Error loading models: {e}")
    lr_model, rf_model, encoders = None, None, None

def create_info_card(image_src, title, description, link, color="pink"):
    """Create a glassmorphic info card with vibrant accents"""
    return dmc.Card(
        children=[
            dmc.CardSection(
                dmc.Image(
                    src=image_src,
                    h=180,
                    fit="cover"
                ),
                className="relative overflow-hidden"
            ),
            dmc.Stack([
                dmc.Text(title, fw=700, size="lg", c="white", className="mt-2"),
                dmc.Text(
                    description,
                    size="sm",
                    c="gray.4",
                    className="mb-3"
                ),
                dmc.Anchor(
                    dmc.Group([
                        dmc.Text("Learn more", size="sm", c=color),
                        DashIconify(icon="carbon:arrow-right", color=color, width=16)
                    ], gap="xs"),
                    href=link,
                    target="_blank",
                    className="no-underline"
                )
            ], gap="xs", p="md")
        ],
        withBorder=False,
        shadow="xl",
        radius="lg",
        className=f"backdrop-blur-md bg-white/10 border border-white/20 hover:bg-white/15 hover:shadow-2xl hover:scale-105 transition-all duration-300"
    )

layout = html.Div([
    dmc.Container([
        # Header Section with gradient text
        dmc.Stack([
            html.Div([
                dmc.Title(
                    "Loan Approval Prediction",
                    order=1,
                    ta="center",
                    className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-teal-400 text-4xl md:text-5xl font-bold mb-2"
                ),
                dmc.Text(
                    "Powered by AI & Machine Learning",
                    ta="center",
                    size="sm",
                    c="gray.4",
                    className="mb-2"
                ),
                dmc.Group([
                    dmc.Badge("Leagacy Random Forest", color="blue", variant="dot", size="lg"),
                    dmc.Badge("Logistic Regression", color="cyan", variant="dot", size="lg"),
                ], justify="center", className="mb-4")
            ]),
            
            dmc.Text(
                "LendingClub enables borrowers to create unsecured personal loans between $1,000 and $40,000. Get instant predictions on your approval odds.",
                ta="center",
                c="gray.3",
                size="md",
                className="max-w-3xl mx-auto mb-8"
            ),
        ], gap="xs", className="mb-10"),

        # Main Form Area with glassmorphism
        dmc.Grid([
            # Left Column: Basic Info
            dmc.GridCol([
                dmc.Card([
                    # Logo section
                    dmc.CardSection(
                        html.Div(
                            html.Img(src='/assets/LC-Logo.png', className="h-14 object-contain mx-auto"),
                            className="p-6 bg-gradient-to-br from-white/10 to-white/5"
                        )
                    ),
                    
                    dmc.Stack([
                        dmc.Group([
                            DashIconify(icon="carbon:data-base", color="cyan", width=24),
                            dmc.Text("Loan Details", fw=700, size="xl", c="white"),
                        ], gap="sm"),
                        
                        dmc.Divider(color="gray.7", className="my-2"),
                        
                        dmc.Select(
                            id='term',
                            label="Loan Term",
                            description="Select your preferred repayment period",
                            data=[{'label': i, 'value': i} for i in [' 36 months', ' 60 months']],
                            value=' 36 months',
                            leftSection=DashIconify(icon="carbon:calendar", width=20),
                            className="w-full"
                        ),
                        
                        dmc.Select(
                            id='emp_length',
                            label="Employment Length",
                            description="How long have you been employed?",
                            data=[{'label': i, 'value': i} for i in ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']],
                            value='< 1 year',
                            leftSection=DashIconify(icon="carbon:badge", width=20),
                            className="w-full"
                        ),
                        
                        dmc.Select(
                            id='grade',
                            label="Credit Grade",
                            description="Your credit score category",
                            data=[{'label': i, 'value': i} for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']],
                            value='A',
                            leftSection=DashIconify(icon="carbon:star", width=20),
                            className="w-full"
                        ),
                        
                        dmc.Select(
                            id='home_ownership',
                            label="Home Ownership",
                            description="Current housing situation",
                            data=[{'label': i, 'value': i} for i in ['RENT', 'OWN', 'MORTGAGE', 'OTHER']],
                            value='RENT',
                            leftSection=DashIconify(icon="carbon:home", width=20),
                            className="w-full"
                        ),
                        
                        dmc.Select(
                            id='purpose',
                            label="Loan Purpose",
                            description="What will you use this loan for?",
                            data=[{'label': i.replace('_', ' ').title(), 'value': i} for i in ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'small_business', 'car', 'vacation', 'moving', 'house', 'wedding', 'renewable_energy', 'educational']],
                            value='debt_consolidation',
                            leftSection=DashIconify(icon="carbon:wallet", width=20),
                            className="w-full"
                        ),
                    ], gap="lg", p="xl")
                ], shadow="xl", radius="xl", withBorder=False, 
                   className="backdrop-blur-md bg-slate-800/50 border border-slate-700/50 hover:bg-slate-800/70 transition-all duration-300")
            ], span={"base": 12, "md": 6}),

            # Right Column: Financial Info & Action
            dmc.GridCol([
                dmc.Card([
                    dmc.CardSection(
                        html.Div(
                            dmc.Image(src='/assets/ap.png', h=220, fit="cover"),
                            className="relative overflow-hidden"
                        )
                    ),
                    
                    dmc.Stack([
                        dmc.Group([
                            DashIconify(icon="carbon:currency-dollar", color="green", width=24),
                            dmc.Text("Financial Information", fw=700, size="xl", c="white"),
                        ], gap="sm"),
                        
                        dmc.Divider(color="gray.7", className="my-2"),
                        
                        dmc.Alert(
                            children=[
                                dmc.Text("Your data is never stored", fw=500, size="sm")
                            ],
                            title="Privacy First",
                            color="blue",
                            variant="light",
                            icon=DashIconify(icon="carbon:security", width=20),
                            className="bg-blue-500/10 border border-blue-500/30"
                        ),

                        dmc.NumberInput(
                            id='annual_inc',
                            label="Annual Income",
                            description="Your total yearly income in USD",
                            min=1000, max=10000000, step=1000,
                            placeholder="e.g., 50000",
                            leftSection=DashIconify(icon="carbon:money", width=20),
                            className="w-full",
                            thousandSeparator=","
                        ),

                        dmc.NumberInput(
                            id='loan_amnt',
                            label="Requested Loan Amount",
                            description="Between $1,000 and $40,000",
                            min=1000, max=40000, step=500,
                            placeholder="e.g., 15000",
                            leftSection=DashIconify(icon="carbon:currency-dollar", width=20),
                            className="w-full",
                            thousandSeparator=","
                        ),

                        dmc.Button(
                            [
                                dmc.Group([
                                    DashIconify(icon="carbon:machine-learning-model", width=20),
                                    dmc.Text("Get AI Prediction", size="md", fw=600)
                                ], gap="xs")
                            ],
                            id="Get Pre-approved",
                            color="cyan",
                            gradient={"from": "blue", "to": "cyan", "deg": 45},
                            variant="gradient",
                            fullWidth=True,
                            size="lg",
                            className="mt-4 shadow-lg shadow-cyan-500/50 hover:shadow-cyan-500/70 hover:scale-105 transition-all duration-300"
                        ),
                        
                        html.Div(id='prediction_result', className="w-full")
                    ], gap="lg", p="xl")
                ], shadow="xl", radius="xl", withBorder=False,
                   className="backdrop-blur-md bg-slate-800/50 border border-slate-700/50 hover:bg-slate-800/70 transition-all duration-300")
            ], span={"base": 12, "md": 6}),
        ], gutter="xl", className="mb-12"),

        # Bottom Info Cards
        dmc.Title("Financial Resources", order=2, ta="center", c="white", className="mb-6"),
        dmc.SimpleGrid(
            cols=3,
            spacing="lg",
            children=[
                create_info_card(
                    "/assets/control_spending.jpg",
                    "Control Spending",
                    "Shop smarter and manage your finances effectively with proven strategies.",
                    "https://www.smartaboutmoney.org/Topics/Spending-and-Borrowing/Control-Spending",
                    "pink"
                ),
                create_info_card(
                    "/assets/debt.jpg",
                    "Manage Debt",
                    "Smart ways to pay off debt and improve your credit score.",
                    "https://www.smartaboutmoney.org/Topics/Spending-and-Borrowing/Deal-With-Debt",
                    "violet"
                ),
                create_info_card(
                    "/assets/Know-Your-Borrowing-Options.jpg",
                    "Borrowing Options",
                    "Understand your options and qualify for better loan terms.",
                    "https://www.smartaboutmoney.org/Topics/Spending-and-Borrowing/Know-Borrowing-Options",
                    "cyan"
                )
            ],
            className="mb-12"
        ),

        # Footer Quote
        dmc.Card([
            dmc.Blockquote(
                children=[
                    html.P("A learning experience is one of those things that says, 'You know that thing you just did? Don't do that.'"),
                    html.Footer([
                        html.Small("— Douglas Adams", className="text-gray-400 italic")
                    ])
                ],
                className="text-gray-300 border-l-4 border-purple-500"
            )
        ], withBorder=False, className="backdrop-blur-md bg-white/5 border border-white/10 p-6", radius="lg"),
        
    ], fluid=True, className="py-12 px-4"),
])

# Callbacks

# Prediction Logic
@app.callback(
    Output('prediction_result', 'children'),
    [Input('Get Pre-approved', 'n_clicks')],
    [State('term', 'value'),
     State('loan_amnt', 'value'),
     State('grade', 'value'),
     State('home_ownership', 'value'),
     State('annual_inc', 'value'),
     State('purpose', 'value'),
     State('emp_length', 'value')],
    prevent_initial_call=True)
def handle_prediction(n_clicks, term, loan_amnt, grade, home_ownership, annual_inc, purpose, emp_length):
    if n_clicks is None:
        return no_update
        
    if all([term, loan_amnt is not None, grade, home_ownership, annual_inc is not None, purpose, emp_length]):
        if None in [lr_model, rf_model, encoders]:
            return dmc.Alert("Models not loaded. Please contact support.", title="System Error", color="red")

        try:
            # Create DataFrame from user input
            user_df = pd.DataFrame([{
                'loan_amnt': loan_amnt,
                'term': term,
                'grade': grade,
                'emp_length': emp_length,
                'home_ownership': home_ownership,
                'annual_inc': annual_inc,
                'purpose': purpose
            }])
            
            # Transform features using saved encoders
            for col in ['term', 'grade', 'home_ownership', 'purpose', 'emp_length']:
                le = encoders[col]
                try:
                    user_df[col] = le.transform(user_df[col].astype(str))
                except ValueError:
                    return dmc.Alert(f'Error: Invalid input value for {col}', title="Validation Error", color="red")

            # Ensure column order matches training
            user_df = user_df[['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']]

            prob_lr = lr_model.predict_proba(user_df)[0][1]
            prob_rf = rf_model.predict_proba(user_df)[0][1]

            prob = (prob_lr + prob_rf*3) / 4
            
            # Create visually appealing result message
            percentage = f"{prob:.1%}"
            emoji = "🎉" if prob > 0.7 else "✅" if prob > 0.5 else "⚠️"
            bg_color = "bg-green-500/10" if prob > 0.5 else "bg-yellow-500/10"
            border_color = "border-green-500/30" if prob > 0.5 else "border-yellow-500/30"
            
            result_content = dmc.Card(
                children=[
                    dmc.Group([
                        dmc.Text("Approval Odds:", fw=500, c="gray.3"),
                        dmc.Badge("High" if prob > 0.7 else "Moderate" if prob > 0.5 else "Low", 
                                 color="green" if prob > 0.5 else "yellow", 
                                 variant="light")
                    ], justify="space-between", className="mb-2"),
                    
                    dmc.Text(f"{percentage}", size="3.5rem", fw=800, ta="center", 
                            className="leading-none text-transparent bg-clip-text bg-gradient-to-r from-green-300 via-emerald-400 to-teal-400 drop-shadow-lg"),
                            
                    dmc.Text(f"Estimated for ${loan_amnt:,}", size="sm", c="dimmed", ta="center", className="mt-1"),
                    
                    dmc.Divider(className="my-3 border-gray-600"),
                    
                    dmc.Text(
                        f"Based on your profile, you have a {percentage} chance of approval.", 
                        size="sm", c="white", ta="center"
                    )
                ],
                className=f"mt-6 border {border_color} {bg_color} backdrop-blur-sm animate-fade-in-up",
                radius="lg",
                p="lg"
            )
            
            return result_content

        except Exception as e:
            print(f"Prediction Error: {e}")
            return dmc.Alert(f'Unable to generate prediction: {str(e)}', title="Error", color="red")
    else:
        return dmc.Alert('Please fill in all fields to get your prediction.', title="Missing Information", color="yellow")
