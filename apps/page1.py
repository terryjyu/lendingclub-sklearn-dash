from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import pathlib
from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

df2 = pd.read_csv(DATA_PATH.joinpath("lc_cleaned_combined.csv"), low_memory=True)

# Data Processing
west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
south_west = ['AZ', 'TX', 'NM', 'OK']
south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

df2['region'] = np.nan
def finding_regions(state):
    if state in west: return 'West'
    elif state in south_west: return 'SouthWest'
    elif state in south_east: return 'SouthEast'
    elif state in mid_west: return 'MidWest'
    elif state in north_east: return 'NorthEast'
    return 'Other'

df2['region'] = df2['addr_state'].apply(finding_regions)

dff2 = df2.groupby('region', as_index=False)[['loan_amnt','funded_amnt','funded_amnt_inv']].sum()
dff3 = df2.groupby('addr_state', as_index=False)[['loan_amnt','funded_amnt','funded_amnt_inv']].sum()

pd.to_datetime(df2.issue_d,format='%b-%Y')
df2['year']=pd.to_datetime(df2.issue_d,format='%b-%Y').dt.year

df2['Charged_Off']=[1 if x=='Charged Off' else 0 for x in df2['loan_status']]
df2['Fully_Paid']=[1 if x=='Fully Paid' else 0 for x in df2['loan_status']]
df2['Fully_Paid_percentage']=[1 if x=='Fully Paid' else 0 for x in df2['loan_status']]

dff4 = df2.groupby(['addr_state','year'], as_index=False)[['Charged_Off','Fully_Paid']].sum()
dff4['Fully_Paid_percentage']=(dff4['Fully_Paid']/(dff4['Fully_Paid']+dff4['Charged_Off'])).round(4)*100


# Table Formatting
from dash.dash_table.Format import Format, Group, Scheme, Symbol
formatted = Format().scheme(Scheme.fixed).precision(0).symbol(Symbol.yes).group(Group.yes).group_delimiter(',')

# Styles for Dark Theme tables
table_header_style = {
    'backgroundColor': 'rgba(30,30,40,0.8)',
    'fontWeight': 'bold',
    'color': 'white',
    'border': '1px solid rgba(255,255,255,0.1)',
    'textAlign': 'left'
}
table_cell_style = {
    'backgroundColor': 'rgba(20,20,30,0.5)',
    'color': '#d1d5db',
    'border': '1px solid rgba(255,255,255,0.05)',
    'padding': '10px',
    'textAlign': 'left'
}
table_data_conditional = [
    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(255,255,255,0.03)'},
    {'if': {'state': 'selected'}, 'backgroundColor': 'rgba(6, 182, 212, 0.2) !important', 'border': '1px solid #06b6d4 !important'},
]

# Helper for Glass Cards
def glass_card(children, title=None):
    content = []
    if title:
        content.append(dmc.Text(title, fw=700, size="xl", c="white", className="mb-4"))
        content.append(dmc.Divider(color="gray.8", className="mb-4"))
    content.extend(children if isinstance(children, list) else [children])
    
    return dmc.Card(
        children=content,
        radius="xl",
        className="backdrop-blur-md bg-white/5 border border-white/10 shadow-xl p-6 h-full"
    )

layout = dmc.Container([
    # Header
    dmc.Stack([
        dmc.Title("Investor Data Exploration", order=1, className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400 text-4xl font-bold"),
        dmc.Text("Analyze historical lending data, regional trends, and risk metrics.", c="gray.4")
    ], className="mb-8 text-center"),

    # Row 1: Region Data and Chart
    dmc.Grid([
        # Region Table
        dmc.GridCol([
            glass_card([
                dmc.Text("Regional Overview", fw=600, c="cyan", className="mb-2"),
                dmc.Text("Select regions to filter the charts.", size="sm", c="gray.5", className="mb-4"),
                dash_table.DataTable(
                    id='datatable_id',
                    data=dff2.to_dict('records'),
                    columns=[{"name": i.title(), "id": i, "type": "numeric", "format": formatted, "selectable": True} for i in dff2.columns],
                    editable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="multi",
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=6,
                    style_as_list_view=True,
                    style_header=table_header_style,
                    style_cell=table_cell_style,
                    style_data_conditional=table_data_conditional,
                    style_table={'overflowX': 'auto'}
                )
            ])
        ], span={"base": 12, "lg": 5}),

        # Region Chart
        dmc.GridCol([
            glass_card([
                dmc.Select(
                    id='linedropdown',
                    label="Select Metric",
                    value='loan_amnt',
                    data=[
                         {'label': 'Loan Amount Applied', 'value': 'loan_amnt'},
                         {'label': 'Amount Funded', 'value': 'funded_amnt'},
                         {'label': 'Total Committed', 'value': 'funded_amnt_inv'}
                    ],
                    className="mb-4 w-64",
                    leftSection=DashIconify(icon="carbon:chart-line", width=20)
                ),
                dcc.Graph(id='linechart', className="rounded-lg overflow-hidden", style={'height': '400px'})
            ])
        ], span={"base": 12, "lg": 7}),
    ], gutter="lg", className="mb-8"),

    # Row 2: State Data & Pie Chart
    dmc.Grid([
        # State Table
        dmc.GridCol([
            glass_card([
                dmc.Text("State Statistics", fw=600, c="cyan", className="mb-2"),
                dmc.Text("Detailed breakdown by state.", size="sm", c="gray.5", className="mb-4"),
                dash_table.DataTable(
                    id='datatable2_id',
                    data=dff3.to_dict('records'),
                    columns=[{"name": i.title(), "id": i, "type": "numeric", "format": formatted, "selectable": True} for i in dff3.columns],
                    editable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="multi",
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    style_as_list_view=True,
                    style_header=table_header_style,
                    style_cell=table_cell_style,
                    style_data_conditional=table_data_conditional,
                    style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'auto'}
                )
            ])
        ], span={"base": 12, "lg": 5}),

        # Pie Chart Area
        dmc.GridCol([
            glass_card([
                dmc.Select(
                    id='piedropdown',
                    label="Select Metric",
                    value='funded_amnt',
                     data=[
                         {'label': 'Loan Amount Applied', 'value': 'loan_amnt'},
                         {'label': 'Amount Funded', 'value': 'funded_amnt'},
                         {'label': 'Total Committed', 'value': 'funded_amnt_inv'}
                    ],
                    className="mb-4 w-64",
                    leftSection=DashIconify(icon="carbon:pie-chart", width=20)
                ),
                dmc.Grid([
                    dmc.GridCol(dcc.Graph(id='piechart', style={'height': '350px'}), span={"base": 12, "md": 6}),
                    dmc.GridCol(dcc.Graph(id='linechart2', style={'height': '350px'}), span={"base": 12, "md": 6}),
                ])
            ])
        ], span={"base": 12, "lg": 7}),
    ], gutter="lg", className="mb-8"),

    # Row 3: Map
    glass_card([
        dmc.Group([
             dmc.Text("Geographic Risk Analysis", fw=700, size="xl", c="white"),
             dmc.Group([
                 dmc.Text("Select Year:", c="gray.4"),
                 dmc.NumberInput(id='input_state', value=2014, min=2007, max=2017, step=1, className="w-32"),
                 dmc.Button("Update Map", id='submit_button', color="cyan", variant="light")
             ], gap="sm")
        ], justify="space-between", className="mb-4"),
        html.Div(id='output_state', className="text-cyan-400 text-sm mb-2"),
        dcc.Graph(id='map', style={'height': '600px'})
    ], title=None),

    html.Div(className="h-8"),

    # Row 4: Box Plot
    glass_card([
        dmc.Grid([
            dmc.GridCol([
                dmc.Stack([
                    dmc.Text("Feature Correlation", fw=700, size="xl", c="white"),
                    dmc.Text("Explore relationships between loan attributes.", size="sm", c="gray.4"),
                    
                    dmc.Text("X-Axis (Categorical)", fw=600, size="sm", className="mt-4"),
                    dmc.Select(
                        id='x-axis',
                        value='grade',
                        label="Select Category",
                        data=[{'value': x, 'label': x.title().replace("_", " ")} for x in ['grade','home_ownership','purpose','emp_length']],
                        clearable=False
                    ),
                
                    dmc.Text("Y-Axis (Numerical)", fw=600, size="sm", className="mt-4"),
                    dmc.SegmentedControl(
                        id='y-axis',
                        value='int_rate',
                        data=[
                            {'label': 'Interest Rate', 'value': 'int_rate'},
                            {'label': 'Annual Income', 'value': 'annual_inc'},
                            {'label': 'Loan Amount', 'value': 'loan_amnt'}
                        ],
                        fullWidth=True,
                        color="cyan"
                    )
                ])
            ], span={"base": 12, "md": 3}),
            
            dmc.GridCol([
                dcc.Graph(id="box-plot")
            ], span={"base": 12, "md": 9})
        ])
    ]),
    
    html.Div(className="h-12")

], fluid=True, className="py-8")


# Callbacks

@app.callback(
    [Output('piechart', 'figure'),
     Output('linechart', 'figure'),
     Output('linechart2', 'figure')
     ],
    [Input('datatable_id', 'selected_rows'),
     Input('datatable2_id', 'selected_rows'),
     Input('piedropdown', 'value'),
     Input('linedropdown', 'value')
     ]
)
def update_data(chosen_rows, chosen_rows2, piedropval, linedropval):
    chosen_rows = chosen_rows or []
    chosen_rows2 = chosen_rows2 or []

    # Region Chart logic
    if len(chosen_rows) == 0:
        df_filterd = dff2.copy()
    else:
        df_filterd = dff2[dff2.index.isin(chosen_rows)]

    # State Chart logic
    if len(chosen_rows2) == 0:
        df_filterd2 = dff3.copy()
        # Default top states for visibility if "State" table is huge?
        # Preserving original logic: "default North Eastern states" mentioned in markdown but code was:
        # if len == 0: df_filterd2 = dff3[dff3['addr_state'].isin([...])] in original?
        # Let's check original logic carefully.
        # Original: if len==0: df_filterd2 = dff3[dff3['addr_state'].isin(['CT', 'NY', ...])]
        df_filterd2 = dff3[dff3['addr_state'].isin(['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME'])]
    else:
        df_filterd2 = dff3[dff3.index.isin(chosen_rows2)]

    list_chosen_regions = df_filterd['region'].tolist()
    df_line = df2[df2['region'].isin(list_chosen_regions)]

    line_chart = px.histogram(title='Loan Statistics by Region',
                              data_frame=df_line,
                              x='year',
                              y=linedropval,
                              color='region',
                              histfunc='sum',
                              template='plotly_dark')

    list_chosen_states = df_filterd2['addr_state'].tolist()
    df_line2 = df2[df2['addr_state'].isin(list_chosen_states)]

    line_chart2 = px.histogram(title='Loan Statistics by State',
                               data_frame=df_line2,
                               x='year',
                               y=piedropval,
                               color='addr_state',
                               histfunc='sum',
                               template='plotly_dark')

    pie_chart = px.pie(title='Loan Amount Distribution',
                       data_frame=df_filterd,
                       names='region',
                       values=piedropval,
                       hole=.3,
                       template='plotly_dark'
                       )

    # Common Layout Updates
    common_layout = {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': 'white'},
        'title_x': 0.5,
        'margin': dict(t=50, b=40, l=40, r=40)
    }
    
    line_chart.update_layout(uirevision='foo', **common_layout)
    line_chart2.update_layout(uirevision='foo', **common_layout)
    pie_chart.update_layout(uirevision='foo', **common_layout)
    
    return (pie_chart, line_chart, line_chart2)


@app.callback(
    [Output('output_state', 'children'),
     Output(component_id='map', component_property='figure')
     ],
    [
        Input(component_id='submit_button', component_property='n_clicks'),
        State(component_id='input_state', component_property='value')],
)
def update_output(num_clicks, val_selected):
    if val_selected is None:
        raise PreventUpdate
    
    df_map = dff4.query("year=={}".format(val_selected))

    map_chart = px.choropleth(df_map, locations="addr_state",
                        color="Fully_Paid_percentage",
                        locationmode='USA-states',
                        hover_name="addr_state",
                        hover_data=['Charged_Off', 'Fully_Paid', 'Fully_Paid_percentage'],
                        scope='usa',
                        title='State Loan Repayment Rates in ' + str(val_selected),
                        color_continuous_scale=px.colors.sequential.Teal, # Changed to Teal for better theme fit
                        template='plotly_dark')

    map_chart.update_layout(
        title=dict(font=dict(size=24), x=0.5, xanchor='center'),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='rgba(0,0,0,0)'
        )
    )

    return (f'Showing distribution for year {val_selected}', map_chart)


@app.callback(
    Output("box-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    # Ensure x is a list if px.box expects it? No, if x is string it's fine.
    # Note: original x-axis was Checklist (list). We changed to Select (string).
    # px.box(x=string) works fine.
    
    pbox = px.box(df2, x=x, y=y, color='loan_status', 
                  category_orders={'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']},
                  template='plotly_dark',
                  color_discrete_sequence=['#22d3ee', '#f472b6', '#a78bfa']) # Cyan, Pink, Purple

    pbox.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        title=f"Relationship: {x.title()} vs {y}",
        title_x=0.5
    )
    return pbox