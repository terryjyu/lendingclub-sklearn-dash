import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import pathlib
from app import app

#get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
#
df2 = pd.read_csv(DATA_PATH.joinpath("lc_cleaned_combined.csv"),low_memory=True)  # Cleaned data

df2['addr_state'].unique()

# Make a list with each of the regions by state.

west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
south_west = ['AZ', 'TX', 'NM', 'OK']
south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

# make regions
df2['region'] = np.nan
def finding_regions(state):
    if state in west:
        return 'West'
    elif state in south_west:
        return 'SouthWest'
    elif state in south_east:
        return 'SouthEast'
    elif state in mid_west:
        return 'MidWest'
    elif state in north_east:
        return 'NorthEast'


df2['region'] = df2['addr_state'].apply(finding_regions)

# making region and state tables for charts
dff2 = df2.groupby('region', as_index=False)[['loan_amnt','funded_amnt','funded_amnt_inv']].sum()

dff3 = df2.groupby('addr_state', as_index=False)[['loan_amnt','funded_amnt','funded_amnt_inv']].sum()

# add a year column to df2 for x-axis in charts
pd.to_datetime(df2.issue_d,format='%b-%Y')
df2['year']=pd.to_datetime(df2.issue_d,format='%b-%Y').dt.year

# make Charged Off, Fully Paid counts, their percentages in table columns
df2['Charged_Off']=[1 if x=='Charged Off' else 0 for x in df2['loan_status']]
df2['Fully_Paid']=[1 if x=='Fully Paid' else 0 for x in df2['loan_status']]
df2['Fully_Paid_percentage']=[1 if x=='Fully Paid' else 0 for x in df2['loan_status']]
# state, year table Charged off/Fully paid, percentages
dff4 = df2.groupby(['addr_state','year'], as_index=False)[['Charged_Off','Fully_Paid']].sum()
dff4['Fully_Paid_percentage']=(dff4['Fully_Paid']/(dff4['Fully_Paid']+dff4['Charged_Off'])).round(4)*100



###############################################################################################
###############################################################################################
###############################################################################################
# ----------------------------------App Layout - Page 1----------------------##################



#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])


    ###########frist row########
data_table1 =html.Div([dcc.Markdown('''
    **Historical Loan Data by Region**

    *Select regions to generate a distribution histogram(default all regions)*
    ''',),
                       dash_table.DataTable(
            id='datatable_id',
            data=dff2.to_dict('records'),
            columns=[
                {"name": i.title(), "id": i, "deletable": False, "selectable": True} for i in dff2.columns
            ],
            tooltip={
        'addr_state': 'Click the box to the left to generate histogram',
        'loan_amnt': 'Loan Amount Applied by Borrower',
        'funded_amnt': 'Amount Funded by Lenders',
        'funded_amnt_inv': 'Total Committed by Investors',
          },css=[{'selector': '.dash-table-tooltip',
                  'rule':'background-color: black; font-family:monospace;'}],
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="multi",
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 6,
            # page_action='none',
            # style_cell={
            # 'whiteSpace': 'normal'
            # },
            # fixed_rows={ 'headers': True, 'data': 0 },
            # virtualization=False,
            style_cell_conditional=[
                {'if': {'column_id': 'region'},
                 'width': '40%', 'textAlign': 'left'},
                {'if': {'column_id': 'loan_amnt'},
                 'width': '20%', 'textAlign': 'left'},
                {'if': {'column_id': 'funded_amnt'},
                 'width': '20%', 'textAlign': 'left'},
                {'if': {'column_id': 'funded_amnt_inv'},
                 'width': '20%', 'textAlign': 'left'},
            ],
            style_as_list_view=False,
            style_cell={'padding':'5px','backgroundColor':'#313539',
                          'color':'white'},
             style_header={'backgroundColor':'#313539',
                            'fontWeight':'bold'},
            style_data_conditional=[
                {
                    'if':{'row_index':'odd'},
                    'backgroundColor':'rgb(50,50,50)'
                },
                {
                     'if':{
                         'column_id':'loan_amnt',

                         'filter_query':'{{loan_amnt}}={}'.format(dff2['loan_amnt'].max())
                     },
                     'backgroundColor':'#9e2f24',
                     'color':'white'

                },
                {
                     'if':{
                         'column_id':'funded_amnt',

                         'filter_query':'{{funded_amnt}}={}'.format(dff2['funded_amnt'].max())
                     },
                     'backgroundColor':'#9e2f24',
                     'color':'white'




                },
{
                     'if':{
                         'column_id':'funded_amnt_inv',

                         'filter_query':'{{funded_amnt_inv}}={}'.format(dff2['funded_amnt_inv'].max())
                     },
                     'backgroundColor':'#9e2f24',
                     'color':'white'




                },


                ]
        )])


#------------------------Chart1-----

chart1=html.Div([

            dcc.Markdown('''
    Choose from dropdown to show:
    '''),
            dcc.Dropdown(id='linedropdown',
                options=[
                         {'label': 'Loan Amount Applied by Borrowers', 'value': 'loan_amnt'},
                         {'label': 'Amount Funded by Lenders', 'value': 'funded_amnt'},
                         {'label': 'Total Committed by Investors', 'value': 'funded_amnt_inv'}
                ],
                value='loan_amnt',
                multi=False,
                clearable=False,
                style={'background-Color':'#212121',
                       'color':'#212121'},
            ),
            dcc.Graph(id='linechart',style={'backgroundColor':'rgb(26,25,25)','paper_bgcolor':'rgb(26,25,25)'}),

            ])  #className='eight columns'





    #############chart2#########

data_table2=html.Div([
            dcc.Markdown('''
    __**Historical Loan Data by State**__

    *Select states to generate a distribution histogram(default North Eastern states) *
    '''),
        dash_table.DataTable(
            id='datatable2_id',
            data=dff3.to_dict('records'),
            columns=[
                {"name": i.title(), "id": i, "deletable": False, "selectable": True} for i in dff3.columns
            ],
            tooltip={
        'addr_state': 'Click the box to the left to generate histogram',
        'loan_amnt': 'Loan Amount Applied by Borrower',
        'funded_amnt': 'Amount Funded by Lenders',
        'funded_amnt_inv': 'Total Committed by Investors',
            },css=[{'selector': '.dash-table-tooltip',
                  'rule':'background-color: black; font-family:monospace;'}],
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="multi",
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 26,
            #page_action='native',
            # style_cell={
            # 'whiteSpace': 'normal'
            # },
            #fixed_rows={ 'headers': True},
            # virtualization=False,
            style_table={'height':'1000px','overflowY':'auto'},
            style_cell_conditional=[
                {'if': {'column_id': 'addr_state'},
                 'width': '40%', 'textAlign': 'left'},
                {'if': {'column_id': 'loan_amnt'},
                 'width': '20%', 'textAlign': 'left'},
                {'if': {'column_id': 'funded_amnt'},
                 'width': '20%', 'textAlign': 'left'},
                {'if': {'column_id': 'funded_amnt_inv'},
                 'width': '20%', 'textAlign': 'left'},
            ],style_cell={'padding':'5px',
                            'backgroundColor':'#313539',
                          'color':'white'},
            style_header={'backgroundColor':'#313539',
                            'fontWeight':'bold'},
            style_data_conditional=[
                {
                    'if':{'row_index':'odd'},
                    'backgroundColor':'rgb(50,50,50)'
                },
                 {
                     'if':{
                         'column_id':'loan_amnt',

                         'filter_query':'{{loan_amnt}}={}'.format(dff3['loan_amnt'].max())
                     },
                     'backgroundColor':'#2f8694',
                     'color':'white'




                },
                {
                     'if':{
                         'column_id':'funded_amnt',

                         'filter_query':'{{funded_amnt}}={}'.format(dff3['funded_amnt'].max())
                     },
                     'backgroundColor':'#2f8694',
                     'color':'white'




                },
                {
                     'if':{
                         'column_id':'funded_amnt_inv',

                         'filter_query':'{{funded_amnt_inv}}={}'.format(dff3['funded_amnt_inv'].max())
                     },
                     'backgroundColor':'#2f8694',
                     'color':'white'




                },



                 {
                     'if':{
                         'column_id':'loan_amnt',

                         'filter_query':'{{loan_amnt}}={}'.format(dff3['loan_amnt'].min())
                     },
                     'backgroundColor':'#924f48',
                     'color':'white'




                },
                {
                     'if':{
                         'column_id':'funded_amnt',

                         'filter_query':'{{funded_amnt}}={}'.format(dff3['funded_amnt'].min())
                     },
                     'backgroundColor':'#924f48',
                     'color':'white'




                },
                {
                     'if':{
                         'column_id':'funded_amnt_inv',

                         'filter_query':'{{funded_amnt_inv}}={}'.format(dff3['funded_amnt_inv'].min())
                     },
                     'backgroundColor':'#924f48',
                     'color':'white'




                },






            ],


        )])





chart2=html.Div([
            dcc.Markdown('''
    Choose from dropdown to show:
    '''),
            dcc.Dropdown(id='piedropdown',
                    options=[
                     {'label': 'Loan Amount Applied by Borrower', 'value': 'loan_amnt'},
                     {'label': 'Amount Funded by Lender', 'value': 'funded_amnt'},
                     {'label': 'Total Committed by Investors', 'value': 'funded_amnt_inv'}
            ],
            value='funded_amnt',
            multi=False,
            clearable=False,
            placeholder='love being dragged....',
            style={'background-Color':'#212121',
                       'color':'#212121'},
        ),
            dcc.Graph(id='linechart2'), #state chart
            dcc.Graph(id='piechart'),   #pie chart

            ])






  ########### map chart ##########


map_chart=dcc.Graph(id='map')
map_io=html.Div([dcc.Input(id='input_state', type='number', inputMode='numeric', value=2014,
                        max=2017, min=2007, step=1, required=True),
                     html.Div(id='output_state'),
                    html.Button(id='submit_button', n_clicks=0, children='Submit'),
            ],style={'align':'centered'})


    ############### box plot#################
box_input=html.Div([
            dcc.Markdown('''
    **Customized Boxplot**
    '''),
        html.P("x-axis:"),
        dcc.Checklist(
        id='x-axis',
        options=[{'value': x, 'label': x}
                 for x in ['grade','home_ownership','purpose','emp_length']],
        value=['grade'],
        labelStyle={'display': 'inline-block'}
    ),
        html.P("y-axis:"),
        dcc.RadioItems(
        id='y-axis',
        options=[{'value': x, 'label': x}
                 for x in ['int_rate', 'annual_inc', 'loan_amnt']],
        value='int_rate',
        labelStyle={'display': 'inline-block'}
    )])

box_plot=dcc.Graph(id="box-plot")


layout = html.Div([
    # 1st row__________________
    dbc.Row([
        dbc.Col(html.H3("Investor Data Exploration Page"), width={'size': 6, 'offset': 5}),
    ]),  # col1
    dbc.Row([
        html.Br(),
        html.Br(),
        ]),
    # 2nd row___________________data table and dropdown+chart
    dbc.Row([
        dbc.Col(data_table1,               # r2c1
                width={'size': 5, 'offset': 1}),
        dbc.Col(chart1,                     #r2c2
                width={'size': 6, 'offset': 0})

    ]),
    # 3rd row__________________
    html.Br(),
    dbc.Row([
        dbc.Col(data_table2, width={'size': 5, 'offset': 1},align='center'),  # r3 col1
        dbc.Col(chart2, width={'size': 6, 'offset': 0},align='center'),  # r3 col2


    ], justify='end',style={'fontcolor':'rgb(255,255,255)'}),
  # 4th row
    dbc.Row([
    dbc.Col(map_chart,width={'size':10,"order":"1"}),
    dbc.Col([html.Br(),html.P('Choose a year:'),map_io],width={'size':2,"order":"last"})
        #dbc.Col(chart2,width={'size':4,"order":"last"}),

    ], justify='end'),
    #5th row
    dbc.Row([

        dbc.Col([box_input,html.P('Choose your x,y to explore relationships: few secs to load your x and y inputs...',style={'color':'green'},)],width={'size': 2, 'offset': 1},align='center'),

        dbc.Col(box_plot,width={'size':9,"order":"last"}),

    ], justify='around'),
],style={'color':'rgb(255,255,255)'})
#__________________________App Call Back- page 1----------##########

# @app.callback(
#     Output(component_id='my-bar', component_property='figure'),
#     [Input(component_id='genre-dropdown', component_property='value'),
#      Input(component_id='sales-dropdown', component_property='value')]
# )
# def display_value(genre_chosen, sales_chosen):
#     dfv_fltrd = dfv[dfv['Genre'] == genre_chosen]
#     dfv_fltrd = dfv_fltrd.nlargest(10, sales_chosen)
#     fig = px.bar(dfv_fltrd, x='Video Game', y=sales_chosen, color='Platform')
#     fig = fig.update_yaxes(tickprefix="$", ticksuffix="M")
#     return fig
@app.callback(
    [Output('piechart', 'figure'),
     Output('linechart', 'figure'),
     Output('linechart2', 'figure')
     ],
    [Input('datatable_id', 'selected_rows'),
     Input('datatable2_id', 'selected_rows'),
     Input('piedropdown', 'value'),
     Input('linedropdown', 'value')
     ],
    #  prevent_initial_call=True
)
def update_data(chosen_rows, chosen_rows2, piedropval, linedropval):
    # Region Chart
    if len(chosen_rows) == 0:
        df_filterd = dff2[dff2['region'].isin(['MidWest', 'NorthEast', 'SouthEast', ' SouthWest', 'West'])]
    else:
        #print(chosen_rows)
        df_filterd = dff2[dff2.index.isin(chosen_rows)]

    # State Chart
    if len(chosen_rows2) == 0:
        df_filterd2 = dff3[dff3['addr_state'].isin(['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME'])]
    else:
        #print(chosen_rows2)
        df_filterd2 = dff3[dff3.index.isin(chosen_rows2)]

    # extract list of chosen regions
    list_chosen_regions = df_filterd['region'].tolist()
    # filter original df according to chosen regions
    # because original df has all the complete month data from 2007-2015
    df_line = df2[df2['region'].isin(list_chosen_regions)]

    line_chart = px.histogram(title='Loan Statistics by Region Over Year',
                              data_frame=df_line,
                              x='year',
                              y=linedropval,
                              color='region',
                              histfunc='sum',
                              labels={'region': 'Regions', 'year': 'year'})

    # extract list of chosen states
    list_chosen_states = df_filterd2['addr_state'].tolist()
    # filter original df according to chosen states
    # because original df has all the complete month data from 2007-2015
    df_line2 = df2[df2['addr_state'].isin(list_chosen_states)]

    line_chart2 = px.histogram(title='Loan Statistics by State Over Year',
                               data_frame=df_line2,
                               x='year',
                               y=linedropval,
                               color='addr_state',
                               histfunc='sum',
                               labels={'addr_state': 'States', 'year': 'year'})

    pie_chart = px.pie(title='Loan Amount Distribution by Region',
                       data_frame=df_filterd,
                       names='region',
                       values=piedropval,
                       hole=.3,
                       labels={'region': 'Regions'}
                       )

    # line_chart = px.line(
    #         data_frame=df_line,
    #         x='year',
    #         y=linedropval,
    #         color='region',
    #         labels={'region':'Regions', 'year':'year'},
    #         )
    line_chart.update_layout(uirevision='foo', title_x=0.5,plot_bgcolor='rgb(39, 43, 48)', paper_bgcolor= 'rgb(39, 43, 48)',font={'color':'white'})
    line_chart2.update_layout(uirevision='foo', title_x=0.5,plot_bgcolor='rgb(39, 43, 48)',paper_bgcolor= 'rgb(39, 43, 48)',font={'color':'white'})
    pie_chart.update_layout(uirevision='foo', title_x=0.5,plot_bgcolor='rgb(39, 43, 48)', paper_bgcolor= 'rgb(39, 43, 48)',font={'color':'white'})
    return (pie_chart, line_chart, line_chart2)

    ##for map layout update


@app.callback(
    [Output('output_state', 'children'),
     Output(component_id='map', component_property='figure')
     ],
    [
        Input(component_id='submit_button', component_property='n_clicks'),
        State(component_id='input_state', component_property='value')],
    #  prevent_initial_call=True
)
def update_output(num_clicks, val_selected):
    if val_selected is None:
        raise PreventUpdate
    else:
        df_map = dff4.query("year=={}".format(val_selected))
        # print(df[:3])

        map = px.choropleth(df_map, locations="addr_state",
                            color="Fully_Paid_percentage",
                            locationmode='USA-states',
                            hover_name="addr_state",
                            hover_data=['Charged_Off', 'Fully_Paid', 'Fully_Paid_percentage'],
                            # projection='equirectangular',
                            scope='usa',
                            title='(Analyzing Risks)\n State Loan Status in ' + str(val_selected),
                            color_continuous_scale=px.colors.sequential.RdBu)

        map.update_layout(title=dict(font=dict(size=28), x=0.5, xanchor='center'),
                          margin=dict(l=60, r=60, t=50, b=50), height=650,
                          plot_bgcolor='rgb(39, 43, 48)', paper_bgcolor= 'rgb(39, 43, 48)'
                          ,font={'color':'white'},geo_bgcolor='rgb(39, 43, 48)')

        return ('Type in a year in the box to see year {} distribution. Button\
                clicked {} times'.format(val_selected, num_clicks), map)


##########-----update box plots
@app.callback(
    Output("box-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    pbox = px.box(df2, x=x, y=y, color='loan_status', category_orders={'grade': {'A', 'B', 'C', 'D', 'E', 'F', 'G'}},)
    pbox.update_layout(plot_bgcolor='rgb(39, 43, 48)', paper_bgcolor= 'rgb(39, 43, 48)',font={'color':'white'})
    return pbox


# ------------------------------------------------------------------
#
# if __name__ == '__main__':
#     app.run_server(debug=True,port=1200)#, use_reloader=False