import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc  #0.11.0
from app import app
from app import server

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

from apps import page1, page2
#

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
        dcc.Location(id='url', refresh=False),

        dcc.Link('Investor EDA Page|', href='/apps/page1'),   # first page
        dcc.Link('Lender Prediction Page', href='/apps/page2'),   #second page
    ],width={'size': 6, 'offset': 1})]),
    dbc.Row([html.Div(id='page-content', children=[])])
])



@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/page1':    #first page .py location
        # app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        return page1.layout
    if pathname == '/apps/page2':       #second page .py location
        # app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        return page2.layout
    else:
        return " ______________________________Feeling lucky today? Click above to learn more about Lending Club:)_________________________________"


if __name__ == '__main__':
    app.run_server(debug=True)
