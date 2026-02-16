from dash import dcc, html
from dash.dependencies import Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from app import app
from app import server
from apps import page1, page2

# Global Background
global_bg = html.Div(
    [
        html.Div(className="fixed inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900", style={"zIndex": -2}),
        html.Div(className="fixed inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))]", style={"zIndex": -1}),
    ]
)

# Navigation Header
header = html.Header(
    className="backdrop-blur-md bg-white/5 border-b border-white/10 sticky top-0 z-50 p-4",
    style={"height": "70px"},
    children=[
        dmc.Container(
            fluid=True,
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Group([
                            html.Img(src='/assets/LC-Logo.png', className="h-8 object-contain bg-white/10 rounded px-2 py-1"),
                            dmc.Text("LendingClub Analysis", c="white", fw=700, size="lg")
                        ], gap="md"),
                        
                        dmc.Group([
                            dcc.Link(
                                dmc.Button(
                                    "Investor EDA",
                                    leftSection=DashIconify(icon="carbon:chart-line-data", width=20),
                                    variant="subtle",
                                    color="cyan",
                                    className="hover:bg-white/10"
                                ),
                                href='/apps/page1'
                            ),
                            dcc.Link(
                                dmc.Button(
                                    "Approval Prediction",
                                    leftSection=DashIconify(icon="carbon:machine-learning-model", width=20),
                                    variant="subtle",
                                    color="purple",
                                    className="hover:bg-white/10"
                                ),
                                href='/apps/page2'
                            )
                        ], gap="sm")
                    ]
                )
            ]
        )
    ]
)

app.layout = dmc.MantineProvider(
    theme={
        "colorScheme": "dark",
        "primaryColor": "cyan",
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        "headings": {"fontFamily": "'Inter', 'Segoe UI', sans-serif", "fontWeight": 700},
    },
    children=[
        global_bg,
        dcc.Location(id='url', refresh=False),
        header,
        html.Div(id='page-content', className="min-h-screen")
    ]
)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/page1':
        return page1.layout
    if pathname == '/apps/page2':
        return page2.layout
    # Default to page 1 or a specific home
    return page1.layout # Changing default to page 1 for convenience, or page 2 if preferred. User didn't specify.

if __name__ == '__main__':
    app.run(debug=True)
