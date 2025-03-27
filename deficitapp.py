# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import numpy as np
import scipy.stats


from dash import Dash, html, dcc, Input, Output, callback, ctx
import dash_bootstrap_components as dbc  # easier layouts
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('imf_clean_data.csv', index_col=0)
data = data.dropna(subset=['Country'])

# app = Dash()
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, "/assets/styles.css"])

def drawTable(data):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.H2("Data"),
                html.P(""),
                dbc.Table.from_dataframe(data, striped=True, bordered=True, hover=True, responsive=True),
            ]),
        ),
    ], style={"maxHeight":"300px", "overflow": "auto"},
        className="table-wrapper", #
    )

# Text field
def drawText(card_id, children="Text"):
    return dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2(children),
                ],
                    style={'textAlign': 'center'},
                id=card_id)
            ])
        )
def create_text_card(title, content):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="card-title"),
                html.P(content, className="card-text")
            ]
        ),
        style={"width": "18rem"}  # Optional: You can adjust the card width here
    )

def create_graph_card(card_id, graph_id, title):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H2(title, className="card-title"),
                dcc.Graph(graph_id)
            ]
        ),
        id=card_id,
        #style={"width": "18rem"}  # Optional: You can adjust the card width here
    )

# Fit a linear regression model
def fit_linear_regression(x, y):
    model = LinearRegression()
    x_reshape = x.reshape(-1, 1)
    model.fit(x_reshape, y)
    return model.predict(x_reshape), model

# Fit a polynomial regression model
def fit_polynomial_regression(x, y, degree=3):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    return model.predict(x_poly), model

# Fit a Ridge regression model
def fit_ridge_regression(x, y, alpha=1):
    model = Ridge(alpha=alpha)
    x_reshape = x.reshape(-1, 1)
    model.fit(x_reshape, y)
    return model.predict(x_reshape), model

def fit_huber_regression(x, y, epsilon=1):
    model = HuberRegressor(epsilon=1)
    x_reshape = x.reshape(-1, 1)
    model.fit(x_reshape, y)
    return model.predict(x_reshape), model

model_dict = {
    # 'linear': fit_linear_regression,
    # 'quadratic': fit_polynomial_regression,
    # 'ridge': fit_ridge_regression,
    'Linear Model with Huber Loss': fit_huber_regression
}

all_countries = data['Country'].unique()
all_incomes = data['Income Level'].dropna().unique()
# print(all_incomes)
max_year = data['year2'].max()
min_year = data['year1'].min()

# SIDE BAR
sidebar_content = html.Div(
    [

        html.H3('Select Income Groups', style={'marginTop': '30px'}),
        dcc.Dropdown(all_incomes,  # all options
                     all_incomes[1:],  # on start
                     multi=True,
                     #style={"Height":"400px", "maxHeight": "400px", "overflowY": "scroll"},
                     id='income-dropdown',
                     ),
        html.H3("Year Range", style={'marginTop': '30px'}),
        dcc.RangeSlider(
            id='year-slider',
            min=min_year,
            max=max(max_year,2025),
            value=[min_year, max_year],
            marks={str(year): str(year) for year in range(min_year, max_year + 2,5)},
            step=None,
        ),
        html.H3('Select Countries', style={'marginTop': '30px'}),
        dbc.Row([
            dbc.Col([
                dbc.Button(
            'Select All',
                    id='reset-button',
                    color='primary',
                    ),
                        ]),
            dbc.Col([
                dbc.Button(
            'Deselect All',
                    id='reset-button2',
                    color='primary',
                    ),
                        ]),
        ]),

        dcc.Dropdown(all_countries, # all options
                     all_countries, # on start
                     searchable=True,
                     multi=True,
                     # style={"maxHeight": "500px", "overflowY": "auto"},
                     id='country-dropdown',
                                          ),

    ],
    # vertical=True,
    # pills=True,
    style={"height": "100vh", "overflow": "auto", 'marginLeft': '10px'},
)

# Define the main content
main_content = html.Div(
    [
        # TITLE
        html.Div([
                html.H1('Deficit vs Inflation Analysis'),
            ]),

        # MAIN CARD WITH PLOTS
        html.Div([
            dbc.Card(
                dbc.CardBody([

                    # STATS
                    dbc.Row([
                        dbc.Col(create_text_card("Card 1", "You clicked 0 times"), width=4, id="card-1"),
                        dbc.Col(create_text_card("Card 2", "You clicked 0 times"), width=4, id="card-2"),
                        dbc.Col(create_text_card("Card 3", "You clicked 0 times"), width=4, id="card-3"),
                    ], align='center'),

                # MAIN TABLE
                    dbc.Row([
                        dbc.Col([
                            #drawTable(data)
                        ]),
                    ], align='center'),

                    html.Br(),html.Br(),

                # PLOT
                    dbc.Row([
                        dbc.Col([
                            create_graph_card("graph-card-1", "main-graph", "Net Surplus vs CPI"),
                        ]),
                    ], align='center'),

                    html.Br(),html.Br(),

                    dbc.Row([
                        dbc.Col([
                            create_graph_card("graph-card-2", "dist-graph-1", "Net Surplus Distribution"),
                        ]),
                        dbc.Col([
                            create_graph_card("graph-card-3", "dist-graph-2", "CPI Distribution"),
                        ]),
                    ]),

                ]), color='dark'
            )
        ])
    ],
    id="main-content",
    style={"height": "100vh", "overflowY": "auto"},
)


# CALLBACKS
@app.callback(
    Output('country-dropdown', 'value'),
    Input('reset-button', 'n_clicks_timestamp'),
    Input('reset-button2', 'n_clicks_timestamp')
)
def reset_countries(n_clicks, n_clicks2):
    button_clicked = ctx.triggered_id
    if button_clicked == 'reset-button':
        return all_countries
    if button_clicked == 'reset-button2':
        return []
    else:
        return all_countries

@callback(
[
        Output('main-graph', "figure"),
        Output('dist-graph-1', "figure"),
        Output('dist-graph-2', "figure"),
        # Output('main-graph', 'figure'),
        Output("card-1", "children"),
        Output("card-2", "children"),
        Output("card-3", "children")
    ],
    Input('year-slider', 'value'),
    Input('country-dropdown', 'value'),
    Input('income-dropdown', 'value')

)
def update_figure(selected_year, countries, incomes):

    # FILTER YEARS
    filtered_df = data[data['year1'] >= selected_year[0] ]
    filtered_df = filtered_df[filtered_df['year2'] <= selected_year[1] ]
    # FILTER COUNTRIES
    filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    # FILTER INCOMES
    filtered_df = filtered_df[filtered_df['Income Level'].isin(incomes)]

    filtered_df['Pop'] = filtered_df['Pop'].fillna(1)
    # showing only Countries with deficit
    # filtered_df['Net Surplus']
    filtered_df = filtered_df.dropna(subset=['Net Surplus','CPI'])
    x = filtered_df['Net Surplus'].to_numpy()
    y = filtered_df['CPI'].to_numpy()

    # MAIN GRAPH
    fig = px.scatter(filtered_df, x="Net Surplus", y="CPI",
                     # size="Pop",
                     color="Income Level", hover_name="NewIndex",
                     hover_data = ["Country", "NewIndex"],
                     # opacity=0.8,
                     # log_x=True,
                     size_max=50,
                     range_x=(-20,20),
                     range_y=(-10,40),
                     height=600,
                     template='ggplot2',

                     )

    # # FIT CURVES
    for model_name in model_dict:
        model_function = model_dict[model_name]
        try:
            y_fitted, model = model_function(x, y)
            # print(model.coef_)
            # print(y_fitted)
            fig.add_scatter(x=x, y=y_fitted, mode='lines', name=model_name, line=dict(width=2, dash='dash'),
                            hovertext=f'Model: {model_name}<br>Coef,:{model.coef_[0]:.3g}<br>Intercept:{model.intercept_:.3g}'
                            )
        except ValueError:
            print("Couldn't fit model")

    fig.update_layout(transition_duration=500)

    # DISTRIBUTIONS
    hist_1 = px.ecdf(
        x=filtered_df['Net Surplus'],
        markers=True, ecdfmode="reversed",
        lines = False,
        marginal="histogram",
        # nbinsx=math.ceil(math.sqrt(len(data))),  # Dynamic number of bins
        opacity=0.75,
    )

    hist_2 = px.ecdf(
        x=filtered_df['CPI'],
        markers=True, ecdfmode="reversed",
        log_x=True,
        log_y=True,
        lines=False,
        marginal="rug",
        # nbinsx=math.ceil(math.sqrt(len(data))),  # Dynamic number of bins
        opacity=0.75,
    )

    # UPDATE CARDS
    card_content = f"You clicked {np.random.random()} times"
    # print(filtered_df.shape[0])
    # print(len(countries))
    corr = scipy.stats.spearmanr(filtered_df['Net Surplus'],filtered_df['CPI'], nan_policy='omit')
    # print(corr)

    # Generate new cards with updated content
    card_1 = create_text_card("Points selected", filtered_df.shape[0])
    card_2 = create_text_card("Countries selected", len(countries))
    card_3 = create_text_card("Spearman Corr. Coef", f"r = {corr[0]:.2g} p = {corr[1]:.2g}")

    return [fig, hist_1, hist_2, card_1, card_2, card_3]

# WHOLE LAYOUT - SIDEBAR + MAIN PAGE
app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(sidebar_content, width=3),
                dbc.Col(main_content, width=9),
            ],
            #style={"height": "100vh", "overflow": "auto"},
        ),
    ],
    #style={"margin": 0},
)

if __name__ == '__main__':
    app.run(debug=False)
