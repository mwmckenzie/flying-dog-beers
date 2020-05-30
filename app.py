import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def load_data(filename):
    infile = open(filename, 'rb')
    return pickle.load(infile)


def save_data(file, filename):
    outfile = open(f'{filename}.pkl', 'wb')
    pickle.dump(file, outfile)
    outfile.close()


class States:
    names = ['New York', 'New Jersey', 'Massachusetts', 'Rhode Island', 'District of Columbia',
             'Connecticut', 'Delaware', 'Illinois', 'Louisiana', 'Maryland', 'Nebraska',
             'Pennsylvania', 'Iowa', 'Michigan', 'South Dakota', 'Indiana', 'Mississippi',
             'Virginia', 'Colorado', 'Georgia', 'Minnesota', 'New Mexico', 'North Dakota',
             'Kansas', 'New Hampshire', 'Alabama', 'Tennessee', 'Ohio', 'Washington',
             'Wisconsin', 'Utah', 'Nevada', 'California', 'Florida', 'North Carolina',
             'Arizona', 'Missouri', 'Arkansas', 'Kentucky', 'South Carolina', 'Texas',
             'Maine', 'Vermont', 'Oklahoma', 'Idaho', 'Wyoming', 'Puerto Rico',
             'West Virginia', 'Oregon', 'Alaska', 'Montana', 'Hawaii']

class Types:
    names = ['Models', 'Forecasts']


class State:

    def __init__(self, name, full_df, data_df):
        self.name = name
        self.full_data = np.array(full_df)
        self.full_range = full_df.index
        self.data = np.array(data_df)
        self.data_range = data_df.index

        self.best_fit_list = None
        self.best_fit_data = None
        self.shortest_model = None

        self.best_fit_mean = None
        self.best_fit_high = None
        self.best_fit_low = None

        self.fit_fwd = None
        self.fit_fwd_range = None

        self.fwd_bias = None
        self.fit_fwd_norm = None


states = load_data('states_data_plotly.pkl')
states_list = States.names

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

app.layout = html.Div([

    html.Div([

    dcc.Graph(id='output_figure'),

    ], style={'width': '75%', 'height': '100%', 'align-items': 'center', 'margin': '0 auto'}),

    html.Div([
        dcc.Dropdown(
            id='state_selection',
            options=[{'label': i, 'value': i} for i in States.names],
            value='New York'
        ),
    ],
        style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='type_selection',
            options=[{'label': i, 'value': i} for i in Types.names],
            value='Models'
        ),
    ],
        style={'width': '50%', 'display': 'inline-block'})

])


@app.callback(
    Output('output_figure', 'figure'),
    [Input('state_selection', 'value'),
     Input('type_selection', 'value')])
def update_value(input_state, input_type):
    state = input_state
    graph_type = input_type

    d = states[state].best_fit_dict
    d1 = states[state].fit_fwd_norm

    fig = go.Figure()

    if graph_type == 'Models':

        for model in d:
            mse = d[model]['mse']
            fig.add_trace(go.Scatter(
                x=d[model]['date'],
                y=d[model]['value'],
                name=f'MSE: {mse}',
                line=dict(shape='linear'),
                connectgaps=True
                #     marker = go.scatter.Marker(color=df['model'])

            ))

        fig.add_trace(go.Scatter(
            x=states[state].data_range,
            y=states[state].data,
            name='Reported Cases',
            mode='lines',
            line=go.scatter.Line(color='rgb(0,0,0)', width=5)
        ))

        fig.update_traces()
        fig.update_layout(title={'text': f'<b>{state} Reported and Modeled Confirmed Cases</b><br>Per 10,000 Population',
                                 'x': .5, 'xanchor': 'center'},
                          legend={
                              'title': 'Double-Click to isolate<br>or<br>single-click to add<br>or subtract from graph<br>',
                              'traceorder': 'reversed'},
                          height=800)  # height=800, width=1200, )  # height=1000, width=1000

        return fig

    elif graph_type == 'Forecasts':

        for model in d1:
            mse = d1[model]['mse']
            fig.add_trace(go.Scatter(
                x=d1[model]['date'],
                y=d1[model]['value_norm'],
                name=f'MSE: {mse}',
                line=dict(shape='linear'),
                connectgaps=True
                #     marker = go.scatter.Marker(color=df['model'])

            ))

        fig.add_trace(go.Scatter(
            x=states[state].data_range,
            y=states[state].data,
            name='Reported Cases',
            mode='lines',
            line=go.scatter.Line(color='rgb(0,0,0)', width=5)
        ))

        fig.update_traces()
        fig.update_layout(title={'text':
                                     f'<b>{state} Reported and Forecast Confirmed Cases</b><br>Per 10,000 Population, Bias Corrected',
                                 'x': .5, 'xanchor': 'center'},
                          legend={
                              'title': 'Double-Click to isolate<br>or single-click to add<br>or subtract from graph<br>',
                              'traceorder': 'reversed'},
                          height=800)  # height=800, width=1200)  # width=1000

        return fig

if __name__ == '__main__':
    app.run_server()
