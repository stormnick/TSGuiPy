import plotly.graph_objs as go

def create_plot():
    trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers')
    layout = go.Layout(title='Sample Plot')
    fig = go.Figure(data=[trace], layout=layout)
    return fig

def create_plot_data(x, y):
    trace = go.Scatter(x=x, y=y, mode='markers')
    layout = go.Layout(title='Sample Plot')
    fig = go.Figure(data=[trace], layout=layout)
    return fig
