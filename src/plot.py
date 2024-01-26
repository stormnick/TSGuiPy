import numpy as np
import plotly.graph_objs as go
import plotly.express as px

def create_plot():
    trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers')
    layout = go.Layout(title='Sample Plot')
    fig = go.Figure(data=[trace], layout=layout)
    return fig

def create_plot_data(x_fitted, y_fitted, x_obs, y_obs, left_line, right_line, centre_line, title):
    # plot fitted as line
    trace = go.Scatter(x=list(x_fitted), y=list(y_fitted), mode='lines', line=dict(color='red'), name='fitted')
    # plot observed data as a scatter plot
    trace_obs = go.Scatter(x=list(x_obs), y=list(y_obs), mode='markers', marker=dict(color='black'), name='observed')
    # plot the left line as a vertical line in green, no label
    trace_left_line = go.Scatter(x=[left_line, left_line], y=[0, 2], mode='lines', line=dict(color='green'), showlegend=False)
    # plot the right line as a vertical line in green
    trace_right_line = go.Scatter(x=[right_line, right_line], y=[0, 2], mode='lines', line=dict(color='green'), showlegend=False)
    # plot the centre line as a vertical line in red
    trace_centre_line = go.Scatter(x=[centre_line, centre_line], y=[0, 2], mode='lines', line=dict(color='blue'), showlegend=False)
    # xlimit is the range of x values to plot
    xlimit = [left_line - 0.1, right_line + 0.1]
    # find y_fitted that is within xlimit
    y_fitted2 = y_fitted[(x_fitted >= xlimit[0]) & (x_fitted <= xlimit[1])]
    max_y = max(max(y_fitted2) + 0.03, 1.03)
    ylimit = min(y_fitted2) - 0.03, max_y
    layout = go.Layout(title=title)
    fig = go.Figure(data=[trace_obs, trace, trace_left_line, trace_right_line, trace_centre_line], layout=layout, layout_xaxis_range=xlimit, layout_yaxis_range=ylimit)
    return fig


def create_plot_data_one_star(wavelength_fitted, flux_fitted, wavelength_observed_rv, flux_observed, left_wavelengths,
                              right_wavelengths, center_wavelengths, title):
    fig = px.scatter(x=wavelength_observed_rv, y=flux_observed, title=title)
    fig.update_layout(
        xaxis_title="Wavelength",
        yaxis_title="Flux",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig