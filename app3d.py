import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

app_iplot = dash.Dash()

app_iplot.layout = html.Div([
    dcc.Input(id='plot_title', value='Type title...', type="text"),
    dcc.Slider(
        id='box_size',
        min=1,
        max=10,
        value=4,
        step=1,
        marks=list(range(0, 10))
    ),
    html.Div([html.Img(id = 'cur_plot', src = '')],
             id='plot_div')
])

@app_iplot.callback(
    Output(component_id='cur_plot', component_property='src'),
    [Input(component_id='plot_title', component_property='value'), Input(component_id = 'box_size', component_property='value')]
)
def update_graph(input_value, n_val):
    fig, ax1 = plt.subplots(1,1)
    np.random.seed(len(input_value))
    ax1.matshow(np.random.uniform(-1,1, size = (n_val,n_val)))
    ax1.set_title(input_value)
    out_url = fig_to_uri(fig)
    return out_url

if __name__ == '__main__':
    app_iplot.run_server()