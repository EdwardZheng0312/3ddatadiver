import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import plotly.graph_objs as go

app = dash.Dash()

file = 'doc/PHASEdata.csv'


df = pd.read_csv(file, header=None, skiprows=1)

z = df[0]
z_approach = z[:500]
z_retract = z[500:]

#phase shift
pslist = []
for k in range(len(z)):
    phaseshift = df.iloc[k,1:]  #[from zero row to the end row, from second column to the last column]
    #print(phaseshift)
    ps = np.array(phaseshift)
    ps_reshape = np.reshape(ps,(48,48))
    pslist.append(ps_reshape)


pslist = pslist

a = np.linspace(0, 47, 48)
b = np.linspace(0, 47, 48)
c = z_approach
x, z, y = np.meshgrid(a, c, b)

#phaseshift information as intensity case
psasas = []
for k in range(len(c)):
    C = pslist[k]
    for i in range(len(a)):
        B = pslist[k][i]
        for j in range(len(b)):
            A = pslist[k][i][j]
            psasas.append(A)
l = psasas

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax = ax.scatter(x, y, z, c=l, alpha=0.4)


app.layout = html.Div([
    dcc.Graph(
        id='3D_plot',
        figure = {ax
        }
    )
])

#app.layout = html.Div([
#    dcc.Graph(
#        id='test_graph',
#        figure={
#            'data': [
#                go.Scatter(
#                    x=df[0],
#                    y=df[1],
#                    mode='markers',
#                    opacity=0.7,
#                    marker={
#                        'size': 15,
#                        'line': {'width': 0.5, 'color': 'white'}
#                    },
#                )
#            ],
#            'layout': go.Layout(
#                xaxis={'title': 'GDP Per Capita'},
#                yaxis={'title': 'Life Expectancy'},
#                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#                legend={'x': 0, 'y': 1},
#            )
#        }
#    )
#])

if __name__ == '__main__':
    app.run_server()