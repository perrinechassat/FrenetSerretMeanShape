from matplotlib.pyplot import legend
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

layout = go.Layout(
    # paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

""" Set of functions for visualization of mean shape, mean curvature, torsion, etc (visualize 3D curves or 2D curves). """

color_list_mean = px.colors.qualitative.Plotly
dict_color = {"True Mean" : color_list_mean[0], "Arithmetic Mean" : px.colors.qualitative.Set1[6], "SRVF Mean" : px.colors.qualitative.Set1[8], "FS Mean" : px.colors.qualitative.Dark24[5], "Extrinsic Mean" : color_list_mean[2], "Individual Mean" : color_list_mean[3], "True Mean 2" : color_list_mean[1]}
color_list = px.colors.qualitative.Plotly

def plot_array_2D(x, array_y, name_ind, legend={"index":False}):
    fig = go.Figure(layout=layout)
    N = array_y.shape[0]
    for i in range(N):
        fig.add_trace(go.Scatter(x=x, y=array_y[i,:], mode='lines', name=name_ind+str(i), line=dict(width=1, color=color_list[(i-9)%9])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

def plot_2array_2D(x, array_y, legend={"index":False}):
    fig = go.Figure(layout=layout)
    N = array_y.shape[0]
    for i in range(N):
        fig.add_trace(go.Scatter(x=x[i], y=array_y[i,:], mode='lines', line=dict(width=2, color=color_list[(i-9)%9])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

def plot_2D(x, y,  legend={"index":False}):
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=2, color=color_list[0])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()


def plot_3D_means_grey(features1, features2, names1, names2):
    fig = go.Figure(layout=layout)

    for i, feat in enumerate(features2):
        feat = np.array(feat)
        fig.add_trace(
            go.Scatter3d(
                x=feat[:,0],
                y=feat[:,1],
                z=feat[:,2],
                mode='lines',
                name=names2[i],
                line=dict(
            width=5,
            color=dict_color[names2[i]],
            )
            )
        )

    for i, feat in enumerate(features1):
        feat = np.array(feat)
        fig.add_trace(
            go.Scatter3d(
                x=feat[:,0],
                y=feat[:,1],
                z=feat[:,2],
                mode='lines',
                line=dict(
                width=0.8,
                dash='solid',
                color='grey',),
                showlegend=False
            )
        )
    fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    yaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    zaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),),
                  )

    fig.show()

def plot_3D_means(features1, features2, names1, names2, path=""):
    fig = go.Figure(layout=layout)

    if names1!='' and names1!="":
        for i, feat in enumerate(features1):
            feat = np.array(feat)
            fig.add_trace(
                go.Scatter3d(
                    x=feat[:,0],
                    y=feat[:,1],
                    z=feat[:,2],
                    mode='lines',
                    name=names1+str(i),
                    line=dict(width=2, color=color_list[(i+4-9)%9])
                )
            )
    else:
        for i, feat in enumerate(features1):
            feat = np.array(feat)
            fig.add_trace(
                go.Scatter3d(
                    x=feat[:,0],
                    y=feat[:,1],
                    z=feat[:,2],
                    mode='lines',
                    line=dict(width=2, color=color_list[(i+4-9)%9]),
                    showlegend=False
                )
            )

    for i, feat in enumerate(features2):
        feat = np.array(feat)
        fig.add_trace(
            go.Scatter3d(
                x=feat[:,0],
                y=feat[:,1],
                z=feat[:,2],
                mode='lines',
                name=names2[i],
                line=dict(
            width=5,
            color=dict_color[names2[i]],
            )
            )
        )
    fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    yaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    zaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),),
                  )
    if path!="":
        fig.write_html(path+"means.html")
    fig.show()


def plot_3D(features, names):
    fig = go.Figure(layout=layout)
    for i, feat in enumerate(features):
        feat = np.array(feat)
        fig.add_trace(
            go.Scatter3d(
                x=feat[:,0],
                y=feat[:,1],
                z=feat[:,2],
                name=names[i],
                mode='lines',
                line=dict(width=3,color=color_list[i])
            )
        )
    fig.show()


def plot_curvatures_grey(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path=""):
    N = len(kappa)
    n = len(kappa_mean)
    fig = go.Figure(layout=layout)
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', name=names1+str(i), opacity=0.4, line=dict(
                width=1,dash='solid',color='grey',),showlegend=False))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, color=dict_color[names_mean[i]])))
        # fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=name[i], line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
    if path!="":
        fig.write_html(path+"kappa.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

    fig = go.Figure(layout=layout)
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', name=names1+str(i), opacity=0.4, line=dict(
                width=1,dash='solid',color='grey',),showlegend=False))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, color=dict_color[names_mean[i]])))
            # fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=name[i], line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
    if path!="":
        fig.write_html(path+"tors.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()


def plot_curvatures(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path=""):
    N = len(kappa)
    n = len(kappa_mean)

    fig = go.Figure(layout=layout)
    if names1!='' and names1!="":
        for i in range(N):
            fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[(i+4-9)%9])))
    else:
        for i in range(N):
            fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', showlegend=False, line=dict(width=2, dash='dot', color=color_list[(i+4-9)%9])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
    if path!="":
        fig.write_html(path+"curv.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

    fig = go.Figure(layout=layout)
    if names1!='' and names1!="":
        for i in range(N):
            fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[(i+4-9)%9])))
    else:
        for i in range(N):
            fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', showlegend=False, line=dict(width=2, dash='dot', color=color_list[(i+4-9)%9])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
    if path!="":
        fig.write_html(path+"tors.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()



def plot_curvatures_raket(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path):
    n_subj = kappa.shape[0]
    n_rept = kappa.shape[1]
    n = len(kappa_mean)

    fig = go.Figure(layout=layout)
    for i in range(n_subj):
        for j in range(n_rept):
            fig.add_trace(go.Scatter(x=s, y=kappa[i,j], mode='lines', name=names1+str(i+1), line=dict(width=2, dash='dot', color=color_list[i])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
    # fig.update_layout(xaxis_title='s', yaxis_title='curvature')
    # fig.update_layout(showlegend=False)
    # fig.write_html(path+"curv.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

    fig = go.Figure(layout=layout)
    for i in range(n_subj):
        for j in range(n_rept):
            fig.add_trace(go.Scatter(x=s, y=tau[i,j], mode='lines', name=names1+str(i+1), line=dict(width=2,  dash='dot', color=color_list[i])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    # fig.update_layout(xaxis_title='s', yaxis_title='torsion')
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
    # fig.update_layout(showlegend=False)
    # fig.write_html(path+"tors.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()



def plot_3D_means_raket(features1, features2, names1, names2, path):
    fig = go.Figure(layout=layout)

    n_subj = features1.shape[0]
    n_rept = features1.shape[1]
    for j in range(n_subj):
        for k in range(n_rept):
            feat = np.array(features1[j,k])
            fig.add_trace(
                go.Scatter3d(
                    x=feat[:,0],
                    y=feat[:,1],
                    z=feat[:,2],
                    mode='lines',
                    opacity=0.4,
                    name=names1+str(j)+' ,'+str(k),
                    showlegend=False,
                    line=dict(width=3,color=color_list[j])
                )
            )

    for i, feat in enumerate(features2):
        feat = np.array(feat)
        fig.add_trace(
            go.Scatter3d(
                x=feat[:,0],
                y=feat[:,1],
                z=feat[:,2],
                mode='lines',
                name=names2[i],
                line=dict(
            width=12,
            color=dict_color[names2[i]],
            )
            )
        )
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
                    scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    yaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    zaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),),
                  )

    fig.write_html(path+".html")
    fig.show()

green_color = ['#008158', '#00AD7A', '#00CC96', '#17E098', '#2FF198']
blue_color = ['#2833A6', '#424DD2', '#636EFA', '#6B90FF', '#76B3FF']
red_color = ['#A62309', '#D2391E', '#EF553B', '#FE6D43', '#FF874E']

def plot_means_cond_raket(s, kappa_mean, tau_mean, name):

    n_cond = len(kappa_mean)
    kappa_T = [kappa_mean[2+i*3] for i in range(5)]
    kappa_M = [kappa_mean[1+i*3] for i in range(5)]
    kappa_S = [kappa_mean[i*3] for i in range(5)]
    tau_T = [tau_mean[2+i*3] for i in range(5)]
    tau_M = [tau_mean[1+i*3] for i in range(5)]
    tau_S = [tau_mean[i*3] for i in range(5)]

    fig = go.Figure(layout=layout)
    for i in range(5):
        fig.add_trace(go.Scatter(x=s, y=kappa_T[i], mode='lines', name=name+str(3+i*3), line=dict(width=2,  color=blue_color[i])))
    for i in range(5):
        fig.add_trace(go.Scatter(x=s, y=kappa_M[i], mode='lines', name=name+str(2+i*3), line=dict(width=2, color=red_color[i])))
    for i in range(5):
        fig.add_trace(go.Scatter(x=s, y=kappa_S[i], mode='lines', name=name+str(1+i*3), line=dict(width=2, color=green_color[i])))
    fig.add_trace(go.Scatter(x=s, y=kappa_mean[-1], mode='lines', name=name+str(16), line=dict(width=2, color=px.colors.qualitative.Set2[5])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

    fig = go.Figure(layout=layout)
    for i in range(5):
        fig.add_trace(go.Scatter(x=s, y=tau_T[i], mode='lines', name=name+str(3+i*3), line=dict(width=2, color=blue_color[i])))
    for i in range(5):
        fig.add_trace(go.Scatter(x=s, y=tau_M[i], mode='lines', name=name+str(2+i*3), line=dict(width=2,  color=red_color[i])))
    for i in range(5):
        fig.add_trace(go.Scatter(x=s, y=tau_S[i], mode='lines', name=name+str(1+i*3), line=dict(width=2, color=green_color[i])))
    fig.add_trace(go.Scatter(x=s, y=tau_mean[-1], mode='lines', name=name+str(16), line=dict(width=2, color=px.colors.qualitative.Set2[5])))
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()


def plot_3D_means_cond_raket(features, name, path):
    fig = go.Figure(layout=layout)

    n_cond = len(features)
    feat_T = [features[2+i*3] for i in range(5)]
    feat_M = [features[1+i*3] for i in range(5)]
    feat_S = [features[i*3] for i in range(5)]

    for i in range(5):
        feat = feat_T[i]
        fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(3+i*3),line=dict(width=8,color=blue_color[i])))
    for i in range(5):
        feat = feat_M[i]
        fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(3+i*3),line=dict(width=8,color=red_color[i])))
    for i in range(5):
        feat = feat_S[i]
        fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(3+i*3),line=dict(width=8,color=green_color[i])))
    feat = features[-1]
    fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(16),line=dict(width=8,color=px.colors.qualitative.Set2[5])))
    fig.update_layout(
    # legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
                    scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    yaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),
                    zaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="grey",
                         gridwidth=0.8,
                         zeroline=False,
                         showbackground=False,),),
                  )

    fig.write_html(path+".html")
    fig.show()
