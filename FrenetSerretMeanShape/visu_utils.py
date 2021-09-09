from matplotlib.pyplot import legend
import numpy as np
import plotly.graph_objs as go
import plotly.express as px


""" Set of functions for visualization of mean shape, mean curvature, torsion, etc (visualize 3D curves or 2D curves). """

color_list_mean = px.colors.qualitative.Plotly
dict_color = {"True Mean" : color_list_mean[0], "Arithmetic Mean" : px.colors.qualitative.Set1[6], "SRVF Mean" : px.colors.qualitative.Set1[8], "FS Mean" : px.colors.qualitative.Dark24[5], "Extrinsic Mean" : px.colors.qualitative.Set1[6], "Individual Mean" : color_list_mean[3]}
color_list = px.colors.qualitative.Plotly

def plot_array_2D(x, array_y, name_ind, legend={"index":False}):
    fig = go.Figure()
    N = array_y.shape[0]
    for i in range(N):
        fig.add_trace(go.Scatter(x=x, y=array_y[i,:], mode='lines', name=name_ind+str(i), line=dict(width=1, color=color_list[(i-9)%9])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.show()

def plot_2array_2D(x, array_y, legend={"index":False}):
    fig = go.Figure()
    N = array_y.shape[0]
    for i in range(N):
        fig.add_trace(go.Scatter(x=x[i], y=array_y[i,:], mode='lines', line=dict(width=2, color=color_list[(i-9)%9])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.show()

def plot_2D(x, y,  legend={"index":False}):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=2, color=color_list[0])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.show()


def plot_3D_means_grey(features1, features2, names1, names2):
    fig = go.Figure()

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
                name=names1+str(i),
                line=dict(
                width=0.8,
                dash='solid',
                color='grey',
            )
            )
        )

    fig.show()

def plot_3D_means(features1, features2, names1, names2, path=""):
    fig = go.Figure()

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
                name=names1+str(i),
                line=dict(width=2, color=color_list[(i+4-9)%9])
            )
        )
    if path!="":
        fig.write_html(path+"means.html")
    fig.show()


def plot_3D(features, names):
    fig = go.Figure()
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


def plot_curvatures_grey(s, kappa, tau, kappa_mean, tau_mean, names_mean,names1, path=""):
    N = len(kappa)
    n = len(kappa_mean)

    fig = go.Figure()
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', name=names1+str(i), line=dict(
                width=1,
                dash='solid',
                color='grey',
            )))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, color=dict_color[names_mean[i]])))

    fig.update_layout(xaxis_title='s', yaxis_title='kappa')
    if path!="":
        fig.write_html(path+"kappa.html")
    fig.show()

    fig = go.Figure()
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', name=names1+str(i), line=dict(
                width=1,
                dash='solid',
                color='grey',
            )))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(xaxis_title='s', yaxis_title='tau')
    if path!="":
        fig.write_html(path+"tors.html")
    fig.show()


def plot_curvatures(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path=""):
    N = len(kappa)
    n = len(kappa_mean)

    fig = go.Figure()
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[(i+4-9)%9])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(xaxis_title='s', yaxis_title='kappa')
    if path!="":
        fig.write_html(path+"curv.html")
    fig.show()

    fig = go.Figure()
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[(i+4-9)%9])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    fig.update_layout(xaxis_title='s', yaxis_title='tau')
    if path!="":
        fig.write_html(path+"tors.html")
    fig.show()



def plot_curvatures_raket(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path):
    n_subj = kappa.shape[0]
    n_rept = kappa.shape[1]
    n = len(kappa_mean)

    fig = go.Figure()
    for i in range(n_subj):
        for j in range(n_rept):
            fig.add_trace(go.Scatter(x=s, y=kappa[i,j], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[i])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    # fig.update_layout(xaxis_title='s', yaxis_title='kappa')
    fig.update_layout(showlegend=False)
    # fig.write_html(path+"curv.html")
    fig.show()

    fig = go.Figure()
    for i in range(n_subj):
        for j in range(n_rept):
            fig.add_trace(go.Scatter(x=s, y=tau[i,j], mode='lines', name=names1+str(i), line=dict(width=2,  dash='dot', color=color_list[i])))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
    # fig.update_layout(xaxis_title='s', yaxis_title='tau')
    fig.update_layout(showlegend=False)
    # fig.write_html(path+"tors.html")
    fig.show()



def plot_3D_means_raket(features1, features2, names1, names2, path):
    fig = go.Figure()

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

    fig.write_html(path+".html")
