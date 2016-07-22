##
## Plotting utilities
##
import matplotlib
import matplotlib.pylab as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import OrderedDict

import numpy as np

import pandas

# def _plot_std_bars(*args, central_data=None, ci=None, data=None, **kwargs):
#     std = data.std(axis=0)
#     ci = np.asarray((central_data - std, central_data + std))
#     kwargs.update({"central_data": central_data, "ci": ci, "data": data})
#     seaborn.timeseries._plot_ci_bars(*args, **kwargs)

# def _plot_std_band(*args, central_data=None, ci=None, data=None, **kwargs):
#     std = data.std(axis=0)
#     ci = np.asarray((central_data - std, central_data + std))
#     kwargs.update({"central_data": central_data, "ci": ci, "data": data})
#     seaborn.timeseries._plot_ci_band(*args, **kwargs)

# seaborn.timeseries._plot_std_bars = _plot_std_bars
# seaborn.timeseries._plot_std_band = _plot_std_band

red = sns.color_palette("Set1")[0]
blue = sns.color_palette("Set1")[1]
green = sns.color_palette("Set1")[2]
lightgreen = "#32cc2d"
lightgrey = "#d3d3d3"
black = "k"
purple = sns.color_palette("Set1")[3]
orange = "#ffa500"
darkgrey = "#a9a9a9"

def set_mpl_fonts():
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Arial'
    matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'


##
## setup fonts
##
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
set_mpl_fonts()

def plot_discrete_nutrient_data(data,
                                data_to_labels={},
                                labels_to_colors={},
                                na_label="NA",
                                na_color="#ffffff",
                                y_val=1,
                                size=2):
    x_axis = range(len(data))
    labels = []
    df = []
    print "Plotting discrete nutrient data for: "
    for n, d in enumerate(data):
        label = na_label
        color = na_color
        if d in data_to_labels:
            label = data_to_labels[d]
        if label in labels_to_colors:
            color = labels_to_colors[label]            
        entry = {"Carbon": label,
                 "color": color,
                 "y": y_val,
                 "x": n}
        df.append(entry)
        labels.append(label)
    df = pandas.DataFrame(df)
    seen_labels = {}
    for label in df["Carbon"]:
        if label in seen_labels:
            continue
        subset = df[df["Carbon"] == label]
        plt.plot(subset["x"], subset["y"], "o",
                 label=label, color=labels_to_colors[label],
                 markersize=size)
        seen_labels[label] = True
#    sns.tsplot(time="x", value="y", condition="Carbon", unit="Carbon",
#               data=df, err_style="unit_points", interpolate=False,
#               color=colors)


def plot_seq(data, schematic_time, t_start,
             data_to_labels,
             labels_to_colors,
             box_height=0.5,
             ax=None,
             y_val=0.5):
    if ax is None:
        ax = plt.gca()
    box_step = 0.5
    y_val = 0.5
    pad = 0.3
    box_width = schematic_time.step_size
    assert t_start in schematic_time.t, "t_start not found."
    t_start_ind = np.where(schematic_time.t == t_start)[0][0]
    time_slice = schematic_time.t[t_start_ind:t_start_ind + len(data)]
    for n, t in enumerate(time_slice):
        label = "NA"
        color = "#999999"
        if data[n] in data_to_labels:
            label = data_to_labels[data[n]]
        if label in labels_to_colors:
            color = labels_to_colors[label]
        x_val = t - box_width*box_step
        rect = patches.Rectangle((x_val, y_val),
                                 box_width,
                                 box_height,
                                 linewidth=0,
                                 facecolor=color)
        p = ax.add_patch(rect)
    return (t_start, x_val)
    

def plot_sudden_switches(time_obj, data,
                         data_to_labels={},
                         labels_to_colors={},
                         na_label="NA",
                         na_color=lightgrey,
                         y_val=0.5,
                         size=2,
                         box_height=0.5,
                         box_step=0.5,
                         labelsize=8,
                         pad=0.3,
                         ax=None,
                         despine=True,
                         tick_len=2.5,
                         with_legend=False,
                         legend_fontsize=8,
                         legend_outside=None):
    box_width = time_obj.step_size
    if ax is None:
        ax = plt.gca()
    color_to_handle = OrderedDict()
    color_to_label = OrderedDict()
    for n, t in enumerate(time_obj.t):
        label = na_label
        color = na_color
        if data[n] in data_to_labels:
            label = data_to_labels[data[n]]
        if label in labels_to_colors:
            color = labels_to_colors[label]
        rect = patches.Rectangle((t - box_width*box_step, y_val),
                                 box_width,
                                 box_height,
                                 linewidth=0,
                                 facecolor=color)
        p = ax.add_patch(rect)
        color_to_handle[color] = p
        color_to_label[color] = label
    plt.xlim(time_obj.t[0] - box_width,
             time_obj.t[-1] + box_width)
    plt.ylim([y_val, y_val + box_height])
    handles = []
    handle_labels = []
#    plt.tick_params(labelsize=labelsize)
    for c in color_to_handle:
        handles.append(color_to_handle[c])
        handle_labels.append(color_to_label[c])
    if with_legend:
        handlelength=0.8
        if legend_outside is not None:
            plt.legend(handles, handle_labels, bbox_to_anchor=legend_outside,
                       handlelength=handlelength,
                       fontsize=legend_fontsize)
        else:
            plt.legend(handles, handle_labels, handlelength=handlelength,
                       fontsize=legend_fontsize)
    if despine:
        sns.despine(trim=True, left=True, ax=ax)
        ax.get_yaxis().set_visible(False)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", pad=pad, length=tick_len,
                       labelsize=labelsize)
    handle_info = {"handle_labels": handle_labels,
                   "handles": handles,
                   "color_to_handle": color_to_handle,
                   "color_to_label": color_to_label}
    return handle_info
        

def get_mat_as_tex(mat):
    if not (mat.shape[0] == mat.shape[1] == 3):
        raise Exception, "Only handles 3x3 matrices."
    mat_tex = \
      r"$\left( \begin{array}{ccc} %.2f & %.2f & %.2f \\ " \
      r"%.2f & %.2f & %.2f \\ " \
      r"%.2f & %.2f & %.2f \end{array} \right)$" \
       %(mat[0, 0], mat[0, 1], mat[0, 2],
         mat[1, 0], mat[1, 1], mat[1, 2],
         mat[2, 0], mat[2, 1], mat[2, 2])
    return mat_tex
