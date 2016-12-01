import os
import paths
import roadrunner
import sbml
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import plot_utils
import seaborn as sns

SBML_MODEL_FNAME = \
  os.path.join(paths.MAIN_DIR,
               "sbml_models",
               "test.xml")
#               "glu_gal_transition_counter.xml")

SBML_SIMPLE_FNAME = \
  os.path.join(paths.MAIN_DIR, "sbml_models",
               "simple_model.xml")

def test_simple_model():
    plot_fname = "/Users/yarden/Desktop/sbml_simple_test.pdf"
    fig = plt.figure(figsize=(3.3, 6))
    sns.set_style("ticks")
    gs = gridspec.GridSpec(3, 1)
    gs.update(hspace=0.5, left=0.19, top=0.55, bottom=0.08)
    panel_label_x = 0.01
    ax1 = plt.subplot(gs[0, 0])
    sbml_model = sbml.SBML(SBML_SIMPLE_FNAME)
    print sbml_model
    t_start = 0
    t_end = 150
    num_time_bins = 500
    times = np.linspace(t_start, t_end, num_time_bins)
    doser = sbml.DoseSched(t_start, t_end, num_time_bins)
    nutrient_val = 50
    # Glu
    doser.add_dose("glu", 0, 49, nutrient_val)
    results = sbml_model.simulate_with_doses(times, doser)
    sns.set_style("ticks")
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    vars_to_plot = ["[glu]", "[glu_act]"]
    offset = 2
    nutrients_to_colors = {"[glu]": plot_utils.lightgreen,
                           "[glu_act]": "red"}
    # linewidth
    lw = 1.5
    for c in vars_to_plot:
        if c != "time":
            plt.plot(results["time"], results[c],
                     label=c,
                     color=nutrients_to_colors[c],
                     linewidth=lw,
                     clip_on=True)
    # time axis
    x_start = t_start
    x_end = t_end
    x_step = 50
    labelspacing = 0.25
    handletextpad = 0.5
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
    ylims = plt.gca().get_ylim()
    plt.ylabel("Conc.")
    plt.savefig(plot_fname)
    
    
def test_model():
    plot_fname = "/Users/yarden/Desktop/sbml_test.pdf"
    print "loading SBML model from %s" %(SBML_MODEL_FNAME)
    fig = plt.figure(figsize=(3.3, 6))
    sns.set_style("ticks")
    gs = gridspec.GridSpec(4, 1)
    gs.update(hspace=0.5, left=0.19, top=0.55, bottom=0.08)
    panel_label_x = 0.01
    ax1 = plt.subplot(gs[0, 0])
    sbml_model = sbml.SBML(SBML_MODEL_FNAME)
    t_start = 0
#    t_end = 250
    t_end = 150
    num_time_bins = 200
    times = np.linspace(t_start, t_end, num_time_bins)
    doser = sbml.DoseSched(t_start, t_end, num_time_bins)
    nutrient_val = 50
    # Glu
    doser.add_dose("Glu", 0, 49, nutrient_val)
    doser.add_dose("Glu", 49, 99, 0)
    # Gal
    doser.add_dose("Gal", 49, 99, nutrient_val)
    doser.add_dose("Gal", 99, 149, 0)
    # Glu
#    doser.add_dose("Glu", 99, 149, nutrient_val)
#    doser.add_dose("Glu", 149, 249, 0)
    # Gal
#    doser.add_dose("Gal", 149, 199, nutrient_val)
#    doser.add_dose("Glu", 199, 200, 0)
    results = sbml_model.simulate_with_doses(times, doser)
    sns.set_style("ticks")
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    vars_to_plot = ["[Glu]", "[Gal]"]
    # offset for despine
    offset = 2
    # x-offset for nutrient plotting
    x_offsets = {"[Glu]": -1,
                 "[Gal]": 1}
    nutrients_to_colors = {"[Glu]": plot_utils.lightgreen,
                           "[Gal]": plot_utils.blue}
    # linewidth
    lw = 1.5
    for c in vars_to_plot:
        if c != "time":
            plt.plot(results["time"] + x_offsets[c], results[c],
                     label=c,
                     color=nutrients_to_colors[c],
                     linewidth=lw,
                     clip_on=True)
    # time axis
    x_start = t_start
    x_end = t_end
    x_step = 50
    labelspacing = 0.25
    handletextpad = 0.5
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
    ylims = plt.gca().get_ylim()
    plt.ylim([0 - 1, 52])
    plt.ylabel("Inputs", fontsize=8)
    ylabel_x = -0.14
    ylabel_y = 0.5
    ax1.yaxis.set_label_coords(ylabel_x, ylabel_y)
    ax1.legend(fontsize=8, handlelength=2.2, handletextpad=handletextpad,
               numpoints=4,
               labelspacing=labelspacing,
               bbox_to_anchor=(1.12, 1.0))
    sns.despine(trim=True, offset=offset)
    ax2 = plt.subplot(gs[1, 0])
    clip_on = False
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    plt.plot(results["time"], results["[Glu_Sensor]"],
             label="[Glu Sensor]",
             linewidth=lw,
             clip_on=clip_on,
             color=plot_utils.lightgreen)
    plt.plot(results["time"], results["[Gal_Sensor]"],
             label="[Gal Sensor]",
             linewidth=lw,
             clip_on=clip_on,
             color=plot_utils.blue)
    ax2.yaxis.set_label_coords(ylabel_x, ylabel_y)
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
    plt.ylim([-0.5, 14])
    plt.yticks(range(0, 14 + 2, 4))
    plt.ylabel("Sensors", fontsize=8)
    ax2.legend(fontsize=8, handlelength=2.2, handletextpad=handletextpad,
               numpoints=4,
               labelspacing=labelspacing,
               bbox_to_anchor=(1.12, 1.1))
    sns.despine(trim=True, offset=offset)
    ax3 = plt.subplot(gs[2, 0])
    plt.plot(results["time"], results["[Glu_to_Gal]"],
             label="[Glu-to-Gal]",
             linewidth=lw,
             clip_on=clip_on,
             color="k")
    line, = plt.plot(results["time"], results["[Gal_to_Glu]"],
             label="[Gal-to-Glu]",
             linewidth=lw,
             clip_on=clip_on,
             linestyle="--",
             color="k")
    line.set_dashes((3, 2))
    ax3.legend(fontsize=8, handlelength=2.2, handletextpad=handletextpad,
               numpoints=4,
               labelspacing=labelspacing,
               bbox_to_anchor=(0.45, 1.1))
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
#    plt.ylim([0, 150])
#    plt.yticks(range(0, 150 + 50, 50))
    ax3.yaxis.set_label_coords(ylabel_x, ylabel_y)
    sns.despine(trim=True, offset=offset)
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Transition counters", fontsize=8)
    ### final plot
    ax4 = plt.subplot(gs[3, 0])
    plt.plot(results["time"], results["[Glu_Activator]"],
             label="[Glu Activator]",
             linewidth=lw,
             color="k",
             clip_on=clip_on)
    plt.plot(results["time"], results["[Gal_Activator]"],
             label="[Gal Activator]",
             linestyle="--",
             linewidth=lw,
             clip_on=clip_on,
             color="k")
    ax4.legend(fontsize=8, handlelength=2.2)
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step),
               fontsize=8)
    sns.despine(trim=True, offset=offset)
    plt.xlabel("Time", fontsize=8)
    plt.savefig(plot_fname)


if __name__ == "__main__":
    test_model()
