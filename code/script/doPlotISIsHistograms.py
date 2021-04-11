
import pdb
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.subplots

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_ISI_in_hist", 
                        type=float,
                        default=500)
    parser.add_argument("--max_ISI_to_plot", 
                        type=float,
                        default=150)
    parser.add_argument("--bin_size", 
                        type=float,
                        default=1)
    parser.add_argument("--data_filename", 
                        help="data filename",
                        default="../../data/66A_int13_14.npz")
    parser.add_argument("--fig_filename_pattern", 
                        help="figure filename pattern",
                        default="../../figures/ISIs.{:s}")
    args = parser.parse_args()

    max_ISI_in_hist = args.max_ISI_in_hist
    max_ISI_to_plot = args.max_ISI_to_plot
    bin_size = args.bin_size
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename, allow_pickle=True)
    female1_spike_times = load_res["Female1_2_spikes_times"]
    female2_spike_times = load_res["Female2_2_spikes_times"]

    female1_ISIs = np.diff(female1_spike_times)
    female1_ISIs[np.where(female1_ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds
    female2_ISIs = np.diff(female2_spike_times)
    female2_ISIs[np.where(female2_ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds

    bins = np.arange(0, max_ISI_in_hist, bin_size)
    female1_ISIs_counts, _ = np.histogram(female1_ISIs, bins)
    female1_ISIs_hist = female1_ISIs_counts/len(female1_ISIs)
    female2_ISIs_counts, _ = np.histogram(female2_ISIs, bins)
    female2_ISIs_hist = female2_ISIs_counts/len(female2_ISIs)

    female1_trace = go.Bar(x=bins, y=female1_ISIs_hist, name="female1")
    female2_trace = go.Bar(x=bins, y=female2_ISIs_hist, name="female2")
    fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(female1_trace, row=1, col=1)
    fig.add_trace(female2_trace, row=2, col=1)
    fig.update_xaxes(title_text="ISI (msec)", range=[0, max_ISI_to_plot])
    fig.update_yaxes(title_text="Probability")

    html_fig_filename = fig_filename_pattern.format("html")
    png_fig_filename = fig_filename_pattern.format("png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
