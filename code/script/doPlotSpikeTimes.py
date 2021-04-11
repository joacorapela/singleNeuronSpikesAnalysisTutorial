
import pdb
import sys
import argparse
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", 
                        help="data filename",
                        default="../../data/66A_int13_14.npz")
    parser.add_argument("--fig_filename_pattern", 
                        help="figure filename pattern",
                        default="../../figures/spike_times.{:s}")
    args = parser.parse_args()

    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename, allow_pickle=True)
    female1_spike_times = load_res["Female1_2_spikes_times"]
    female1_duration_sec = load_res["Female1_2_duration_sec"].item()
    female2_spike_times = load_res["Female2_2_spikes_times"]
    female2_duration_sec = load_res["Female2_2_duration_sec"].item()

    female1_spike_rate = len(female1_spike_times)/female1_duration_sec
    female2_spike_rate = len(female2_spike_times)/female2_duration_sec

    female1_trace = go.Scatter(x=female1_spike_times,
                               y=np.ones_like(female1_spike_times),
                               mode="markers",
                               name="female1 ({:.02f} spikes/sec)".format(female1_spike_rate))
    female2_trace = go.Scatter(x=female2_spike_times,
                               y=2*np.ones_like(female2_spike_times),
                               mode="markers",
                               name="female2 ({:.02f} spikes/sec)".format(female2_spike_rate))
    fig = go.Figure()
    fig.add_trace(female1_trace)
    fig.add_trace(female2_trace)
    fig.update_xaxes(title_text="Spike Time (msec)")
    fig.update_yaxes(showticklabels=False)

    html_fig_filename = fig_filename_pattern.format("html")
    png_fig_filename = fig_filename_pattern.format("png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
