
import pdb
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
sys.path.append("../src")
import resampling

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_size",
                        type=int,
                        help="increments bin size (msec)",
                        default="50")
    parser.add_argument("--nResamples",
                        type=int,
                        help="number of resamples to compute Fano factors CIs",
                        default=2000)
    parser.add_argument("--significance",
                        type=float,
                        help="significance for Fano factors CIs",
                        default=.05)
    parser.add_argument("--data_filename", 
                        help="data filename",
                        default="../../data/66A_int13_14.npz")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/increments.{:s}")
    args = parser.parse_args()

    bin_size = args.bin_size
    nResamples = args.nResamples
    significance = args.significance
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename, allow_pickle=True)
    female1_spike_times = load_res["Female1_2_spikes_times"]
    female1_duration_sec = load_res["Female1_2_duration_sec"].item()
    female2_spike_times = load_res["Female2_2_spikes_times"]
    female2_duration_sec = load_res["Female2_2_duration_sec"].item()

    female1_bins = np.arange(0, female1_duration_sec*1e3, bin_size)
    female1_increments, _ = np.histogram(female1_spike_times, female1_bins)
    female2_bins = np.arange(0, female2_duration_sec*1e3, bin_size)
    female2_increments, _ = np.histogram(female2_spike_times, female2_bins)

    def fano_factor(increments):
        answer = increments.var() / increments.mean()
        return answer

    female1_ff = fano_factor(increments=female1_increments)
    female1_ff_boot = resampling.get_bootstrap_sample(sample=female1_increments,
                                                      statistic=fano_factor,
                                                      nResamples=nResamples)
    female1_ff_95CI = resampling.compute_bootstrap_CI(boot_sample=female1_ff_boot,
                                                      significance=significance)
    female2_ff = fano_factor(increments=female2_increments)
    female2_ff_boot = resampling.get_bootstrap_sample(sample=female2_increments,
                                                      statistic=fano_factor,
                                                      nResamples=nResamples)
    female2_ff_95CI = resampling.compute_bootstrap_CI(boot_sample=female2_ff_boot,
                                                      significance=significance)

    title = "Female1 Fano Factor: {:.02f} ({:.02f}, {:.02f}), Female2 Fano Factor: {:.02f} ({:.02f}, {:.02f})".format(female1_ff, female1_ff_95CI[0], female1_ff_95CI[1], female2_ff, female2_ff_95CI[0], female2_ff_95CI[1])

    female1_trace = go.Scatter(x=female1_bins,
                               y=female1_increments,
                               mode="markers",
                               name="female1")
    female2_trace = go.Scatter(x=female2_bins,
                               y=female2_increments,
                               mode="markers",
                               name="female2")
    fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(female1_trace, row=1, col=1)
    fig.add_trace(female2_trace, row=2, col=1)
    fig.update_xaxes(title_text="Time (msec)")
    fig.update_yaxes(title_text="Spikes Count")
    fig.update_layout(title=title)

    html_fig_filename = fig_filename_pattern.format("html")
    png_fig_filename = fig_filename_pattern.format("png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
