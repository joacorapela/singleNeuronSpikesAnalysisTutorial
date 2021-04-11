
import pdb
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
sys.path.append("../src")
import timeSeries

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time_sec",
                        type=float,
                        help="start time (sec) to include in the autocorrelation calculation",
                        default=0.0)
    parser.add_argument("--end_time_sec",
                        type=float,
                        help="end time (sec) to include in the autocorrelation calculation",
                        default=30.0)
    parser.add_argument("--bin_size",
                        type=int,
                        help="increments bin size (msec)",
                        default="1")
    parser.add_argument("--num_lags",
                        type=int,
                        help="number of lags in autocorrelation",
                        default="100")
    parser.add_argument("--ylim",
                        help="y axes limits",
                        default="[-.1,.1]")
    parser.add_argument("--data_filename",
                        help="data filename",
                        default="../../data/66A_int13_14.npz")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/autocorrelations.{:s}")
    parser.add_argument("--diff_fig_filename_pattern",
                        help="difference figure filename pattern",
                        default="../../figures/diffAutocorrelations.{:s}")
    args = parser.parse_args()

    start_time_sec = args.start_time_sec
    end_time_sec = args.end_time_sec
    bin_size = args.bin_size
    num_lags = args.num_lags
    ylim = [float(str) for str in args.ylim[1:-1].split(",")]
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern
    diff_fig_filename_pattern = args.diff_fig_filename_pattern

    load_res = np.load(data_filename, allow_pickle=True)
    female1_spike_times = load_res["Female1_2_spikes_times"]
    female1_duration_sec = load_res["Female1_2_duration_sec"].item()
    if female1_duration_sec<start_time_sec:
        raise ValueError("start_time_sec={:f} should be less than female1_duration_sec={:f}".format(start_time_sec, female1_duration_sec))
    if female1_duration_sec>end_time_sec:
        female1_duration_sec = end_time_sec
    female1_spike_times = female1_spike_times[np.logical_and(start_time_sec*1e3<=female1_spike_times, female1_spike_times<=female1_duration_sec*1e3).nonzero()[0]]

    female2_spike_times = load_res["Female2_2_spikes_times"]
    female2_duration_sec = load_res["Female2_2_duration_sec"].item()
    if female2_duration_sec<start_time_sec:
        raise ValueError("start_time_sec={:f} should be less than female2_duration_sec={:f}".format(start_time_sec, female2_duration_sec))
    if female2_duration_sec>end_time_sec:
        female2_duration_sec = end_time_sec
    female2_spike_times = female2_spike_times[np.logical_and(start_time_sec*1e3<=female2_spike_times, female2_spike_times<=female2_duration_sec*1e3).nonzero()[0]]

    female1_bins = np.arange(start_time_sec*1e3, female1_duration_sec*1e3, bin_size)
    female1_increments, _ = np.histogram(female1_spike_times, female1_bins)
    female1_significance = 2.0/np.sqrt(len(female1_bins))
    female1_autocorrelations = timeSeries.autocorr(female1_increments, num_lags)

    female2_bins = np.arange(start_time_sec*1e3, female2_duration_sec*1e3, bin_size)
    female2_increments, _ = np.histogram(female2_spike_times, female2_bins)
    female2_significance = 2.0/np.sqrt(len(female2_bins))
    female2_autocorrelations = timeSeries.autocorr(female2_increments, num_lags)

    diff_autocorrelations = female1_autocorrelations-female2_autocorrelations
    diff_std = np.sqrt(1/len(female1_bins)+1/len(female2_bins))
    diff_significance = 2*diff_std

    lags_msec = np.arange(num_lags)*bin_size
    female1_trace = go.Scatter(x=lags_msec,
                               y=female1_autocorrelations,
                               mode="markers",
                               name="female1")
    female2_trace = go.Scatter(x=lags_msec,
                               y=female2_autocorrelations,
                               mode="markers",
                               name="female2")
    fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(female1_trace, row=1, col=1)
    fig.add_hline(y=female1_significance, row=1, col=1, line_dash="dash")
    fig.add_hline(y=-female1_significance, row=1, col=1, line_dash="dash")
    fig.add_trace(female2_trace, row=2, col=1)
    fig.add_hline(y=female2_significance, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-female2_significance, row=2, col=1, line_dash="dash")
    fig.update_xaxes(title_text="Lag (msec)")
    fig.update_yaxes(title_text="Autocorrelation", range=ylim)

    html_fig_filename = fig_filename_pattern.format("html")
    png_fig_filename = fig_filename_pattern.format("png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()

    diff_fig = go.Figure()
    diff_trace = go.Scatter(x=lags_msec,
                            y=diff_autocorrelations,
                            mode="markers")
    diff_fig.add_trace(diff_trace)
    diff_fig.add_hline(y=diff_significance, line_dash="dash")
    diff_fig.add_hline(y=-diff_significance, line_dash="dash")
    diff_fig.update_xaxes(title_text="Lag (msec)")
    diff_fig.update_yaxes(title_text="Autocorrelation Difference", range=ylim)

    html_fig_filename = diff_fig_filename_pattern.format("html")
    png_fig_filename = diff_fig_filename_pattern.format("png")
    diff_fig.write_html(html_fig_filename)
    diff_fig.write_image(png_fig_filename)

    diff_fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
