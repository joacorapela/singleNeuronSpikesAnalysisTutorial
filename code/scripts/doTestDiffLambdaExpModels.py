
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
    parser.add_argument("--nResamples", 
                        help="number of resamples for bootstrap hypothesis test",
                        type=int,
                        default=2000)
    parser.add_argument("--max_ISI_in_hist", 
                        type=float,
                        default=500)
    parser.add_argument("--max_ISI_to_plot", 
                        type=float,
                        default=150)
    parser.add_argument("--num_bins", 
                        type=int,
                        default=30)
    parser.add_argument("--data_filename", 
                        help="data filename",
                        default="../../data/66A_int13_14.npz")
    parser.add_argument("--fig_filename_pattern", 
                        help="figure filename pattern",
                        default="../../figures/diffLambdaExpModel.{:s}")
    args = parser.parse_args()

    nResamples = args.nResamples
    max_ISI_in_hist = args.max_ISI_in_hist
    max_ISI_to_plot = args.max_ISI_to_plot
    num_bins = args.num_bins
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename, allow_pickle=True)
    female1_spike_times = load_res["Female1_2_spikes_times"]
    female2_spike_times = load_res["Female2_2_spikes_times"]

    female1_ISIs = np.diff(female1_spike_times)
    female1_ISIs[np.where(female1_ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds
    female2_ISIs = np.diff(female2_spike_times)
    female2_ISIs[np.where(female2_ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds

    all_ISIs = np.hstack((female1_ISIs, female2_ISIs))

    def difference_lambdas(samples,
                           Nfemale1=len(female1_ISIs),
                           NFemale2=len(female2_ISIs)):
        female1_ISIs = samples[:Nfemale1]
        female2_ISIs = samples[Nfemale1:]
        female1_lambda = 1/female1_ISIs.mean()
        female2_lambda = 1/female2_ISIs.mean()
        answer = female1_lambda-female2_lambda
        return answer

    observed_difference_lambdas = difference_lambdas(samples=all_ISIs)
    null_hyp_samples = resampling.get_bootstrap_sample(
        sample=all_ISIs, statistic=difference_lambdas, nResamples=nResamples)
    sign = resampling.compute_bootstrap_HT_sign(
        null_hyp_samples=null_hyp_samples,
        observed=observed_difference_lambdas,
        two_sided=True)

    title = "Significance (two-sided): {:.04f}".format(sign)

    hist_trace = trace0 = go.Histogram(x=null_hyp_samples, nbinsx=num_bins, histnorm='probability')
    fig = go.Figure()
    fig.add_trace(hist_trace)
    fig.add_vline(x=observed_difference_lambdas, line_dash="dash")
    fig.update_layout(title=title)
    fig.update_xaxes(title_text=r"$\lambda$")
    fig.update_yaxes(title_text="Probability")

    html_fig_filename = fig_filename_pattern.format("html")
    png_fig_filename = fig_filename_pattern.format("png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
