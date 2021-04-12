
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
                        default="../../figures/diffParamsInvGaussianModel_{:s}.{:s}")
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

    def difference_lambda(samples,
                          Nfemale1=len(female1_ISIs),
                          NFemale2=len(female2_ISIs)):
        female1_ISIs = samples[:Nfemale1]
        female2_ISIs = samples[Nfemale1:]
        female1_mu = female1_ISIs.mean()
        female1_lambda = 1/(1/female1_ISIs-1/female1_mu).mean()
        female2_mu = female2_ISIs.mean()
        female2_lambda = 1/(1/female2_ISIs-1/female2_mu).mean()
        lambda_diff = female1_lambda-female2_lambda
        return lambda_diff

    observed_difference_lambda = difference_lambda(samples=all_ISIs)
    null_hyp_samples_lambda = resampling.get_bootstrap_sample(
        sample=all_ISIs, statistic=difference_lambda, nResamples=nResamples)
    sign_lambda = resampling.compute_bootstrap_HT_sign(
        null_hyp_samples=null_hyp_samples_lambda,
        observed=observed_difference_lambda,
        two_sided=True)

    def difference_mu(samples,
                      Nfemale1=len(female1_ISIs),
                      NFemale2=len(female2_ISIs)):
        female1_ISIs = samples[:Nfemale1]
        female2_ISIs = samples[Nfemale1:]
        female1_mu = female1_ISIs.mean()
        female2_mu = female2_ISIs.mean()
        mu_diff = female1_mu-female2_mu
        return mu_diff

    observed_difference_mu = difference_mu(samples=all_ISIs)
    null_hyp_samples_mu = resampling.get_bootstrap_sample(
        sample=all_ISIs, statistic=difference_mu, nResamples=nResamples)
    sign_mu = resampling.compute_bootstrap_HT_sign(
        null_hyp_samples=null_hyp_samples_mu,
        observed=observed_difference_mu,
        two_sided=True)

    title = "Significance (two-sided): {:.04f}".format(sign_lambda)
    hist_trace = trace0 = go.Histogram(x=null_hyp_samples_lambda, nbinsx=num_bins, histnorm='probability')
    fig = go.Figure()
    fig.add_trace(hist_trace)
    fig.add_vline(x=observed_difference_lambda, line_dash="dash")
    fig.update_layout(title=title)
    fig.update_xaxes(title_text=r"$\lambda$")
    fig.update_yaxes(title_text="Probability")

    html_fig_filename = fig_filename_pattern.format("lambda", "html")
    png_fig_filename = fig_filename_pattern.format("lambda", "png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()

    title = "Significance (two-sided): {:.04f}".format(sign_mu)
    hist_trace = trace0 = go.Histogram(x=null_hyp_samples_mu, nbinsx=num_bins, histnorm='probability')
    fig = go.Figure()
    fig.add_trace(hist_trace)
    fig.add_vline(x=observed_difference_mu, line_dash="dash")
    fig.update_layout(title=title)
    fig.update_xaxes(title_text=r"$\mu$")
    fig.update_yaxes(title_text="Probability")

    html_fig_filename = fig_filename_pattern.format("mu", "html")
    png_fig_filename = fig_filename_pattern.format("mu", "png")
    fig.write_html(html_fig_filename)
    fig.write_image(png_fig_filename)

    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
