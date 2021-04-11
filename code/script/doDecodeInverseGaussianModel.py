
import pdb
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
import plotly.express as px
sys.path.append("../src")
import probabilisticModels
import classifiers
import statMetrics

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nResamples",
                        help="number of resamples for confusion_matrix",
                        type=int,
                        default=100)
    parser.add_argument("--percentage_train", 
                        help="percentage train for confusionmatrix",
                        type=float,
                        default=.8)
    parser.add_argument("--randomize_ISIs", 
                        help="randomize ISI across classes",
                        action="store_true")
    parser.add_argument("--data_filename", 
                        help="data filename",
                        default="../../data/66A_int13_14.npz")
    parser.add_argument("--fig_filename_pattern", 
                        help="figure filename pattern",
                        default="../../figures/inverseGaussianFit_randomized_ISIs{:d}.{:s}")
    args = parser.parse_args()

    nResamples = args.nResamples
    percentage_train = args.percentage_train
    randomize_ISIs = args.randomize_ISIs
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename, allow_pickle=True)
    female1_spike_times = load_res["Female1_2_spikes_times"]
    female2_spike_times = load_res["Female2_2_spikes_times"]
    interactions_labels = ["female1", "female2"]

    female1_ISIs = np.diff(female1_spike_times)
    female1_ISIs[np.where(female1_ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds
    female2_ISIs = np.diff(female2_spike_times)
    female2_ISIs[np.where(female2_ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds

    if randomize_ISIs:
        all_ISIs = np.concatenate((female1_ISIs, female2_ISIs))
        suffled_all_ISIs = np.random.permutation(all_ISIs)
        female1_ISIs = suffled_all_ISIs[len(female1_ISIs):]
        female2_ISIs = suffled_all_ISIs[:len(female1_ISIs)]

    confusion_matrix = np.zeros((2,2))
    classifier = classifiers.NaiveBayes()

    for i in range(nResamples):
        shuffled_female1_ISIs = np.random.permutation(female1_ISIs)
        shuffled_female2_ISIs = np.random.permutation(female2_ISIs)
        train_female1_ISIs = shuffled_female1_ISIs[:round(len(shuffled_female1_ISIs)*percentage_train)]
        test_female1_ISIs = shuffled_female1_ISIs[round(len(shuffled_female1_ISIs)*percentage_train):]
        train_female2_ISIs = shuffled_female2_ISIs[:round(len(shuffled_female2_ISIs)*percentage_train)]
        test_female2_ISIs = shuffled_female2_ISIs[round(len(shuffled_female2_ISIs)*percentage_train):]
        classifier.train(x=[train_female1_ISIs, train_female2_ISIs],
                         y=interactions_labels,
                         model_class=probabilisticModels.InverseGaussian)
        classified_female1 = classifier.classify(x=test_female1_ISIs)
        if classified_female1==interactions_labels[0]:
            confusion_matrix[0,0] += 1 # TP
        else:
            confusion_matrix[0,1] += 1 # FN
        classified_female2 = classifier.classify(x=test_female2_ISIs)
        if classified_female2==interactions_labels[1]:
            confusion_matrix[1,1] += 1 # TN
        else:
            confusion_matrix[1,0] += 1 # FN

    confusion_matrix_metrics = statMetrics.get_confusion_matrix_metrics(confusion_matrix=confusion_matrix)

    fig = px.imshow(confusion_matrix,
                        labels=dict(y="Decoded Interaction", x="True Interaction"),
                        x=interactions_labels,
                        y=interactions_labels,
                        zmin=0.0, zmax=nResamples)
    fig.update_layout(
        title="Precision: {:.02f}, Recall: {:.02f}, f1-score: {:.02f}".format(*confusion_matrix_metrics)
    )

    htmlFigFilename = fig_filename_pattern.format(randomize_ISIs, "html")
    pngFigFilename = fig_filename_pattern.format(randomize_ISIs, "png")
    fig.write_html(htmlFigFilename)
    fig.write_image(pngFigFilename)
    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
