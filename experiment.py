from sklearn.model_selection import train_test_split
import numpy as np

from imblearn.datasets import fetch_datasets

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE

from NESMOTE.oversampler import NESMOTE
from NESMOTE.util import std_euclid_distance, std_euclid_wavg

from time import time

def compute_classwize_accuracy(true, pred):
    class_counts = {}
    for tval, pval in zip(true, pred):
        if class_counts.get(tval) is None:
            class_counts[tval] = [1, 0]
        else:
            class_counts[tval] = [class_counts[tval][0] + 1, class_counts[tval][1]]
        if tval == pval:
            class_counts[tval] = [class_counts[tval][0], class_counts[tval][1] + 1]
    res = {}
    for c in class_counts.keys():
        # total // correct // ratio
        res[c] = [class_counts[c][0], class_counts[c][1], class_counts[c][0] / len(true)]
    return res

def get_scores(cum_res):
    balanced_accuracy = 0
    guesses = 0
    correct = 0
    for c in cum_res.keys():
        balanced_accuracy += cum_res[c][1] / cum_res[c][0] * (1 - cum_res[c][2])
        guesses += cum_res[c][0]
        correct += cum_res[c][1]
    return balanced_accuracy, correct / guesses



datasets = fetch_datasets()

class NotOverSampler:
    def __init__(self):
        self.trivial = True
    
    def fit_resample(self, X, y):
        return X, y



overs = {
    "Baseline" : NotOverSampler(),
    "RandomOS" : RandomOverSampler(),
    "IL-SMOTE" : SMOTE(),
    "NE - MAX" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "cliques", "wrap": "all"}),
    "NE - ALL" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "cliques", "wrap": "max"}),
    "NE - KNN" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "neighbors", "wrap": "max"}),
    "NE-SMOTE" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "smote", "wrap": "all"}),
}


def run_trial(dataset, oversampler, classifier, nruns=5):
    cum = None
    for _ in range(nruns):
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=.25)
        X_train_OS, y_train_OS = oversampler.fit_resample(X_train, y_train)
        classifier.fit(X_train_OS, y_train_OS)
        preds = classifier.predict(X_test)
        ca = compute_classwize_accuracy(y_test, preds)
        if cum is None:
            cum = ca
        else:
            for c in cum.keys():
                cum[c][0] += ca[c][0]
                cum[c][1] += ca[c][1]
    return get_scores(cum)

OS_names = ["Baseline", "RandomOS", "IL-SMOTE", "NE - max", "NE - all", "NE - KNN", "NE-SMOTE"]
ds_names = ["ecoli", "optical_digits", "abalone", "sick_euthyroid", "spectrometer", "car_eval_34",
            "us_crime", "yeast_ml8", "scene", "car_eval_4", "thyroid_sick", "wine_quality", "solar_flare_m0",
            "oil", "yeast_me2", "ozone_level", "abalone_19"]
suites = sorted(ds_names)
kn_values = [2, 3, 5, 8, 10]

results_acc = {}
results_bacc = {}

for dataset_name in suites:
    results_acc[dataset_name] = {}
    results_bacc[dataset_name] = {}
    for OS_name in OS_names:
        results_acc[dataset_name][OS_name] = 0
        results_bacc[dataset_name][OS_name] = 0

run = 1
for kneighbors in kn_values:
    print(f"\n<> Started run {run}/5 with k_neighbors={kneighbors} <>")
    run += 1

    oversamplers = {
        "Baseline" : NotOverSampler(),
        "RandomOS" : RandomOverSampler(),
        "IL-SMOTE" : SMOTE(k_neighbors=kneighbors),
        "NE - max" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "cliques", "wrap": "all", "k-neighbors": kneighbors}),
        "NE - all" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "cliques", "wrap": "max", "k-neighbors": kneighbors}),
        "NE - KNN" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "neighbors", "wrap": "max", "k-neighbors": kneighbors}),
        "NE-SMOTE" : NESMOTE(std_euclid_distance, std_euclid_wavg, params={"groupby": "smote", "wrap": "all", "k-neighbors": kneighbors}),
    }

    for dataset_name in suites:
        print(f"\n-> running {len(oversamplers.keys())} trials from suite {dataset_name}")
        dataset = datasets[dataset_name]

        for OS_name in oversamplers.keys():
            
            start = time()

            try:
                oversampler = oversamplers[OS_name]

                bacc, acc = run_trial(dataset, oversampler, RandomForestClassifier())

                if bacc > results_bacc[dataset_name][OS_name]:
                    results_bacc[dataset_name][OS_name] = bacc
                if acc > results_acc[dataset_name][OS_name]:
                    results_acc[dataset_name][OS_name] = acc
            
                end = time()
                print(f"--> trial {OS_name} completed in {round(end - start, 2)}")
            except Exception as e:
                print(f"--> Exception caught while running trial {OS_name}")
                print("----", e)

out_file = open("results.out", "w")

out_file.write("{:15} ".format(""))
for OS_name in OS_names:
    out_file.write("  {:8} ".format(OS_name))
out_file.write("\n")

for dataset_name in suites:
    out_file.write("{:15} ".format(dataset_name))
    for OS_name in OS_names:
        res = results_bacc[dataset_name][OS_name]
        if res > 0:
            out_file.write("{:10.4f} ".format(res))
        else:
            out_file.write("{:10} ".format("--"))
    out_file.write("\n")

out_file.close()        
