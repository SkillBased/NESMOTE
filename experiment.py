from sklearn.model_selection import train_test_split
import numpy as np

from imblearn.datasets import fetch_datasets

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE

from NESMOTE.util import std_euclid_distance, std_euclid_wavg

from NESMOTE.oversample import Pipeline
from NESMOTE.oversample import ANNProcessor, MCProcessor, SMOTEProcessor
from NESMOTE.oversample import UniformSampler, StandardSampler
from NESMOTE.neighbors import RingQuery, NENN

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
    minor, major = 0, 0
    for c in cum_res.keys():
        if cum_res[c][2] < 0.5:
            minor += cum_res[c][1] / cum_res[c][0]
        else:
            major += cum_res[c][1] / cum_res[c][0]
    return minor, major



datasets = fetch_datasets()

class NotOverSampler:
    def __init__(self):
        self.trivial = True
    
    def fit_resample(self, X, y):
        return X, y


def run_trial(dataset, oversampler, classifier, nruns=5):
    cum = None
    for _ in range(nruns):
        X, y = dataset.data, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.3)
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


def main():
    OS_names = ["Baseline",
                "RandomOS",
                "IL-SMOTE",
                "NE-maxSP",
                "NE-maxAG",
                "NE-allSP",
                "NE-allAG",
                "SP-SMOTE", 
                "AG-SMOTE"]

    ds_names = ["ecoli", "optical_digits", "abalone", "sick_euthyroid", "spectrometer", "car_eval_34",
                "us_crime", "yeast_ml8", "car_eval_4", "thyroid_sick", "wine_quality", "solar_flare_m0",
                "yeast_me2", "ozone_level", "abalone_19"]
    #ds_names = ["ecoli", "oil"]
    suites = sorted(ds_names)
    kn_values = [3, 5, 8]

    results_macc = {}
    results_Macc = {}

    for dataset_name in suites:
        results_macc[dataset_name] = {}
        results_Macc[dataset_name] = {}
        for OS_name in OS_names:
            results_macc[dataset_name][OS_name] = 0
            results_Macc[dataset_name][OS_name] = 0
    
    run = 1
    for kneighbors in kn_values:
        print(f"\n<> Started run {run}/{len(kn_values)} with k_neighbors={kneighbors} <>")
        run += 1

        oversamplers = {
            "Baseline" : NotOverSampler(),
            "RandomOS" : RandomOverSampler(),
            "IL-SMOTE" : SMOTE(k_neighbors=kneighbors),
            "NE-allAG" : Pipeline(RingQuery(std_euclid_distance), ANNProcessor(), UniformSampler(std_euclid_wavg)),
            "NE-maxSP" : Pipeline(RingQuery(std_euclid_distance), MCProcessor(), StandardSampler(std_euclid_wavg)),
            "NE-maxAG" : Pipeline(RingQuery(std_euclid_distance), MCProcessor(), UniformSampler(std_euclid_wavg)),
            "NE-allSP" : Pipeline(RingQuery(std_euclid_distance), ANNProcessor(), StandardSampler(std_euclid_wavg)),
            "SP-SMOTE" : Pipeline(RingQuery(std_euclid_distance), SMOTEProcessor(), StandardSampler(std_euclid_wavg)),
            "AG-SMOTE" : Pipeline(RingQuery(std_euclid_distance), SMOTEProcessor(), UniformSampler(std_euclid_wavg)),
        }

        for dataset_name in suites:
            print(f"\n-> running {len(oversamplers.keys())} trials from suite {dataset_name}")
            dataset = datasets[dataset_name]

            for OS_name in oversamplers.keys():
                
                start = time()

                try:
                    oversampler = oversamplers[OS_name]

                    macc, Macc = run_trial(dataset, oversampler, RandomForestClassifier())

                    if macc > results_macc[dataset_name][OS_name]:
                        results_macc[dataset_name][OS_name] = macc
                    if Macc > results_Macc[dataset_name][OS_name]:
                        results_Macc[dataset_name][OS_name] = Macc
                
                    end = time()
                    print(f"--> trial {OS_name} completed in {round(end - start, 2)}")
                except Exception as e:
                    print(f"--> Exception caught while running trial {OS_name}")
                    print("----", e)

    out_file = open("results.out", "w")

    out_file.write("{:15} &".format("minor"))
    for OS_name in OS_names:
        out_file.write("  {:8} &".format(OS_name))
    out_file.write("\n")

    for dataset_name in suites:
        out_file.write("{:15} &".format(dataset_name))
        for OS_name in OS_names:
            res = results_macc[dataset_name][OS_name]
            if res > 0:
                out_file.write("{:10.4f} &".format(res))
            else:
                out_file.write("{:10} &".format("--"))
        out_file.write("\n")

    out_file.write("\n")

    out_file.write("{:15} &".format("major"))
    for OS_name in OS_names:
        out_file.write("  {:8} &".format(OS_name))
    out_file.write("\n")

    for dataset_name in suites:
        out_file.write("{:15} &".format(dataset_name))
        for OS_name in OS_names:
            res = results_Macc[dataset_name][OS_name]
            if res > 0:
                out_file.write("{:10.4f} &".format(res))
            else:
                out_file.write("{:10} &".format("--"))
        out_file.write("\n")

    out_file.close()  

if __name__ == '__main__':
    main()