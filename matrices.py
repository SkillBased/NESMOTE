from sklearn.model_selection import train_test_split
import numpy as np

from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.spd_matrices import SPDMetricAffine

from imblearn.datasets import fetch_datasets

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from scipy.linalg import logm

from NESMOTE.oversample import Pipeline
from NESMOTE.oversample import ANNProcessor, MCProcessor, SMOTEProcessor
from NESMOTE.oversample import UniformSampler, StandardSampler
from NESMOTE.neighbors import RingQuery

from time import time

metric = SPDMetricAffine(12)
mean = FrechetMean(metric)

def matrix_distance(a, b):
    global metric
#    print(a, b, flush=True)
    d = metric.dist(a, b)
#    print("dist ret", flush=True)
    return d

def matrix_mean(weights, points):
    global mean
    mean.fit(X=points, weights=weights)
    return mean.estimate_


data = np.load("corrs.npy")
labels = np.load("labels.npy")
label = labels[2]

class Dataset:
    def __init__(self, data, label) -> None:
        self.data = data
        self.target = label

class NotOverSampler:
    def __init__(self):
        self.trivial = True
    
    def fit_resample(self, X, y):
        return X, y

notos = NotOverSampler()
rdmos = RandomOverSampler()
mcstd = Pipeline(RingQuery(matrix_distance), MCProcessor(), StandardSampler(matrix_mean))
annag = Pipeline(RingQuery(matrix_distance), ANNProcessor(), UniformSampler(matrix_mean))

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

def log_transform(mat):
    lg = logm(mat)
    res = []
    for x in range(12):
        for y in range(12):
            if y < x:
                continue
            res.append(lg[x, y])
    return np.array(res)

def run_trial(dataset, oversampler, classifier, nruns=5):
    cum = None
    for _ in range(nruns):
        X, y = dataset.data, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.3)
#        print("split", flush=True)
        X_train_flat = np.array([])
        if isinstance(oversampler, RandomOverSampler):
            X_train_flat = np.array([log_transform(x) for x in X_train])
            X_train_flat, y_train_OS = oversampler.fit_resample(X_train_flat, y_train)
        else:
            X_train_OS, y_train_OS = oversampler.fit_resample(X_train, y_train)
            X_train_flat = np.array([log_transform(x) for x in X_train_OS])
#        print("oversample", flush=True)
        classifier.fit(X_train_flat, y_train_OS)
#        print("fit", flush=True)
        X_test_flat = np.array([log_transform(x) for x in X_test])
#        print("logm test", flush=True)
        preds = classifier.predict(X_test_flat)
#        print("predict", flush=True)
        ca = compute_classwize_accuracy(y_test, preds)
        if cum is None:
            cum = ca
        else:
            for c in cum.keys():
                cum[c][0] += ca[c][0]
                cum[c][1] += ca[c][1]
    return get_scores(cum)

datasets = {}
for idx, row in enumerate(data):
    datasets["set" + str(idx)] = Dataset(row, label)

def main():
    OS_names = ["Baseline",
                "RandomOS",
                "NE-maxSP",
                "NE-maxAG",
                "NE-allSP",
                "NE-allAG",
                "SP-SMOTE", 
                "AG-SMOTE"]

    ds_names = datasets.keys()
    suites = sorted(ds_names)
    kn_values = [3, 5, 8]
    kn_values = [5]

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
            "NE-allAG" : Pipeline(RingQuery(matrix_distance), ANNProcessor(), UniformSampler(matrix_mean)),
            "NE-maxSP" : Pipeline(RingQuery(matrix_distance), MCProcessor(), StandardSampler(matrix_mean)),
            "NE-maxAG" : Pipeline(RingQuery(matrix_distance), MCProcessor(), UniformSampler(matrix_mean)),
            "NE-allSP" : Pipeline(RingQuery(matrix_distance), ANNProcessor(), StandardSampler(matrix_mean)),
            "SP-SMOTE" : Pipeline(RingQuery(matrix_distance), SMOTEProcessor(), StandardSampler(matrix_mean)),
            "AG-SMOTE" : Pipeline(RingQuery(matrix_distance), SMOTEProcessor(), UniformSampler(matrix_mean)),
        }

        for dataset_name in suites:
            print(f"\n-> running {len(oversamplers.keys())} trials from suite {dataset_name}")
            dataset = datasets[dataset_name]

            for OS_name in oversamplers.keys():
                
                start = time()

                try:
                    oversampler = oversamplers[OS_name]

                    macc, Macc = run_trial(dataset, oversampler, LogisticRegression(), 2)

                    if macc > results_macc[dataset_name][OS_name]:
                        results_macc[dataset_name][OS_name] = macc
                    if Macc > results_Macc[dataset_name][OS_name]:
                        results_Macc[dataset_name][OS_name] = Macc
                
                    end = time()
                    print(f"--> trial {OS_name} completed in {round(end - start, 2)}")
                except Exception as e:
                    print(f"--> Exception caught while running trial {OS_name}")
                    print("----", e)

    out_file = open("spheric.out", "w")

    out_file.write("{:15} ".format("minor"))
    for OS_name in OS_names:
        out_file.write("  {:8} ".format(OS_name))
    out_file.write("\n")

    for dataset_name in suites:
        out_file.write("{:15} ".format(dataset_name))
        for OS_name in OS_names:
            res = results_macc[dataset_name][OS_name]
            if res > 0:
                out_file.write("{:10.4f} ".format(res))
            else:
                out_file.write("{:10} ".format("--"))
        out_file.write("\n")

    out_file.write("\n")

    out_file.write("{:15} ".format("major"))
    for OS_name in OS_names:
        out_file.write("  {:8} ".format(OS_name))
    out_file.write("\n")

    for dataset_name in suites:
        out_file.write("{:15} ".format(dataset_name))
        for OS_name in OS_names:
            res = results_Macc[dataset_name][OS_name]
            if res > 0:
                out_file.write("{:10.4f} ".format(res))
            else:
                out_file.write("{:10} ".format("--"))
        out_file.write("\n")

    out_file.close()  

if __name__ == '__main__':
    main()