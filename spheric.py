from sklearn.model_selection import train_test_split
import numpy as np

from imblearn.datasets import fetch_datasets

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import RandomOverSampler

from NESMOTE.oversample import Pipeline
from NESMOTE.oversample import ANNProcessor, MCProcessor, SMOTEProcessor
from NESMOTE.oversample import UniformSampler, StandardSampler
from NESMOTE.neighbors import RingQuery

from random import randint
from time import time

def base_positive(x, y, z):
    return x > 0.0 and y > 0.0 and z > 0.0

def thin_positive(x, y, z):
    return x > 0.33 and y > 0.33 and z > 0.33

def thin_negative(x, y, z):
    return x > -0.33 and y > -0.33 and z > -0.33

def symmetric(x, y, z):
    return thin_positive(x, y, z) or thin_negative(x, y, z)

def head(x, y, z):
    return x > 0.6

def scale_point(point):
    x, y, z = point
    length = (x * x + y * y + z * z) ** 0.5
    res = [x / length, y / length, z / length]
    return np.array(res)

def generate_point(class_selector):
    x, y, z = 0, 0, 0
    while x == 0 and y == 0 and z == 0:
        x, y, z = randint(-100, 100), randint(-100, 100), randint(-100, 100)
    pt = scale_point([x, y, z])
    xr, yr, zr = pt
    pt_class = 1 if class_selector(xr, yr, zr) else 0
    return pt, pt_class

def spheric_distance(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    horde = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
    angle = 2 * np.arcsin(horde / 2)
    return angle

def spheric_mean(weigths, points):
    x, y, z, = 0, 0, 0
    for w, p in zip(weigths, points):
        x0, y0, z0 = p
        x += x0 * w
        y += y0 * w
        z += z0 * w
    return scale_point([x, y, z])

class NotOverSampler:
    def __init__(self):
        self.trivial = True
    
    def fit_resample(self, X, y):
        return X, y

notos = NotOverSampler()
rdmos = RandomOverSampler()
mcstd = Pipeline(RingQuery(spheric_distance), MCProcessor(), StandardSampler(spheric_mean))
annag = Pipeline(RingQuery(spheric_distance), ANNProcessor(), UniformSampler(spheric_mean))

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.target = []

    def make(self, n, selector, ratio = None):
        if ratio is None:
            ratio = np.log2(n) ** 2
        print(ratio)
        ones = 0
        for _ in range(n):
            x0, y0 = generate_point(selector)
            if y0 == 1:
                if ones >= (1 / ratio) * n:
                    continue
                ones += 1
            self.data.append(x0)
            self.target.append(y0)
        self.data = np.array(self.data)
        self.target = np.array(self.target)
        return self

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

datasets = {
    "base0512" : Dataset().make( 512, base_positive, 40),
    "base2048" : Dataset().make(2048, base_positive, 90),
    "base8192" : Dataset().make(8192, base_positive, 140),
    "thin0512" : Dataset().make( 512, thin_positive, 40),
    "thin2048" : Dataset().make(2048, thin_positive, 90),
    "thin8192" : Dataset().make(8192, thin_positive),
    "symm0512" : Dataset().make( 512, symmetric, 40),
    "symm2048" : Dataset().make(2048, symmetric, 90),
    "symm8192" : Dataset().make(8192, symmetric, 140),
    "head0512" : Dataset().make( 512, head, 40),
    "head2048" : Dataset().make(2048, head, 90),
    "head8192" : Dataset().make(8192, head, 140),
}

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
            "NE-allAG" : Pipeline(RingQuery(spheric_distance), ANNProcessor(), UniformSampler(spheric_mean)),
            "NE-maxSP" : Pipeline(RingQuery(spheric_distance), MCProcessor(), StandardSampler(spheric_mean)),
            "NE-maxAG" : Pipeline(RingQuery(spheric_distance), MCProcessor(), UniformSampler(spheric_mean)),
            "NE-allSP" : Pipeline(RingQuery(spheric_distance), ANNProcessor(), StandardSampler(spheric_mean)),
            "SP-SMOTE" : Pipeline(RingQuery(spheric_distance), SMOTEProcessor(), StandardSampler(spheric_mean)),
            "AG-SMOTE" : Pipeline(RingQuery(spheric_distance), SMOTEProcessor(), UniformSampler(spheric_mean)),
        }

        for dataset_name in suites:
            print(f"\n-> running {len(oversamplers.keys())} trials from suite {dataset_name}")
            dataset = datasets[dataset_name]

            for OS_name in oversamplers.keys():
                
                start = time()

                try:
                    oversampler = oversamplers[OS_name]

                    macc, Macc = run_trial(dataset, oversampler, SGDClassifier())

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