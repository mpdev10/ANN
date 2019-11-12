import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    pickle_name = input()
    pkl_file = open(pickle_name, 'rb')
    run_results = pkl.load(pkl_file)

    labels = []

    fig, ax1 = plt.subplots()

    x = np.arange(0, len(run_results[0].averages))
    ax1.set_xlabel('Liczba epok')
    ax1.set_ylabel('Dokładność [%]')

    for i in range(0, len(run_results)):
        label = input()
        labels.append(label)

    for i in range(0, len(run_results)):
        print("{0} average test score: {1}; std: {2}".format(labels[i], run_results[i].test_average,
                                                             run_results[i].test_std_deviation))
        ax1.plot(x, run_results[i].averages, label=labels[i])
    ax1.legend()
    ax1.margins(0)

    plt.show()
