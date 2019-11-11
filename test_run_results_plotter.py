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
        ax1.plot(x, run_results[i].averages, label=labels[i])

    plt.show()
