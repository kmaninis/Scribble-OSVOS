""" Analyse Global Summary
"""
import json

import matplotlib.pyplot as plt


def main():
    with open('results/summary.json', 'r') as fp:
        summary = json.load(fp)

    print('AUC: \t{:.3f}'.format(summary['auc']))
    th = summary['jaccard_at_threshold']['threshold']
    jac = summary['jaccard_at_threshold']['jaccard']
    print('J@{}: \t{:.3f}'.format(th, jac))

    time = summary['curve']['time']
    jaccard = summary['curve']['jaccard']

    plt.plot(time, jaccard)
    plt.ylim([0, 1])
    plt.xlim([0, max(time)])
    plt.xlabel('Accumulated Time (s)')
    plt.ylabel(r'Jaccard ($\mathcal{J}$)')
    plt.axvline(th, c='k')
    plt.show()


if __name__ == '__main__':
    main()
