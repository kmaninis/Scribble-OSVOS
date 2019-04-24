import os
import json
import matplotlib.pyplot as plt
from mypath import Path

METRIC_TXT = {'J': 'J',
              'F': 'F',
              'J_AND_F': 'J&F'}


def main():
    metric = 'J_AND_F'

    with open(os.path.join(Path.save_root_dir(), 'summary.json'), 'r') as f:
        summary = json.load(f)

    print('AUC: \t{:.3f}'.format(summary['auc']))
    th = summary['metric_at_threshold']['threshold']
    jac = summary['metric_at_threshold'][metric]
    print('J@{}: \t{:.3f}'.format(th, jac))

    time = summary['curve']['time']
    jaccard = summary['curve'][metric]

    plt.plot(time, jaccard)
    plt.ylim([0, 1])
    plt.xlim([0, max(time)])
    plt.xlabel('Accumulated Time (s)')
    plt.ylabel(r'$\mathcal{' + METRIC_TXT[metric] + '}$')
    plt.axvline(th, c='k')
    plt.show()


if __name__ == '__main__':
    main()
