#!/usr/bin/env python
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from kl_planning.util import file_util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, required=True)
    parser.add_argument('--max_time', type=int, default=35)
    args = parser.parse_args()

    file_util.check_path_exists(args.pickle, "Pickle file")

    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)

    kl = data['kl_divergence'][0][:args.max_time + 1] # Adding 1 because it fills out span

    sns.set_style("darkgrid")

    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 3)
    sns.lineplot(np.arange(args.max_time + 1), kl, lw=5)
    ax.set_xlim(0, args.max_time)
    ax.set_xlabel('Timesteps', fontsize=20)
    ax.set_ylabel('KL Divergence', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.show()
