from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

from optint.data import synthetic_instance, gen_dag
from optint.run import run
from optint.visualize import *

import warnings
warnings.filterwarnings('ignore')
seed = 1234
np.random.seed(seed)

# generate problem instantiation
nnodes = 10
sigma_square = np.ones(nnodes)
DAG_type = 'complete'

num_instances = 2
a_size = 5

problems = []
graph = gen_dag(nnodes=nnodes, DAG_type=DAG_type)
for _ in range(num_instances):
	problem = synthetic_instance(
		nnodes=nnodes, 
		DAG_type=DAG_type,
		std=True,
		sigma_square=sigma_square, 
		a_size=a_size,
		a_target_nodes=[i+nnodes//2 for i in range(a_size)], 
		prefix_DAG=graph
		)
	problems.append(problem)
# Add noise level parameter
# noise_levels = [0.1, 0.5, 1.0]  # Different noise standard deviations
noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]

# Create subplots for each acquisition function
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Run experiments for each noise level
MSEs_by_noise = {noise: [] for noise in noise_levels}
for noise in noise_levels:
    opts = Namespace(n=1, T=50, W=0, R=20, time=True, noise_std=noise)
    MSEs = []  
	
    # options for testing
    acqs = ['greedy', 'maxv','cv', 'civ', 'civ', 'ei', 'mi', 'ucb']
    measure = [None, None, None, 'unif', 'ow', None, None, None]
    known_variance = [False, True, True, True, True, False, True, True]
    name = ['greedy', 'maxv', 'cv', 'civ', 'civow', 'ei', 'mi', 'ucb']

    As = []
    Probs = []
    Times = []
    for i in range(num_instances):
        problem = problems[i]

        A = {}
        Prob = {}
        Time = {}

        opts.active = False
        A['passive'], Prob['passive'], Time['passive'] = run(problem, opts)

        opts.active = True
        for a in range(len(acqs)):
            opts.acq = acqs[a]
            opts.measure = measure[a]
            opts.known_noise = known_variance[a]
            A[name[a]], Prob[name[a]], Time[name[a]]  = run(problem, opts)

        print(f'Graph {i+1}')

        As.append(A)
        Probs.append(Prob)
        Times.append(Time)


    mu_MSEs = []
    for i in range(num_instances):
        Prob = Probs[i]
        problem = problems[i]
        mu_mses = {k:[] for k in ['passive', 'greedy', 'maxv', 'cv', 'civ', 'civow', 'ei', 'mi','ucb']}
        for r in range(opts.R):
            for k in ['passive', 'greedy', 'maxv', 'cv', 'civ', 'civow', 'ei', 'mi', 'ucb']:
                mse = []
                for prob in Prob[k][r]:
                    errs = abs(np.dot(problem.A,np.dot(np.eye(problem.nnodes)-np.array(prob['mean']),problem.mu_target)) - problem.mu_target)
                    mse.append(np.linalg.norm(np.concatenate(errs)) / np.linalg.norm(problem.mu_target))
                mu_mses[k].append(mse)
        mu_MSEs.append(mu_mses)
    MSEs_by_noise[noise] = mu_MSEs

for i, acq in enumerate(['greedy', 'maxv', 'cv', 'civ', 'civow', 'ei', 'mi', 'ucb']):
    ax = axes[i]
    for noise in noise_levels:
        mean = np.array([np.array(MSEs_by_noise[noise][j][acq]).mean(axis=0) 
                        for j in range(num_instances)]).mean(axis=0)
        std = np.array([np.array(MSEs_by_noise[noise][j][acq]).std(axis=0) 
                       for j in range(num_instances)]).mean(axis=0)
        
        ax.plot(range(opts.T), mean, label=f'noise={noise}')
        ax.fill_between(range(opts.T), mean - std, mean + std, alpha=0.2)
    
    ax.set_yscale('log')
    # ax.set_title(labels[i+1]) 
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/noise_comparison.pdf')

plt.clf()
fig, ax = plt.subplots(1,1,figsize=(5.6,3.8))
plt.rcParams.update({'font.size': 12})

plt.yscale('log')

labels = ['Random', 'Greedy', 'EI-Int', 'MI-Int', 'UCB-Int', 'MaxV', 'CV', 'CIV', 'CIV-OW']
colors = ['#069AF3', '#9ACD32', 'magenta','pink', 'turquoise', 'black', 'grey', 'orange', '#C79FEF']
markers = ['^', 'o', 'o', 'o', 'o', 'o', 'o', 's', 's']
linestyles = ['--','--', '--', '--','--','--', '--','-', '-']
for i, k in enumerate(['passive', 'greedy', 'ei', 'mi', 'ucb', 'maxv', 'cv', 'civ', 'civow']):
	mean = np.array([np.array(mu_MSEs[i][k]).mean(axis=0) for i in range(num_instances)]).mean(axis=0)
	std = np.array([np.array(mu_MSEs[i][k]).std(axis=0) for i in range(num_instances)]).mean(axis=0)
	plt.plot(range(opts.T), mean, label=labels[i], linewidth=2, color=colors[i], marker=markers[i], markersize=0, linestyle=linestyles[i])
	plt.fill_between(range(opts.T), mean - std, mean + std, alpha=.2, color=colors[i])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
# plt.ylim(0.01,1)

plt.legend(loc='upper right')
plt.xlabel(r'time step $t$')
plt.ylabel(r'$||\mu_t^*-\mu^*||_2~/~||\mu^*||_2$ (log scale)')
# plt.title(f'square distance to target mean')
plt.tight_layout()

plt.savefig(f'figures/additional_baseline/relative-rmse_{DAG_type}-{nnodes}-{a_size}.pdf')