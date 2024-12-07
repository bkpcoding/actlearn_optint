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

num_instances = 10
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
noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]

# Create subplots for each acquisition function
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes = axes.flatten()

# Run experiments for each noise level
MSEs_by_noise = {noise: [] for noise in noise_levels}
for noise in noise_levels:
    opts = Namespace(n=1, T=50, W=0, R=20, time=True, noise_std=noise)
    MSEs = []

    # options for testing - only greedy, civ, civow
    acqs = ['greedy', 'civ', 'civ']
    measure = [None, 'unif', 'ow']
    known_variance = [False, True, True]
    name = ['greedy', 'civ', 'civow']

    As = []
    Probs = []
    Times = []
    for i in range(num_instances):
        problem = problems[i]

        A = {}
        Prob = {}
        Time = {}

        opts.active = True
        for a in range(len(acqs)):
            opts.acq = acqs[a]
            opts.measure = measure[a]
            opts.known_noise = known_variance[a]
            A[name[a]], Prob[name[a]], Time[name[a]] = run(problem, opts)

        print(f'Graph {i+1}')

        As.append(A)
        Probs.append(Prob)
        Times.append(Time)

    mu_MSEs = []
    for i in range(num_instances):
        Prob = Probs[i]
        problem = problems[i]
        mu_mses = {k:[] for k in ['greedy', 'civ', 'civow']}
        for r in range(opts.R):
            for k in ['greedy', 'civ', 'civow']:
                mse = []
                for prob in Prob[k][r]:
                    errs = abs(np.dot(problem.A,np.dot(np.eye(problem.nnodes)-np.array(prob['mean']),problem.mu_target)) - problem.mu_target)
                    # mse.append(np.linalg.norm(np.concatenate(errs)) / np.linalg.norm(problem.mu_target))
                    mse.append(np.linalg.norm(np.concatenate(errs)))
                mu_mses[k].append(mse)
        mu_MSEs.append(mu_mses)
    MSEs_by_noise[noise] = mu_MSEs

for i, acq in enumerate(['greedy', 'civ', 'civow']):
    ax = axes[i]
    for noise in noise_levels:
        mean = np.array([np.array(MSEs_by_noise[noise][j][acq]).mean(axis=0) 
                        for j in range(num_instances)]).mean(axis=0)
        std = np.array([np.array(MSEs_by_noise[noise][j][acq]).std(axis=0) 
                       for j in range(num_instances)]).mean(axis=0)
        
        ax.plot(range(opts.T), mean, label=f'noise={noise}')
        ax.fill_between(range(opts.T), mean - std, mean + std, alpha=0.2)
    
    ax.set_yscale('log')
    ax.set_title(acq)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/noise_comparison_{DAG_type}-{nnodes}-{a_size}-{num_instances}.pdf')

plt.clf()
fig, ax = plt.subplots(1,1,figsize=(5.6,3.8))
plt.rcParams.update({'font.size': 12})

plt.yscale('log')

labels = ['Greedy', 'CIV', 'CIV-OW']
colors = ['#9ACD32', 'orange', '#C79FEF']
markers = ['o', 's', 's']
linestyles = ['--','-', '-']
for i, k in enumerate(['greedy', 'civ', 'civow']):
    mean = np.array([np.array(mu_MSEs[i][k]).mean(axis=0) for i in range(num_instances)]).mean(axis=0)
    std = np.array([np.array(mu_MSEs[i][k]).std(axis=0) for i in range(num_instances)]).mean(axis=0)
    plt.plot(range(opts.T), mean, label=labels[i], linewidth=2, color=colors[i], marker=markers[i], markersize=0, linestyle=linestyles[i])
    plt.fill_between(range(opts.T), mean - std, mean + std, alpha=.2, color=colors[i])

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.legend(loc='upper right')
plt.xlabel(r'time step $t$')
plt.ylabel(r'$||\mu_t^*-\mu^*||_2~/~||\mu^*||_2$ (log scale)')
plt.tight_layout()

plt.savefig(f'figures/noise_comparison_{DAG_type}-{nnodes}-{a_size}_{num_instances}.pdf')

plt.clf()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({'font.size': 12})

methods = ['greedy', 'civ', 'civow']
method_labels = ['Greedy', 'CIV', 'CIV-OW']
colors = ['lightcoral', 'cornflowerblue', 'lightgreen', 'orange', 'purple']

# For each method
for method_idx, method in enumerate(methods):
    ax = axes[method_idx]
    
    # Get final values for each noise level
    final_means = []
    final_stds = []
    
    for noise in noise_levels:
        # Calculate final values (last time step) for current method and noise level
        final_values = []
        for i in range(num_instances):
            instance_values = np.array(MSEs_by_noise[noise][i][method])[:, -1]  # Get last timestep
            final_values.extend(instance_values)
        
        final_means.append(np.mean(final_values))
        final_stds.append(np.std(final_values))
    
    # Create bar plot
    x_pos = range(len(noise_levels))
    ax.bar(x_pos, final_means, yerr=final_stds, capsize=5, 
           color=colors[:len(noise_levels)], alpha=0.7)
    
    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n}' for n in noise_levels], rotation=45)
    ax.set_yscale('log')
    ax.set_title(method_labels[method_idx])
    ax.set_xlabel('Noise Level')
    if method_idx == 0:
        ax.set_ylabel('Final Error (log scale)')
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/final_error_comparison_{DAG_type}-{nnodes}-{a_size}-{num_instances}.pdf')
