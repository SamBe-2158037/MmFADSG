import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dsd
from super_greedy_set_linear import *
from init_graph import *
import super_greedy_set_linear
import utils
from tqdm import tqdm
import argparse


def main():

    parser = argparse.ArgumentParser(description='Binary search Tester')
  
    parser.add_argument('--dataset-name', type=str, default='cosponsorBel', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--target_alpha', type=float, default=0.08, metavar='R', help='Target Diversity Level')
    parser.add_argument('--epsilon', type=float, default=1e-2, metavar='R', help='Tolerance')
    parser.add_argument('--alpha', type=float, default=0.05, metavar='R', help='Diversity Level')
    parser.add_argument('--num-iterations', type=int, default=None, metavar='N', help='Number of bisection iterations')
    parser.add_argument('--linearization-method', type=int, default=1, metavar='L', help='Linearization method (1: finite difference approximation, 2: block ordering (ascending), 3: symmetric subgradient, 4: block ordering (descending))')

    args = parser.parse_args()

    dataset_name = copy.deepcopy(args.dataset_name)
    target_alpha = copy.deepcopy(args.target_alpha)
    epsilon = copy.deepcopy(args.epsilon)
    alpha = copy.deepcopy(args.alpha)
    num_iterations = copy.deepcopy(args.num_iterations)
    linearization_method = copy.deepcopy(args.linearization_method)

    lam_max = 2000
    G, protected_nodes = init_graph(dataset_name)


    

    n = G.number_of_nodes()
    print('Number of Nodes:', n)
    m = G.number_of_edges()
    print('Number of Edges:', m)


    num_outer = 10
    T = 5
    mu = 1

    lam_max_ = lam_max
    lam_final = []
    lam_min_ = 0

    lam_mid_vec = []
    diversity_vec = []
    density_vec = []
    num_of_nodes = []
    num_of_protected_vec = []
    protected_portion_in_prot_vec = []
    fairness_vec = []
    LSE_values = []
    target_reached = []
    density_reached = []

    L = len(protected_nodes)
    target_etha_range = np.linspace(0, 1/L, 11)

    if(num_iterations is not None):
        max_iters = num_iterations
    else:
        max_iters = int(np.ceil(np.log2(lam_max/epsilon)))
    
    for target_etha in target_etha_range:
        lam_max_ = lam_max
        lam_min_ = 0
        converged = False
        for i in tqdm(range(max_iters)):
            # ---- Pick midpoint and run solver there ----
            lam_mid = (lam_max_ + lam_min_) / 2
            lam_mid_vec.append(lam_mid)

            super_greedy_pp_R = super_greedy_pp_lse_DC_SSP(
                G, G, protected_nodes, lam=lam_mid,
                mu=mu, alpha=alpha,
                num_passes=T, num_outer=num_outer
            )
            super_greedy_pp_R_G = G.subgraph(super_greedy_pp_R[0])
            super_greedy_pp_R_nodes = list(super_greedy_pp_R_G.nodes())
            super_greedy_pp_R_nodes.sort()
            LSE_value = super_greedy_pp_R[2]
            induced_protected_nodes = super_greedy_pp_R[2]

            # ---- Compute metrics ----
            num_of_nodes.append(len(super_greedy_pp_R_nodes))

            density = utils.compute_density(super_greedy_pp_R_G)
            density_vec.append(density)


            fairness = sum([utils.find_num_common(super_greedy_pp_R_nodes, protected_nodes[i_])
                            for i_ in range(len(protected_nodes))])
            fairness_vec.append(fairness)

            num_of_protected = min([utils.find_num_common(super_greedy_pp_R_nodes, protected_nodes[i_])
                                    for i_ in range(len(protected_nodes))])
            num_of_protected_vec.append(num_of_protected)

            # ---- The diversity value used for bisection ----
            if len(super_greedy_pp_R_G) > 0:
                diversity = num_of_protected / len(super_greedy_pp_R_G)
            else:
                diversity = 0.0
            diversity_vec.append(diversity)

            protected_portion_in_prot = num_of_protected / G.number_of_nodes()
            protected_portion_in_prot_vec.append(protected_portion_in_prot)
            LSE_values.append(LSE_value)

            # ---- Bisection on diversity vs target alpha ----
            if abs(diversity - target_etha) < epsilon:
                converged = True
                print(f"Converged at iteration {i} with lambda={lam_mid}, diversity={diversity}, target etha={target_etha}")
                target_reached.append(diversity)
                density_reached.append(density)
                lam_final.append(lam_mid)
                break

            if diversity < target_etha:
                # not diverse enough -> need more fairness pressure -> larger lambda
                lam_min_ = lam_mid
            else:
                # too diverse -> reduce fairness pressure -> smaller lambda
                lam_max_ = lam_mid
            
        if not converged:
            print(f"Reached max iterations with lambda={lam_mid}, diversity={diversity}, target etha={target_etha}")
            target_reached.append(diversity)
            density_reached.append(density)
            lam_final.append(lam_mid)
    # ============================================================
    # Plot: same layout as parameter sweep, but with bisection points
    # ============================================================
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(target_etha_range, target_reached, marker='o', label='Bisection Reached Diversity')
    plt.plot(target_etha_range, target_etha_range, linestyle='--', color='red', label='Target Diversity')
    plt.plot(target_etha_range,[target_etha+epsilon for target_etha in target_etha_range], linestyle=':', color='green', label='Target Diversity error bound')
    plt.plot(target_etha_range,[target_etha-epsilon for target_etha in target_etha_range], linestyle=':', color='green')
    plt.legend()
    plt.xlabel('Target Diversity')
    plt.ylabel('Reached Diversity')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(lam_final, target_reached, marker='s', color='blue', label='Final Bisection Points')
    plt.xlabel('Lambda at Final Bisection Point')
    plt.ylabel('Diversity at Final Bisection Point')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dataset_name}/bisection_sweep_results_{dataset_name}.pdf')
    plt.show()


if __name__ == '__main__':
    main()