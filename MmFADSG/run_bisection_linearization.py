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

    parser = argparse.ArgumentParser(description='Bisection Tester')
    parser.add_argument('--dataset-name', type=str, default='cosponsorBel', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--target_alpha', type=float, default=0.08, metavar='R', help='Target Diversity Level')
    parser.add_argument('--epsilon', type=float, default=1e-2, metavar='R', help='Tolerance')
    parser.add_argument('--alpha', type=float, default=0.05, metavar='R', help='Diversity Level')
    parser.add_argument('--num-iterations', type=int, default=None, metavar='N', help='Number of bisection iterations')
    parser.add_argument('linearization_method', type=int, default=1, metavar='L', help='Linearization method (1: finite difference approximation, 2: block ordering (ascending), 3: symmetric subgradient, 4: block ordering (descending))')

    args = parser.parse_args()

    dataset_name = copy.deepcopy(args.dataset_name)
    target_alpha = copy.deepcopy(args.target_alpha)
    epsilon = copy.deepcopy(args.epsilon)
    alpha = copy.deepcopy(args.alpha)
    num_iterations = copy.deepcopy(args.num_iterations)
    linearization_method = copy.deepcopy(args.linearization_method)


    lam_max = 1000
    G, protected_nodes = init_graph(dataset_name)

    n = G.number_of_nodes()
    print('Number of Nodes:', n)
    m = G.number_of_edges()
    print('Number of Edges:', m)



    num_outer = 10
    T = 5
    mu = 1

    lam_max_ = lam_max
    lam_min_ = 0

    lam_mid_vec = []
    diversity_vec = []
    density_vec = []
    num_of_nodes = []
    num_of_protected_vec = []
    protected_portion_in_prot_vec = []
    fairness_vec = []
    LSE_values = []

    if(num_iterations is not None):
        max_iters = num_iterations
    else:
        max_iters = int(np.ceil(np.log2(lam_max/epsilon)))
    S0 = G  
    for i in tqdm(range(max_iters)):
        # ---- Pick midpoint and run solver there ----
        lam_mid = (lam_max_ + lam_min_) / 2
        lam_mid_vec.append(lam_mid)

        super_greedy_pp_R = super_greedy_pp_lse_DC_SSP(
            G, S0, protected_nodes, lam=lam_mid,
            mu=mu, alpha=alpha,
            num_passes=T, num_outer=num_outer, lin_method=linearization_method
        )
        #S = G.subgraph(super_greedy_pp_R[0])
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
        if abs(diversity - target_alpha) < epsilon:
            print(f"Converged at iteration {i} with lambda={lam_mid}, diversity={diversity}, target alpha={target_alpha}")
            break

        if diversity < target_alpha:
            # not diverse enough -> need more fairness pressure -> larger lambda
            lam_min_ = lam_mid
        else:
            # too diverse -> reduce fairness pressure -> smaller lambda
            lam_max_ = lam_mid
        
    if i == max_iters - 1:
        print(f"Reached max iterations with lambda={lam_mid}, diversity={diversity}, target alpha={target_alpha}")

    L = len(protected_nodes)
    lam_arr = np.array(lam_mid_vec)
    div_arr = np.array(diversity_vec)
    density_arr = np.array(density_vec)
    Lvec = [1.0 / L for _ in range(len(lam_arr))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # ---- Left: minimum represented proportion ----
    ax1.set_facecolor('#f0f0f0')
    ax1.scatter(lam_arr, div_arr, color="tab:blue", s=60, edgecolor='black',
                label="Super-Greedy++ LSE")
    ax1.plot(lam_arr, Lvec, color="tab:red", linestyle="dashed", label="1/L")
    ax1.axhline(target_alpha, color="tab:green", linestyle="dotted", label=r"target $\alpha$")
    ax1.set_xlabel(r"$\lambda$", fontsize=20)
    ax1.set_ylabel(
        r"$\underset{\ell}{\mathrm{min}} \; \dfrac{|\mathcal{S} \cap \mathcal{S}_{\ell}|}{|\mathcal{S}|}$",
        fontsize=20)
    ax1.legend(fontsize=16)

    # ---- Right: density ----
    ax2.set_facecolor('#f0f0f0')
    ax2.scatter(lam_arr, density_arr, color="tab:blue", s=60, edgecolor='black')
    ax2.set_xlabel(r"$\lambda$", fontsize=20)
    ax2.set_ylabel(r"$\rho(\mathcal{S})$", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"bisection_{dataset_name}.svg", format='svg', bbox_inches='tight')
    plt.show()

    # Save Variables
    variables_dict = {
        'target_alpha': target_alpha,
        'lam_mid': lam_mid_vec,
        'induced': super_greedy_pp_R,
        'induced_protected': induced_protected_nodes,
        'protected': protected_nodes,
        'max_iters': max_iters,
        'num_of_nodes': num_of_nodes,
        'density': density_vec,
        'diversity': diversity_vec,
        'num_of_protected': num_of_protected_vec,
        'protected_portion_in_sub': diversity_vec,
        'protected_portion_in_prot': protected_portion_in_prot_vec,
        'fairness': fairness_vec,
        'LSE_values': LSE_values
    }


    save_folder = 'logs/bisection_/' + dataset_name
    save_path = save_folder + '/' + dataset_name  + '_' + str(target_alpha) + '_log.npy'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)


if __name__ == '__main__':
    main()