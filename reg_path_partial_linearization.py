import os
import copy
import numpy as np
import networkx as nx
from super_greedy_set_linear import (
    super_greedy_pp_lse_DC_SSP,
    default_alpha,
)
from init_graph import *
import utils
from tqdm import tqdm
import argparse


def main():

    parser = argparse.ArgumentParser(description='Fair Densest Subgraph — DC composition LSE')
    parser.add_argument('--dataset-name', type=str, default='cosponsorBel', metavar='S',
                        help='Desired Dataset Name')
    parser.add_argument('--num-outer', type=int, default=10, metavar='N',
                        help='Number of SSP outer iterations per lambda')
    parser.add_argument('--mu', type=float, default=3, metavar='M',
                        help='LSE temperature parameter (larger = closer to min)')
    parser.add_argument('--alpha', type=float, default=None, metavar='A',
                        help='corretion coefficient for Q_cross in linearization (default: 1/mu)')
    parser.add_argument('--lam-max', type=float, default=200.0,
                        help='Maximum lambda for sweep')
    parser.add_argument('--num-lam', type=int, default=int(160),
                        help='Number of lambda points in sweep')
    parser.add_argument('--num-passes', type=int, default=5,
                        help='Number of SuperGreedy++ passes per run')
    parser.add_argument('--linearization-method',type=int, default=2, metavar='L',
                        help='Linearization method (1: finite difference approximation, 2: block ordering (ascending), 3: symmetric subgradient, 4: block ordering (descending))')
    parser.add_argument('--initial-graph', type=int, default=0, metavar='I',
                        help='Initial graph for linearization (0: cold start, 1: densest subgraph,2:relinearize at each lambda)')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    num_outer = args.num_outer
    mu = args.mu
    alpha = args.alpha
    lam_max = args.lam_max
    num_lam = args.num_lam
    T = args.num_passes
    linearization_method = args.linearization_method
    initial_graph = args.initial_graph
    G, protected_nodes = init_graph(dataset_name)

    L = len(protected_nodes)
    if alpha is None:
        alpha = default_alpha(mu, L)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f'Dataset: {dataset_name}')
    print(f'Nodes: {n},  Edges: {m},  Groups: {L}')
    print(f'Number of protected nodes: {[len(group) for group in protected_nodes]}')
    print(f'mu={mu},  alpha={alpha:.6f},  lam_max={lam_max},  num_lam={num_lam}')
    print(f'SuperGreedy++ passes={T},  SSP outer iters={num_outer}')

    lam_vec = np.linspace(0,(lam_max), num_lam)


    density_vec = []
    num_of_nodes = []
    num_of_protected_vec = []
    protected_portion_in_sub_vec = []
    protected_portion_in_prot_vec = []
    neg_lse_values = []
    fairness_vec = []
    cost = []
    r_vec = []
    nonfairddensityvec = []
    group_sizes_vec = []

    print('Running DC-Composed LSE SuperGreedy++...')
    
    if initial_graph == 1 or initial_graph == 2:
        result = super_greedy_pp_lse_DC_SSP(
                G, G, protected_nodes, lam=0.0,
                mu=mu, alpha=alpha,
                num_passes=T, num_outer=num_outer
                ,lin_method=linearization_method
            )

        S0 = G.subgraph(result[0])
    elif initial_graph == 0:
        S0 = G.copy()
    else:
        raise ValueError(f"Unknown initial graph option: {initial_graph}")
    
    for Lam in tqdm(lam_vec):
        result = super_greedy_pp_lse_DC_SSP(
            G, S0, protected_nodes, lam=Lam,
            mu=mu, alpha=alpha,
            num_passes=T, num_outer=num_outer,
            lin_method=linearization_method
        )
        result_G = G.subgraph(result[0])

        # Output becomes new linearization point for next lambda
        if initial_graph == 2:
            S0 = G.subgraph(result[0])
        

        # ---- Record metrics ----
        nonfairddensity = utils.compute_density(result_G)
        nonfairddensityvec.append(nonfairddensity)

        super_greedy_pp_R_nodes = sorted(result_G.nodes())

        neg_lse_val = result[3]   # true (-r) value at best subgraph
        cost_value = result[1]    # true density: [2e + lam*(-r)] / |S|

        num_of_nodes.append(len(super_greedy_pp_R_nodes))
        neg_lse_values.append(neg_lse_val)
        density_vec.append(nonfairddensity)
        cost.append(cost_value)

        fairness = sum([
            utils.find_num_common(super_greedy_pp_R_nodes, protected_nodes[i])
            for i in range(len(protected_nodes))
        ])
        fairness_vec.append(fairness)

        induced_protected_nodes = result[2]


        gsizes = [len(induced_protected_nodes[i]) for i in range(L)]
        group_sizes_vec.append(gsizes)

        r = utils.compute_r(result_G, protected_nodes, mu=mu)
        r_vec.append(r)

        num_of_protected = min(gsizes)
        num_of_protected_vec.append(num_of_protected)

        protected_portion_in_sub = (num_of_protected / len(super_greedy_pp_R_nodes)
                                    if len(super_greedy_pp_R_nodes) > 0 else 0.0)
        protected_portion_in_sub_vec.append(protected_portion_in_sub)

        total_prot = sum(len(protected_nodes[i]) for i in range(len(protected_nodes)))
        protected_portion_in_prot = (num_of_protected / total_prot
                                     if total_prot > 0 else 0.0)
        protected_portion_in_prot_vec.append(protected_portion_in_prot)

    print('Done.')

    variables_dict = {
        'lam': lam_vec,
        'induced': result,
        'num_of_nodes': num_of_nodes,
        'density': density_vec,
        'num_of_protected': num_of_protected_vec,
        'protected_portion_in_sub': protected_portion_in_sub_vec,
        'protected_portion_in_prot': protected_portion_in_prot_vec,
        'fairness': fairness_vec,
        'r': r_vec,
        'neg_lse_values': neg_lse_values,
        'cost_value': cost,
        'nonfairddensity': nonfairddensityvec,
        'group_sizes': group_sizes_vec,
        'mu': mu,
        'alpha': alpha,
    }

    save_folder = 'logs/' + dataset_name
    save_path = save_folder + '/' + dataset_name + '_' +  'log.npy'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)
    print(f'Saved to {save_path}')


if __name__ == '__main__':
    main()