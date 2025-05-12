"""
HyperGraph

The class HyperGraph is a representation of
hypergraphs as edge-sequences that allows to 
access relevant statistics of the hypergraph.

The child class HyperGraphAsymmetricSI inherits
all properties of HyperGraph and defines a
simulation environment for a non-linear version
of the Susceptible-Infected (SI) model.


2024/06/05 --- ML, SD, JE
"""


from collections import defaultdict
import copy
from itertools import combinations, compress
import numpy as np
from numpy.typing import ArrayLike
import scipy.special as special
import scipy.stats as stats
from typing import List, Dict, Set, Union, Tuple, Callable
import warnings

from SamplableSet import *
from HyperGraphHelper import *

def nested_defaultdict():
    return defaultdict(int)

class HyperGraph:

    def __init__(self, nodes:List, edges:List, group:List=None, fast:bool=False):
        """Create a HyperGraph object that stores the edgelist
           as well as main summary statistics.
           
           Input
           nodes - list of nodes (integer indices)
           edges - list of edges represented as sets or tuples.
           group - list of node memberships

           Output
           h     - hypergraph object
        """

        # turn edges to tuples
        if fast:
            sorted_edges = edges
        else:
            sorted_edges = [tuple(sorted(e)) for e in edges]
        

        self.edges = sorted_edges                   # edges in the hypergraph
        self.rank = len(max(self.edges, key=len))   # maximum edge size
        self.M = len(self.edges)                    # total number of edges

        self.nodes = nodes                          # nodes of the hypergraph
        self.N = len(nodes)                         # total number of nodes

        if np.any(group==None):
            self.group = self.N * [1]
        else:
            self.group = group

        n1 = sum(self.group)
        self.n = [self.N - n1, n1]

        # number of edges of a given size
        self.m = defaultdict(nested_defaultdict)

        # degree of a node
        self.degree = dict(zip(range(0,self.N), self.N*[0]))

        # size of each edge
        self.edge_size = dict(zip(self.edges, self.M*[0]))
        self.edge_type = dict(zip(self.edges, self.M*[0]))

        # incidence matrix
        self.incidence = {i : copy.copy([]) for i in self.nodes}

        # k-core (graph projection) and k-m-core dictionaries
        self.k_m_core_r = {}        # size-independent hypercoreness
        self.k_m_core_rw = {}       # frequency-based hypercoreness
        self.k_core = {}

        # update edge sizes and node degrees
        for e in self.edges:

            k = len(e)

            t = 0
            for i in e:
                i = int(i)
                
                self.degree[i] += 1
                self.incidence[i].append(e)

                if self.group[i]!=None:
                    t += self.group[i]
            
            self.m[k][t] += 1
            self.edge_size[e] = k
            self.edge_type[e] = t
    
    def clique_projection(self):
        """Calculate the clique projection of the hypergraph.
        
        Output
        h_clique - the clique projection of the hypergraph        
        """

        new_edges = set({})

        # for each edge create every pairwise edge among its members
        for e in self.edges:
            
            # note that combinations returns sorted tuples
            for (u,v) in combinations(e, 2):
                new_edges.add((u,v))

        h_clique = HyperGraph(nodes=self.nodes, edges=new_edges, group=self.group)

        return h_clique
    
    def set_k_m_core(self, verbose:bool=False)->Dict[int,Dict[int,int]]:
        """Determine the hypercoreness [Mancastroppa et al. (2023)] of each node by systematically finding each of the (k, m)-cores, i.e.,
           the shells of nodes that have at least degree k in the subgraph of hyperedges of size at least m, upon
           removal of nodes with degree less than k. 

           Output
           km_coreness - dictionary keyed by edge size and with values being dictionaries of node indices indicating the maximum k s.t.
                         i is in the km-core.
        """

        # create a copy of the hypergrpah in which nodes can be deleted
        h = copy.deepcopy(self)

        # stores the km-coreness as a dictionary of dictionaries of form {m : {i: maximum k s.t. i is in the km-core}}
        km_coreness = {}
        
        # iterate over edge sizes
        for mstar in range(2, h.rank + 1):

            km_coreness[mstar] = {i:0 for i in h.nodes}

            # iterate over degree values
            for kstar in range(1, max(h.degree.values()) + 1):

                core_nodes = set(h.nodes)
                core_edges = set(h.edges)
                core_degree = copy.copy(h.degree)
                core_incidence = {i:set(ei) for i,ei in h.incidence.items()}

                new_nodes = copy.copy(core_nodes)
                new_edges = copy.copy(core_edges)

                # remove edges below edge size mstar and adjust the node degrees accordingly
                for e in core_edges:

                    if len(e) < mstar:

                        new_edges.remove(e)

                        for v in e:

                            core_degree[v] -= 1
                            core_incidence[v].remove(e)

                # continue removing nodes and edges until the core is stable
                stable = False
                iteration = 0
                while not stable:

                    for v in core_nodes:
                        
                        # remove nodes that do not satisfy the degree condition
                        if core_degree[v] < kstar:

                            new_nodes.remove(v)

                            # if a node is removed all edges it belongs to need to be changed
                            # if an edge is to small after removal of the node v it is removed
                            # and the degree of all it's member nodes is adjusted.
                            for e in core_incidence[v]:
                                
                                new_edges.remove(e)
                                enew = tuple(sorted([u for u in e if u!=v]))

                                for u in enew:
                                    core_degree[u] -= 1
                                    core_incidence[u].remove(e)

                                if enew not in new_edges and len(enew) >= mstar:
                                    
                                    new_edges.add(enew)

                                    for u in enew:

                                            core_incidence[u].add(enew)
                                            core_degree[u] += 1
                                
                    
                    iteration += 1
                    if verbose: print(iteration, end='\r')
                    
                    # check whether the core is still changing
                    if new_nodes == core_nodes:

                        stable = True
                        for v in core_nodes:

                            # update if nodes are still in the km-core
                            km_coreness[mstar][v] = kstar
                    
                    core_nodes = copy.copy(new_nodes)
                    core_edges = copy.copy(new_edges)
                
                # stop if no more nodes are left
                if len(new_nodes) == 0:

                    if verbose: print(f"m:{mstar}, k:{kstar - 1}", end='\n')
                    
                    break
        
        # store the weighted and unweighted hypercoreness

        self.k_m_core_r = {v:0. for v in self.nodes}
        self.k_m_core_rw = {v:0. for v in self.nodes}


        for s in range(2, self.rank +1):

            kmax = np.max(list(km_coreness[s].values()))
            psi = np.sum(list(self.m[s].values()))/self.M

            for v in self.nodes:
                self.k_m_core_r[v] += km_coreness[s][v]/kmax
                self.k_m_core_rw[v] += psi*km_coreness[s][v]/kmax
        
        return km_coreness

    def set_k_core(self, verbose:bool=False):
        """Sets the k-core of the graph projection of the hypergraph.

        Output
        k_coreness - dictionary of node indices and their k-coreness
        """

        # determine the clique projection of the hypergraph
        h_clique = self.clique_projection()

        # determine the k-core of the clique projection as the k2-hypercore
        km_coreness = h_clique.set_k_m_core(verbose=verbose)

        self.k_core = {v : km_coreness[2][v] for v in self.nodes}

        return self.k_core

    def power_inequality(self) -> float:
        """Calculates the power inequality (generalized from Avin et al., 2015), i.e. the ratio of first
           moments of the degree distribution, of the hypergraph H. The function assumes that group 0
           is the majority group.

           Output
           power inequality
        """

        # get the group membership of all nodes
        group = np.asarray([self.group[int(v)] for v in self.nodes])

        # get the node degrees
        degree = np.asarray([self.degree[int(v)] for v in self.nodes])

        return np.mean(degree[group == 1]) / np.mean(degree[group == 0])

    def moment_glass_ceiling(self) -> float:
        """Calculate the moment glass ceiling (generalized from Avin et al. (2015)) of a hypergraph, i.e. the
           ratio of second moments of the degree distribution. The function assumes that group 0 is the majority
           group.

        Output
        moment glass ceiling

        """

        # get the group membership of all nodes
        group = np.array([self.group[int(v)] for v in self.nodes])

        # get the node degrees
        degree = np.array([self.degree[int(v)] for v in self.nodes])

        return np.mean(np.power(degree[group == 1], 2)) / np.mean(np.power(degree[group == 0], 2))

    
    def homophily(self, alternative:bool=False, exact_baseline:bool=False) -> Dict:
        """Calculate affinity and baseline scores of the homophily measure proposed
            by Veldt, Kleinberg, & Benson (2023).

            Input
            alternative    - whether to use the main or alternative score
            exact_baseline - whether to use the exact baseline score or the numerically more stable
                            asymptotic (N>>1) baseline score

            Output
            h - homophily score, i.e., affinity score normalized to baseline
        """

        h = {0 : defaultdict(lambda: defaultdict(float)),     # majority homophily score
             1 : defaultdict(lambda: defaultdict(float))}     # minority homophily score
        
        # calculate the fraction of minority nodes
        nu = self.n[1] / self.N

        # iterate over the edge sizes that exist in the hypergraph
        for k in self.m.keys():

            # use the alterantive homophily score
            if alternative:

                # calculate the total number of edges of size k
                mk0 = np.sum([self.m[k][t] for t in self.m[k].keys() if k - t > 0])
                mk1 = np.sum([self.m[k][t] for t in self.m[k].keys() if t > 0])

                # iterate over the existing edge types
                for t in self.m[k].keys():
                    
                    # calculate the affinity scores (not normalized to baseline)
                    if k - t > 0 : h[0][k][k - t] = self.m[k][t] / mk0
                    if t > 0 : h[1][k][t] = self.m[k][t] / mk1

                    # calculate the baseline scores
                    if exact_baseline:
                        b0 = special.binom(self.n[0], k - t) * special.binom(self.n[1], t) / (special.binom(self.N, k) - special.binom(self.n[1], k))
                        b1 = special.binom(self.n[1], t) * special.binom(self.n[0], k - t) / (special.binom(self.N, k) - special.binom(self.n[0], k))

                    else:
                        b0 = (1.0 - nu) ** (k - t) * nu ** t * special.binom(k, k - t) * (1.0 / (1.0 - nu ** k))
                        b1 = nu ** t * (1.0 - nu) ** (k - t) * special.binom(k, t) * (1.0 / (1.0 - (1.0 - nu) ** k))
                    
                    # calculate the homophily score by normalizing the affinity score to baseline
                    if k - t > 0:
                        # print(h[0][k][k - t])
                        # print(b0)
                        h[0][k][k - t] /= b0
                        # print(h[0][k][k - t])
                        # print()
                    if t > 0:
                        # print(h[1][k][t])
                        # print(b1)
                        h[1][k][t] /= b1
                        # print(h[1][k][t])
                        # print()

            
            # use the standard homophily score
            else:
                
                # calcuate the contribution of size k edges to the sum of degrees of nodes in either group
                mk0 = np.sum([(k - t) * self.m[k][t] for t in self.m[k].keys() if k - t > 0])
                mk1 = np.sum([t * self.m[k][t] for t in self.m[k].keys() if t > 0])

                # iterate over edge types
                for t in self.m[k].keys():

                    # calculate the affinity scores (not compared to baseline)
                    if k - t > 0: h[0][k][k - t] = (k - t) * self.m[k][t] / mk0
                    if t > 0: h[1][k][t] = t * self.m[k][t] / mk1

                    # calculate the baseline scores
                    if exact_baseline:
                        b0 = special.binom(self.n[0] - 1, k - t - 1) * special.binom(self.n[1], t) / special.binom(self.N - 1, k - 1)
                        b1 = special.binom(self.n[1] - 1, t - 1) * special.binom(self.n[0], k - t) / special.binom(self.N - 1, k - 1)
                    else:
                        b0 = (1.0 - nu) ** (k - t - 1) * nu ** t * special.binom(k - 1, k - t - 1)
                        b1 = nu ** (t - 1) * (1.0 - nu) ** (k - t) * special.binom(k - 1, t - 1)
                    
                    # calculate the homophily score by normalizing the affinity score to baseline
                    if k - t > 0: h[0][k][k - t] /= b0
                    if t > 0: h[1][k][t] /= b1
    
        return h
    
def CSCM(n:ArrayLike, m:Dict[Tuple[int],int], degree_sampler:Callable[...,List], params:Tuple=(), erased:bool=True, *, seed:int)->Tuple[List,List,List]:
    """Generate a hypergraph from the colored soft configuration model or its erased version.
    
    Input
    n - array in which entry i is the number of nodes in group i
    edge_counts - dictionary of edge counts of a given type.
    degree_sampler - function to sample the average degrees. Needs to return list.
    params - other parameters of the degree distribution
    erased - whether to erase multi-edges
    seed - seed for random number generator

    Output
    edges - the edge list of the hypergraph
    group - the group membership of nodes 
    kappa - the hidden degrees of all nodes
    """

    assert len(n)==2, "Only two groups are supported."

    # calculate the total number of nodes
    N = np.sum(n)

    nodes = {0: list(range(0,n[0])),
             1: list(range(n[0], N))}
    
    group = n[0]*[0] + n[1]*[1]

    # calculate the total number of edges
    mtot = np.sum([sum(list(m[k].values())) for k in m.keys()])

    # calculate the average degree
    kbar = {0: 0, 1: 0}
    for k, mk in m.items():
        kbar[0] += np.sum([(k-t)*mt for t, mt in mk.items()])/n[0]
        kbar[1] += np.sum([t*mt for t, mt in mk.items()])/n[1]

    # sample the expected degree of each node.
    kappa = {g: degree_sampler(n[g], kbar[g], *params) for g in [0,1]}

    sub_nodes = {g : {i : kappa_i for i, kappa_i in zip(nodes[g], kappa[g]) if kappa_i > 0} for g in [0, 1]}

    # build a SamplableSet for each node group
    SamplableSet.seed(seed)
    s = {g: SamplableSet(min(sub_nodes[g].values()),
                         max(sub_nodes[g].values()),
                         sub_nodes[g]) for g in [0,1]}

    # for each edge size and edge type sample its composition
    # as SamplableSets deletes elements upon sammpling without 
    # replacement we sample with replacement and check whether
    # the node is already in the edge.
    edges = []

    for k in m.keys():
        
        for t, mt in m[k].items():

            for _ in range(mt):

                e = set({})
                size_e = 0
                
                if t>0:
                    
                    # select group 1 nodes
                    while size_e < t:
                        i, _ = s[1].sample(n_samples=1, replace=True)

                        if i not in e:
                            e.add(i)
                            size_e += 1
                
                if t<k:

                    # select group 0 nodes
                    while size_e < k:
                        i, _ = s[0].sample(n_samples=1, replace=True)

                        if i not in e:
                            e.add(i)
                            size_e += 1

                e = tuple(sorted(list(e)))
                edges.append(e)

    
    # remove multi-edges
    if erased:
        edges = list(set(edges))
    
    return edges, group, kappa


def select_seeds(heuristic, H, num, p_m=None):
    """Selects a given number of seeds from the hypergraph based on a provided heuristic.

    Input
    heuristic - a string determining the type of seeding strategy (options: {'degree', 'hyper_core', 'k_core', 'random'})
    H - a hypergraph object
    num - the number of desired seeds
    p_m - the proportion of seeds from group 1

    Output
    seed_range - a numpy array listing the indices of the seeds
    """

    # determine number of seeds from each group
    if p_m is not None:
        num_m = int(num * p_m)
        num_M = num - num_m

    if heuristic == 'degree':
        # based on largest degree
        sorted_nodes = np.argsort(np.array([H.degree[i] for i in H.nodes]))
        sorted_nodes_desc = sorted_nodes[::-1]

        if p_m:
            group = np.asarray([H.group[v] for v in H.nodes])
            sorted_group = group[sorted_nodes]
            group_desc = sorted_group[::-1]

            seeds_M = sorted_nodes_desc[group_desc == 0][:num_M]
            seeds_m = sorted_nodes_desc[group_desc == 1][:num_m]
            seed_range = np.concatenate([seeds_M, seeds_m])
        else:
            seed_range = sorted_nodes_desc[:num]

    elif heuristic == 'hyper_core':
        # based on hypercoreness; r for size-independent, r_w takes size into account

        if len(H.k_m_core_r) == 0:
            H.set_k_m_core()
        r = H.k_m_core_r
        r_w = H.k_m_core_rw

        if p_m:
            group = np.asarray([H.group[v] for v in H.nodes])
            r_M, r_w_M, r_m, r_w_m = {}, {}, {}, {}
            for key in r.keys():
                if group[key] == 0:
                    r_M[key] = r[key]
                    r_w_M[key] = r_w[key]
                else:
                    r_m[key] = r[key]
                    r_w_m[key] = r_w[key]

            # size-independent hypercoreness
            seeds_r_M = sorted(r_M, key=r_M.get, reverse=True)[:num_M]
            seeds_r_m = sorted(r_m, key=r_m.get, reverse=True)[:num_m]
            seed_range_r = np.concatenate([seeds_r_M, seeds_r_m])
            # frequency-based hypercoreness
            seeds_r_w_M = sorted(r_w_M, key=r_w_M.get, reverse=True)[:num_M]
            seeds_r_w_m = sorted(r_w_m, key=r_w_m.get, reverse=True)[:num_m]
            seed_range_r_w = np.concatenate([seeds_r_w_M, seeds_r_w_m])

        else:
            seed_range_r = sorted(r, key=r.get, reverse=True)[:num]
            seed_range_r_w = sorted(r_w, key=r_w.get, reverse=True)[:num]

        seed_range_r = np.array(seed_range_r)
        seed_range_r = seed_range_r.astype(int)
        # seed_range_r_w = seed_range_r_w.astype(int)
        # seed_range = [seed_range_r, seed_range_r_w]
        seed_range = seed_range_r

    elif heuristic == 'k_core':
        # based on k core connectedness of projected graph
        if len(H.k_core) == 0:
            H.set_k_core()
        k = H.k_core

        if p_m:
            group = np.asarray([H.group[v] for v in H.nodes])
            M, m = {}, {}
            for key in k.keys():
                if group[key] == 0:
                    M[key] = k[key]
                else:
                    m[key] = k[key]

            seeds_M = sorted(M, key=M.get, reverse=True)[:num_M]
            seeds_m = sorted(m, key=m.get, reverse=True)[:num_m]

            seed_range = np.concatenate([seeds_M, seeds_m])

        else:
            seed_range = sorted(k, key=k.get, reverse=True)[:num]

        seed_range = np.array(seed_range)
        seed_range = seed_range.astype(int)

    else:
        # select randomly
        group = np.asarray([H.group[v] for v in H.nodes])
        M = H.nodes[group == 0]
        m = H.nodes[group == 1]

        seed_range = np.zeros(H.N)
        unique, counts = np.unique(seed_range, return_counts=True)

        while len(unique) == 1 or counts[1] != num:
            # note: this implementation assumes that the nodes are ordered by group (e.g., group = [0 0 0 ... 1 1 1]
            if p_m:
                seeds_M = stats.bernoulli.rvs(num_M / len(M), size=len(M))
                seeds_m = stats.bernoulli.rvs(num_m / len(m), size=len(m))
                seed_range = np.concatenate([seeds_M, seeds_m])

            else:
                seed_range = stats.bernoulli.rvs(num / H.N, size=H.N)

            unique, counts = np.unique(seed_range, return_counts=True)

        seed_range = np.argwhere(seed_range == 1).flatten()

    return seed_range

class HyperGraphAsymmetricSI(HyperGraph):

    def __init__(self, nodes:List[int], edges:List[Set[int]], group:List[int], lams:float, nus:float, fast:bool=False, samplable_set:bool=True, seed:int=42):
        """Create a simulation object that allows to simulate the asymmetric Susceptible-
        Infected (SI) model on hypergraphs using Gillespie's direct method.
        The asymmetric hypergraph SI model uses different infection rates and social reinforcement parameters,
        depending on whether a node is infected by an ingroup or outgroup node.

        Input
        nodes - integer indices of hypergraph nodes.
        edges - list of edges represented as sets of nodes.
        group - list of group memberships of the nodes s.t. entry i is the group of node i
        lams  - matrix of rate of infection parameters
        nus   - matrix of social reinforcement parameters
        samplable_set - whether to use the SamplableSet data structure to store the transition rates
        seed  - seed for random number generators.

        Out
        hSI   - HyperGraphSI simulation object
        """

        super().__init__(nodes, edges, group, fast=fast)

        self.node_state = np.zeros(self.N)                                  # 1: infectious 0: susceptible
        self.lam = lams                                                     # rate of infection parameters
        self.nu = nus                                                       # social reinforcement parameters
        self.samplable_set = samplable_set                                  # whether to use a SamplableSet to store the transition rates

        self.rng = np.random.default_rng(seed=seed)                         # properly seeded random number generator

        self.edge_to_idx = {e : idx for idx, e in enumerate(self.edges)}    # to access the edge state fast
        self.idx_to_edge = {idx : e for e, idx in self.edge_to_idx.items()} # to access the edge members fast

        if self.samplable_set:
            self.edge_state = {0:np.zeros(self.M),  1:np.zeros(self.M)}      # number of infectious individuals of each type in each edge
        else:
            self.edge_state = None
    
    def set_initial_condition(self, seeds:ArrayLike):
        """Resets the node and edge states s.t. only the seed nodes are infected.
        
        Input
        seeds - the indices of the seed nodes.
        """

        # set the node state
        self.node_state = np.zeros(self.N)    # 1: infectious 0: susceptible
        self.node_state[seeds] = 1            # set the seed nodes to infected

        # update the edge state to maintain a consistent instance.
        if not self.samplable_set:
            self.edge_state = {0:np.zeros(self.M), 1:np.zeros(self.M)}
            for e in self.edges:
                for j in [0, 1]:
                    self.edge_state[j][self.edge_to_idx[e]] += np.sum(self.node_state[[v for v in e if self.group[v]==j]])
    
    def reset_params(self, lams:float=None, nus:float=None):
        """Set the parameter of the contagion process.
        
        Input
        lams - matrix of infection rate parameters
        nus  - matrix of non-linearity parameters
        """

        if self.lam != None:
            self.lam = lams
        if self.nu != None:
            self.nu = nus
    
    def simulate(self, maxiter:int=int(1e6), verbose:bool=False) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Runs a simulation until the absorbing state is reached or the maximum number of iterations is exceeded.
           
        Input
        maxiter - maximum number of iterations
        verbose - whether to print the elapsed time and iterations

        Output
        state_seq - entry i is an array holding the state of the system after the ith iteration
        time_seq  - entry i is the time elapsed until the ith iteration
        node_seq  - entry i is the node infected in the ith iteration (excluding the seed nodes)
        group_seq - entry i is the node infected in the ith iteration (excluding the seed nodes)
        type_seq  - entry i is the type of infection event that happened in the ith iteration
        """

        # maximum number of iterations before breaking the simulation
        maxiter = int(maxiter)

        # store the initial conditions
        state_seq = [np.sum(self.node_state)]
        time_seq = [0]
        node_seq = [None]
        group_seq = [None]
        event_type_seq = [None]
        edge_type_seq = [None]
        edge_size_seq = [None]

        # simulation using the SamplableSet data structure
        if self.samplable_set:

            # degine the types of transitions that are possible (target, source)
            event_types = [(0,0), (0,1), (1,0), (1,1)]

            # determine the maximum and minimum rate
            max_rate = np.max(self.lam) * np.max([(s - i) * np.power(i, nu) for nu in np.ravel(self.nu) for s in range(2, self.rank+1) for i in range(1, s)])
            min_rate = np.min(self.lam) * np.min([(s - i) * np.power(i, nu) for nu in np.ravel(self.nu) for s in range(2, self.rank+1) for i in range(1, s)])

            # initialize a samplable set that stores the transition rates for all possible events
            init_dict = {}
            for e in self.edges:
                
                # determine the number of nodes of each type in the edge
                e_numnodes = (self.edge_size[e] - self.edge_type[e], self.edge_type[e])
                e_infected = tuple(np.sum([self.node_state[v] for v in e if self.group[v]==g]) for g in [0, 1])
                
                for event_type, target_source in enumerate(event_types):

                    target, source = target_source 
                    rate = self.lam[target, source] * (e_numnodes[target] - e_infected[target]) * np.power(e_infected[source], self.nu[target, source])
                    
                    if rate >= min_rate:
                        init_dict[(event_type, self.edge_to_idx[e])] = rate

            # initialize the samplable set
            beta = SamplableSet(min_rate, max_rate, init_dict)

            # run the simulations
            t = 0.0
            for iteration in range(1, maxiter):

                if verbose: print(iteration)

                # Determine the global transition rate, i.e. the rate at which the outcome "anything happens" occurs
                beta_tot = beta.total_weight()

                # Terminate the simulation if no further transitions can take place
                if len(beta) == 0:
                    break

                # Use inverse cdf sampling to determine the time to the next event
                u = self.rng.random()
                dt = - np.log(1.0 - u)/beta_tot
                t += dt
                time_seq.append(t)

                # sample a specific edge and event type with probability proportional to their contribution to the rate
                sample, _ = beta.sample(n_samples=1, replace=True)
                event_type, ie_idx = sample

                target, source = event_types[event_type]
                ie = self.idx_to_edge[ie_idx]

                # store information about the infection event regarding the edge
                event_type_seq.append(event_type)
                edge_size_seq.append(self.edge_size[ie])
                edge_type_seq.append(self.edge_type[ie])
                
                # sample the node that will be infected
                iv = self.rng.choice([v for v in ie if self.node_state[v]==0 and self.group[v]==target])

                # store information about the infection event regarding the node
                node_seq.append(iv)
                group_seq.append(target)

                # update node state and store the node state at time t
                self.node_state[iv] = 1
                state_seq.append(np.sum(self.node_state))

                # update the rates for affected edges
                for e in self.incidence[iv]:

                    # determine the number of nodes of each type in the edge
                    e_numnodes = (self.edge_size[e] - self.edge_type[e], self.edge_type[e])
                    e_infected = tuple(np.sum([self.node_state[v] for v in e if self.group[v]==g]) for g in [0, 1])
                    e_idx = self.edge_to_idx[e]
                    
                    # update the rates of the affected edges
                    for event_type, target_source in enumerate(event_types):

                        target, source = target_source
                        rate = self.lam[target, source] * (e_numnodes[target] - e_infected[target]) * np.power(e_infected[source], self.nu[target, source])

                        if rate >= min_rate:
                            beta[(event_type, e_idx)] = rate
                        else:
                            beta.erase((event_type, e_idx))
        
        # simulations without using the samplable data structure
        else:

            tolerance = 1e-15 # values below this number are regarded as 0 from the perspective of the stopping criterion.

            # to enable access to multiple edge-sizes and edge-compositions simultaneously
            edge_sizes = {0: np.array([len([v for v in e if self.group[v]==0]) for e in self.edges]),
                        1: np.array([len([v for v in e if self.group[v]==1]) for e in self.edges])}
            
            # infection rates by type
            Beta_bb = self.lam[0,0] * (edge_sizes[0] - self.edge_state[0]) * np.power(self.edge_state[0], self.nu[0,0])  # group 0 node infects group 0 node
            Beta_rb = self.lam[0,1] * (edge_sizes[0] - self.edge_state[0]) * np.power(self.edge_state[1], self.nu[0,1])  # group 1 node infects group 0 node
            Beta_br = self.lam[1,0] * (edge_sizes[1] - self.edge_state[1]) * np.power(self.edge_state[0], self.nu[1,0])  # group 0 node infects group 1 node
            Beta_rr = self.lam[1,1] * (edge_sizes[1] - self.edge_state[1]) * np.power(self.edge_state[1], self.nu[1,1])  # group 1 node infects group 1 node

            t = 0
            for iteration in range(1, maxiter):

                if verbose: print(iteration)

                # Step 1: Draw two numbers uniformly at random from the unit interval
                u1, u2 = self.rng.random(size=2)

                # Step 2: Determine the global transition rate, i.e. the rate at which the outcome
                # "anything has happened at all" takes place. There are four possible events that can
                # happen corresponding to the four infection channels: 0 -> 0, 0 -> 1, 1 -> 0, 1 -> 1.
                if t>0:
                    edge_state_0 = self.edge_state[0][affected_edges]
                    edge_state_1 = self.edge_state[1][affected_edges]

                    edge_sizes_0 = edge_sizes[0][affected_edges] 
                    edge_sizes_1 = edge_sizes[1][affected_edges]

                    Beta_bb[affected_edges] = self.lam[0,0] * (edge_sizes_0 - edge_state_0) * np.power(edge_state_0, self.nu[0,0])  # group 0 node infects group 0 node
                    Beta_rb[affected_edges] = self.lam[0,1] * (edge_sizes_0 - edge_state_0) * np.power(edge_state_1, self.nu[0,1])  # group 1 node infects group 0 node
                    Beta_br[affected_edges] = self.lam[1,0] * (edge_sizes_1 - edge_state_1) * np.power(edge_state_0, self.nu[1,0])  # group 0 node infects group 1 node
                    Beta_rr[affected_edges] = self.lam[1,1] * (edge_sizes_1 - edge_state_1) * np.power(edge_state_1, self.nu[1,1])  # group 1 node infects group 1 node

                Beta = np.cumsum(np.concatenate([Beta_bb, Beta_rb, Beta_br, Beta_rr]))

                # Step 3: Terminate if no futher actions can happen, i.e., in any edge either all nodes are infected
                # or no node is infected. This will result in all betas being zero.
                if Beta[-1] <= tolerance:
                    break

                # Step 4: Sample the time at which the next event happens. This is done via inverse 
                # cdf sampling of the exponential distribution. Once the ellapsed time is determined
                # update the timer and store. 
                dt = - np.log(1.0 - u1)/Beta[-1]
                t += dt
                time_seq.append(t)

                # Step 5: Determine what type of event has happened. Each possible event contributes linearly
                # to the overall event rate. We use (somewhat slow) linear search to sample an event with probability
                # proportional to the amount it contributes to the overall event rate.
                # As events no longer map one to one to edges we need to determine the edge and the infection channel within
                # the edge. Then an appropriate node is selected uniformly at random among the elligible nodes.
                event_idx = np.min(np.where(Beta>=Beta[-1]*u2)[0])  # which event has happened
                event_type = event_idx // self.M                    # which infection channel does this event belong to
                ie_idx = event_idx % self.M                         # index of edge is affected by the event
                ie = self.idx_to_edge[ie_idx]                       # edge affected by the event

                event_type_seq.append(event_type)
                edge_size_seq.append(self.edge_size[ie])
                edge_type_seq.append(np.sum([self.group[v] for v in ie]))

                # events that infect a group 0 node
                if event_type == 0 or event_type == 1:
                    ig = 0                              
                # events that infect a group 1 node
                elif event_type == 2 or event_type == 3:
                    ig = 1
                # you messed up
                else:
                    raise ValueError('Invalid event type')

                # choose a random susceptible node from the right group within the event triggering edge to infect.
                iv = self.rng.choice([v for v in self.edges[ie_idx] if self.node_state[v]==0 and self.group[v]==ig])
                node_seq.append(iv)
                group_seq.append(self.group[iv])

                # Step 6: A transition of a node from susceptible to infected modifies the infection rates of all
                # edges the node belongs to. To be able to compute them correctly in Step 2 we need to update both
                # node and the group-specific edge state.
                self.node_state[iv] = 1
                affected_edges = [self.edge_to_idx[e] for e in self.incidence[iv]]
                self.edge_state[ig][affected_edges] += 1

                # save the state for output
                state_seq.append(np.sum(self.node_state))
        
        return np.asarray(state_seq), np.asarray(time_seq), np.asarray(node_seq), np.asarray(group_seq), np.asarray(event_type_seq), np.asarray(edge_size_seq), np.asarray(edge_type_seq)