"""
Code to compute various graph kernels. This is just used to test our variants by plugging in the kernels. 
For more optimal workflow, import kernels from https://github.com/ysig/GraKeL which is much faster. 
"""
from functools import partial
import warnings
import numpy as np
import scipy.sparse.linalg as linalg
from scipy import sparse
import networkx as nx
import h5py
from numpy.linalg import inv
from scipy.spatial.distance import pdist, squareform
import logging

# Create and configure logger
# Creating an object
# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def zscore(proj, samples=None, mean=None, std=None):
    """
    :param proj:
    :param samples:
    :param mean: mean of samples, ignored if samples argument is provided
    :param std: std of samples, ignored if samples argument is provided
    :return zscore of samples:
    """
    if not (samples is not None or (mean is not None and std is not None)):
        raise ValueError(
            "Either samples argument or (mean,std) arguments pair has to be provided to zscore function"
        )
    if samples is not None:
        return (proj - samples.mean(axis=1)) / samples.std(axis=1)
    else:
        return (proj - mean) / std


def onetail(proj, samples):
    """
    :param proj:
    :param samples:
    :return pvalue: fraction of times the node projection is greater than random samples
    i.e. a one-tailed test.
    """
    return (proj[:, None] > samples).sum(axis=1) / samples.shape[0]


def istvan(proj, samples=None, mean=None, std=None):
    """
    :param proj:
    :param samples:
    :param mean: mean of samples, ignored if samples argument is provided
    :param std: std of samples, ignored if samples argument is provided
    :return istvan:
    """
    raise NotImplementedError


# Notice: The article references refer to the papers from which the kernel equations have been considered for
# implementation here, NOT the original paper where the kernel has been proposed.

# Random Walk (RW) kernel (Cowen et al., 2017)
def random_walk_kernel(A, nRw):
    Asp = sparse.csc_matrix(A / A.sum(axis=0))
    return np.asarray((Asp**nRw).todense())


# Random Walk with Restart (RWR) kernel (Cowen et al., 2017)
def random_walk_with_restart_kernel(A, alpha):
    if alpha != 0.0:
        Dinv = np.diag(1 / A.sum(axis=0))
        W = np.dot(A, Dinv)
        I = np.eye(A.shape[0])
        return alpha * np.linalg.inv((I - (1 - alpha) * W))
    else:
        return np.tile(A.sum(axis=1, keepdims=True) / A.sum(), [1, A.shape[0]])


# Diffusion State Distance (DSD) (Cowen et al., 2017)
# Note: this is a distance matrix, NOT a kernel (similarity)!
def diffusion_state_distance(adjacency, nRw):

    adjacency = np.asmatrix(adjacency)
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    p = adjacency / degree
    if nRw >= 0:
        c = np.eye(n)
        for i in range(nRw):
            c = np.dot(c, p) + np.eye(n)
        return squareform(pdist(c, metric="cityblock"))
    else:
        pi = degree / degree.sum()
        return squareform(pdist(inv(np.eye(n) - p - pi.T), metric="cityblock"))


# Heat kernel (HK) (Cowen et al., 2017)
def heat_kernel(A, t):
    D = np.diag(A.sum(axis=0))
    W = D - A
    Wtexp = sparse.csc_matrix(-1 * t * W)
    return linalg.expm(Wtexp)


# Interconnectedness (ICN) kernel (Hsu et al., 2011)
def interconnected_kernel(A):
    Dinv = np.sqrt(A.sum(axis=0))
    Asp = sparse.csc_matrix(A)
    return np.asarray((Asp**2 + Asp * 2).todense()) / (Dinv[:, None] * Dinv)


def istvan_kernel(A):
    raise NotImplementedError


def istvan_kernel2(A):
    raise NotImplementedError


class GIDMapper:
    """
    A simple class to keep track of various nodes in a graph by creating a dictionary of nodes,
    essentially giving an order to the nodes.
    """

    def __init__(self, nodelist):
        self.nodelist = nodelist
        self._gid2id_dict = {gid: i for i, gid in enumerate(nodelist)}
        self._dtype = type(nodelist[0])

    def id2gid(self, ids):
        if isinstance(ids, int):
            return self.nodelist[ids]
        else:  # if it is a list of ids
            return [int(self.nodelist[i]) for i in ids]

    def gid2id(self, gids):
        if isinstance(gids, self._dtype):
            return self._gid2id_dict[gids]
        else:  # if it is a list of gids
            return [self._gid2id_dict[gid] for gid in gids]


class GraphKernel:
    def __init__(
        self, graph=None, savefile=None, nodelist=None, weight="weight", verbose_level=1
    ):
        """
        Instantiates a GraphKernel object. Can be initialized by passing a networkx graph or from a savefile
        Parameters
        ----------
        graph : NetworkX graph or numpy ndarray
            Input graph as NetworkX graph or as numpy adjacency matrix (numpy array)
        savefile : str
            Path of savefile
        nodelist : list
            List of node IDs to be used for rows/columns ordering in the adjacency matrix. If None nodelist = graph.nodes().
            This parameter is best left as default
        weight : str
            Name of the NetworkX edge property to be considered as edge weight for weighted graphs
            If graph is provided as numpy adjacency matrix this parameter is ignored
        """
        if savefile is None:
            logging.info("Initializing GraphKernel...")
            if isinstance(graph, np.ndarray) or isinstance(
                graph, np.matrixlib.defmatrix.matrix
            ):
                self.adj = np.asarray(graph)
                if nodelist is None:
                    nodelist = range(self.adj.shape[0])
            else:  # if NetworkX graph
                self.adj = np.array(
                    nx.adjacency_matrix(
                        graph, nodelist=nodelist, weight=weight
                    ).todense()
                )
                if nodelist is None:
                    nodelist = graph.nodes()
            self.nodelist = nodelist
            self.gm = GIDMapper(nodelist=nodelist)
            self.kernels = {}
            logging.info("Complete.", newline=True)
        else:
            self.load(savefile)

    def eval_random_walk_kernel(self, nRw):
        """
        Simple Random Walk kernel.
        Parameters
        ----------
        nRw : int
            Number of steps of random walk
        Returns
        -------
        str
            Kernel ID (KID) to identify the corresponding kernel
        """
        kid = "rw_" + str(nRw)
        if kid not in self.kernels:
            logging.info(
                "Initializing RW kernel (this may take a while)...",
            )
            self.kernels[kid] = random_walk_kernel(self.adj, nRw)
            logging.info("Complete.")
        return kid

    def eval_random_walk_with_restart_kernel(self, alpha):
        """
        Random Walk with Restart kernel.
        Parameters
        ----------
        alpha : float
            Restart probability of random walk
        Returns
        -------
        str
            Kernel ID (KID) to identify the corresponding kernel
        """
        kid = "rwr_" + str(alpha)
        if kid not in self.kernels:
            logging.info("Initializing RWR kernel (this may take a while)...")
            self.kernels[kid] = random_walk_with_restart_kernel(self.adj, alpha)
            logging.info("Complete.")
        return kid

    def eval_diffusion_state_distance(self, nRw):
        """
        Diffusion State Distance kernel, as defined in Cao et al., 2013.
        Parameters
        ----------
        nRw : int
            Number of steps of random walk
        Returns
        -------
        str
            Kernel ID (KID) to identify the corresponding kernel
        """
        kid = "dsd_" + str(nRw)
        if kid not in self.kernels:
            logging.info("Initializing DSD kernel (this may take a while)...")
            self.kernels[kid] = diffusion_state_distance(self.adj, nRw)
            logging.info("Complete.")
        return kid

    def eval_heat_kernel(self, t):
        """
        Heat kernel, as defined in Cowen et al., 2017
        Parameters
        ----------
        t : float
            Diffusion time value
        Returns
        -------
        str
            Kernel ID (KID) to identify the corresponding kernel
        """
        kid = "hk_" + str(t)
        if kid not in self.kernels:
            logging.info("Initializing heat kernel (this may take a while)...")
            self.kernels[kid] = heat_kernel(self.adj, t)
            logging.info("Complete.")
        return kid

    def eval_istvan_kernel(self):
        kid = "ist"
        if kid not in self.kernels:
            logging.info("Initializing Istvan kernel (this may take a while)...")
            self.kernels[kid] = istvan_kernel(self.adj)
            logging.info("Complete.")
        return kid

    def eval_istvan2_kernel(self):
        kid = "ist2"
        if kid not in self.kernels:
            logging.info("Initializing Istvan kernel (this may take a while)...")
            self.kernels[kid] = istvan_kernel2(self.adj)
            logging.info("Complete.")
        return kid

    def eval_interconnected_kernel(self):
        kid = "icn"
        if kid not in self.kernels:
            logging.info("Initializing ICN kernel (this may take a while)...")
            self.kernels[kid] = interconnected_kernel(self.adj)
            logging.info("Complete.")
        return kid

    def eval_kernel_statistics(
        self,
        kernel,
        n_samples=None,
        rdmmode="CONFIGURATION_MODEL",
        n_edge_rewirings=None,
    ):
        """
        Pre-computes the approximate kernel statistics necessary to apply the CONFIGURATION_MODEL and EDGE_REWIRING corrections to the projection scores
        Parameters
        ----------
        kernel : str
            Kernel ID (KID) of the chosen kernel.
        n_samples : int
            Number of random samples to calculate.
        rdmmode : str
            Randomization mode for the network edges. To be invoked prior to calling get_projection function with correction mode 'CONFIGURATION_MODEL' or 'EDGE_REWIRING'.
            Options are:
                - 'CONFIGURATION_MODEL': generates a configuration model sample of the network
                - 'EDGE_REWIRING': generates each sample by swapping (n_edge_rewirings) times a pair of randomly selected edges. Connectivity of the final graph is enforced.
        n_edge_rewirings : int
            Number of times the pairs of edges are swapped in a sample.
        """
        if rdmmode == "CONFIGURATION_MODEL":

            def gen_func(graph):
                degseq = [nx.degree(graph, node) for node in self.nodelist]
                return nx.relabel_nodes(
                    nx.configuration_model(degseq),
                    mapping={i: self.nodelist[i] for i in range(len(self.nodelist))},
                )

            statprefix = "cm"
        elif rdmmode == "EDGE_REWIRING":

            def gen_func(graph):
                rdmgraph = graph.copy()
                nx.connected_double_edge_swap(rdmgraph, nswap=n_edge_rewirings)
                return rdmgraph

            statprefix = "er"
        else:
            raise ValueError(
                "Incorrect rdmmode parameter selected ({}): possible modes are CONFIGURATION_MODEL, EDGE_REWIRING.".format(
                    rdmmode
                )
            )
        if n_samples is None:
            raise ValueError(
                "n_samples argument must be provided if samples is set to 'CONFIGURATION_MODEL'."
            )
        graph = self.rebuild_nx_graph()

        # Welford's method (ca. 1960) to calculate running variance
        N = n_samples
        M = 0
        S = 0
        try:
            from tqdm import tnrange  # if tqdm is present use tqdm progress bar

            rangefunc = tnrange
        except ImportError:  # if tqdm is not present it will fallback on standard loop
            rangefunc = range
        logging.info("Calculating kernel statistics (this may take a long while)...")
        for k in rangefunc(N):
            rdmgraph = gen_func(graph)
            rdmadj = np.asarray(
                nx.adjacency_matrix(rdmgraph, nodelist=self.nodelist).todense()
            )
            kmatrix = self.kid2func(kernel)(A=rdmadj)
            x = kmatrix
            oldM = M
            M = M + (x - M) / (k + 1)
            S = S + (x - M) * (x - oldM)
        self.kernels[kernel + "_" + statprefix + "mean"] = M
        self.kernels[kernel + "_" + statprefix + "var"] = np.sqrt(S / (N - 1))
        logging.info("Complete")

    def onehot_encode(self, nodeset, norm=False):
        if (
            len(nodeset) == self.adj.shape[0]
        ):  # if the seedgenes are already in vector form (also for weighted configurations)
            vec = nodeset
        else:
            vec = np.zeros(self.adj.shape[0])
            vec[self.gm.gid2id(nodeset)] = 1
        if norm:
            vec /= vec.sum()
        return vec

    def vec2dict(self, nodevec):
        return {self.gm.id2gid(i): nodevec[i] for i in range(len(nodevec))}

    def dict2vec(self, nodedict):
        return np.array([nodedict[node] for node in self.nodelist])

    def get_projection(
        self,
        seedset,
        kernel,
        destset=None,
        correction=False,
        rdm_seedsets=None,
        significance_formula="ZSCORE",
        norm=False,
        return_dict=True,
    ):
        """
        Computes the projection (similarity) of the nodes in the seedset with the destset nodes. If destset is None the projection to the full network is computed.
        Parameters
        ----------
        seedset : list
            List of source nodes
        kernel : str or numpy matrix
            Kernel ID (KID) of the chosen kernel or numpy kernel matrix A_ij where represents the projection from node j (source) to node i (destination)
        destset : list
            List of destination nodes. If return_dict is False this parameter is ignored.
        correction : str or False
            Projection score correction to account for statistical biases such as high degree.
            Options are:
                - False: no correction
                - 'SEEDSET_SIZE': the final score vector is divided by the number of source nodes
                - 'DEGREE_CENTRALITY': the score of each destination node is divided by its degree
                - 'RDM_SEED': statistical significance of the score is evaluated by considering random samples of the source nodes in seedset.
                    Random samples of seedset have to be provided through the rdm_seedsets parameter.
                    In this mode the output projection nodes will be the significance values of the uncorrected scores,
                    calculated according to the formula specified by the significance_formula parameter
                - 'CONFIGURATION MODEL': statistical significance of the score is evaluated by comparing to random configuration model samples of the network.
                    This mode can be called only after having called the eval_kernel_statistics method with rdmmode='CONFIGURATION_MODEL'
                - 'EDGE_REWIRING': statistical significance of the score is evaluated by comparing to samples generated by random rewiring of the network edges.
                    This mode can be called only after having called the eval_kernel_statistics method with rdmmode='EDGE_REWIRING'
        rdm_seedsets : list of lists
            List of lists containing random samples of the seedset list. To be used for statistical significance evaluation.
            If correction is not set to 'RDM_SEED' this parameter is ignored
        significance_formula : str
            Formula to calculate statistical significance.
            Options are:
                - 'ZSCORE': (value - mean) / std.dev.
                - 'ISTVAN': Not implemented
        norm : bool
            Whether to normalize the output projection vector. Useful if comparing projections of several source nodesets with different sizes
        return_dict: bool
            Whether the output projection has to be returned as a {node_id : value} dict or as a dense N-dim vector
        Returns
        -------
        dict or list
            Output projection from seedset nodes to destset nodes
        """
        if isinstance(kernel, str):
            kid = kernel
            kernel = self.kernels[kernel]
        else:
            kid = None
        seedvec = self.onehot_encode(seedset)
        if not correction:
            nodevec = np.dot(kernel, seedvec)
        elif correction == "SEEDSET_SIZE":  # number of genes in the seed set
            nodevec = np.dot(kernel, seedvec) / seedvec.sum()
        elif correction == "DEGREE_CENTRALITY":  # degree centrality
            k0 = self.adj.sum(axis=1)
            nodevec = np.dot(kernel, seedvec) / k0
        elif correction == "RDM_SEED":
            if rdm_seedsets is None:
                raise ValueError(
                    "rdm_seedsets param must be set when in RDM_SEED mode!"
                )
            samples_proj = self.get_projections_batch(rdm_seedsets, kernel)
            if significance_formula == "ZSCORE":
                nodevec = zscore(np.dot(kernel, seedvec), samples_proj)
            elif significance_formula == "ONETAIL":
                nodevec = onetail(np.dot(kernel, seedvec), samples_proj)
            elif significance_formula == "ISTVAN":
                nodevec = istvan(np.dot(kernel, seedvec), samples_proj)
            else:
                raise ValueError(
                    "Incorrect significance formula selected ({}): possible modes are ZSCORE, ONETAIL, ISTVAN.".format(
                        significance_formula
                    )
                )
        elif correction == "CONFIGURATION_MODEL" or correction == "EDGE_REWIRING":
            statprefix = "cm" if correction == "CONFIGURATION_MODEL" else "er"
            if (
                kid is None
                or kid + "_" + statprefix + "mean" not in self.kernels.keys()
                or kid + "_" + statprefix + "var" not in self.kernels.keys()
            ):
                raise ValueError(
                    "CONFIGURATION_MODEL/EDGE_REWIRING correction can be invoked only for pre_calculated kernels and kernel statistics. Call eval_kernel_statistics() function on the selected kernel to make this mode accessible."
                )
            mean, var = (
                self[kid + "_" + statprefix + "mean"],
                self[kid + "_" + statprefix + "var"],
            )
            if significance_formula == "ZSCORE":
                nodevec = zscore(
                    np.dot(kernel, seedvec),
                    mean=np.dot(mean, seedvec),
                    std=np.sqrt(np.dot(var, seedvec**2)),
                )
            elif significance_formula == "ISTVAN":
                nodevec = istvan(
                    np.dot(kernel, seedvec),
                    mean=np.dot(mean, seedvec),
                    std=np.sqrt(np.dot(var, seedvec**2)),
                )
            else:
                raise ValueError(
                    "Incorrect significance formula selected ({}): possible modes are ZSCORE, ISTVAN.".format(
                        significance_formula
                    )
                )
        else:
            raise ValueError(
                "Incorrect mode selected ({}): possible modes are SEEDSET_SIZE, DEGREE_CENTRALITY, RDM_SEED, CONFIGURATION_MODEL, EDGE_REWIRING.".format(
                    correction
                )
            )
        if norm:
            nodevec /= nodevec.sum()
        if return_dict:
            valuedict = self.vec2dict(nodevec)
            if destset is not None:
                return {
                    key: value for key, value in valuedict.iteritems() if key in destset
                }
            else:
                return valuedict
        else:
            return nodevec

    def get_projections_batch(self, seedsets, kernel):
        """
        Evaluates list of projections from list of sets of source nodes
        Parameters
        ----------
        seedsets : list of lists
            Each list is a set of source nodes to evaluate a distribution of projections
        kernel : str or numpy matrix
            Kernel ID (KID) of the chosen kernel or numpy kernel matrix A_ij where represents the projection from node j (source) to node i (destination)
        Returns
        -------
        list of numpy arrays
            N x N_samples numpy matrix, where N_samples is len(seedsets), and each column is a numpy vector of projection scores
        """
        if isinstance(kernel, str):
            kernel = self.kernels[kernel]
        seedvecs = np.array(map(self.onehot_encode, seedsets)).T
        samples = np.dot(kernel, seedvecs)
        return samples

    def get_projection_statistics(self, seedsets, kernel):
        """
        Evaluates mean and standard deviation of projections from seedsets source nodes to network nodes
        Parameters
        ----------
        seedsets : list of lists
            Each list is a set of source nodes to evaluate a distribution of projections
        kernel : str or numpy matrix
            Kernel ID (KID) of the chosen kernel or numpy kernel matrix A_ij where represents the projection from node j (source) to node i (destination)
        Returns
        -------
        (numpy array, numpy array)
            Mean and std.dev. vectors of seedsets projections distribution
        """

        samples = self.get_projections_batch(seedsets=seedsets, kernel=kernel)
        return samples.mean(axis=1), samples.std(axis=1)

    def get_ranking(self, projection, candidateset=None, descending=True):
        """
        Evaluates ranking of nodes from a projection vector or dict
        Parameters
        ----------
        projection : dict or numpy array
            Projection dict/vector obtained with get_projection function
        candidate_set : list
            Set of destination nodes to consider for the ranking. If None all network nodes are considered
        ascending : bool
            Whether to order in ascending or descending scores (ascending for similarity matrices, descending for distances such as DSD)
        Returns
        -------
        list
            List of network nodes ordered by increasing rank
        """
        if isinstance(projection, dict):
            projection = self.dict2vec(projection)
        direction = -1 if descending else 1
        ranking = self.gm.id2gid(np.argsort(projection)[::direction])
        if candidateset is None:
            return ranking
        else:
            excludeset = set(self.nodelist) - set(candidateset)
            for elem in excludeset:
                ranking.remove(elem)
            return ranking

    def available_kernels(self):
        """
        Returns list of KIDs of kernels cached in GraphKernel object. To directly obtain a kernel matrix use the getitem operator
        e.g.
            kernel = gk[kid]   where gk is a GraphKernel instance and kid is the Kernel ID
        Returns
        -------
        list
            List of Kernel IDs cached in GraphKernel instance
        """
        return self.kernels.keys()

    def get_average_projection(self, sourceset, destset, kernel):
        """
        Evaluates average projection from sourceset to destset using kernel matrix
        Parameters
        ----------
        sourceset : list
            List of source node IDs
        destset : list
            List of destination node IDs
        kernel : str or numpy matrix
            KID or kernel matrix
        Returns
        -------
        float
            Average projection from sourceset to destset
        """
        if isinstance(kernel, str):
            kernel = self.kernels[kernel]
        seedvec = self.onehot_encode(sourceset, norm=True)
        nodevec = np.dot(kernel, seedvec)
        return (self.onehot_encode(destset, norm=True) * nodevec).sum()

    def save(self, filename, kidlist=None, description=None):
        """
        Save kernels to file
        Parameters
        ----------
        filename : str
            Path of savefile (.h5 format)
        kidlist : list
            List of KIDs to save on file. If None all kernels are saved
        description : str
            Optional description text embedded in the kernel savefile
        """

        if kidlist is None:
            kidlist = self.kernels.keys()
        elif isinstance(kidlist, str):
            kidlist = [kidlist]
        logging.info("Saving kernels...")
        with h5py.File(filename, "w") as hf:
            for kid in kidlist:
                hf.create_dataset(kid, data=self.kernels[kid], compression="gzip")
            hf.create_dataset("nodelist", data=self.nodelist)
            hf.create_dataset("adjacency", data=self.adj, compression="gzip")
            if description is not None:
                hf.create_dataset("description", data=description)
        logging.info("Complete.")

    def load(self, filename):
        """
        Load kernels from file
        Parameters
        ----------
        filename : str
            Path of savefile
        """

        if hasattr(self, "kernels") and len(self.kernels) > 0:
            warnings.warn("Loaded GraphKernel is overwriting an existing kernel set.")
        logging.info("Loading kernels...")
        with h5py.File(filename, "r") as hf:
            data = {}
            for key in hf.keys():
                if key == "description":
                    data["description"] = hf["description"].value
                elif key == "nodelist":
                    self.nodelist = hf["nodelist"][:]
                elif key == "adjacency":
                    self.adj = hf["adjacency"][:]
                else:
                    data[key] = hf[key][:]
        self.kernels = data
        self.gm = GIDMapper(nodelist=self.nodelist)
        logging.info("Complete.")

    def rebuild_nx_graph(self):
        graph = nx.from_numpy_matrix(self.adj)
        return nx.relabel_nodes(
            graph, {i: self.nodelist[i] for i in range(len(self.nodelist))}
        )

    def kid2func(self, kid):
        kid = kid.split("_")
        if kid[0] == "rw":
            return partial(random_walk_kernel, nRw=int(kid[1]))
        elif kid[0] == "rwr":
            return partial(random_walk_with_restart_kernel, alpha=float(kid[1]))
        elif kid[0] == "hk":
            return partial(heat_kernel, t=float(kid[1]))
        elif kid[0] == "dsd":
            return partial(diffusion_state_distance, nRw=int(kid[1]))
        elif kid[0] == "ist":
            return istvan_kernel

    def __getitem__(self, kid):
        return self.kernels[kid]
