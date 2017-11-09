import numpy as np

# Bytes per batch to keep batches around the right size for good performance
_BYTES_PER_BATCH = 262144
# A 256 element array that acts as a lookup table that maps an 8-bit number
# `i` to the number of one-bits in the binary representation of `i`
_BIT_COUNT = np.unpackbits(np.arange(256).astype(np.uint8)[:, None],
                           axis=1).sum(axis=1).astype(int)


def bitwise_hamming_distance_arg_min(x, y, batch_size=None):
    """
    Compute the argmin of the bit-wise hamming distance between samples
    in `x` and samples in `y`. Returns the index of the sample in `y`
    that has the smallest hamming distance to each sample in `x`.

    Parameters
    ----------
    x: NumPy array, dtype=np.uint8
        The samples to test, a (N,bits) array where 8 bits are packed into
        each byte element
    y: NumPy array, dtype=np.uint8
        The samples to test against, a (M,bits) array where 8 bits are packed
        into each byte element
    batch_size: int or none
        The size of mini-batch from `x` to process at once, or `None` to
        estimate based on memory usage

    Returns
    -------
    Tuple(closest_ndx, dist) where
        closest_ndx is an array of dtype int and shape (N,) in which each
        element corresponds to a sample in `x` and gives the index of
        the sample from `y` whose hamming distance is the minimim
        and dist is an array of dtype int that gives the hamming distance
        from each sample in `x` to its closest neighbour in `y`
    """
    dist = np.zeros((len(x),), dtype=int)
    closest_ndx = np.zeros((len(x),), dtype=int)
    if batch_size is None:
        bytes_per_sample = (x.shape[1] + 1) * (len(y) + 1)
        batch_size = max(_BYTES_PER_BATCH // bytes_per_sample, 32)
    y = y.transpose()[None, :, :]
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i:i + batch_size]
        x_batch = x_batch[:, :, None]
        delta = np.bitwise_xor(x_batch, y)
        dist_batch = _BIT_COUNT[delta].sum(axis=1)
        dist[i:i + batch_size] = dist_batch.min(axis=1)
        closest_ndx[i:i + batch_size] = dist_batch.argmin(axis=1)
    return closest_ndx, dist


def binary_mode(x, batch_size=None):
    """
    Compute the mode of the packed binary samples in `x`

    Parameters
    ----------
    x: NumPy array, dtype=np.uint8
        Samples; (N,bits) array where 8 bits are packed into each byte element
    batch_size: int or none
        The size of mini-batch from `x` to process at once, or `None` to
        estimate based on memory usage

    Returns
    -------
    NumPy array, dtype=int, shape (N,)
        A (bits,) shape array that provides the packed binary mode of the
        samples in `x`.
    """
    if len(x) == 0:
        raise ValueError('Must have at least 1 sample')
    else:
        zeros = None
        ones = None
        if batch_size is None:
            bytes_per_sample = (x.shape[1] * 8)
            batch_size = max(_BYTES_PER_BATCH // bytes_per_sample, 32)
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i + batch_size]
            x_batch_bits = np.unpackbits(x_batch, axis=1)
            o = x_batch_bits.sum(axis=0)
            z = np.logical_not(x_batch_bits).sum(axis=0)
            zeros = (zeros + z) if zeros is not None else z
            ones = (ones + o) if ones is not None else o
        return np.packbits(ones >= zeros, axis=0)


class BinaryKModes(object):
    """
    Binary K-Modes
    """
    def __init__(self, n_clusters=10, rng=None, max_iter=128, verbose=False):
        """
        Constructor

        Parameters
        ----------
        n_clusters: int (default=10)
            The number of clusters to find
        rng: np.random.RandomState or None
            The random number generate used to initialise the clusters,
            use the default `np.random` if None
        max_iter: int (default=128)
            The maximum number of iterations to perform
        verbose: True
            Report progress
        """
        self.n_clusters = n_clusters
        if rng is None:
            rng = np.random
        self.rng = rng
        self.max_iter = max_iter
        self.verbose = verbose

        # Boolean array indicating that >= 1 samples are assigned to a
        # cluster
        self._assigned_clusters = np.array([False] * n_clusters)
        # Integer array giving the cluster index of each sample
        self._cluster_assignments = None
        # Integer array giving the hamming distance from each sample to its
        # closest (assigned) cluster
        self._cluster_distances = None
        # Cluster centres, named to fit with scikit-learn KMeans
        self.cluster_centers_ = None

    def _fill_empty_clusters(self, Xb, cluster_modes):
        """
        Choose samples from `Xb` to act as cluster modes for clusters with
        no assigned samples
        """
        # Count of the number of iterations that have passed with no change
        num_iters_no_change = 0
        # Number of clusters
        num_assigned = self._assigned_clusters.sum()
        # Until all clusters have samples
        while num_assigned < self.n_clusters:
            # Get the number of clusters we need to re-assign
            num_unassigned = self.n_clusters - num_assigned

            if self._cluster_distances is not None:
                # We have sample->cluster distances
                # Select samples whose distance to their cluster is > 0;
                # use these as modes for new clusters
                ndx = np.arange(len(Xb))[self._cluster_distances > 0]
                if len(ndx) == 0:
                    # We cannot find samples to use
                    # This probably means that there are less unique samples
                    # than clusters
                    return cluster_modes
                # Randomly shuffle these samples
                self.rng.shuffle(ndx)
                # Select the number that we need
                ndx = ndx[:num_unassigned]
            else:
                # Initial step: choose samples randomly
                ndx = self.rng.permutation(len(Xb))[:num_unassigned]
            # Use the chosen samples as new cluster modes
            unassigned = np.logical_not(self._assigned_clusters)
            unassigned_ndx = np.arange(len(cluster_modes))[unassigned]
            cluster_modes[unassigned_ndx[:len(ndx)]] = Xb[ndx]

            # Re-compute cluster assignments and distances
            self._cluster_assignments, self._cluster_distances = \
                bitwise_hamming_distance_arg_min(Xb, cluster_modes)
            self._notify_cluster_assignments_changed()

            num_assigned = self._assigned_clusters.sum()

        return cluster_modes

    def _notify_cluster_assignments_changed(self):
        self._assigned_clusters = np.bincount(self._cluster_assignments,
                                              minlength=self.n_clusters) > 0

    def fit(self, X, packed_input=False):
        """
        Fit the model

        Parameters
        ----------
        X: NumPy array
            A (N,M) dtype=bool array where N is the number of samples and M
            is the number of features if `packed_input` is False
            Or a (N,Q) dtype=unit8 array where N is the number of samples and
            M is the number of bytes needed to store the bits of a binary
            packed sample if `packed_input` is True
        packed_input: bool
            if True, then `X`
        :param X:
        :return:
        """
        if packed_input:
            Xb = X
        else:
            Xb = np.packbits(X.astype(bool), axis=1)

        # Initialise cluster modes
        cluster_modes = np.zeros((self.n_clusters, Xb.shape[1]), dtype=np.uint8)
        cluster_modes = self._fill_empty_clusters(Xb, cluster_modes)

        # Iterate
        for i in range(self.max_iter):
            if self.verbose:
                print('Binary K-Modes iteration {}...'.format(i))
            # Re-compute modes
            for cls_i in range(self.n_clusters):
                samples = Xb[self._cluster_assignments == cls_i]
                if len(samples) > 0:
                    cluster_modes[cls_i] = binary_mode(samples)

            # Re-assign clusters
            clus_assign, clus_dist = bitwise_hamming_distance_arg_min(
                Xb, cluster_modes)

            if (self._cluster_assignments == clus_assign).all():
                break
            else:
                self._cluster_assignments = clus_assign
                self._cluster_distances = clus_dist
                self._notify_cluster_assignments_changed()
                cluster_modes = self._fill_empty_clusters(Xb, cluster_modes)

        self.cluster_centers_ = np.unpackbits(
            cluster_modes, axis=1)[:, :X.shape[1]]
        return self
