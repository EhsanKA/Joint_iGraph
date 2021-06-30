from networkx.drawing.layout import  rescale_layout
import numpy as np
import numbers


def joint_layout(
    A1,
    A2,
    factor=0.1,
    k1=None,
    k2=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    scale=1,
    dim=2,
    seed=10,
):
    print(dim)
    center = np.zeros(2)
    seed = check_random_state(seed)

    pos1, pos2 = _find_positions(
        A1, A2, k1, k2, factor, iterations, threshold, dim, seed
    )
    if fixed is None and scale is not None:
        pos1 = rescale_layout(pos1, scale=scale) + center
        pos2 = rescale_layout(pos2, scale=scale) + center
    return pos1, pos2


def _find_positions(
    A1, A2, k1=None, k2=None, factor=0.1, iterations=50, threshold=1e-4, dim=2, seed=None
):
    nnodes1, _ = A1.shape
    nnodes2, _ = A2.shape
    pos1 = np.asarray(seed.rand(nnodes1, dim), dtype=np.float32)
    pos2 = np.asarray(seed.rand(nnodes2, dim), dtype=np.float32)

    # optimal distance between nodes
    if k1 is None:
        k1 = np.sqrt(1.0 / nnodes1)
    if k2 is None:
        k2 = np.sqrt(1.0 / nnodes2)

    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t1 = max(max(pos1.T[0]) - min(pos1.T[0]), max(pos1.T[1]) - min(pos1.T[1])) * 0.1
    t2 = max(max(pos2.T[0]) - min(pos2.T[0]), max(pos2.T[1]) - min(pos2.T[1])) * 0.1

    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt1 = t1 / float(iterations + 1)
    dt2 = t2 / float(iterations + 1)
    # delta1 = np.zeros((pos1.shape[0], pos1.shape[0], pos1.shape[1]), dtype=A1.dtype)
    # delta2 = np.zeros((pos2.shape[0], pos2.shape[0], pos2.shape[1]), dtype=A2.dtype)

    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        err1, pos1 = _iteration(A1, pos1, k1, t1, dt1, nnodes1)
        err2, pos2 = _iteration(A2, pos2, k2, t2, dt2, nnodes2)
        delta = (pos1 - pos2)/2
        pos1 -= factor*delta*((iterations-iteration)/iterations*1.0)
        pos2 += factor*delta*((iterations-iteration)/iterations*1.0)

        if err1 < threshold or err2 < threshold:
            break
    return pos1, pos2

def _iteration(A, pos, k, t, dt,nnodes):
    # matrix of difference between points
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # distance between points
    distance = np.linalg.norm(delta, axis=-1)
    # enforce minimum distance of 0.01
    np.clip(distance, 0.01, None, out=distance)
    # displacement "force"
    displacement = np.einsum(
        "ijk,ij->ik", delta, (k * k / distance ** 2 - A * distance / k)
    )
    # update positions
    length = np.linalg.norm(displacement, axis=-1)
    length = np.where(length < 0.01, 0.1, length)
    delta_pos = np.einsum("ij,i->ij", displacement, t / length)
    pos += delta_pos
    # cool temperature
    t -= dt
    err = np.linalg.norm(delta_pos) / nnodes

    return err, pos

# def rescale_layout(pos, scale=1):
#     """Returns scaled position array to (-scale, scale) in all axes.
#
#     The function acts on NumPy arrays which hold position information.
#     Each position is one row of the array. The dimension of the space
#     equals the number of columns. Each coordinate in one column.
#
#     To rescale, the mean (center) is subtracted from each axis separately.
#     Then all values are scaled so that the largest magnitude value
#     from all axes equals `scale` (thus, the aspect ratio is preserved).
#     The resulting NumPy Array is returned (order of rows unchanged).
#
#     Parameters
#     ----------
#     pos : numpy array
#         positions to be scaled. Each row is a position.
#
#     scale : number (default: 1)
#         The size of the resulting extent in all directions.
#
#     Returns
#     -------
#     pos : numpy array
#         scaled positions. Each row is a position.
#
#     See Also
#     --------
#     rescale_layout_dict
#     """
#     # Find max length over all dimensions
#     lim = 0  # max coordinate for all axes
#     for i in range(pos.shape[1]):
#         pos[:, i] -= pos[:, i].mean()
#         lim = max(abs(pos[:, i]).max(), lim)
#     # rescale to (-scale, scale) in all directions, preserves aspect
#     if lim > 0:
#         for i in range(pos.shape[1]):
#             pos[:, i] *= scale / lim
#     return pos


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed