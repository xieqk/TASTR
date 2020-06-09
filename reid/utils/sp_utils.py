import numpy as np
from sklearn import cluster


def gaussian_sp(delta_t, miu, sigma=65):
    x, u, sig = delta_t, miu, sigma
    p = np.exp(-(x-u)**2 / (2*sig**2))
    p = max(0.0001, p)
    return p

def flat_gaussian_sp(delta_t, miu, sigma=30):
    if delta_t <= 0:
        return 0.0001
    else:
        x, u, sig = delta_t, miu, sigma
        p = np.exp(-(max(np.abs(x-u), sig)-sig)**2 / (2*sig**2))
        p = max(0.0001, p)
        return p

def delta_time(q_tmins, q_tmaxs, g_tmins, g_tmaxs):
    m, n = q_tmins.shape[0], g_tmins.shape[0]
    delta_t = np.full((m, n), -1.)
    for i in range(m):
        for j in range(n):
            t1 = g_tmins[j] - q_tmaxs[i]
            t2 = q_tmins[i] - g_tmaxs[j]
            # if t1*t2 < 0:
            #     delta_t[i, j] = max(t1, t2)
            delta_t[i, j] = max(t1, t2)
    return delta_t

def compute_sp(delta_t, q_camids, g_camids, time_mat, sigma=0.7, use_flat=False):
    m, n = q_camids.shape[0], g_camids.shape[0]
    p_mat = np.full((m, n), 1.0)
    for i in range(m):
        for j in range(n):
            if use_flat:
                T = time_mat[q_camids[i], g_camids[j]]
                sigma_final = max(sigma * T, 5.0)
                p_mat[i, j] = flat_gaussian_sp(delta_t[i, j], time_mat[q_camids[i], g_camids[j]], sigma=sigma_final)
            else:
                T = time_mat[q_camids[i], g_camids[j]]
                sigma_final = max(sigma * T, 5.0)
                p_mat[i, j] = gaussian_sp(delta_t[i, j], time_mat[q_camids[i], g_camids[j]], sigma=sigma_final)
    return p_mat

def kmeans_1d_k(data, k):
    init = np.linspace(min(data), max(data), k)
    x = np.reshape(data, (-1,1))
    init = np.reshape(init, (-1,1))
    kmeans = cluster.KMeans(n_clusters=k, init=init, n_init=1)
    kmeans.fit(x)
    res = kmeans.predict(x)
    return res
