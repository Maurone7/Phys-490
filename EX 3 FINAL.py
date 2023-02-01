import numpy as np

def w(x):
    x_sq = np.sum(x**2)
    if (x_sq > 1):
        return 0.
    else:
        return np.exp(-x_sq)


def f(x, y):
    return x[0]**2 + y[0]**2


def initialize():
    return np.random.uniform(-0.5, 0.5, 3)


def propose_new(x, theta):
    return x + theta * np.random.uniform(-1, 1, 3)


def monte_carlo(w, f, theta, n_sample, discard_m):
    x = initialize()
    wx = w(x)
    fs = []
    accept_num = 0
    for jj in range(discard_m+n_sample):
        y = propose_new(x, theta)
        wy = w(y)
        z = propose_new(y, theta)
        wz = w(z)
        if (wx>wy):
            alpha = wy/wx
            xi = np.random.uniform(0, 1)
            if (alpha >= xi):
                x = y
                wx = wy
                accept_num += 1
        else:
            x = y
            wx = wy
            accept_num += 1

        fs.append(f(x, y))

    accept_prob = accept_num/n_sample
    fs = np.array(fs)
    fs = fs[discard_m:]
    return np.sum(fs)/n_sample, accept_prob

if __name__ == '__main__':
    theta = 1.2
    n_sample = 100000
    discard_m = 2000
    integral, accept_prob = monte_carlo(w, f, theta, n_sample, discard_m)
    print("Integration = ", integral)
    print("Acceptance probability is", accept_prob)