

# Gaussian function for curve fitting
def gaussian(x, a, b, sigma, c):
    return a * np.exp(-((x - b) ** 2) / (2 * sigma**2)) + c


def gaussian_fit(y, x):
    mean_arg = np.argmax(y)  # np.argmin(y)
    mean = x[mean_arg]
    sigma = 0.5  # np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    popt, pcov = curve_fit(
        gaussian,
        x,
        y,
        bounds=(
            (0, min(x), -np.inf, -np.inf),
            (np.inf, max(y), np.inf, np.inf),
        ),
        p0=[max(y) - min(y), mean, sigma, min(y)],
    )
    return popt


def correct_ofs_and_amp(data_array, amp, ofs):
    return (data_array - ofs) / amp


def gaussian_2d(xy, A, B, C):
    x = xy[:, 0]
    y = xy[:, 1]
    envelope = np.exp(-(A * x**2 + 2 * B * x * y + C * y**2))
    return envelope


def gaussian_2d_fit(x, y, z, bounds, guess):
    x_mesh, y_mesh = np.meshgrid(x, y)
    xdata = np.c_[x_mesh.flatten(), y_mesh.flatten()]
    popt, pcov = curve_fit(
        gaussian_2d,
        xdata,
        z.flatten(),
        bounds=bounds,
        p0=guess,
    )
    return popt


def get_2d_guassian_sigma(x, y, z):
    bounds = [
        (0, -np.inf, 0),
        (
            np.inf,
            np.inf,
            np.inf,
        ),
    ]
    guess = (20, -50, 40)
    # z = correct_ofs_and_amp(z, 1.0, 1 * min(z.flatten()))
    opt = gaussian_2d_fit(x, y, z, bounds, guess)
    A, B, C = opt[:3]
    cov_matrix = np.array([[A, B], [B, C]])
    results = LA.eig(cov_matrix)
    lamb1, lamb2 = results[0]
    vec1, vec2 = results[1]
    sig1, sig2 = 1 / (2 * lamb1) ** 0.5, 1 / (2 * lamb2) ** 0.5
    return (sig1, sig2, vec1, vec2, opt)
