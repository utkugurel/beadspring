def chi_squared(observed, expected, scaling):
    """
Compute the chi-squared value between observed and expected data.

Parameters
----------
observed : np.ndarray
    Observed data values.
expected : np.ndarray
    Expected data values.
scaling : np.ndarray or float
    Scaling factors for the denominator in chi-squared formula.

Returns
-------
float
    The chi-squared value.

Examples
--------
>>> chi_squared_value = chi_squared(observed, expected, scaling)
>>> print(f"Chi2: {chi_squared_value:.2f}")
"""
    return ((observed - expected) ** 2 / scaling).sum()

def oneparam_fit(function, x, y):
    """
    Performs a fit on (x, y) as the specified function with one fit parameter
    Returns fitted parameter and quality factor q
    """

    popt, _ = curve_fit(function, x, y)
    p = popt[0]

    yexp = function(x, p)
    chi_squared_value = chi_squared(y, yexp, yexp)
    dof = len(x) - 1
    if chi_squared_value <= 0:
        q = 1.0
    else:
        q = 1 - gammainc(dof / 2.0, chi_squared_value / 2.0)

    return p, q

def fit_msd_with_quality_control(t, msd, msd_std, plot=False, title="MSD"):
    """
    Increases begin point of fitting (time_log, msd) until quality factor
    is above a 1/2. Returns 3D diffusion coefficient and its uncertainty
    """

    def linear(x, b):
        return x + b

    def diffusion(t, D):
        return 6 * D * t

    log_t = np.log10(t)  # TODO: Add a checkpoint to prevent zeros in these arrays
    log_msd = np.log10(msd)

    # compute bounds of std
    msd_min = msd - msd_std
    msd_max = msd + msd_std

    # loop over begin points and compute fit until quality factor >1/2
    start_index = -1
    quality_factor = 0
    while quality_factor < 1 / 2:
        start_index += 1
        t_selection = log_t[start_index:]
        msd_selection = log_msd[start_index:]
        _, quality_factor = oneparam_fit(linear, t_selection, msd_selection)

    # perform fitting on linear to obtain D and its bounds
    D, _ = oneparam_fit(diffusion, t[start_index:], msd[start_index:])
    diffusion_min, _ = oneparam_fit(diffusion, t[start_index:], msd_min[start_index:])
    diffusion_max, _ = oneparam_fit(diffusion, t[start_index:], msd_max[start_index:])

    diffusion_sigma = (diffusion_max - diffusion_min) / 2
    diffusion_uncertainty = diffusion_sigma / np.sqrt(len(log_msd) - 1)

    return D, diffusion_uncertainty

def fit_line_with_fixed_slope(x, y):
    """
    Fit a straight line to data points by forcing the slope to 1.

    Parameters:
    x (array-like): Independent variable data points.
    y (array-like): Dependent variable data points.

    Returns:
    float: The y-intercept of the fitted line.
    """
    x = np.array(x)
    y = np.array(y)

    # Calculate the y-intercept b
    b = np.mean(y - x)

    return b