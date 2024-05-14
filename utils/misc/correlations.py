import scipy.stats
import scipy.optimize
import numpy as np
from matplotlib import pyplot as plt
from utils.image_processing.image_tools import normalize_array
from utils.logging import log_warn
from utils.misc.miscelaneous import float2str3


# constants and defs
CORRELATIONS_EPS = 1e-6

SROCC_FIELD = 'SROCC'
KROCC_FIELD = 'KROCC'
PLCC_FIELD = 'PLCC'
RMSE_FIELD = 'RMSE'
PLCC_NOFIT_FIELD = 'PLCC_NOFIT'
RMSE_NOFIT_FIELD = 'RMSE_NOFIT'


def compute_correlations(a, b, normalize=True):
    if normalize:
        aa = normalize_array(a)
        bb = normalize_array(b)
    else:
        aa = a.copy()
        bb = b.copy()
    spearman = scipy.stats.spearmanr(aa, bb).correlation
    kendall = scipy.stats.kendalltau(aa, bb).correlation

    pearson_nofit = scipy.stats.pearsonr(aa, bb)[0]
    rmse_nofit = ((aa - bb) ** 2).mean() ** 0.5

    # apply linear fitting before computing PLCC and RMSE
    try:
        fit_function = FitFunction(bb, aa)
        bb = fit_function(bb)  # target
    except OverflowError as e:
        log_warn("Overflow during logistic fit", e)

    pearson = scipy.stats.pearsonr(aa, bb)[0]
    rmse = ((aa - bb) ** 2).mean() ** 0.5

    return {
        SROCC_FIELD: spearman,
        KROCC_FIELD: kendall,
        PLCC_FIELD: pearson,
        RMSE_FIELD: rmse,
        PLCC_NOFIT_FIELD: pearson_nofit,
        RMSE_NOFIT_FIELD: rmse_nofit,
    }


########################################################################################################################

class FitFunction:
    def __init__(self,
                 source, target,
                 fit_function_to_use=1,
                 residuals_func='L1',
                 pguess=None,
                 ):
        if fit_function_to_use == 1:
            self.pguess = (1.0, 1.0, np.median(source), 1.0, np.median(target))
            self.fit_function = FitFunction.fit_function1
            self.info = 'Fit coeffs (p0, p1, p2, p3, p4) in ' \
                        '"y = p0 * (0.5 - 1. / (1 + np.exp(p1 * (x - p2)))) + abs(p3) * x + p4"'
            # abs() needed to prevent sign flipping near 0
        elif fit_function_to_use == 2:
            self.pguess = (1.0, 1.0, np.median(source), np.median(target))
            self.fit_function = FitFunction.fit_function2
            self.info = 'Fit coeffs (p0, p1, p2, p3) in ' \
                        '"y = p0 / (1 + np.exp(-p1 * (x - p2))) + p3"'
        elif fit_function_to_use == 3:
            self.pguess = (1.0, 0.0, 1.0, 0.0)
            self.fit_function = FitFunction.fit_function3
            self.info = 'Fit coeffs (p0, p1) ' \
                        '"y = (p0 * x^2 + abs(p1) * x) + p2"'  # abs() needed to prevent sign flipping near 0
        elif fit_function_to_use == 4:
            self.pguess = (1.0, 1.0, 0.0)
            self.fit_function = FitFunction.fit_function4
            self.info = 'Fit coeffs (p0, p1, p2) in ' \
                        '"p0 / (p1 + exp(-p2*x))"'  # abs() needed to prevent sign flipping near 0
        else:
            raise ValueError("Unsupported fit function.")

        if pguess is not None:
            self.pguess = pguess

        if residuals_func == "L1":
            self.regularization = 1
        elif residuals_func == "L2":
            self.regularization = 2
        else:
            raise ValueError("Unsupported regularization.")

        self.p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
            self.residuals, self.pguess, args=(source, target), full_output=True)

        if np.isnan(np.array(self.p)).any():
            raise OverflowError("Fitting failed: result contains NaNs.")

    def __call__(self, x, p=None):
        if p is None:
            return self.fit_function(self.p, x)
        else:
            return self.fit_function(p, x)

    def __str__(self):
        return "FitFunction: {}\n" \
               "init p={}\n" \
               "fit p={}".format(
            self.info,
            [float2str3(f) for f in self.pguess],
            [float2str3(f) for f in self.p])

    def residuals(self, p, x, y):
        return (y - self(x, p)) ** self.regularization

    def get_info(self):
        return self.info

    @staticmethod
    def fit_function1(p, x):
        p0, p1, p2, p3, p4 = p[:5]
        y = p0 * (0.5 - 1. / (1. + np.exp(p1 * (x - p2) + CORRELATIONS_EPS))) + abs(p3) * x + p4
        return y

    @staticmethod
    def fit_function2(p, x):
        p0, p1, p2, p3 = p[:4]
        y = p0 / (1 + np.exp(-p1 * (x - p2))) + p3
        return y

    @staticmethod
    def fit_function3(p, x):
        p0, p1, p2, p3 = p[:4]
        y = p0 * np.sqrt(abs(x - p1 + CORRELATIONS_EPS)) + p2 * (x - p1) ** 2 + p3
        return y

    @staticmethod
    def fit_function4(p, x):
        p0, p1, p2 = p[:3]
        return p0 / (p1 + np.exp(-x)) + p2


def fit_values(source, target, verbose=False):
    """
    fit source array to target array, and return fitted values
    :param source:
    :param target:
    :param verbose:
    :return:
    """
    # normalize model predictions then fit to results of the subjective study
    # source_norm = normalize_array(source)
    # target_norm = normalize_array(target)
    try:
        return fit_regression(source, target, verbose)
    except OverflowError:
        log_warn("Regression failed, returning unfitted input.")
        return source.copy(), None


def fit_regression(source, target, max_fit_error=0.2, verbose=False):
    fit_function = FitFunction(source, target)
    source_fit = fit_function(source)

    if verbose:
        print(fit_function.info)
        print(fit_function.p)

        # plot the fitted vs unfitted points
        plt.plot(target, target, 'g-')
        plt.plot(source, target, 'bo', label='nofit', markersize=5, alpha=0.8)
        plt.plot(source_fit, target, 'ro', label='fit', markersize=4, alpha=0.8)
        plt.xlabel('a')
        plt.ylabel('b')
        plt.legend()
        plt.show()

        # plot the fitted function between 0-1
        ar = np.arange(0, 1, 0.001)
        ar_fit = fit_function(ar)
        plt.plot(ar, ar_fit)
        plt.plot(ar, ar)
        plt.plot(source, source_fit, 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Fit function 0-1')
        plt.show()

    res_max = np.abs(source_fit - target).max()
    max_allowed = max_fit_error * (target.max() - target.min())
    if res_max > max_allowed:
        log_warn("fit max error [{}] exceeds allowed [{}]).".format(
            res_max, max_allowed))

    return source_fit, fit_function
