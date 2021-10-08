import scipy.stats
import scipy.optimize
import numpy as np

from utils.image_tools import normalize_array


CORRELATIONS_EPS = 1e-6


def compute_correlations(a, b, normalize=True, do_fit=True):
    if normalize:
        aa = normalize_array(a)
        bb = normalize_array(b)
    else:
        aa = a.copy()
        bb = b.copy()
    spearman = scipy.stats.spearmanr(aa, bb).correlation
    kendall = scipy.stats.kendalltau(aa, bb).correlation
    if do_fit:
        # apply linear fitting before computing PLCC and RMSE
        try:
            fit_function = FitFunction(aa, bb)
            aa = fit_function(aa)
        except OverflowError as e:
            print("WARNING:", e)
    pearson = scipy.stats.pearsonr(aa, bb)[0]
    rmse = ((aa - bb) ** 2).mean() ** 0.5
    return spearman, kendall, pearson, rmse


########################################################################################################################

class FitFunction:
    def __init__(self,
                 source, target,
                 fit_function_to_use=1,
                 residuals_func='L1',
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
            self.info = 'Fit coeffs (p0, p1) in ' \
                        '"y = (p0 * x^2 + abs(p1) * x) + p2"'  # abs() needed to prevent sign flipping near 0
        elif fit_function_to_use == 4:
            self.pguess = (1.0, 1.0, 0.0)
            self.fit_function = FitFunction.fit_function4
            self.info = 'Fit coeffs (p0, p1, p2) in ' \
                        '"p0 / (p1 + exp(-p2*x))"'  # abs() needed to prevent sign flipping near 0

        else:
            raise ValueError("Unsupported fit function.")

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

    def residuals(self, p, x, y):
        return (y - self(x, p)) ** self.regularization

    def get_info(self):
        return self.info

    @staticmethod
    def fit_function1(p, x):
        p0, p1, p2, p3, p4 = p[:5]
        y = p0 * (0.5 - 1. / (1 + np.exp(p1 * (x - p2) + CORRELATIONS_EPS))) + abs(p3) * x + p4
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


def fit_values(source, target):
    """
    fit source array to target array, and return fitted values
    :param source:
    :param target:
    :return:
    """
    # normalize model predictions then fit to results of the subjective study
    # source_norm = normalize_array(source)
    # target_norm = normalize_array(target)
    try:
        return fit_regression(source, target)
    except OverflowError:
        print("Warning: regression failed, returning unfitted input.")
        return source.copy(), None


def fit_regression(source, target):
    fit_function = FitFunction(source, target)
    source_fit = fit_function(source)

    return source_fit, fit_function
