import numpy as np                    # NumPy 2.1.3
from scipy.stats import norm          # SciPy 1.15.2
from scipy.stats import qmc
import pyDOE3                         # pyDOE3 1.4.0

# Contents:
# (1) General utility functions
# (2) Random sampling functions (within bounds and around a specific point with uniform or LHS)
# (3) Multi-fidelity test functions  (Forrester and Park functions, ...)
# (4) Check of the nested design input sets
# (5) Performance metrics for UQ  (RMSE, Q^2, IAE_CI, IAE_PI, LOO variants)
# (6) Parametric mean GP functions (linear predictors)
# (7) Kernels (RBF and Matérn)
# (8) Parametric noise "eta" functions (Homoscedastic, exp of linear predictor)


#################### General utility functions ####################

def check_shape(name, arr, expected_shape, verbose=True):
    """ Check if the shape of the array in argument corresponds to the expected shape.
    Args:
        name (str): Name of the array of interest
        arr (array): Array of interest
        expected_shape (shape): Expected numpy shape
        verbose (bool, optional): If True, displays the message. Defaults to True.
    """
    if verbose:
        print(f"{name}: got {arr.shape}, expected {expected_shape}")
    assert arr.shape == expected_shape, "Shape mismatch!"


#################### Random sampling functions ####################

def sample_initial_points(bounds, n_samples=10, method="uniform", random_state=None):
    """ Sample initial parameter vectors within the given bounds.
    Args:
        bounds (list of tuple): List of (lower, upper) pairs for each parameter.
        n_samples (int): Number of samples.
        method (str): Sampling method: 'uniform', 'LHS simple' or 'LHS maximin'.
        random_state (int or None): Random seed for reproducibility.
    Returns: samples ((n_samples, n_params) array): Each row is an initial parameter vector within bounds.
    """
    rng = np.random.RandomState(random_state)  
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    if method == "uniform":
        samples = rng.uniform(0, 1, size=(n_samples, dim))
    elif method == "LHS simple":
        samples = qmc.LatinHypercube(d=dim, seed=random_state).random(n_samples)
    elif method == "LHS maximin":
        samples = pyDOE3.lhs(dim, samples=n_samples, criterion='maximin', random_state=random_state)
    else:
        raise ValueError("method must be 'uniform', 'LHS simple' or 'LHS maximin'")
    samples = lower + samples * (upper - lower)
    return samples


def sample_around_point(params, bounds, n_samples=10, proximity=0.1, method_near="uniform", random_state=None):
    """ Sample random points around a given point 'params' and within the specified bounds.
    Args:
        params ((d,) array): Base parameter.
        bounds (list of tuple): List of (lower, upper) pairs for each parameter.
        n_samples (int): Number of points near 'params'.
        proximity (float): Relative variation around 'params' (default corresponds to +/- 10%).
        method_near (str): 'uniform' or 'normal' for near samples.
        random_state (int or None): Random seed for reproducibility.
    Returns:
        samples ((n_samples,d) array): Resulting random samples.
    """
    rng = np.random.RandomState(random_state) 
    params = np.asarray(params)
    d = len(params)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    if method_near not in ["uniform", "normal"]:
        raise ValueError("method_near must be 'uniform' or 'normal'")
    near_samples = []
    for i in range(d):
        p = params[i]
        delta = proximity * abs(p) if p != 0 else proximity * (upper[i] - lower[i]) / 2
        low = max(lower[i], p - delta)
        high = min(upper[i], p + delta)
        if method_near == "uniform":
            near_col = rng.uniform(low, high, size=n_samples)
        else:  # Normal
            sigma = delta / 2 
            near_col = rng.normal(p, sigma, size=n_samples)
            near_col = np.clip(near_col, lower[i], upper[i])
        near_samples.append(near_col)
    near_samples = np.column_stack(near_samples)
    return near_samples


#################### Multi-fidelity functions ####################

#####  1D  #####

### Linear relation between LF and HF
# Forrester's function (same expression as in Le Gratiet (thesis), other forms exist)
def Forrester_LF(X):   
    """ Computes the Low-fidelity Forrester function
    Args: X ((N,D=1) array): Input values (in [0,1])
    Returns: (N,1) array: Function values
    """
    return 0.5*(6*X-2)**2 * np.sin(12*X-4) + 10*(X-0.5) - 5

def Forrester_HF(X): 
    """ Computes the High-fidelity Forrester function
    Args: X ((N,D=1) array): Input values (in [0,1])
    Returns: (N,1) array: Function values
    """
    return 2*Forrester_LF(X) - 20*X + 20   # the +20 term is not in Cutajar et al.[2019]

def Sine_LF(X): # taken from "Uncertainty-aware multi-fidelity surrogate modeling with noisy data" (Giannoukou et al., MF-PCE)
    """ Computes the Low-fidelity function (Sine wave)
    Args: X ((N,D=1) array): Input values (in [0,2])
    Returns: (N,1) array: Function values
    """
    return np.sin(2*np.pi*X)

def Sine_HF(X):
    """ Computes the High-fidelity function (Perturbed sine wave)
    Args: X ((N,D=1) array): Input values (in [0,2])
    Returns: (N,1) array: Function values
    """
    return (0.25*X - np.sqrt(2)) * np.sin(2*np.pi*X + np.pi)

#####  4D  #####

# Park function 
def ParkA_HF(X):
    """ Computes the High-fidelity Park function 
    Args: X ((N,D=4) array): Input values (in [0,1]^4)
    Returns: (N,1) array: Function values
    """
    x1, x2, x3, x4 = X[:,0], X[:,1], X[:,2], X[:,3]
    f_hf = 0.5 * (np.sqrt(x1**2 + (x2 + x3**2) * x4) - x1) + (x1 + 3*x4) * np.exp(1+np.sin(x3))   # (N,) shape
    return f_hf.reshape(-1,1)

def ParkA_LF(X):
    """ Computes the Low-fidelity Park function 
    Args: X ((N,D=4) array): Input values (in [0,1]^4)
    Returns: (N,1) array: Function values
    """
    x1, x2, x3, x4 = X[:,0], X[:,1], X[:,2], X[:,3]
    f_hf = ParkA_HF(X)[:,0]    # (N,) shape
    f_lf = (1 + 0.1*np.sin(x1)) * f_hf - 2*x1 + x2**2 + x3**2 + 0.5  # (N,) shape
    return f_lf.reshape(-1,1)


#####  8D  #####

def Borehole_fun(X, A, B):
    """ Computes the "general" Borehole function (LF and HF functions are special cases)
    Args: 
        X ((N,D=8) array): Input values (in [0,1]^8, to be rescaled inside the function).
        A, B : floats
    Returns: (N,1) array: Function values (Water flow rate, in m^3/year).
    """
    lower = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])    # Usual bounds
    upper = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])
    # lower = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 1500])      # Bounds used in Xiong et al. (2013)
    # upper = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 15000])
    Xs = lower + (upper - lower) * X
    x1, x2, x3, x4, x5, x6, x7, x8 = Xs[:,0], Xs[:,1], Xs[:,2], Xs[:,3], Xs[:,4], Xs[:,5], Xs[:,6], Xs[:,7]
    num = A * x3 * (x4 - x6)
    denom = np.log(x2/x1) * (B + (2*x7*x3 / (np.log(x2/x1)*x1**2*x8)) + x3/x5)
    return (num / denom).reshape(-1,1)

def Borehole_LF(X):
    return Borehole_fun(X, A=2*np.pi, B=1.)

def Borehole_HF(X):
    return Borehole_fun(X, A=5., B=1.5)

#################### (Non)-nested input sets ####################


def Nested_inputs_hypothesis(X_H, X_L, tol=1e-10):
    """ Checks if X_H is a subset of X_L, and returns the corresponding indexes
    Args:
        X_H ((N_H,D) array): High-fidelity input set
        X_L ((N_L,D) array): Low-fidelity input set
        tol (float, optional): Numerical tolerance for which values are considered "close". Defaults to 1e-10.
    Returns: (bool, (N_H,) bool array, (N_H,) int array, float)  
    """
    # Euclidean distances
    diff = X_H[:, None, :] - X_L[None, :, :]  # shape (N_H, N_L, D)
    dist = np.sqrt(np.sum(diff**2, axis=2))   # shape (N_H, N_L)
    min_dist = np.min(dist, axis=1)           # shape (N_H,)
    indices = np.argmin(dist, axis=1)         # shape (N_H,) 
    membership = (min_dist <= tol * np.ones(X_H.shape[0]))
    is_subset = np.all(membership)
    indices = np.where(membership, indices, -1)  # puts -1 as the index value if the entry in membership is False
    # Nesting Threshold, if lower than tol, we consider that the sets are numerically nested.
    Nesting_Threshold = np.max(min_dist)   
    return is_subset, membership, indices, Nesting_Threshold


#################### Performance metrics for UQ ####################

# Metrics when the true function is available

# Root Mean Squared Error
def RMSE(X_test, code, meta):
    """ Computes the Root mean squared error for a test set.
    Args:
        X_test ((N_test,D) array): Test input points
        code (function: (N_test,D) -> (N_test,1)): Numerical code to be evaluated at X_test 
        meta ((N_test,1) array): Predictions of the metamodel at X_test
    Returns: float: RMSE value
    """
    true_test = code(X_test).reshape(-1,1)  # shape (N_test, 1) 
    return np.sqrt(np.mean((meta - true_test)**2))

# Predictivity coefficent
def Qsquared(X_test, code, meta):
    """ Computes the predictivity Q^2 coefficient for a test set.
    Args:
        X_test ((N_test,D) array): Test input points
        code (func: (N_test,D) -> (N_test,1)): Numerical code to be evaluated at X_test 
        meta ((N_test,1) array): Predictions of the metamodel at X_test
    Returns: float: Q^2 value
    """
    true_test = code(X_test).reshape(-1,1)  # shape (N_test, 1)
    mean_test = np.mean(true_test)
    num = np.mean((meta - true_test)**2)
    denom = np.mean((true_test - mean_test)**2) 
    return 1 - num/denom

# Predictive Variance Adequacy
def PVA(X_test, code, meta_mean, meta_std):
    """ Computes the PVA for a test set.
    Args:
        X_test ((N_test,D) array): Test input points
        code (func: (N_test,D) -> (N_test,1)): Numerical code to be evaluated at X_test
        meta_mean ((N_test,1) array): Mean predictions of the GP metamodel at X_test
        meta_std ((N_test,1) array): Standard deviations of the GP metamodel at X_test
    Returns: (float): PVA value
    """
    true_test = code(X_test).reshape(-1,1)  # shape (N_test, 1)
    frac = (meta_mean - true_test) / meta_std    # shape (N_test, 1)
    return np.abs(np.log(np.mean(frac**2)))

# Integrated Absolute Errors  (rely on the CI/PI for GPs) 
def IAE_CI(X_test, code, meta_mean, meta_std, alphas=np.arange(0.001, 1.0, 0.001)):
    """ Computes the IAE and Coverage Probabilities for GPs from the CIs on the latent function.
    Args:
        X_test ((N_test,D) array): Test input points.
        code (func: (N_test,D) -> (N_test,1)): Numerical code to be evaluated at X_test.
        meta_mean ((N_test,1) array): Mean predictions of the GP metamodel at X_test.
        meta_std ((N_test,1) array): Standard deviations of the GP metamodel at X_test.
        alphas ((K,) array, optional): Target coverage values. Defaults to alpha=0.001, 0.002, ..., 0.998, 0.999.
    Returns: (float, (K,) array, (K,) array): IAE value, alphas and corresponding empirical CICP values.
    """
    y_mean = meta_mean.T             # shape (1, N_test)
    y_true = code(X_test).T          # shape (1, N_test)
    y_std = meta_std.T               # shape (1, N_test)
    z = norm.ppf(0.5 + alphas[:, None] / 2)  # shape (K, 1)  
    lower = y_mean - z * y_std               # shape (K, N_test)
    upper = y_mean + z * y_std               # shape (K, N_test)
    inside = (y_true >= lower) & (y_true <= upper)  # shape (K, N_test)
    coverage = inside.mean(axis=1)          # shape (K,)
    abs_error = np.abs(coverage - alphas)   # shape (K,)
    iae = np.trapz(abs_error, alphas)
    return iae, alphas, coverage

def IAE_PI(X_test, code, meta_mean, meta_std, code_noise, meta_noise, alphas=np.arange(0.001, 1.0, 0.001)):
    """ Computes the IAE and Coverage Probabilities for GPs from the PIs on the noisy observations.
        Warning: The function generates noisy Gaussian perturbations of the true output values when called !
    Args:
        X_test ((N_test,D) array): Test input points.
        code (func: (N_test,D) -> (N_test,1)): Numerical code to be evaluated at X_test.
        meta_mean ((N_test,1) array): Mean predictions of the GP metamodel at X_test.
        meta_std ((N_test,1) array): Standard deviations of the GP metamodel at X_test.
        code_noise ((N_test,1) array): True noise variance values at X_test.
        meta_noise ((N_test,1) array): Estimated noise variance values at X_test.
        alphas ((K,) array, optional): Target coverage values. Defaults to alpha=0.001, 0.002, ..., 0.998, 0.999.
    Returns: (float, (K,) array, (K,) array): IAE value, alphas and corresponding empirical PICP values.
    """
    samples_test = code(X_test) + np.sqrt(code_noise) * np.random.normal(0,1,size=X_test.shape[0]).reshape(-1,1) # shape (N_test,1)
    y_mean = meta_mean.T             # shape (1, N_test)
    y_noisy = samples_test.T         # shape (1, N_test)              
    meta_std = np.sqrt(meta_std**2 + meta_noise)    # shape (N_test,1)
    y_std = meta_std.T                       # shape (1, N_test) 
    z = norm.ppf(0.5 + alphas[:, None] / 2)  # shape (K, 1)
    lower = y_mean - z * y_std               # shape (K, N_test)
    upper = y_mean + z * y_std               # shape (K, N_test)
    inside = (y_noisy >= lower) & (y_noisy <= upper)  # shape (K, N_test)
    coverage = inside.mean(axis=1)           # shape (K,)
    abs_error = np.abs(coverage - alphas)    # shape (K,)
    iae = np.trapz(abs_error, alphas)
    return iae, alphas, coverage


# Metrics with proxy targets, such as LOO-CV mean and variance :
def Qsquared_loo(z, mean_loo):
    """ Computes the predictivity Q^2 coefficient between (noisy) targets and a prediction vector. 
    Args:
        z ((N,1) array): Training output set.
        mean_loo ((N,1) array): LOO-CV predictive mean on the input training set.
    Returns: (float): Q^2 value.
    """
    mean_z = np.mean(z)
    num = np.mean((z - mean_loo)**2)
    denom = np.mean((z - mean_z)**2) 
    return 1 - num/denom

def IAE_loo(z, mean_loo, var_loo, alphas=np.arange(0.001, 1.0, 0.001)):
    """ Computes the IAE and Coverage Probabilities for GPs.
    Args:
        z ((N,1) array): Training output set.
        mean_loo ((N,1) array): LOO-CV predictive mean on the input training set.
        var_loo ((N,1) array): LOO-CV predictive variance on the input training set.
        alphas ((K,) array, optional): Target coverage values. Defaults to alpha=0.001, 0.002, ..., 0.998, 0.999.
    Returns: (float, (K,) array, (K,) array): IAE value, alphas and corresponding empirical CP values.
    """
    z_mean = mean_loo.T                       # shape (1, N)
    z_std = np.sqrt(var_loo).T    # shape (1, N)
    q = norm.ppf(0.5 + alphas[:, None] / 2)   # shape (K, 1)  
    lower = z_mean - q * z_std                # shape (K, N_test)
    upper = z_mean + q * z_std                # shape (K, N_test)
    inside = (z.T >= lower) & (z.T <= upper)  # shape (K, N_test)
    coverage = inside.mean(axis=1)            # shape (K,)
    abs_error = np.abs(coverage - alphas)     # shape (K,)
    iae = np.trapz(abs_error, alphas)
    return iae, alphas, coverage


#################### Parametric mean GP functions (linear predictors) ####################

# We define several parametric functions that serves as GP means.
# These functions are "linear predictors", i.e., functions of the form: m(x) = <f(x), beta>,
# where: - beta is a column vector of p parameters to estimate, shape: (p, 1)
#        - f(x) is a basis of functions: f(x) = (f1(x), ..., fp(x)).T where fj outputs a scalar for all j.
# We want to compute : F = (f_j(x_i)) for (i,j) and the scalar product <f(x), beta>.
# The user does not have to specify a value for beta since we do not need an initial value for optimizing the GP
# in most cases.

class Constant_basis():   
    def __init__(self, beta=None):
        """ The function f is: f(x) = 1
        Args: beta ((p=1,1) array, optional): Parameter. Defaults to None.
        """
        self.expected_Xdim = None 
        self.p = 1
        self.beta = beta 
        if beta is not None :
            check_shape('beta', beta, (self.p,1), verbose=False)
    def basis(self, X):
        """ Computes the Fmatrix (only composed of ones in this case)
        Args: X ((N,D) array): Input set
        Returns: (N,p=1) array: Fmatrix
        """
        return np.ones((X.shape[0], self.p)) 
    def mean_function(self, X):
        """ Computes the scalar product <f(x), beta> for each x (shape (D,)) in X
        Args: X ((N,D) array): Input set
        Returns: ((N,1) array): Vector of N components that are all equal to beta.
        """
        if self.beta is None:
            raise ValueError(f"No value attributed to beta ! (beta=None)")
        return self.basis(X) @ self.beta  
    
class Polynomial_basis():   
    def __init__(self, beta=None, p=1):
        """ The function f is: f(x) = (1, x, x^2, x^3, ..., x^(p-1)).T
        Args:
            beta ((p,1) array, optional): Parameters to estimate. Defaults to None.
            p (int): Number of parameters/coefficients to estimate (one more than the degree of the polynomial)
        """
        self.expected_Xdim = 1  # X must be (N, 1)
        self.beta = beta
        if beta is None:
            self.p = p
        else:
            assert beta.shape[1] == 1, f'{beta.shape} is not a valid shape for beta'
            self.p = np.size(beta)
            if p != self.p:
                print(f'Warning: the chosen value for p was {p} but it was changed to {self.p} !')
    def basis(self, X):
        """ Computes the Fmatrix 
        Args: X ((N,D=1) array): Input set, raises an error if X is not of the right shape.
        Returns: (N,p) array: Fmatrix, first column should be only composed of ones.
        """
        return X**np.arange(self.p)    
    def mean_function(self, X):
        """ Computes the scalar product <f(x), beta>=beta0 + beta1*x + beta2*x**2 + ... for each x (shape (1,)) in X
        Args: X ((N,D=1) array): Input set
        Returns: ((N,1) array): Vector containing the N scalar products
        """
        if self.beta is None:
            raise ValueError(f"No value attributed to beta ! (beta=None)")
        return self.basis(X) @ self.beta   
    
class Affine2D_basis():    
    def __init__(self, beta=None):
        """ The function f is: f(x1, x2) = (1, x1, x2)
        Args: beta ((p=3,1) array, optional): Parameters to estimate. Defaults to None.
        """
        self.expected_Xdim = 2  # X must be (N, 2)
        self.p = 3
        self.beta = beta
        if beta is not None :
            check_shape('beta', beta, (self.p,1), verbose=False)
    def basis(self, X):
        """ Computes the Fmatrix 
        Args: X ((N,D=2) array): Input set, raises an error if X is not of the right shape.
        Returns: (N,p=3) array: Fmatrix, first column should be only composed of ones.
        """
        X = np.atleast_2d(X)  
        return np.column_stack((np.ones(X.shape[0]), X)) 
    def mean_function(self, X):
        """ Computes the scalar product <f(x), beta>=beta0 + beta1*x1 + beta2*x2 for each x=(x1,x2) in X
        Args: X ((N,D=2) array): Input set
        Returns: ((N,1) array): Vector containing the N scalar products
        """
        if self.beta is None:
            raise ValueError(f"No value attributed to beta ! (beta=None)")
        return self.basis(X) @ self.beta  
    
class Sum_basis:
    def __init__(self, basis1, basis2):
        """Creates a linear predictor defined as the sum of two (compatible) linear predictors.
        Args:
            basis1, basis2 (linear predictors): Mean functions with their respective attributes.
        """
        self.basis1 = basis1
        self.basis2 = basis2
        # The two functions must be defined on the same input space.
        if (basis1.expected_Xdim is not None) and (basis2.expected_Xdim is not None):
            if basis1.expected_Xdim != basis2.expected_Xdim:
                raise ValueError(f"Incompatible input dimensions: {basis1.expected_Xdim} vs {basis2.expected_Xdim}")
        self.expected_Xdim = basis1.expected_Xdim or basis2.expected_Xdim
        self.p = basis1.p + basis2.p
        if (basis1.beta is not None) and (basis2.beta is not None):
            self.beta = np.vstack((basis1.beta, basis2.beta))
        else:
            self.beta = None

    def basis(self, X):
        """ Computes the Fmatrix 
        Args: X ((N,D) array): Input set, raises an error if X is not of the right shape.
        Returns: (N,p=p1+p2) array: Fmatrix
        """
        F1 = self.basis1.basis(X)
        F2 = self.basis2.basis(X)
        return np.hstack((F1, F2))

    def mean_function(self, X):
        """ Computes the scalar product <f(x), beta> = <f1(x), beta1> + <f2(x), beta2>.
        Args: X ((N,D) array): Input set
        Returns: ((N,1) array): Vector containing the N scalar products.
        """
        if self.beta is None:
            raise ValueError("No value attributed to beta ! (beta=None)")
        beta1 = self.beta[:self.basis1.p]
        beta2 = self.beta[self.basis1.p:]
        return self.basis1.basis(X) @ beta1 + self.basis2.basis(X) @ beta2

#################### Kernels ####################

# We define several kernels that serves as GP covariance functions.
# These kernels are of the form: k(x,x') = sigma2 * r(x,x'; theta),
# where: - r is a correlation function, it often takes as input the distance ||x-x'||.
#        - sigma2 (> 0) is the "kernel variance" parameter, it controls the magnitude of the variance of the GP
#        - theta is the length scaler parameter, it controls the correlation between two outputs of the GP
#          depending on the corresponding inputs. It is a vector with only positive entries.
# We want to compute : the correlation matrix R = r(x_i, y_j) and/or the covariance matrix K = k(x_i, y_j)
# The user does not always have to specify a value for sigma2 since we do not need an initial value for optimizing the GP
# in most cases. We consider here the "tensorized" version, where we take the product over the dimension on 1D correlation functions.


class RBF_kernel():
    def __init__(self, length_scale, variance=None):
        """ Definition of the Radial Basis Function (RBF) / Gaussian / Squared exponential kernel.
        Args:
            length_scale (float or (D,) array): Length scale(s) parameter(s).
            variance (float, optional): Kernel variance parameter. Defaults to None.
        """
        self.theta = np.atleast_1d(length_scale) 
        if variance is not None:
            self.sigma2 = np.atleast_1d(variance)

    def correlation(self, X1, X2):
        """ Computes the correlation matrix from two input sets. 
        Args:
            X1 ((N1,D) array): First input set
            X2 ((N2,D) array): Second input set
        Returns: (N1,N2) array: Correlation matrix
        """
        theta = self.theta
        assert X1.shape[1] == X2.shape[1], 'Input sets must be defined on spaces of same dimensions'
        # sqdist_theta[i,j,d]= (x1_i^d - x2_j^d)**2 / theta_d**2
        sqdist_theta = ((X1[:, None, :] - X2[None, :, :]) / theta) ** 2  # shape (N1, N2, D)
        R = np.exp(-0.5 * sqdist_theta)  # shape (N1, N2, D)
        return np.prod(R, axis=2) 
    
    def covariance(self, X1, X2):
        """ Computes the covariance matrix from two input sets. 
        Args:
            X1 ((N1,D) array): First input set
            X2 ((N2,D) array): Second input set
        Returns: (N1,N2) array: Covariance matrix
        """
        return self.sigma2 * self.correlation(X1, X2)
    
    def derivative_factor(self, X1, X2):
        """ Computes the term R(d) such that (d/dtheta_d)r = R(d) * r for each dimension d. 
        Args:
            X1 ((N1,D) array): First input set
            X2 ((N2,D) array): Second input set
        Returns: (N1,N2,D) array: Derivative factor term.
        """
        sqdist_theta = (((X1[:, None, :] - X2[None, :, :]) / self.theta)**2)  # shape (N1, N2, D)
        return sqdist_theta / self.theta
  
class Matérn_kernel():
    def __init__(self, regularity, length_scale, variance=None):
        """ Definition of the Matérn kernel.
        Args:
            regularity (str): Regularity of the kernel, either '3/2' or '5/2'.
            length_scale (float or (D,) array): Length scale(s) parameter(s).
            variance (float, optional): Kernel variance parameter. Defaults to None.
        """
        self.theta = np.atleast_1d(length_scale)
        if variance is not None:
            self.sigma2 = np.atleast_1d(variance)
        self.regularity = regularity
        if self.regularity == '3/2':
            self.Build_Corr_Matrix = lambda sqdist_theta: (1 + np.sqrt(3*sqdist_theta)) * np.exp(-np.sqrt(3*sqdist_theta))
        elif self.regularity == '5/2':
            self.Build_Corr_Matrix = lambda sqdist_theta: (1 + np.sqrt(5*sqdist_theta) + (5*sqdist_theta/3)) * np.exp(-np.sqrt(5*sqdist_theta))
        else:
            raise ValueError("Regularity must be '3/2' or '5/2'")

    def correlation(self, X1, X2):
        """ Computes the correlation matrix from two input sets. 
        Args:
            X1 ((N1,D) array): First input set
            X2 ((N2,D) array): Second input set
        Returns: (N1,N2) array: Correlation matrix
        """
        theta = self.theta
        sqdist_theta = ((X1[:, None, :] - X2[None, :, :]) / theta) ** 2
        R = self.Build_Corr_Matrix(sqdist_theta)
        return np.prod(R, axis=2)
        
    def covariance(self, X1, X2):
        """ Computes the covariance matrix from two input sets. 
        Args:
            X1 ((N1,D) array): First input set
            X2 ((N2,D) array): Second input set
        Returns: (N1,N2) array: Covariance matrix
        """
        return self.sigma2 * self.correlation(X1, X2)
    
    def derivative_factor(self, X1, X2):
        """ Computes the term R(d) such that (d/dtheta_d)r = R(d) * r for each dimension d. 
        Args:
            X1 ((N1,D) array): First input set
            X2 ((N2,D) array): Second input set
        Returns: (N1,N2,D) array: Derivative factor term.
        """
        sqdist_theta = (((X1[:, None, :] - X2[None, :, :]) / self.theta)**2)  # shape (N1, N2, D)
        if self.regularity == '3/2':
            return np.sqrt(3*sqdist_theta)*(1 - (1+np.sqrt(3*sqdist_theta))**-1) / self.theta
        if self.regularity == '5/2':
            return np.sqrt(5*sqdist_theta)*(1 - (1+np.sqrt(20*sqdist_theta)/3) * (1+np.sqrt(5*sqdist_theta)+5*sqdist_theta/3)**-1) / self.theta
            

#################### Parametric noise functions ####################

# We define several parametric "eta" noise functions for heteroscedastic GP regression.
# We suppose the noise variance function is of the form: sigma^2_epsilon(x) = sigma2 * eta(x), so the covariance
# matrix can be factorized as: COV = sigma2 * CORR, which lets us derive a formula for the MLE of sigma2.
# where: - sigma2 is the kernel variance parameter (), shape: (1,)
#        - eta is a positive function that depends on a vector parameter zeta, shape (len_z,)
# The user has to specify an initial value for zeta.

class Homoscedastic():
    def __init__(self, params):
        """ The function is: eta(x) = exp(zeta0), the noise variance does not depend on X. 
        Args: params ((len_z=1,) array): Parameter to estimate.  (equivalent to optimize log eta)
        """
        self.expected_Xdim = None  
        self.zeta = np.asarray(params)  # if a 1D python list is given, convert it to a 1D array
        if self.zeta.ndim != 1:
            raise ValueError(f"zeta must be 1D (shape (1,)), got shape {self.zeta.shape}")
        if self.zeta.size != 1:
            raise ValueError(f"zeta must contain exactly one value, got {self.zeta.size}.")
    def compute(self, X):  
        """ Computes the eta noise function for each x (shape (1,)) in X
        Args: X ((N,D=1) array): Input set
        Returns: ((N,1) array): Vector containing the N eta noise values (all equal)
        """
        N = X.shape[0]
        return np.exp(self.zeta[0]) * np.ones(shape=(N,1))
    def gradient(self, X):
        """ Computes the gradient of the eta function for each x (shape (1,)) in X and each component of zeta.
        Args: X ((N,D=1) array): Input set
        Returns: ((N,len_z=1) array): Gradient values (equal to the function).
        """
        N = X.shape[0]
        return np.exp(self.zeta[0]) * np.ones(shape=(N,1))

class Homoscedastic_classic():
    def __init__(self, params):
        """ The function is: eta(x) = zeta0, the noise variance does not depend on X. 
        Args: params ((len_z=1,) array): Parameter to estimate. 
        """
        self.expected_Xdim = None  
        self.zeta = np.asarray(params)  # if a 1D python list is given, convert it to a 1D array
        if self.zeta.ndim != 1:
            raise ValueError(f"zeta must be 1D (shape (1,)), got shape {self.zeta.shape}")
        if self.zeta.size != 1:
            raise ValueError(f"zeta must contain exactly one value, got {self.zeta.size}.")
    def compute(self, X):  
        """ Computes the eta noise function for each x (shape (1,)) in X
        Args: X ((N,D=1) array): Input set
        Returns: ((N,1) array): Vector containing the N eta noise values (all equal)
        """
        N = X.shape[0]
        return self.zeta[0] * np.ones(shape=(N,1))
    def gradient(self, X):
        """ Computes the gradient of the eta function for each x (shape (1,)) in X and each component of zeta.
        Args: X ((N,D=1) array): Input set
        Returns: ((N,len_z=1) array): Gradient values (all equal to one).
        """
        N = X.shape[0]
        return np.ones(shape=(N,1))

class Exp_polynomial():  
    def __init__(self, params):
        """ The function is: eta(x) = exp(zeta0 + zeta1*x + zeta2*x^2 + ...)
        Args: params ((len_z,) array): Parameters to estimate.  
        """
        self.expected_Xdim = 1  # X must be (N, 1)
        self.zeta = np.asarray(params)  # if a 1D python list is given, convert it to a 1D array
        if self.zeta.ndim != 1:
            raise ValueError(f"zeta must be 1D (shape (len_z,)), got shape {self.zeta.shape}")
        if self.zeta.size == 0:
            raise ValueError("zeta cannot be empty (zeta must be > 0).")  
    def compute(self, X):  
        """ Computes the eta noise function for each x (shape (1,)) in X
        Args: X ((N,D=1) array): Input set
        Returns: ((N,1) array): Vector containing the N eta noise values
        """
        if X.shape[1] != 1 :
            raise ValueError(f"X is not 1D!")
        len_z = self.zeta.shape[0]
        return np.exp((X**np.arange(len_z)) @ self.zeta.reshape(-1,1))   
    def gradient(self, X):
        """ Computes the gradient of the eta function for each x (shape (1,)) in X and each component of zeta.
        Args: X ((N,D=1) array): Input set
        Returns: ((N,len_z) array): Gradient values.
        """
        len_z = self.zeta.shape[0]
        return (X**np.arange(len_z)) * self.compute(X)
    
class Exp_and_Power():  
    def __init__(self, params):
        """ The function is: eta(x) = exp(zeta0 + zeta1*log(|x|) + zeta2*x)
        Args: params ((len_z,) array): Parameters to estimate.  
        """
        self.expected_Xdim = 1  # X must be (N, 1)
        self.zeta = np.asarray(params)  # if a 1D python list is given, convert it to a 1D array
        if self.zeta.ndim != 1:
            raise ValueError(f"zeta must be 1D (shape (len_z,)), got shape {self.zeta.shape}")
        if self.zeta.size != 3:
            raise ValueError(f"zeta must contain exactly three values, got {self.zeta.size}.")   
    def compute(self, X):  
        """ Computes the eta noise function for each x (shape (1,)) in X
        Args: X ((N,D=1) array): Input set
        Returns: ((N,1) array): Vector containing the N eta noise values
        """
        zeta = self.zeta
        return np.exp(zeta[0] + zeta[1] * np.log(np.abs(X)) + zeta[2] * X)   
    def gradient(self, X):
        """ Computes the gradient of the eta function for each x (shape (1,)) in X and each component of zeta.
        Args: X ((N,D=1) array): Input set
        Returns: ((N,3) array): Gradient values.
        """
        N = X.shape[0]
        grad = np.zeros((N,3))
        eta = self.compute(X)[:,0]
        grad[:, 0] = eta
        grad[:, 1] = eta * np.log(np.abs(X[:,0]))
        grad[:, 2] = eta * X[:,0]
        return grad

class Log_affine_log():  
    def __init__(self, params, eta0):
        """ The function is: eta(x) = exp(zeta0) * exp(zeta1 * log eta0(x)), or
        log eta(x) = zeta0 + zeta1 * log eta0(x).
        Args: 
            params ((len_z=2,) array): Parameters to estimate.  
            eta0 (eta function)
        """
        self.eta0 = eta0
        self.expected_Xdim = eta0.expected_Xdim 
        self.zeta = np.asarray(params)  # if a 1D python list is given, convert it to a 1D array
        if self.zeta.ndim != 1:
            raise ValueError(f"zeta must be 1D (shape (len_z,)), got shape {self.zeta.shape}")
        if self.zeta.size != 2:
            raise ValueError(f"zeta must contain exactly two values, got {self.zeta.size}.")  
    def compute(self, X):  
        """ Computes the eta noise function for each x (shape (1,)) in X.
        Args: X ((N,D) array): Input set.
        Returns: ((N,1) array): Vector containing the N eta noise values.
        """
        zeta0 = self.zeta[0]
        zeta1 = self.zeta[1]
        return np.exp(zeta0 + zeta1 * np.log(self.eta0.compute(X))) 
    def gradient(self, X):
        """ Computes the gradient of the eta function for each x (shape (1,)) in X and each component of zeta.
        Args: X ((N,D) array): Input set
        Returns: ((N,len_z=2) array): Gradient values.
        """
        N = X.shape[0]
        grad = np.zeros((N,2))
        eta = self.compute(X)[:,0]
        grad[:, 0] = eta
        grad[:, 1] = eta * np.log(self.eta0.compute(X)[:,0])
        return grad
    
class Softplus_inv_affine():  
    def __init__(self, params, eta0):
        """ The function is: eta(x) = Sp(zeta0 + zeta1 * Sp^-1(eta0(x))), where Sp is the softplus function:
        Sp(x) = log(exp(x) + 1), and Sp^-1(y) = log(exp(y) - 1).
        Args: 
            params ((len_z=2,) array): Parameters to estimate.  
            eta0 (eta function)
        """
        self.eta0 = eta0
        self.expected_Xdim = eta0.expected_Xdim 
        self.zeta = np.asarray(params)  # if a 1D python list is given, convert it to a 1D array
        if self.zeta.ndim != 1:
            raise ValueError(f"zeta must be 1D (shape (len_z,)), got shape {self.zeta.shape}")
        if self.zeta.size != 2:
            raise ValueError(f"zeta must contain exactly two values, got {self.zeta.size}.")  
    def compute(self, X):  
        """ Computes the eta noise function for each x (shape (1,)) in X.
        Args: X ((N,D) array): Input set.
        Returns: ((N,1) array): Vector containing the N eta noise values.
        """
        zeta0 = self.zeta[0]
        zeta1 = self.zeta[1]
        inside = np.log(np.exp(self.eta0.compute(X)) - 1.)
        return np.log(np.exp(zeta0 + zeta1 * inside) + 1.)
    def gradient(self, X):
        """ Computes the gradient of the eta function for each x (shape (1,)) in X and each component of zeta.
        Args: X ((N,D) array): Input set
        Returns: ((N,len_z=2) array): Gradient values.
        """
        N = X.shape[0]
        grad = np.zeros((N,2))
        term = 1. - np.exp(-self.compute(X)[0])
        grad[:, 0] = term
        grad[:, 1] = term * np.log(np.exp(self.eta0.compute(X)[0]) - 1.)
        return grad
