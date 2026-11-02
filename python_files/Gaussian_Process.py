from utils import *

import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm  # version 4.67.1


################################################################
################################################################

#################### Gaussian Process class ####################

################################################################
################################################################

class GPR():
    def __init__(self, mean, kernel, X, z, noise=None, eta_fn=None, REML=False, random_init=False):
        """ Initiates a general GP regression (GPR) model which is fitted using the (marginal) log-likelihood and the 
            L-BFGS-B optimizer to fit the parameters for which no MLE formula is given. Depending on the situation
            regarding the noise, the parametrization can be different (focuses either on the cov or corr matrix).
        Args:
            mean (linear predictor): Mean regression function, linear predictor with a function basis (parameter: beta).
            kernel (kernel): Kernel, RBF or Matérn (parameters: sigma2, theta).
            X ((N,D) array): Input training set.
            z ((N,1) array): Output training set.
            noise (multiple types, defaults to None): Different noise situations:
                - If noise=None, noiseless GPR.
                - If noise='opt' (or any str) and eta_fn=None, homoscedastic GPR by default (parameter: zeta = log(eta)),
                otherwise, parametric heteroscedastic GPR with the chosen eta function.
                - If noise is a (N,1) array, the noise part is fixed and not affected in the optimization.
            eta_fn (None or Noise eta function, optional): Uses the specified noise eta function (parameter: zeta). Defaults to None.
            REML (bool): If True, optimizes the GP using the REML and not the usual "plug-in" MLE.
            random_init (bool): If True, all initial values for the parameters are random in the multi-start.
        """
        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D), got shape {X.shape}")
        N, D = X.shape[0], X.shape[1]
        if z.ndim != 2:
            raise ValueError(f"z must be a 2D column vector, got shape {z.shape}")
        if z.shape[1] != 1 or z.shape[0] != N:
            raise ValueError(f"z must be of shape {(N,1)}, got shape {z.shape}")
        if type(noise) == np.ndarray:
            if noise.ndim != 2:
                raise ValueError(f"noise must be a 2D column vector, got shape {noise.shape}")
            if noise.shape[1] != 1 or noise.shape[0] != N:
                raise ValueError(f"noise must be of shape {(N,1)}, got shape {noise.shape}")
        self.X = X
        self.z = z
        self.noise = noise
        self.noise_type = type(self.noise)  # is either equal to type(None), str ('opt') or np.ndarray
        self.N = N
        self.D = D 
        self.REML = REML
        self.random_init = random_init
        # Mean function check:
        mean_Xdim = mean.expected_Xdim
        if mean_Xdim is not None:
            if D != mean_Xdim:
                raise ValueError(f'Mean function not compatible with input dimension: got {D}, expected {mean_Xdim}.')
        p = mean.p
        if not REML:
            if N < p:
                raise ValueError(f'MLE undefined: number of data samples ({N}) must be greater or equal to the size of the mean parameter vector ({p}).')
            self.dim_residuals = N
        else:
            if N < p+1 :
                raise ValueError(f'REML undefined: number of data samples ({N}) must be greater than the size of the mean parameter vector ({p}).')
            self.dim_residuals = N - p
        # Kernel check: if theta is scalar but X is multi-dim, assume isotropic kernel
        if kernel.theta.shape[0] == 1 and D > 1:  
            kernel.theta = np.full(D, kernel.theta[0])
        self.mean = mean             
        self.kernel = kernel
        # Noise eta function check: (useful only if noise='opt')
        if eta_fn is None:
            eta_fn = Homoscedastic(params=np.array([0.]))
        noise_Xdim = eta_fn.expected_Xdim
        if noise_Xdim is not None:
            if D != noise_Xdim:
                raise ValueError(f'Noise function not compatible with input dimension: got {D}, expected {noise_Xdim}.')
        self.eta_fn = eta_fn
        self.Fmatrix = self.mean.basis(X)  # Mean regression matrix (only data-dependant)
        self.Lmatrix = None  # Lower triangular Choesky decomposition matrix of the covariance matrix on 
                             # the training set X, updated when computing the loss function, 
                             # stored in the GP class to avoid duplicate computations when making predictions.
        self.call_fit = False
        self.jitter = 1e-10    # default, can be changed when calling the fit function.
        # Default parameter bounds:
        max_theta = np.max(self.X, axis=0) - np.min(self.X, axis=0)
        min_theta = np.zeros(D)
        for d in range(D):
            min_theta[d] = min(pdist(np.unique(self.X[:,d]).reshape(-1,1)))
        #min_theta = np.min(pdist(self.X)) * np.ones_like(self.kernel.theta)  # Ensure shape consistency
        self.theta_bounds = list(zip(min_theta, max_theta))
        self.zeta_bounds = [(-40., 10.)] * self.eta_fn.zeta.size   # the choice of the bounds is quite important
        self.sigma2_bounds = [(1e-30, 1e5)]   # only useful when the noise term is "known"
        self.bounds = None
        self.init_point = None
        # Determine NLML function and gradient
        self.nlml_fun = None
        self.grad_nlml = None
        if self.noise is None:
            self.nlml_fun = self.nlml_noiseless
            self.grad_nlml = self.grad_noiseless
            self.bounds = self.theta_bounds
            self.init_point = self.kernel.theta
        elif self.noise_type == str:
            self.nlml_fun = self.nlml_noisy_unknown
            self.grad_nlml = self.grad_noisy_unknown
            self.bounds = self.theta_bounds + self.zeta_bounds
            self.init_point = np.concatenate([self.kernel.theta, self.eta_fn.zeta])
        else:
            self.nlml_fun = self.nlml_noisy_known
            self.grad_nlml = self.grad_noisy_known
            self.bounds = self.theta_bounds + self.sigma2_bounds
            if not self.random_init:
                if self.kernel.sigma2 is None:
                    raise ValueError(f"No value attributed to the kernel variance ! (sigma2=None)")
                self.init_point = np.concatenate([self.kernel.theta, self.kernel.sigma2])

    def number_of_optimized_parameters(self):
        """ Returns: int: Number of all parameters to be optimized, either numerically or by an explicit MLE formula.
        """
        D = self.D
        p = self.mean.p
        return p + (D + 1) + int(self.noise_type==str) * self.eta_fn.zeta.shape[0]

    def get_all_params(self):
        """ Returns: ((p,1) array, (1,) array, (D,) array, (len_z,) array) : Outputs all the different parameters.
        """
        beta = self.mean.beta
        sigma2 = self.kernel.sigma2
        theta = self.kernel.theta
        zeta = self.eta_fn.zeta
        return beta, sigma2, theta, zeta
    
    def get_numerical_params(self):
        """ Returns the vectors of numerically optimized parameters (used then as input in the nlml function for instance),
            hence, its contents vary depending on the noise optimization situation.
            Type: (n_params,) array
        """
        theta = self.kernel.theta
        if self.noise is None:
            params = theta
        elif self.noise_type == str:
            params = np.concatenate([theta, self.eta_fn.zeta])
        else:
            params = np.concatenate([theta, self.kernel.sigma2])
        return params

    def get_noise_variance(self):
        """ Computes the noise variance at the training input set with the given parameter values.
        Returns: (N,1) array: Array containing the noise variance values.
        """
        N = self.N
        if self.noise is None:
            noise_estim = np.zeros((N,1))
        elif self.noise_type == str:
            noise_estim = self.kernel.sigma2 * self.eta_fn.compute(self.X)
        else:
            noise_estim = self.noise
        return noise_estim

    def __str__(self):
        """ Prints the all the parameter values in the class with their corresponding names. 
        """
        beta, sigma2, theta, zeta = self.get_all_params()
        if self.noise_type != str:
            param_names = ["mean_params", "kernel_variance", "length_scale"]
            param_values = [beta, sigma2, theta]
        if self.noise_type == str and self.eta_fn.zeta.size == 1:
            param_names = ["mean_params", "kernel_variance", "length_scale", "eta_fn_params", "noise_variance"]
            param_values = [beta, sigma2, theta, zeta, np.exp(zeta)*sigma2]
        if self.noise_type == str and self.eta_fn.zeta.size > 1:
            param_names = ["mean_params", "kernel_variance", "length_scale", "eta_fn_params"]
            param_values = [beta, sigma2, theta, zeta]
        header = f"{'Parameter Name':<20}{'Value':<20}"
        separator = "-" * len(header)
        rows = []
        for name, value in zip(param_names, param_values):
            value_str = str(value)
            lines = value_str.split("\n")
            rows.append(f"{name:<20}{lines[0]:<20}")
            rows.extend(f"{'':<20}{line:<20}" for line in lines[1:])
        return "\n".join([header, separator] + rows)
    
    # Negative log-marginal likelihoods (NLML) for the different noise situations and parametrizations (Corr or Cov):
    # Noiseless (Corr), Noise parameters to optimize (Corr) and Fixed noise variance (Cov)
    def nlml_noiseless(self, params):
        """ Computes the NLML for the noiseless case, params only contains theta, stores in the GP class the
            MLE/REML values for (beta, sigma2).
        Args: 
            params ((D,) array): Array containing all parameters to be optimized by scipy.
        Returns: float: NLML value.
        """
        kernel = self.kernel
        kernel.theta = params
        F = self.Fmatrix  # shape (N,p)
        N = self.N
        dres = self.dim_residuals
        R = kernel.correlation(self.X, self.X) + self.jitter * np.eye(N) # shape (N,N), SPD
        L = cholesky(R, lower=True, check_finite=False)  # shape (N,N), Lower triangular
        u_F = solve_triangular(L, F, lower=True, check_finite=False)          # = L^-1 F
        u_z = solve_triangular(L, self.z, lower=True, check_finite=False)     # = L^-1 z
        FRinvF = u_F.T @ u_F   # shape (p,p)
        beta = np.linalg.solve(FRinvF, u_F.T @ u_z)  # MLE formula for beta, shape (p,1)
        omega = u_z - u_F @ beta   # L^-1 (z - F @ beta)
        sigma2 = (omega.T @ omega)[0] / dres         # MLE/REML formula for sigma2, shape (1,)
        self.mean.beta = beta      # Store MLE values 
        self.kernel.sigma2 = sigma2
        self.Lmatrix = np.sqrt(sigma2) * L
        log_det_R = 2 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * (dres*np.log(sigma2) + log_det_R + dres*np.log(2*np.pi)+dres)
        if self.REML:
            _, log_det_FRinvF = np.linalg.slogdet(FRinvF)
            nlml = nlml + 0.5 * log_det_FRinvF
        return nlml[0]
    
    def nlml_noisy_unknown(self, params):
        """ Computes the NLML for the noisy case with optimization of the noise parameters, params contains (theta, zeta),
            stores in the GP class the MLE/REML values for (beta, sigma2).
        Args: 
            params ((D+len_z,) array): Array containing all parameters to be optimized by scipy.
        Returns: float: NLML value.
        """
        kernel = self.kernel
        D = self.D
        theta, zeta = params[:D], params[D:]
        kernel.theta = theta
        self.eta_fn.zeta = zeta
        F = self.Fmatrix    # shape (N,p)
        N = self.N
        dres = self.dim_residuals
        R = kernel.correlation(self.X, self.X) + self.jitter*np.eye(N) + np.diag(self.eta_fn.compute(self.X)[:,0])  # shape (N,N), SPD
        L = cholesky(R, lower=True, check_finite=False)    # shape (N,N), Lower triangular
        u_F = solve_triangular(L, F, lower=True, check_finite=False)          # = L^-1 F
        u_z = solve_triangular(L, self.z, lower=True, check_finite=False)     # = L^-1 z
        FRinvF = u_F.T @ u_F   # shape (p,p)
        beta = np.linalg.solve(FRinvF, u_F.T @ u_z)  # MLE formula for beta, shape (p,1)
        omega = u_z - u_F @ beta   # L^-1 (z - F @ beta)
        sigma2 = (omega.T @ omega)[0] / dres         # MLE/REML formula for sigma2, shape (1,)
        self.mean.beta = beta      # Store MLE values 
        self.kernel.sigma2 = sigma2
        self.Lmatrix = np.sqrt(sigma2) * L
        log_det_R = 2 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * (dres*np.log(sigma2) + log_det_R + dres*np.log(2*np.pi)+dres)
        if self.REML:
            _, log_det_FRinvF = np.linalg.slogdet(FRinvF)
            nlml = nlml + 0.5 * log_det_FRinvF
        return nlml[0]

    def nlml_noisy_known(self, params):
        """ Computes the NLML for the noisy case with "known" noise, params contains (theta, sigma2),
            stores in the GP class the MLE values for beta.
        Args: 
            params ((D+1,) array): Array containing all parameters to be optimized by scipy
        Returns: float: NLML value.
        """
        kernel = self.kernel
        D = self.D
        theta, sigma2 = params[:D], params[D:]
        kernel.theta = theta
        kernel.sigma2 = sigma2
        F = self.Fmatrix    # shape (N,p)
        N = self.N
        K = kernel.covariance(self.X, self.X) + self.jitter*np.eye(N) + np.diag(self.noise[:,0])   # shape (N,N), SPD
        L = cholesky(K, lower=True, check_finite=False)   # shape (N,N), Lower triangular
        u_F = solve_triangular(L, F, lower=True, check_finite=False)          # = L^-1 F
        u_z = solve_triangular(L, self.z, lower=True, check_finite=False)     # = L^-1 z
        FKinvF = u_F.T @ u_F   # shape (p,p)
        beta = np.linalg.solve(FKinvF, u_F.T @ u_z)  # MLE formula for beta, shape (p,1)
        omega = u_z - u_F @ beta   # L^-1 (z - F @ beta)
        self.mean.beta = beta      # Store MLE values 
        self.Lmatrix = L
        log_det_K = 2 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * ((omega.T @ omega)[0] + log_det_K + N*np.log(2*np.pi))
        if self.REML:
            print('REML is not implemented for this case, MLE is used instead.')
        return nlml[0]
    
    
    # Gradient of Negative log-marginal likelihoods (NLML) for the different noise situations and parametrizations (Corr or Cov):
    def grad_noiseless(self, params):
        """ Computes the gradient of the NLML for the noiseless case, params only contains theta, stores in the GP class the
            MLE/REML values for (beta, sigma2).
        Args: 
            params ((D,) array): Array containing all parameters to be optimized by scipy.
        Returns: grad ((D,) array): Gradient values.
        """
        kernel = self.kernel
        kernel.theta = params
        F = self.Fmatrix  # shape (N,p)
        N, D = self.N, self.D
        dres = self.dim_residuals
        Correl = kernel.correlation(self.X, self.X)
        R = Correl + self.jitter * np.eye(N) # shape (N,N), SPD
        L = cholesky(R, lower=True, check_finite=False)  # shape (N,N), Lower triangular
        u_F = solve_triangular(L, F, lower=True, check_finite=False)        # L^-1 F
        u_z = solve_triangular(L, self.z, lower=True, check_finite=False)   # L^-1 z
        FRinvF = u_F.T @ u_F         # shape (p,p)
        beta = np.linalg.solve(FRinvF, u_F.T @ u_z)  # MLE formula for beta, shape (p,1)
        omega = u_z - u_F @ beta                     # L^-1 (z - F @ beta)
        sigma2 = (omega.T @ omega)[0] / dres         # MLE/REML formula for sigma2, shape (1,)
        self.mean.beta = beta      # Store MLE values 
        self.kernel.sigma2 = sigma2
        self.Lmatrix = np.sqrt(sigma2) * L
        dR = self.kernel.derivative_factor(self.X, self.X)   # shape (N,N,D)
        alpha_R = solve_triangular(L.T, omega, check_finite=False) / np.sqrt(sigma2)
        grad = np.zeros(D)   # shape (D,)
        if self.REML:
            L_p = cholesky(FRinvF, lower=True, check_finite=False)  # shape (p,p), Lower triangular
            v_F = solve_triangular(L_p, F.T, lower=True, check_finite=False) 
        for d in range(D):
            dR_d = dR[:,:,d] * Correl
            term1 = alpha_R.T @ dR_d @ alpha_R
            R_inv_dR = solve_triangular(L.T, solve_triangular(L, dR_d, lower=True, check_finite=False),
                                            check_finite=False)  # solves R X = dR_d
            term2 = np.trace(R_inv_dR)
            if self.REML:
                Aux = solve_triangular(L.T, solve_triangular(L, (v_F.T @ v_F) @ R_inv_dR, lower=True, check_finite=False),
                                            check_finite=False) 
                term2 = term2 - np.trace(Aux)
            grad[d] = -0.5 * (term1 - term2)
        return grad
    
    def grad_noisy_unknown(self, params):
        """ Computes the Gradient NLML for the noisy case with optimization of the noise parameters, params contains (theta, zeta),
            stores in the GP class the MLE/REML values for (beta, sigma2).
        Args: 
            params ((D+len_z,) array): Array containing all parameters to be optimized by scipy.
        Returns: grad ((D+len_z,) array): Gradient values.
        """
        kernel = self.kernel
        D = self.D
        theta, zeta = params[:D], params[D:]
        kernel.theta = theta
        self.eta_fn.zeta = zeta
        W = zeta.shape[0]
        F = self.Fmatrix    # shape (N,p)
        N = self.N
        dres = self.dim_residuals
        Correl = kernel.correlation(self.X, self.X)
        R = Correl + self.jitter*np.eye(N) + np.diag(self.eta_fn.compute(self.X)[:,0])  # shape (N,N), SPD
        L = cholesky(R, lower=True, check_finite=False)    # shape (N,N), Lower triangular
        u_F = solve_triangular(L, F, lower=True, check_finite=False)        # L^-1 F
        u_z = solve_triangular(L, self.z, lower=True, check_finite=False)   # L^-1 z
        FRinvF = u_F.T @ u_F         # shape (p,p)
        beta = np.linalg.solve(FRinvF, u_F.T @ u_z)  # MLE formula for beta, shape (p,1)
        omega = u_z - u_F @ beta                     # L^-1 (z - F @ beta)
        sigma2 = (omega.T @ omega)[0] / dres         # MLE/REML formula for sigma2, shape (1,)
        self.mean.beta = beta      # Store MLE values 
        self.kernel.sigma2 = sigma2
        self.Lmatrix = np.sqrt(sigma2) * L
        dR_dtheta = self.kernel.derivative_factor(self.X, self.X)   # shape (N,N,D)
        dR_dzeta = self.eta_fn.gradient(self.X)    # shape (N,W) 
        alpha_R = solve_triangular(L.T, omega, check_finite=False) / np.sqrt(sigma2)    # shape (N,1)
        grad = np.zeros(D+W)   # shape (D+W,)
        if self.REML:
            L_p = cholesky(FRinvF, lower=True, check_finite=False)  # shape (p,p), Lower triangular
            v_F = solve_triangular(L_p, F.T, lower=True, check_finite=False) 
        for d in range(D):
            dR_d = dR_dtheta[:,:,d] * Correl
            term1 = alpha_R.T @ dR_d @ alpha_R
            R_inv_dR = solve_triangular(L.T, solve_triangular(L, dR_d, lower=True, check_finite=False),
                                        check_finite=False)  # solves R X = dR_d
            #R_inv_dR = cho_solve((L, True), dR_d)   # solves R X = dR_d 
            term2 = np.trace(R_inv_dR)
            if self.REML:
                Aux = solve_triangular(L.T, solve_triangular(L, (v_F.T @ v_F) @ R_inv_dR, lower=True, check_finite=False),
                                            check_finite=False) 
                term2 = term2 - np.trace(Aux)
            grad[d] = -0.5 * (term1 - term2)
        for w in range(W):
            dR_w = dR_dzeta[:,w]
            # exploit the diagonal structure of dR for noise parameters
            term1 = np.sum(dR_w * alpha_R[:,0]**2)   
            Linv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)   # shape (N,N)
            diag_Rinv = np.sum(Linv**2, axis=0)
            term2 = np.sum(diag_Rinv * dR_w)
            if self.REML:
                v_F_Rinv = v_F @ Linv.T @ Linv
                diag_aux = np.sum(v_F_Rinv**2, axis=0)
                term2 = term2 - np.sum(diag_aux * dR_w)
            grad[D+w] = -0.5 * (term1 - term2)
        return grad
    
    def grad_noisy_known(self, params):
        """ Computes the Gradient NLML for the noisy case with "known" noise, params contains (theta, sigma2),
            stores in the GP class the MLE values for beta.
        Args: 
            params ((D+1,) array): Array containing all parameters to be optimized by scipy
        Returns: grad ((D+1,) array): Gradient values.
        """
        kernel = self.kernel
        D = self.D
        theta, sigma2 = params[:D], params[D:]
        kernel.theta = theta
        kernel.sigma2 = sigma2
        F = self.Fmatrix    # shape (N,p)
        N = self.N
        Covar = kernel.covariance(self.X, self.X)
        K = Covar + self.jitter*np.eye(N) + np.diag(self.noise[:,0])   # shape (N,N), SPD
        L = cholesky(K, lower=True, check_finite=False)   # shape (N,N), Lower triangular
        u_F = solve_triangular(L, F, lower=True, check_finite=False)        # L^-1 F
        u_z = solve_triangular(L, self.z, lower=True, check_finite=False)   # L^-1 z
        FKinvF = u_F.T @ u_F         # shape (p,p)
        beta = np.linalg.solve(FKinvF, u_F.T @ u_z)  # MLE formula for beta, shape (p,1)
        omega = u_z - u_F @ beta                     # L^-1 (z - F @ beta)
        self.mean.beta = beta      # Store MLE values 
        self.Lmatrix = L
        alpha_K = solve_triangular(L.T, omega, check_finite=False)  # shape (N,1)
        dK_dtheta = self.kernel.derivative_factor(self.X, self.X)   # shape (N,N,D)
        grad = np.zeros(D+1)   # shape (D+1,)
        for d in range(D):
            dK_d = dK_dtheta[:,:,d] * Covar
            term1 = alpha_K.T @ dK_d @ alpha_K
            K_inv_dK = solve_triangular(L.T, solve_triangular(L, dK_d, lower=True, check_finite=False),
                                        check_finite=False)  # solves K X = dK_d 
            #K_inv_dK = cho_solve((L, True), dK_d)   
            term2 = np.trace(K_inv_dK)
            grad[d] = -0.5 * (term1 - term2)
        dK_sig = Covar / sigma2
        term1 = alpha_K.T @ dK_sig @ alpha_K
        K_inv_dK = solve_triangular(L.T, solve_triangular(L, dK_sig, lower=True, check_finite=False),
                                        check_finite=False)    
        term2 = np.trace(K_inv_dK)
        grad[-1] = -0.5 * (term1 - term2)
        #grad[-1] = -0.5 * (omega.T @ omega - N) / sigma2
        if self.REML:
            print('REML is not implemented for this case, MLE is used instead.')
        return grad
           
    def LOO_CV_prediction(self):
        """ Computes the "fast leave-one-out cross-validation formulas" for the predictive mean and variance when the 
            mean function is already known (i.e., no LOO-CV optimization scheme) from the saved parameters.
        Returns: ((N,1) array, (N,1) array) : Predictive LOO-CV mean and variance.
        """
        beta = self.mean.beta
        L, F = self.Lmatrix, self.Fmatrix   # nlml_fun must have been called at least once, so that Lmatrix =/= None.
        # compute K^-1 (z - F beta)
        numerator = cho_solve((L, True), self.z - F @ beta, check_finite=False)   # shape (N,1)
        Linv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)   # shape (N,N)
        diag_Kinv = np.sum(Linv**2, axis=0).reshape(-1,1)  # shape (N,1)
        mean_loo = self.z - numerator / diag_Kinv          # shape (N,1)
        var_loo = 1. / diag_Kinv                           # shape (N,1)
        return mean_loo, var_loo        

    def run_minimize(self, x0, idx=None, fun=None, jac=None, bounds=None):
        """ Runs one instance of the minimize routine
        Args:
            x0 ((n_params,) array): Array containing the parameters to be optimized
            idx (None or int, optional): Run index. Defaults to None.
            fun (callable, optional): Objective function. Defaults to None.
            jac (callable, optional): Gradient of the objective function. Defaults to None.
        Returns:
            (scipy res): Scipy object that resumes the optimization run.
        """
        res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bounds)
        msg = f"Start {idx}: fun_value={res.fun:.4f}, nb_fun_eval={res.nfev}, nb_grad_eval={res.njev}"
        return res, msg 

    # Multi-start optimization, sequential or parallel
    def fit(self, n_starts=10, n_cores=1, jitter=1e-10, sampling_method='LHS maximin', rng_seed=0, verbose=False):
        """ Fits / Optimizes the GP parameters using the training data.
        Args:
            n_starts (int, optional): Number of starts in the multi-start. Defaults to 10.
            n_cores (int, optional): Number of CPU cores used in parallel computing, one means sequential. Defaults to 1.
            jitter (float): Adds a small jitter / nugget effect on the corr or cov matrix (avoid ill-conditioning)
            sampling_method (str): Sampling method for the initial points: 'uniform', 'LHS simple' or 'LHS maximin'.
            rng_seed (int): Sets the RNG seed for the random sampling of the initial points.
            verbose (bool, optional): Prints the results of the optimization steps. Defaults to False.
        Returns:
            best, nlml_value (scipy res, float): Returns the result of the best optimization in terms of loss function, and NLML
                                                 value when all parameters are "optimal".
        """
        self.jitter = jitter
        bounds = self.bounds
        nlml_fun = self.nlml_fun
        grad_nlml = self.grad_nlml
        if not self.random_init:
            if n_starts == 1:
                initial_points_list = [self.init_point]
            else:
                # Random initial points within the bounds (Warning: the random initial points depend on the value n_starts!)
                initial_points = sample_initial_points(bounds, n_samples=n_starts-1, method=sampling_method, random_state=rng_seed)
                initial_points_list = [self.init_point] + [x for x in initial_points]
        else:
            if n_starts == 1:
                initial_points = sample_initial_points(bounds, n_samples=n_starts, method="uniform", random_state=rng_seed)
            else:
                initial_points = sample_initial_points(bounds, n_samples=n_starts, method=sampling_method, random_state=rng_seed)
            initial_points_list = [x for x in initial_points]
        # Number of cores
        n_available = multiprocessing.cpu_count()    # Number of (logical) CPU cores available
        n_cores_used = min(n_cores, n_available)
        if n_cores_used < n_cores:
            print(f"Warning: {n_cores} cores were called, but only {n_cores_used} are used.")
        # Optimization
        if n_cores_used > 1:  # Parallel (warning: BLAS is already multi-threaded by default, it may lead to different results)
            if verbose:
                print('Parallel optimization')
            results = Parallel(n_jobs=n_cores, backend="loky")(
                delayed(self.run_minimize)(x0, idx=i, fun=nlml_fun, jac=grad_nlml, bounds=bounds) 
                for i, x0 in enumerate(initial_points_list))
            if verbose:
                for _, msg in results:
                    print(msg)
            results = [r[0] for r in results]
        else:            # Sequential
            if verbose:
                print('Sequential optimization')
            results = []
            for i, x0 in enumerate(initial_points_list):
                res, msg = self.run_minimize(x0, idx=i, fun=nlml_fun, jac=grad_nlml, bounds=bounds)
                results.append(res)
                if verbose:
                    print(msg)
        self.all_results = results
        best = min(results, key=lambda r: r.fun)
        # Stores the parameters (by computing the nlml value at the optimal parameter values) and the "Cov" matrix.
        nlml_value = nlml_fun(best.x)
        self.fitted = True
        return best, nlml_value   

    def prediction(self, X_test, only_diag_cov=False, Inv=False):
        """ Computes the mean and variance prediction with the kriging equations, obtained after Gaussian conditioning. 
        Args:
            X_test ((N_test,D) array): Input test set.
            only_diag_cov (bool, optional): If True, computes only the diagonal of the covariance matrix prediction,
            useful when N_test is quite large (> 10^3). Defaults to False.
            Inv (bool, optional): If True, uses direct inversion of the covariance matrix, never do that in practice,
            this option is present only to present the analytical formula. Defaults to False.
        Returns: mu ((N_test,1) array), cov ((N_test,N_test) or (N_test,1) array) : predictive mean and cov values.
        """
        assert self.X.shape[1] == X_test.shape[1], f"The test input set must have the right dimension, got {X_test.shape[1]}, expected {self.X.shape[1]}."
        m_train = self.Fmatrix @ self.mean.beta    # shape (N,1)
        m_test = self.mean.mean_function(X_test)   # shape (N_test,1)
        K_cross = self.kernel.covariance(self.X, X_test)   # shape (N,N_test)
        L = self.Lmatrix    # shape (N,N)
        if only_diag_cov :
            K_test = self.kernel.sigma2 * np.ones((X_test.shape[0],1))  # Only for stationary kernels (ok for RBF and Matérn)
        else :                                                   # shape (N_test,1)
            K_test = self.kernel.covariance(X_test, X_test)      # shape (N_test,N_test)
        if Inv : 
            K_inv = np.linalg.inv(L @ L.T)
            mu = m_test + K_cross.T @ K_inv @ (self.z - m_train)
            cov = K_test - K_cross.T @ K_inv @ K_cross           # shape (N_test,N_test)
        else : 
            alpha = cho_solve((L, True), self.z - m_train, check_finite=False)
            mu = m_test + K_cross.T @ alpha   # shape (N_test,1)
            u = solve_triangular(L, K_cross, lower=True, check_finite=False)   # shape (N,N_test)
            if only_diag_cov :
                cov = K_test - np.sum(u**2, axis=0).reshape(-1,1)  # shape (N_test,1)
            else :
                cov = K_test - u.T @ u
        return mu, cov   
