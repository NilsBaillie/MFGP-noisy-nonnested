from utils import *
from Gaussian_Process import *

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


######################################################################
###################################################################### 

#################### Recursive MF-GP AR(1) model  ####################

######################################################################
######################################################################

class Multifi_GP_AR1():
    def __init__(self, LF_GP, HF_GP, scale, index_com_dim=None, random_init=False):
        """ Initiates a general multi-fidelity GP AR(1) model (recursive formulation) from two GPR objects and a parametric 
            scaling function. The LF-GP is supposed to be already optimized by MLE (and NSGA2 + k-means if wanted). Depending 
            on the situation, the HF parameters can be optimized in several different ways: if we are in the "simplified case", 
            the HF-GP is fitted by MLE as for the LF-GP, otherwise, we use the alternative EM-based optimization of the likelihood.
        Args:
            LF_GP (GPR): Low-fidelity GP with its mean, kernel, training points and noise type.
            HF_GP (GPR): High-fidelity GP with its mean, kernel, training points and noise type.
            scale (linear predictor): Scaling factor, linear predictor with a function basis (parameter: beta_rho) 
            index_com_dim (None or list): List containing the indices of the common input dimensions between LF and HF codes, if None, 
            it is assumed that all dimensions are common (i.e., no projection from the HF input space to the LF one).
            random_init (bool): If True, the EM is initialized randomly.
        """
        self.LF_GP = LF_GP
        self.HF_GP = HF_GP
        self.scale = scale
        self.simplified_case = LF_GP.noise is None 
        self.random_initial_points = []   # saves the random initial points in the multi-start 
        self.random_init = random_init
        self.n_safeguard = 0
        # Check LF-HF compatibility:
        N_L, D_L = LF_GP.X.shape
        N_H, D_H = HF_GP.X.shape
        q = scale.p
        if D_L > D_H:
            raise ValueError(f'The HF input dimension ({D_H}) must be greater than or equal to the LF input dimension ({D_L}).')
        # Check scaling factor:
        scale_Xdim = scale.expected_Xdim
        if scale_Xdim is not None:
            if D_H != scale_Xdim:
                raise ValueError(f'Scaling factor not compatible with HF input dimension: got {D_H}, expected {scale_Xdim}.')
        if N_H < q:
            raise ValueError(f'MLE undefined: number of HF data samples ({N_H}) must be greater than or equal to the size of the scale parameter vector ({scale.p}).')
        if HF_GP.REML:
            if N_H < q + HF_GP.mean.p + 1:
                raise ValueError(f'REML undefined: number of HF data samples ({N_H}) must be greater than the total size of the mean parameter vector ({scale.p+HF_GP.mean.p}).')
            HF_GP.dim_residuals = N_H - q - HF_GP.mean.p
        self.scale.Fmatrix = self.scale.basis(HF_GP.X)  # equivalent of the Fmatrix for the "rho", denoted "G_L".
        
        if D_L == D_H:
            # Check nested design sets:
            is_subset, membership, indices, Nesting_Threshold = Nested_inputs_hypothesis(HF_GP.X, LF_GP.X, tol=1e-10)
            self.simplified_case = is_subset and (LF_GP.noise is None)
            self.index_com_dim = range(D_L)
        else: 
            self.simplified_case = False
            self.index_com_dim = index_com_dim
            if np.size(index_com_dim) != D_L:
                raise ValueError(f'The number of indices of common input dimension must be equal to the LF input dimension: got {np.size(index_com_dim)}, expected {D_L}.')
        
        if self.simplified_case:
            z_L_selected = LF_GP.z[indices]
            G_Lprod = self.scale.Fmatrix * (z_L_selected @ np.ones((q,1)).T)
            self.Hmatrix = np.hstack([G_Lprod,  HF_GP.Fmatrix])  # shape (N_H, q+p_H)
            # Define the "alternative" HF-GP: in the simplified case, the algorithm, i.e., optimization and prediction
            # is (almost) identical to the one used in the single-fidelity case, but applied on a particular GP whose mean
            # function is the combination of the scale term and the mean of the "initial" HF-GP and whose kernel
            # has an equal correlation structure to the one of the "initial" HF-GP.
            self.alt_HF_GP = GPR(Sum_basis(scale, HF_GP.mean), HF_GP.kernel, HF_GP.X, HF_GP.z, HF_GP.noise, HF_GP.eta_fn)
            self.alt_HF_GP.Fmatrix = self.Hmatrix
        else : 
            if not self.random_init:
                if HF_GP.mean.beta is None:
                    raise ValueError(f"No value attributed to the HF mean function, needed for EM ! (beta_H=None)") 
                if scale.beta is None:
                    raise ValueError(f"No value attributed to the scaling factor, needed for EM ! (beta_rho=None)") 
                if HF_GP.kernel.sigma2 is None:
                    raise ValueError(f"No value attributed to the HF kernel variance, needed for EM ! (sigma2_H=None)") 
            self.EM_Hmatrix = None  # shape (N_H, q+p_H), equivalent of the Hmatrix, updated at each EM iteration
            self.EM_Sigma = None    # must be of shape (N_H, N_H) in the EM iterations, updated at each EM iteration
            # Need the values of the predictive LF mean and covariance at the HF training input points.
            meanLF_X_H, covLF_X_H = self.LF_GP.prediction((self.HF_GP.X)[:, self.index_com_dim], only_diag_cov=False, Inv=False)
            self.meanLF_X_H = meanLF_X_H   # shape (N_H, 1)
            self.covLF_X_H = covLF_X_H     # shape (N_H, N_H)
            self.meanAR_X_H = None  # shape (N_H,1), a priori mean of the MF-GP 
            self.covAR_X_H = None   # shape (N_H,N_H,N_H), a priori covariance of the MF-GP
            if HF_GP.REML:                  # shape (N_H, q+p_H)
                self.REML_EM_Hmatrix = np.hstack((self.scale.Fmatrix*(meanLF_X_H @ np.ones((q,1)).T), HF_GP.Fmatrix))    

            if HF_GP.noise is None:
                self.EM_fun = self.EM_proxy_noiseless
                self.EM_grad = self.grad_EM_noiseless
            elif HF_GP.noise_type == str:
                self.EM_fun = self.EM_proxy_noisy_unknown
                self.EM_grad = self.grad_EM_noisy_unknown
            else:
                self.EM_fun = self.EM_proxy_noisy_known
                self.EM_grad = self.grad_EM_noisy_known
            self.log_lik_values = []  # stores the (restricted) log-likelihood values at each EM step (should be increasing)

    def number_of_optimized_parameters(self):
        """ Returns: int: Number of all parameters to be optimized.
        """
        q = self.scale.p
        p1 = self.LF_GP.number_of_optimized_parameters()
        p2 = self.HF_GP.number_of_optimized_parameters()
        return q + p1 + p2

    def __str__(self):
        """ Prints the all the parameter values in the class with their corresponding names, LF and HF. 
        """
        gp_L, gp_H = self.LF_GP, self.HF_GP
        # LF parameters
        beta, sigma2, theta, zeta = gp_L.get_all_params()
        if gp_L.noise_type != str:
            param_names = ["LF_mean_params", "LF_kernel_variance", "LF_length_scale"]
            param_values = [beta, sigma2, theta]
        if gp_L.noise_type == str and gp_L.eta_fn.zeta.size == 1:
            param_names = ["LF_mean_params", "LF_kernel_variance", "LF_length_scale", "LF_eta_fn_params", "LF_noise_variance"]
            param_values = [beta, sigma2, theta, zeta, np.exp(zeta)*sigma2]
        if gp_L.noise_type == str and gp_L.eta_fn.zeta.size > 1:
            param_names = ["LF_mean_params", "LF_kernel_variance", "LF_length_scale", "LF_eta_fn_params"]
            param_values = [beta, sigma2, theta, zeta]
        # HF parameters
        beta, sigma2, theta, zeta = gp_H.get_all_params()
        param_names = param_names + ['scaling_params']
        param_values = param_values + [self.scale.beta]
        if gp_H.noise_type != str:
            param_names = param_names + ["HF_mean_params", "HF_kernel_variance", "HF_length_scale"]
            param_values = param_values + [beta, sigma2, theta]
        if gp_H.noise_type == str and gp_H.eta_fn.zeta.size == 1:
            param_names = param_names + ["HF_mean_params", "HF_kernel_variance", "HF_length_scale", "HF_eta_fn_params", "HF_noise_variance"]
            param_values = param_values + [beta, sigma2, theta, zeta, np.exp(zeta)*sigma2]
        if gp_H.noise_type == str and gp_H.eta_fn.zeta.size > 1:
            param_names = param_names + ["HF_mean_params", "HF_kernel_variance", "HF_length_scale", "HF_eta_fn_params"]
            param_values = param_values + [beta, sigma2, theta, zeta]
        header = f"{'Parameter Name':<20}{'Value':<20}"
        separator = "-" * len(header)
        rows = []
        for name, value in zip(param_names, param_values):
            value_str = str(value)
            lines = value_str.split("\n")
            rows.append(f"{name:<20}{lines[0]:<20}")
            rows.extend(f"{'':<20}{line:<20}" for line in lines[1:])
        return "\n".join([header, separator] + rows)


    def EM_proxy_noiseless(self, params):
        """ Computes the Negative Qfunction value for the noiseless case, params only contains theta, the updated EM values
            for beta_H, beta_rho, sigma2_H are saved in the class. Supposes that EM_Hmatrix and EM_Sigma are not None.
        Args: params ((D,) array): Array containing all parameters to be optimized by scipy.
        Returns: (float): Negative EM_Q value.
        """
        gp_H = self.HF_GP
        kernel = gp_H.kernel
        kernel.theta = params
        N_H = gp_H.N
        q, p_H = self.scale.p, gp_H.mean.p
        dres = gp_H.dim_residuals
        G_L = self.scale.Fmatrix    # shape (N_H,q)
        R = kernel.correlation(gp_H.X, gp_H.X) + self.jitter * np.eye(N_H) # shape (N_H,N_H), SPD
        L = cholesky(R, lower=True, check_finite=False)  # shape (N_H,N_H), Lower triangular
        R_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N_H), lower=True, check_finite=False),
                                    lower=False, check_finite=False)
        if gp_H.REML:
            H_H = self.REML_EM_Hmatrix
            HRinvH = H_H.T @ R_inv @ H_H    # shape (q+p_H, q+p_H)
            R_inv = R_inv @ (np.eye(N_H) - H_H @ np.linalg.inv(HRinvH) @ H_H.T @ R_inv) + self.jitter * np.eye(N_H)
        H_Ht = self.EM_Hmatrix   # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma  # shape (N_H, N_H)
        T_11 = G_L.T @ (R_inv * Sigma_t) @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        HRinvH_plus_T = H_Ht.T @ (R_inv @ H_Ht) + Tmatrix
        beta_rho_H_t1 = np.linalg.solve(HRinvH_plus_T, H_Ht.T @ (R_inv @ gp_H.z))   # EM update formula, shape (q+p_H,1)
        z_center = gp_H.z - H_Ht @ beta_rho_H_t1
        sigma2_t1 = (z_center.T @ (R_inv @ z_center) + beta_rho_H_t1.T @ (Tmatrix @ beta_rho_H_t1))[0] / dres  # EM update, shape (1,)
        beta_rho, beta_H = beta_rho_H_t1[0:q], beta_rho_H_t1[q:]
        gp_H.mean.beta = beta_H
        self.scale.beta = beta_rho
        gp_H.kernel.sigma2 = sigma2_t1
        gp_H.Lmatrix = np.sqrt(sigma2_t1) * L
        log_det_R = 2 * np.sum(np.log(np.diag(L)))
        nQfct = 0.5 * (dres*np.log(sigma2_t1) + log_det_R + dres*np.log(2*np.pi) + dres)
        if gp_H.REML:
            #_, log_det_FRinvF = np.linalg.slogdet(HRinvH_plus_T)
            _, log_det_FRinvF = np.linalg.slogdet(HRinvH)
            nQfct = nQfct + 0.5 * log_det_FRinvF
        return nQfct[0]
    
    def EM_proxy_noisy_unknown(self, params):
        """ Computes the Negative Qfunction value for the noisy case with optimization of the noise parameters, 
            params contains (theta, zeta), the updated EM values for beta_H, beta_rho, sigma2_H are saved in the class. 
            Supposes that EM_Hmatrix and EM_Sigma are not None.
        Args: params ((D+len_z,) array): Array containing all parameters to be optimized by scipy.
        Returns: float: Negative EM_Q value.
        """
        gp_H = self.HF_GP
        kernel = gp_H.kernel
        D = gp_H.D
        theta, zeta = params[:D], params[D:]
        kernel.theta = theta
        gp_H.eta_fn.zeta = zeta
        N_H = gp_H.N
        q, p_H = self.scale.p, gp_H.mean.p
        dres = gp_H.dim_residuals
        G_L = self.scale.Fmatrix    # shape (N_H,q)
        R = kernel.correlation(gp_H.X, gp_H.X) + self.jitter * np.eye(N_H) + np.diag(gp_H.eta_fn.compute(gp_H.X)[:,0]) # shape (N_H,N_H), SPD
        L = cholesky(R, lower=True, check_finite=False)  # shape (N_H,N_H), Lower triangular
        R_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N_H), lower=True, check_finite=False),
                                lower=False, check_finite=False)
        if gp_H.REML:
            H_H = self.REML_EM_Hmatrix
            HRinvH = H_H.T @ R_inv @ H_H    # shape (q+p_H, q+p_H)
            R_inv = R_inv @ (np.eye(N_H) - H_H @ np.linalg.inv(HRinvH) @ H_H.T @ R_inv) + self.jitter * np.eye(N_H)
        H_Ht = self.EM_Hmatrix   # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma  # shape (N_H, N_H)
        T_11 = G_L.T @ (R_inv * Sigma_t) @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        HRinvH_plus_T = H_Ht.T @ (R_inv @ H_Ht) + Tmatrix
        beta_rho_H_t1 = np.linalg.solve(HRinvH_plus_T, H_Ht.T @ (R_inv @ gp_H.z))   # EM update formula, shape (q+p_H,1)
        z_center = gp_H.z - H_Ht @ beta_rho_H_t1
        sigma2_t1 = (z_center.T @ (R_inv @ z_center) + beta_rho_H_t1.T @ (Tmatrix @ beta_rho_H_t1))[0] / dres  # EM update, shape (1,)
        beta_rho, beta_H = beta_rho_H_t1[0:q], beta_rho_H_t1[q:]
        gp_H.mean.beta = beta_H
        self.scale.beta = beta_rho
        gp_H.kernel.sigma2 = sigma2_t1
        gp_H.Lmatrix = np.sqrt(sigma2_t1) * L
        log_det_R = 2 * np.sum(np.log(np.diag(L)))
        nQfct = 0.5 * (dres*np.log(sigma2_t1) + log_det_R + dres*np.log(2*np.pi) + dres)
        if gp_H.REML:
            #_, log_det_FRinvF = np.linalg.slogdet(HRinvH_plus_T)
            _, log_det_FRinvF = np.linalg.slogdet(HRinvH)
            nQfct = nQfct + 0.5 * log_det_FRinvF
        return nQfct[0]
  

    def EM_proxy_noisy_known(self, params):
        """ Computes the Negative Qfunction value for the noisy case with "known" noise, params contains (theta, sigma2), 
            the updated EM values for beta_H, beta_rho are saved in the class. 
            Supposes that EM_Hmatrix and EM_Sigma are not None.
        Args: params ((D+1,) array): Array containing all parameters to be optimized by scipy.
        Returns: float: Negative EM_Q value.
        """
        gp_H = self.HF_GP
        kernel = gp_H.kernel
        D = gp_H.D
        theta, sigma2 = params[:D], params[D:]
        kernel.theta = theta
        kernel.sigma2 = sigma2
        N_H = gp_H.N
        q, p_H = self.scale.p, gp_H.mean.p
        G_L = self.scale.Fmatrix    # shape (N_H, q)
        K = kernel.covariance(gp_H.X, gp_H.X) + self.jitter * np.eye(N_H) + np.diag(gp_H.noise[:,0]) # shape (N_H,N_H), SPD
        L = cholesky(K, lower=True, check_finite=False)  # shape (N_H,N_H), Lower triangular
        K_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N_H), lower=True, check_finite=False),
                                lower=False, check_finite=False)
        H_Ht = self.EM_Hmatrix      # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma     # shape (N_H, N_H)
        T_11 = G_L.T @ (K_inv * Sigma_t) @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        HKinvH_plus_T = H_Ht.T @ (K_inv @ H_Ht) + Tmatrix
        beta_rho_H_t1 = np.linalg.solve(HKinvH_plus_T, H_Ht.T @ (K_inv @ gp_H.z))   # EM update formula, shape (q+p_H,1)
        z_center = gp_H.z - H_Ht @ beta_rho_H_t1
        beta_rho, beta_H = beta_rho_H_t1[0:q], beta_rho_H_t1[q:]
        gp_H.mean.beta = beta_H
        self.scale.beta = beta_rho
        gp_H.Lmatrix = L
        log_det_K = 2 * np.sum(np.log(np.diag(L)))
        scalar_prod = (z_center.T @ (K_inv @ z_center) + beta_rho_H_t1.T @ (Tmatrix @ beta_rho_H_t1))[0]
        nQfct = 0.5 * (scalar_prod + log_det_K + N_H*np.log(2*np.pi))
        return nQfct[0]   

    def estim_with_diag_corr(self):
        """ Computes the estimator/update formulas for beta_{rho,H} and sigma2 when the correlation matrix is equal to the 
            identity matrix (up to a multiplicative constant).
        Returns: (p,1) array, (1,) array: beta and sigma2 values.
        """
        gp_H = self.HF_GP
        N_H = gp_H.N
        q, p_H = self.scale.p, gp_H.mean.p
        G_L = self.scale.Fmatrix    # shape (N_H,q)
        H_Ht = self.EM_Hmatrix   # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma  # shape (N_H, N_H)
        T_11 = G_L.T @ (np.eye(N_H) * Sigma_t) @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        beta = np.linalg.solve(H_Ht.T @ H_Ht + Tmatrix, H_Ht.T @ gp_H.z)
        z_center = gp_H.z - H_Ht @ beta
        sigma2 = (z_center.T @ z_center + beta.T @ Tmatrix @ beta) / N_H
        return beta, sigma2[0]

    def grad_EM_noiseless(self, params):
        """ Computes the Gradient of the Negative Qfunction value for the noiseless case, params only contains theta, 
        the updated EM values for beta_H, beta_rho, sigma2_H are saved in the class. 
        Supposes that EM_Hmatrix and EM_Sigma are not None.
        Args: params ((D,) array): Array containing all parameters to be optimized by scipy.
        Returns: grad ((D,) array): Gradient values.
        """
        gp_H = self.HF_GP
        kernel = gp_H.kernel
        kernel.theta = params
        N_H = gp_H.N
        q, p_H = self.scale.p, gp_H.mean.p
        dres = gp_H.dim_residuals
        G_L = self.scale.Fmatrix    # shape (N_H,q)
        Correl = kernel.correlation(gp_H.X, gp_H.X)
        R = Correl + self.jitter * np.eye(N_H) # shape (N_H,N_H), SPD
        L = cholesky(R, lower=True, check_finite=False)  # shape (N_H,N_H), Lower triangular
        R_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N_H), lower=True, check_finite=False),
                                lower=False, check_finite=False)
        if gp_H.REML:
            H_H = self.REML_EM_Hmatrix
            HRinvH = H_H.T @ R_inv @ H_H    # shape (q+p_H, q+p_H)
            R_inv = R_inv @ (np.eye(N_H) - H_H @ np.linalg.inv(HRinvH) @ H_H.T @ R_inv) + self.jitter * np.eye(N_H)
        H_Ht = self.EM_Hmatrix   # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma  # shape (N_H, N_H)
        T_11 = G_L.T @ (R_inv * Sigma_t) @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        HRinvH_plus_T = H_Ht.T @ (R_inv @ H_Ht) + Tmatrix
        beta_rho_H_t1 = np.linalg.solve(HRinvH_plus_T, H_Ht.T @ (R_inv @ gp_H.z))   # EM update formula, shape (q+p_H,1)
        z_center = gp_H.z - H_Ht @ beta_rho_H_t1
        R_inv_z_center = R_inv @ z_center
        sigma2_t1 = (z_center.T @ R_inv_z_center + beta_rho_H_t1.T @ (Tmatrix @ beta_rho_H_t1))[0] / dres  # EM update, shape (1,)
        beta_rho, beta_H = beta_rho_H_t1[0:q], beta_rho_H_t1[q:]
        gp_H.mean.beta = beta_H
        self.scale.beta = beta_rho
        gp_H.kernel.sigma2 = sigma2_t1
        gp_H.Lmatrix = np.sqrt(sigma2_t1) * L
        alpha_R = R_inv_z_center / np.sqrt(sigma2_t1)
        dR = gp_H.kernel.derivative_factor(gp_H.X, gp_H.X)   # shape (N_H,N_H,D)
        D = gp_H.X.shape[1]
        grad = np.zeros(D)
        rho_vect = self.scale.mean_function(gp_H.X)
        # if gp_H.REML:
        #     L_qp = cholesky(HRinvH_plus_T, lower=True, check_finite=False)  # shape (q+p_H,q+p_H), Lower triangular
        #     v_H = solve_triangular(L_qp, H_Ht.T, lower=True, check_finite=False) 
        for d in range(D):
            dR_d = dR[:,:,d] * Correl
            term1 = alpha_R.T @ (dR_d @ alpha_R)
            R_inv_dR = R_inv @ dR_d
            term2 = np.trace(R_inv_dR)
            Had = (R_inv_dR @ R_inv) * Sigma_t
            # if gp_H.REML:
            #     Aux = solve_triangular(L.T, solve_triangular(L, (v_H.T @ v_H) @ R_inv_dR, lower=True, check_finite=False),
            #                                 check_finite=False) 
            #     term2 = term2 - np.trace(Aux)
            #     dT_dtheta = np.block([[-G_L.T @ Had @ G_L,    np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
            #                           [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
            #     Aux2 = solve_triangular(L_qp.T, solve_triangular(L_qp, dT_dtheta, lower=True, check_finite=False),
            #                                 check_finite=False)
            #     term2 = term2 + np.trace(Aux2)
            term3 = rho_vect.T @ Had @ rho_vect / sigma2_t1
            grad[d] = -0.5 * (term1 - term2 + term3)
        return grad
    
    def grad_EM_noisy_unknown(self, params):
        """ Computes the Gradient of the Negative Qfunction value for the noisy case with optimization of the noise parameters, 
            params contains (theta, zeta), the updated EM values for beta_H, beta_rho, sigma2_H are saved in the class. 
            Supposes that EM_Hmatrix and EM_Sigma are not None.
        Args: params ((D+len_z,) array): Array containing all parameters to be optimized by scipy.
        Returns: grad ((D+len_z,) array): Gradient values.
        """
        gp_H = self.HF_GP
        kernel = gp_H.kernel
        D = gp_H.D
        theta, zeta = params[:D], params[D:]
        kernel.theta = theta
        gp_H.eta_fn.zeta = zeta
        N_H = gp_H.N
        W = zeta.shape[0]
        q, p_H = self.scale.p, gp_H.mean.p
        dres = gp_H.dim_residuals
        G_L = self.scale.Fmatrix    # shape (N_H,q)
        Correl = kernel.correlation(gp_H.X, gp_H.X)
        R = Correl + self.jitter * np.eye(N_H) + np.diag(gp_H.eta_fn.compute(gp_H.X)[:,0]) # shape (N_H,N_H), SPD
        L = cholesky(R, lower=True, check_finite=False)  # shape (N_H,N_H), Lower triangular
        R_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N_H), lower=True, check_finite=False),
                                lower=False, check_finite=False)
        if gp_H.REML:
            H_H = self.REML_EM_Hmatrix
            HRinvH = H_H.T @ R_inv @ H_H    # shape (q+p_H, q+p_H)
            R_inv = R_inv @ (np.eye(N_H) - H_H @ np.linalg.inv(HRinvH) @ H_H.T @ R_inv) + self.jitter * np.eye(N_H)
        H_Ht = self.EM_Hmatrix   # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma  # shape (N_H, N_H)
        T_11 = G_L.T @ (R_inv * Sigma_t) @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        HRinvH_plus_T = H_Ht.T @ (R_inv @ H_Ht) + Tmatrix
        beta_rho_H_t1 = np.linalg.solve(HRinvH_plus_T, H_Ht.T @ (R_inv @ gp_H.z))   # EM update formula, shape (q+p_H,1)
        z_center = gp_H.z - H_Ht @ beta_rho_H_t1
        R_inv_z_center = R_inv @ z_center
        sigma2_t1 = (z_center.T @ R_inv_z_center + beta_rho_H_t1.T @ (Tmatrix @ beta_rho_H_t1))[0] / dres  # EM update, shape (1,)
        beta_rho, beta_H = beta_rho_H_t1[0:q], beta_rho_H_t1[q:]
        gp_H.mean.beta = beta_H
        self.scale.beta = beta_rho
        gp_H.kernel.sigma2 = sigma2_t1
        gp_H.Lmatrix = np.sqrt(sigma2_t1) * L
        alpha_R = R_inv_z_center / np.sqrt(sigma2_t1)
        dR_dtheta = kernel.derivative_factor(gp_H.X, gp_H.X)   # shape (N_H,N_H,D)
        dR_dzeta = gp_H.eta_fn.gradient(gp_H.X)    # shape (N_H,W)
        grad = np.zeros(D+W)
        rho_vect = self.scale.mean_function(gp_H.X)
        # if gp_H.REML:
        #     L_qp = cholesky(HRinvH_plus_T, lower=True, check_finite=False)  # shape (q+p_H,q+p_H), Lower triangular
        #     v_H = solve_triangular(L_qp, H_Ht.T, lower=True, check_finite=False) 
        for d in range(D):
            dR_d = dR_dtheta[:,:,d] * Correl
            term1 = alpha_R.T @ (dR_d @ alpha_R)
            R_inv_dR = R_inv @ dR_d
            term2 = np.trace(R_inv_dR)
            Had = (R_inv_dR @ R_inv) * Sigma_t
            # if gp_H.REML:
            #     Aux = solve_triangular(L.T, solve_triangular(L, (v_H.T @ v_H) @ R_inv_dR, lower=True, check_finite=False),
            #                                 check_finite=False) 
            #     term2 = term2 - np.trace(Aux)
            #     dT_dtheta = np.block([[-G_L.T @ Had @ G_L,    np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
            #                           [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
            #     Aux2 = solve_triangular(L_qp.T, solve_triangular(L_qp, dT_dtheta, lower=True, check_finite=False),
            #                                 check_finite=False)
            #     term2 = term2 + np.trace(Aux2)
            term3 = rho_vect.T @ Had @ rho_vect / sigma2_t1
            grad[d] = -0.5 * (term1 - term2 + term3)
        for w in range(W):
            dR_w = dR_dzeta[:,w]
            # exploit the diagonal structure of dR for noise parameters
            term1 = np.sum(dR_w * alpha_R[:,0]**2)
            R_inv_dR = R_inv @ np.diag(dR_w)
            term2 = np.trace(R_inv_dR)
            Had = (R_inv_dR @ R_inv) * Sigma_t
            # if gp_H.REML:
            #     Aux = solve_triangular(L.T, solve_triangular(L, (v_H.T @ v_H) @ R_inv_dR, lower=True, check_finite=False),
            #                                 check_finite=False) 
            #     term2 = term2 - np.trace(Aux)
            #     dT_dtheta = np.block([[-G_L.T @ Had @ G_L,    np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
            #                           [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
            #     Aux2 = solve_triangular(L_qp.T, solve_triangular(L_qp, dT_dtheta, lower=True, check_finite=False),
            #                                 check_finite=False)
            #     term2 = term2 + np.trace(Aux2)
            term3 = rho_vect.T @ Had @ rho_vect / sigma2_t1
            grad[D+w] = -0.5 * (term1 - term2 + term3)
        return grad
    
    def grad_EM_noisy_known(self, params):
        """ Computes the Gradient of the Negative Qfunction value for the noisy case with "known" noise, 
            params contains (theta, sigma2), the updated EM values for beta_H, beta_rho are saved in the class. 
            Supposes that EM_Hmatrix and EM_Sigma are not None.
        Args: params ((D+1,) array): Array containing all parameters to be optimized by scipy.
        Returns: grad ((D+1,) array): Gradient values.
        """
        gp_H = self.HF_GP
        kernel = gp_H.kernel
        D = len(kernel.theta)
        theta, sigma2 = params[:D], params[D:]
        kernel.theta = theta
        kernel.sigma2 = sigma2
        N_H = len(gp_H.z)
        q, p_H = self.scale.p, gp_H.mean.p
        G_L = self.scale.Fmatrix    # shape (N_H, q)
        Covar = kernel.covariance(gp_H.X, gp_H.X)
        K = Covar + self.jitter * np.eye(N_H) + np.diag(gp_H.noise[:,0]) # shape (N_H,N_H), SPD
        L = cholesky(K, lower=True, check_finite=False)  # shape (N_H,N_H), Lower triangular
        K_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N_H), lower=True, check_finite=False),
                                lower=False, check_finite=False)
        H_Ht = self.EM_Hmatrix      # shape (N_H, q+p_H)
        Sigma_t = self.EM_Sigma     # shape (N_H, N_H)
        K_inv_Sigma = K_inv * Sigma_t
        T_11 = G_L.T @ K_inv_Sigma @ G_L    # shape (q,q)
        Tmatrix = np.block([[T_11,                  np.zeros((q,p_H))],      # shape (q+p_H,q+p_H)
                            [np.zeros((p_H,q)),     np.zeros((p_H,p_H))]])
        HKinvH_plus_T = H_Ht.T @ (K_inv @ H_Ht) + Tmatrix
        beta_rho_H_t1 = np.linalg.solve(HKinvH_plus_T, H_Ht.T @ (K_inv @ gp_H.z))   # EM update formula, shape (q+p_H,1)
        beta_rho, beta_H = beta_rho_H_t1[0:q], beta_rho_H_t1[q:]
        gp_H.mean.beta = beta_H
        self.scale.beta = beta_rho
        gp_H.Lmatrix = L
        z_center = gp_H.z - H_Ht @ beta_rho_H_t1
        alpha_K = K_inv @ z_center
        dK_dtheta = gp_H.kernel.derivative_factor(gp_H.X, gp_H.X)   # shape (N_H,N_H,D)
        grad = np.zeros(D+1)
        rho_vect = self.scale.mean_function(gp_H.X)
        for d in range(D):
            dK_d = dK_dtheta[:,:,d] * Covar
            term1 = alpha_K.T @ dK_d @ alpha_K
            K_inv_dK = K_inv @ dK_d   
            term2 = np.trace(K_inv_dK)
            term3 = rho_vect.T @ ((K_inv_dK @ K_inv) * Sigma_t) @ rho_vect
            grad[d] = -0.5 * (term1 - term2 + term3)
        dK_sig = Covar / sigma2
        term1 = alpha_K.T @ dK_sig @ alpha_K
        K_inv_dK = K_inv @ dK_sig   
        term2 = np.trace(K_inv_dK)
        term3 = rho_vect.T @ ((K_inv_dK @ K_inv) * Sigma_t) @ rho_vect
        grad[-1] = -0.5 * (term1 - term2 + term3)
        #grad[-1] = -0.5 * (z_center.T @ alpha_K - N_H + rho_vect.T @ (K_inv_Sigma) @ rho_vect) / sigma2
        return grad   

    def LOO_CV_prediction(self):
        """ Computes the "fast leave-one-out cross-validation formulas" for the predictive mean and variance when the 
            mean function is already known (i.e., no LOO-CV optimization scheme) from the saved parameters.
        Returns: ((N_H,1) array, (N_H,1) array) : Predictive LOO-CV mean and variance.
        """
        if self.simplified_case:
            mean_loo, var_loo = self.alt_HF_GP.LOO_CV_prediction()
        # Compute m_AR, K_AR:
        gp_H = self.HF_GP
        L_H = gp_H.Lmatrix
        rho_tr = self.scale.mean_function(gp_H.X)
        m_AR = rho_tr * self.meanLF_X_H + gp_H.mean.mean_function(gp_H.X)   # shape (N_H, 1)
        K_AR = (rho_tr @ rho_tr.T) * self.covLF_X_H + L_H @ L_H.T           # shape (N_H, N_H)
        L_AR = cholesky(K_AR, lower=True, check_finite=False)
        # Uses the "fast LOO-CV" formulas for the MF framework with non-nested and noisy design sets.
        numerator = cho_solve((L_AR, True), gp_H.z - m_AR, check_finite=False)   # shape (N_H,1)
        Linv = solve_triangular(L_AR, np.eye(L_AR.shape[0]), lower=True, check_finite=False)   # shape (N_H,N_H)
        diag_Kinv = np.sum(Linv**2, axis=0).reshape(-1,1)  # shape (N_H,1)
        mean_loo = gp_H.z - numerator / diag_Kinv          # shape (N_H,1)
        var_loo = 1. / diag_Kinv                           # shape (N_H,1)
        return mean_loo, var_loo

    def run_minimize(self, x0, idx=None, fun=None, jac=None, bounds=None):
        """ Runs one instance of the minimize routine
        Args:
            x0 ((n_params,) array): Array containing the parameters to be optimized.
            idx (None or int, optional): Run index. Defaults to None.
            fun (callable, optional): Objective function. Defaults to None.
            jac (callable, optional): Gradient of the objective function. Defaults to None.
        Returns:
            (scipy res): Scipy object that resumes the optimization run.
        """
        res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bounds)
        msg = f"Start {idx}: fun_value={res.fun:.4f}, nb_fun_eval={res.nfev}, nb_grad_eval={res.njev}"
        return res, msg

    def fit(self, EM_iter=10, n_starts=10, n_cores=1, jitter=1e-10, sampling_method='LHS maximin', 
            rng_seed=0, verbose=False, tol=None):
        """ Fits / Optimizes the HF-GP parameters using the training data.
        Args:
            EM_iter (int): Number of EM iterations.
            n_starts (int, optional): Number of starts in the multi-start. Defaults to 10.
            n_cores (int, optional): Number of CPU cores used in parallel computing, 1 means sequential. Defaults to 1.
            jitter (float): Adds a small jitter / nugget effect on the corr or cov matrix (avoid ill-conditioning)
            sampling_method (str): Sampling method for the initial points: 'uniform', 'LHS simple' or 'LHS maximin'.
            rng_seed (int): Sets the RNG seed for the random sampling of the initial points.
            verbose (bool, optional): Prints the results of the optimization steps. Defaults to False.
            tol (float, optional): Tolerance on the relative log-likelihood difference. Defaults to None.
        Returns:
            (scipy res, float): Returns the result of the best optimization in terms of the loss function, and the HF NLL
                                value when all parameters are "optimal" (general case, otherwise the NLML value).
        """
        self.jitter = jitter
        gp_H = self.HF_GP
        if self.simplified_case:
            best, nlml_value = self.alt_HF_GP.fit(n_starts, n_cores, jitter, sampling_method, rng_seed, verbose)
            q = self.scale.p
            beta_rhoH = self.alt_HF_GP.mean.beta
            beta_rho, beta_H = beta_rhoH[0:q], beta_rhoH[q:]
            self.scale.beta = beta_rho
            self.HF_GP.mean.beta = beta_H
            return best, nlml_value
        # Number of cores
        n_available = multiprocessing.cpu_count()    # Number of (logical) CPU cores available
        n_cores_used = min(n_cores, n_available)
        if n_cores_used < n_cores:
            print(f"Warning: {n_cores} cores were called, but only {n_cores_used} are used.")
        if verbose:
            if n_cores_used > 1:
                print('Parallel optimization')
            else:
                print('Sequential optimization')
        bounds = gp_H.bounds
        # Compute the EM utility matrices for the initial values of the HF parameters. 
        G_L = self.scale.Fmatrix                                  # shape (N_H, q)
        N_H, p_H, q = gp_H.N, gp_H.mean.p, self.scale.p
        if self.random_init:  # Only works if zeta is a scalar (i.e. homoscedastic case)
            D_H = gp_H.X.shape[1]
            random_init_bounds = [(-60, 10)] + gp_H.theta_bounds + [(-40., 10.)] + [(-50., 50.)]*p_H + [(-10., 10.)]*q
            rand_init_points = sample_initial_points(random_init_bounds, n_samples=1, method="uniform", random_state=rng_seed)[0,:]
            self.random_initial_points.append(rand_init_points)
            # gp_H.kernel.sigma2 = np.exp(rand_init_points[0])
            # gp_H.kernel.theta = rand_init_points[1:(D_H+1)]
            gp_H.eta_fn.zeta = rand_init_points[(D_H+1):(D_H+2)]
            # gp_H.mean.beta = rand_init_points[(D_H+2):(D_H+2+p_H)].reshape(-1,1)
            # self.scale.beta = rand_init_points[(D_H+2+p_H):].reshape(-1,1)
        K_0 = gp_H.kernel.covariance(gp_H.X, gp_H.X) + jitter * np.eye(N_H) + np.diag(gp_H.get_noise_variance()[:,0])  # shape (N_H, N_H)
        beta_0 = gp_H.mean.beta                                   # shape (p_H, 1)
        rho_0 = self.scale.mean_function(gp_H.X)                  # shape (N_H, 1)
        mean_Z = rho_0 * self.meanLF_X_H + gp_H.Fmatrix @ beta_0  # shape (N_H, 1)
        Sig_YZ = (np.ones((N_H,1)) @ rho_0.T) * self.covLF_X_H    # shape (N_H, N_H)
        Sig_ZZ = (rho_0 @ rho_0.T) * self.covLF_X_H + K_0         # shape (N_H, N_H)
        # Compute (for MLE):
        # Sig_ZZ_inv = np.linalg.inv(Sig_ZZ)
        # Mu = self.meanLF_X_H + Sig_YZ @ Sig_ZZ_inv @ (gp_H.z - mean_Z)
        # Sigma = self.covLF_X_H - Sig_YZ @ Sig_ZZ_inv @ Sig_YZ.T
        L_ZZ = cholesky(Sig_ZZ, lower=True, check_finite=False)
        u = solve_triangular(L_ZZ, Sig_YZ.T, lower=True, check_finite=False)  # = L_ZZ^-1 Sig_YZ.T
        if not gp_H.REML:
            u_lik = solve_triangular(L_ZZ, gp_H.z - mean_Z, lower=True, check_finite=False)
            alpha = solve_triangular(L_ZZ.T, u_lik, check_finite=False)  
            Mu = self.meanLF_X_H + Sig_YZ @ alpha                 # shape (N_H, 1) 
            Sigma = self.covLF_X_H - u.T @ u                      # shape (N_H, N_H)
            # Compute the initial value of the log-likelihood on Z_H, it should increase with the EM iterations
            log_lik_Z = -0.5 * (2*np.sum(np.log(np.diag(L_ZZ))) + N_H*np.log(2*np.pi) + u_lik.T @ u_lik)[0,0]
        else:
            z_H = gp_H.z
            u_lik = solve_triangular(L_ZZ, z_H, lower=True, check_finite=False)
            u_HH = solve_triangular(L_ZZ, self.REML_EM_Hmatrix, lower=True, check_finite=False)
            L_HSH = cholesky(u_HH.T @ u_HH, lower=True, check_finite=False)
            v = solve_triangular(L_HSH, u_HH.T @ u, lower=True, check_finite=False)
            v_lik = solve_triangular(L_HSH, u_HH.T @ u_lik, lower=True, check_finite=False)
            log_det_Sig = 2*np.sum(np.log(np.diag(L_ZZ)))
            log_det_HSH = 2*np.sum(np.log(np.diag(L_HSH)))
            Mu = self.meanLF_X_H + (u.T @ u_lik - v.T @ v_lik + jitter * Sig_YZ @ z_H)    # shape (N_H, 1) 
            Sigma = self.covLF_X_H - (u.T @ u - v.T @ v + jitter * Sig_YZ @ Sig_YZ.T)     # shape (N_H, N_H)
            # Compute the restricted log-likelihood on Z_H, it should increase with the EM iterations
            log_lik_Z = -0.5*(log_det_Sig+log_det_HSH+(N_H-p_H-q)*np.log(2*np.pi)+(u_lik.T@u_lik-v_lik.T@v_lik+jitter*z_H.T@z_H))[0,0]

        self.EM_Hmatrix = np.hstack((G_L * (Mu @ np.ones((q,1)).T), gp_H.Fmatrix))    # shape (N_H, q+p_H)
        self.EM_Sigma = Sigma           # shape (N_H, N_H)
        self.log_lik_values.append(log_lik_Z)
        if verbose:
            print(f'log-likelihood value: {log_lik_Z}')
        if not gp_H.random_init:
            if n_starts == 1:
                    initial_points_list = [gp_H.init_point]
            else:
                # Random initial points within the bounds 
                initial_points = sample_initial_points(bounds, n_samples=n_starts-1, method=sampling_method, random_state=rng_seed)
                initial_points_list = [gp_H.init_point] + [x for x in initial_points]
        else:
            if n_starts == 1:
                initial_points = sample_initial_points(bounds, n_samples=n_starts, method="uniform", random_state=rng_seed)
            else:
                initial_points = sample_initial_points(bounds, n_samples=n_starts, method=sampling_method, random_state=rng_seed)
            initial_points_list = [x for x in initial_points]
        self.random_initial_points.append(initial_points_list)
        ##### EM iterations #####
        for t in range(1, EM_iter + 1):
            if verbose:
                print(f"EM iteration n°{t}")
            x_current = gp_H.get_numerical_params().copy()
            # Optimization
            if n_cores_used > 1:  # Parallel
                results = Parallel(n_jobs=n_cores, backend="loky")(
                    delayed(self.run_minimize)(x0, idx=i, fun=self.EM_fun, jac=self.EM_grad, bounds=bounds) 
                    for i, x0 in enumerate(initial_points_list))
                if verbose:
                    for _, msg in results:
                        print(msg)
                results = [r[0] for r in results]
            else:            # Sequential
                results = []
                for i, x0 in enumerate(initial_points_list):
                    res, msg = self.run_minimize(x0, idx=i, fun=self.EM_fun, jac=self.EM_grad, bounds=bounds)
                    results.append(res)
                    if verbose:
                        print(msg)
            self.all_results = results
            finite = [r for r in results if np.isfinite(r.fun)]
            best = min(finite, key=lambda r: r.fun) if finite else None
            f_current = self.EM_fun(x_current)
            if best is None or best.fun > f_current:   # Imposes monotonicity of the log-likelihood
                x_next = x_current
                self.n_safeguard += 1
            else:
                x_next = best.x
            EM_value = self.EM_fun(x_next)  # returns -Q_alt(xi^(t+1); xi^(t)) and stores the updated parameters
            # Update EM utility matrices
            L_t = gp_H.Lmatrix
            beta_t = gp_H.mean.beta                                          # shape (p_H, 1)
            rho_t = self.scale.mean_function(gp_H.X)                         # shape (N_H, 1)
            mean_Z = rho_t * self.meanLF_X_H + gp_H.Fmatrix @ beta_t         # shape (N_H, 1)
            Sig_YZ = (np.ones((N_H,1)) @ rho_t.T) * self.covLF_X_H           # shape (N_H, N_H)
            Sig_ZZ = (rho_t @ rho_t.T) * self.covLF_X_H + L_t @ L_t.T        # shape (N_H, N_H)
            # Compute (for MLE):
            # Sig_ZZ_inv = np.linalg.inv(Sig_ZZ)
            # Mu = self.meanLF_X_H + Sig_YZ @ Sig_ZZ_inv @ (gp_H.z - mean_Z)
            # Sigma = self.covLF_X_H - Sig_YZ @ Sig_ZZ_inv @ Sig_YZ.T
            L_ZZ = cholesky(Sig_ZZ, lower=True, check_finite=False)
            u = solve_triangular(L_ZZ, Sig_YZ.T, lower=True, check_finite=False)  # = L_ZZ^-1 Sig_YZ.T
            if not gp_H.REML:
                u_lik = solve_triangular(L_ZZ, gp_H.z - mean_Z, lower=True, check_finite=False)
                alpha = solve_triangular(L_ZZ.T, u_lik, check_finite=False)  
                Mu = self.meanLF_X_H + Sig_YZ @ alpha                 # shape (N_H, 1) 
                Sigma = self.covLF_X_H - u.T @ u                      # shape (N_H, N_H)
                # Compute the initial value of the log-likelihood on Z_H, it should increase with the EM iterations
                log_lik_Z = -0.5 * (2*np.sum(np.log(np.diag(L_ZZ))) + N_H*np.log(2*np.pi) + u_lik.T @ u_lik)[0,0]
            else:
                z_H = gp_H.z
                u_lik = solve_triangular(L_ZZ, z_H, lower=True, check_finite=False)
                u_HH = solve_triangular(L_ZZ, self.REML_EM_Hmatrix, lower=True, check_finite=False)
                L_HSH = cholesky(u_HH.T @ u_HH, lower=True, check_finite=False)
                v = solve_triangular(L_HSH, u_HH.T @ u, lower=True, check_finite=False)
                v_lik = solve_triangular(L_HSH, u_HH.T @ u_lik, lower=True, check_finite=False)
                log_det_Sig = 2*np.sum(np.log(np.diag(L_ZZ)))
                log_det_HSH = 2*np.sum(np.log(np.diag(L_HSH)))
                Mu = self.meanLF_X_H + (u.T @ u_lik - v.T @ v_lik + jitter * Sig_YZ @ z_H)    # shape (N_H, 1) 
                Sigma = self.covLF_X_H - (u.T @ u - v.T @ v + jitter * Sig_YZ @ Sig_YZ.T)     # shape (N_H, N_H)
                # Compute the restricted log-likelihood on Z_H, it should increase with the EM iterations
                log_lik_Z = -0.5*(log_det_Sig+log_det_HSH+(N_H-p_H-q)*np.log(2*np.pi)+(u_lik.T@u_lik-v_lik.T@v_lik+jitter*z_H.T@z_H))[0,0]
                # Update only if it is not the last iteration
            if t != EM_iter:
                self.EM_Hmatrix = np.hstack((G_L * (Mu @ np.ones((q,1)).T), gp_H.Fmatrix))    # shape (N_H, q+p_H)
                self.EM_Sigma = Sigma           # shape (N_H, N_H)
            self.log_lik_values.append(log_lik_Z)
            if verbose:
                print(f'log-likelihood value: {log_lik_Z}')
            # Random initial points within the bounds 
            initial_points = sample_initial_points(bounds, n_samples=n_starts, method=sampling_method, random_state=rng_seed)
            initial_points_list = [x for x in initial_points]
            self.random_initial_points.append(initial_points_list)
            # Stopping criterion
            if tol is not None and len(self.log_lik_values) >= 2:
                l_prev = self.log_lik_values[-2]
                l_curr = self.log_lik_values[-1]
                EM_gap = np.abs(l_curr-l_prev) / max(np.abs(l_prev), np.abs(l_curr), 1.)  # Similar to Scipy's L-BFGS-B
                if EM_gap < tol:
                    if verbose:
                        print(f"EM stopped after iteration {t}.")
                    break
        self.HF_GP.fitted = True
        return best, -log_lik_Z

    def prediction(self, X_test, only_diag_cov=False, Inv=False):
        """ Computes the predictive mean and variance with the general co-kriging equations, obtained after 
            Gaussian conditioning. In the general case, we introduce many intermediate vectors and matrices in order
            to alleviate the notational complexity, they are deleted when no longer needed to avoid a high-memory cost. 
        Args:
            X_test ((N_test,D) array): Input test set
            only_diag_cov (bool, optional): If True, computes only the diagonal of the covariance matrix prediction,
            useful when N_test is quite large (> 10^3). Defaults to False.
            Inv (bool, optional): If True, uses direct inversion of the covariance matrix, never do that in practice,
            this option is present only to present the analytical formula. Defaults to False.
        Returns: 
            mu_L, mu_H ((N_test,1) arrays): LF and HF predictive means at X_test.
            cov_L, cov_H ((N_test,N_test) or (N_test,1) arrays): LF and HF predictive covariances at X_test.
        """
        gp_L, gp_H = self.LF_GP, self.HF_GP
        mu_L_test, cov_L_test = gp_L.prediction(X_test[:, self.index_com_dim], only_diag_cov, Inv)
        K_H_cross = gp_H.kernel.covariance(gp_H.X, X_test)   # shape (N_H, N_test)
        rho_test = self.scale.mean_function(X_test)          # shape (N_test, 1)
        if self.simplified_case:
            if only_diag_cov :
                K_test = gp_H.kernel.sigma2 * np.ones((X_test.shape[0], 1))  # Only for stationary kernels (ok for RBF and Matérn)
            else :                                                   # shape (N_test,1)
                K_test = gp_H.kernel.covariance(X_test, X_test)      # shape (N_test,N_test)
            H_H = self.alt_HF_GP.Fmatrix
            L = self.alt_HF_GP.Lmatrix
            alpha = cho_solve((L, True), gp_H.z - H_H @ self.alt_HF_GP.mean.beta, check_finite=False)
            mu_delta = gp_H.mean.mean_function(X_test) + K_H_cross.T @ alpha   # shape (N_test,1)
            mu_H = rho_test * mu_L_test + mu_delta                             # shape (N_test,1)
            u = solve_triangular(L, K_H_cross, lower=True, check_finite=False)   # shape (N_H,N_test)
            if only_diag_cov :
                cov_delta = K_test - np.sum(u**2, axis=0).reshape(-1,1)  # shape (N_test,1)
                cov_H = rho_test**2 * cov_L_test + cov_delta             # shape (N_test,1)
            else :
                cov_delta = K_test - u.T @ u                               # shape (N_test,N_test)
                cov_H = (rho_test @ rho_test.T) * cov_L_test + cov_delta   # shape (N_test,N_test)
        else :
            # Goal: compute K_AR(X,X') = rho(X)rho(X').T \odot V_{Y_L}(X,X') + K_H(X,X') and
            # m_AR(X) = rho(X) \odot m_{Y_L}(X) + f(X).T @ beta at (X_test,X_test), (X_H,X_H) and (X_H,X_test) in order
            # to obtain the general covariance matrix. We proceed in several steps:
            ### Compute V_{Y_L}(X_H, X_test)
            # K_L(X_L, X_test), shape (N_L, N_test),  duplicate kernel computation (LF kriging on X_test)
            K_L_cross_L = gp_L.kernel.covariance(gp_L.X, X_test[:, self.index_com_dim])   
            L_L = gp_L.Lmatrix
            # (K_L(X_L,X_L) + LF noise)^-1 @ K_L(X_L, X_test), shape (N_L, N_test)
            K_L_inv_times_K_L_cross_L = solve_triangular(L_L.T, solve_triangular(L_L, K_L_cross_L, lower=True), lower=False)
            del K_L_cross_L
            # K_L(X_H, X_test), shape (N_H, N_test)
            K_L_cross_H = gp_L.kernel.covariance((gp_H.X)[:, self.index_com_dim], X_test[:, self.index_com_dim])
            # K_L(X_L, X_H), shape (N_L, N_H), duplicate kernel computation (LF kriging on X_H)
            K_L_tr_LH = gp_L.kernel.covariance(gp_L.X, (gp_H.X)[:, self.index_com_dim]) 
            cov_L_cross = K_L_cross_H - K_L_tr_LH.T @ K_L_inv_times_K_L_cross_L  # Target matrix, shape (N_H, N_test)
            del K_L_cross_H, K_L_tr_LH, K_L_inv_times_K_L_cross_L
            ### Compute m_AR, K_AR on X_H
            rho_tr_H = self.scale.mean_function(gp_H.X)   # shape (N_H, 1)
            m_tr_AR = rho_tr_H * self.meanLF_X_H + gp_H.mean.mean_function(gp_H.X)   # shape (N_H, 1)
            self.meanAR_X_H = m_tr_AR
            L_H = gp_H.Lmatrix    # shape (N_H, N_H)
            K_tr_AR = (rho_tr_H @ rho_tr_H.T) * self.covLF_X_H + L_H @ L_H.T + self.jitter*np.eye(len(gp_H.z))  # shape (N_H, N_H)
            self.covAR_X_H = K_tr_AR
            Chol = cholesky(K_tr_AR, lower=True, check_finite=False)
            del K_tr_AR
            alpha = cho_solve((Chol, True), gp_H.z - m_tr_AR, check_finite=False)  # shape (N_H, N_test)   
            del m_tr_AR
            ### Compute m_AR, K_AR on X_test
            m_test = rho_test * mu_L_test + gp_H.mean.mean_function(X_test)   # shape (N_test,1)
            K_cross_AR = (rho_tr_H @ rho_test.T) * cov_L_cross + gp_H.kernel.covariance(gp_H.X, X_test)  # shape (N_H,N_test)
            del rho_tr_H, cov_L_cross
            mu_H = m_test + K_cross_AR.T @ alpha  # shape (N_test,1)
            del m_test, alpha
            # K_AR(X_test, X_test): shape (N_test,N_test) or (N_test,1) if only_diag_cov
            if only_diag_cov :     # Only for stationary kernels !
                K_test_AR = rho_test**2 * cov_L_test + gp_H.kernel.sigma2*np.ones((X_test.shape[0],1))   # shape (N_test,1)
            else :
                K_test_AR = (rho_test @ rho_test.T) * cov_L_test + gp_H.kernel.covariance(X_test, X_test)
            del rho_test
            u = solve_triangular(Chol, K_cross_AR, lower=True, check_finite=False)  # shape (N_H,N_test)
            del K_cross_AR, Chol
            if only_diag_cov :
                cov_H = K_test_AR - np.sum(u**2, axis=0).reshape(-1,1)   # shape (N_test,1)
            else :
                cov_H = K_test_AR - u.T @ u   # = K_test_AR - K_cross_AR.T @ inv(K_tr_AR) @ K_cross_AR
        return mu_L_test, cov_L_test, mu_H, cov_H


#########################################################################
######################################################################### 

#################### Non-recursive MF-GP AR(1) model ####################

#########################################################################
#########################################################################



class Multifi_GP_AR1_KO():
    def __init__(self, LF_GP, HF_GP, scale, random_init=False):
        """ Initiates a general multi-fidelity GP AR(1) model (non-recursive formulation, or Kennedy-O'Hagan formulation) 
            from two GPR objects and a parametric scaling value (must be Constant_basis). All parameters (LF and HF) are optimized jointly 
            using the likelihood, only the general case is considered here (hom. noise terms and non-nested DoEs). 
            Zetas are equal to the log noise variance (not log(eta)).
        Args:
            LF_GP (GPR): Low-fidelity GP with its mean, kernel, training points (noise fun not used).
            HF_GP (GPR): High-fidelity GP with its mean, kernel, training points (noise fun not used).
            scale (linear predictor): Scaling factor, only scalar for now (parameter: beta_rho).
            random_init (bool): If True, all initial values for the parameters are random in the multi-start.
        """
        self.LF_GP = LF_GP
        self.HF_GP = HF_GP
        self.scale = scale
        self.random_initial_points = []   # saves the random initial points in the multi-start 
        self.random_init = random_init
        LF_GP.random_init = random_init
        HF_GP.random_init = random_init
        # Check LF-HF compatibility:
        N_L, D_L = LF_GP.X.shape
        N_H, D_H = HF_GP.X.shape
        self.z_LH = np.block([[LF_GP.z], 
                              [HF_GP.z]])
        q = scale.p  # always equal to 1.
        if D_L > D_H:
            raise ValueError(f'The HF input dimension ({D_H}) must be greater than or equal to the LF input dimension ({D_L}).')
        # Check scaling factor:
        scale_Xdim = scale.expected_Xdim
        # if scale_Xdim is not None:
        #     if D_H != scale_Xdim:
        #         raise ValueError(f'Scaling factor not compatible with HF input dimension: got {D_H}, expected {scale_Xdim}.')
        
        self.nlml_fun = self.nlml_fun
        self.grad_fun = self.grad_fun
        # sigma2_L, sigma_H, theta_L, theta_H, zeta_L, zeta_H, beta_rho bounds:
        self.bounds = [(-60, 10)] + [(-60, 10)] + LF_GP.theta_bounds + HF_GP.theta_bounds + [(-40., 10.)] + [(-40., 10.)] + [(-10., 10.)]
        if not random_init:
            if scale.beta is None:
                raise ValueError(f"No value attributed to the scaling factor, needed for the optimization ! (beta_rho=None)") 
            if HF_GP.kernel.sigma2 is None:
                raise ValueError(f"No value attributed to the HF kernel variance, needed for the optimization ! (sigma2_H=None)") 
            self.init_point = np.concatenate([np.log(LF_GP.kernel.sigma2), np.log(HF_GP.kernel.sigma2), LF_GP.kernel.theta, 
                                            HF_GP.kernel.theta, LF_GP.eta_fn.zeta, HF_GP.eta_fn.zeta, scale.beta[0]])

    def number_of_optimized_parameters(self):
        """ Returns: int: Number of all parameters to be optimized.
        """
        q = self.scale.p
        p1 = self.LF_GP.mean.p
        p2 = self.HF_GP.mean.p
        return q + p1 + p2 + 4

    def __str__(self):
        """ Prints the all the parameter values in the class with their corresponding names, LF and HF. 
        """
        gp_L, gp_H = self.LF_GP, self.HF_GP
        # LF parameters
        beta, sigma2, theta, zeta = gp_L.get_all_params()
        param_names = ["LF_mean_params", "LF_kernel_variance", "LF_length_scale", "LF_noise_variance"]
        param_values = [beta, sigma2, theta, np.exp(zeta)]
        # HF parameters
        beta, sigma2, theta, zeta = gp_H.get_all_params()
        param_names = param_names + ['scaling_params']
        param_values = param_values + [self.scale.beta]
        param_names = param_names + ["HF_mean_params", "HF_kernel_variance", "HF_length_scale", "HF_noise_variance"]
        param_values = param_values + [beta, sigma2, theta, np.exp(zeta)]
        header = f"{'Parameter Name':<20}{'Value':<20}"
        separator = "-" * len(header)
        rows = []
        for name, value in zip(param_names, param_values):
            value_str = str(value)
            lines = value_str.split("\n")
            rows.append(f"{name:<20}{lines[0]:<20}")
            rows.extend(f"{'':<20}{line:<20}" for line in lines[1:])
        return "\n".join([header, separator] + rows)

    def nlml_fun(self, params):
        """ Computes the NLML, stores in the GP classes the MLE values for (beta_L, beta_H). 
            params contains log sigma2_L, log sigma2_H, theta_L, theta_H, zeta_L, zeta_H, beta_rho.
        Args: 
            params ((2+2D+3,) array): Array containing all parameters to be optimized by scipy.
        Returns: float: NLML value.
        """
        gp_L, gp_H = self.LF_GP, self.HF_GP
        kernel_L, kernel_H = gp_L.kernel, gp_H.kernel
        D_L, D_H = gp_L.D, gp_H.D
        sigma2_L, sigma2_H = np.exp(params[0]), np.exp(params[1])
        theta_L, theta_H = params[2:(2+D_L)], params[(2+D_L):(2+D_L+D_H)]
        zeta_L, zeta_H, beta_rho = params[-3], params[-2], params[-1]
        kernel_L.theta = theta_L
        kernel_H.theta = theta_H
        kernel_L.sigma2 = sigma2_L
        kernel_H.sigma2 = sigma2_H
        gp_L.eta_fn.zeta = zeta_L
        gp_H.eta_fn.zeta = zeta_H
        self.scale.beta = beta_rho.reshape(-1,1)
        N_L, N_H = gp_L.N, gp_H.N
        X_L, X_H = gp_L.X, gp_H.X
        p_L, p_H = gp_L.mean.p, gp_H.mean.p
        F_L, F_H = gp_L.Fmatrix, gp_H.Fmatrix              # shapes (N_L,p_L) and (N_H,p_H)
        F_LH = np.block([[F_L,  np.zeros((N_L, p_H))],     # shape (N_L+N_H,p_L+p_H)
                         [beta_rho*gp_L.mean.basis(X_H), F_H]])
        K_L = kernel_L.covariance(X_L, X_L)
        K_LH = kernel_L.covariance(X_L, X_H)
        K_H = kernel_H.covariance(X_H, X_H) 
        V_H = (beta_rho**2)*kernel_L.covariance(X_H, X_H) + K_H + np.exp(zeta_H)*np.eye(N_H)
        V_LH = np.block([[K_L+np.exp(zeta_L)*np.eye(N_L), K_LH*beta_rho],
                         [beta_rho*K_LH.T,                V_H]]) + self.jitter * np.eye(N_L+N_H)
        L = cholesky(V_LH, lower=True, check_finite=False)    # shape (N_L+N_H,N_L+N_H), Lower triangular
        u_F = solve_triangular(L, F_LH, lower=True, check_finite=False)          # = L^-1 F
        u_z = solve_triangular(L, self.z_LH, lower=True, check_finite=False)     # = L^-1 z
        FVinvF = u_F.T @ u_F   # shape (p_L+p_H,p_L+p_H)
        beta_LH = np.linalg.solve(FVinvF, u_F.T @ u_z)  # MLE formula for betas, shape (p_L+p_H,1)
        omega = u_z - u_F @ beta_LH   # L^-1 (z - F @ beta)
        gp_L.mean.beta = beta_LH[0:p_L]      # Store MLE values 
        gp_H.mean.beta = beta_LH[p_L:]     
        self.Lmatrix = L
        log_det_V = 2 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * ((omega.T @ omega)[0] + log_det_V + (N_L+N_H)*np.log(2*np.pi))
        return nlml[0]
    
    def grad_fun(self, params):
        """ Computes the gradient, stores in the GP classes the MLE values for (beta_L, beta_H). 
            params contains log sigma2_L, log sigma2_H, theta_L, theta_H, zeta_L, zeta_H, beta_rho.
        Args: 
            params ((2+2D+3,) array): Array containing all parameters to be optimized by scipy.
        Returns: (2+2D+3,) array): Gradient values.
        """
        gp_L, gp_H = self.LF_GP, self.HF_GP
        kernel_L, kernel_H = gp_L.kernel, gp_H.kernel
        D_L, D_H = gp_L.D, gp_H.D
        sigma2_L, sigma2_H = np.exp(params[0]), np.exp(params[1])
        theta_L, theta_H = params[2:(2+D_L)], params[(2+D_L):(2+D_L+D_H)]
        zeta_L, zeta_H, beta_rho = params[-3], params[-2], params[-1]
        kernel_L.theta = theta_L
        kernel_H.theta = theta_H
        kernel_L.sigma2 = sigma2_L
        kernel_H.sigma2 = sigma2_H
        gp_L.eta_fn.zeta = zeta_L
        gp_H.eta_fn.zeta = zeta_H
        self.scale.beta = beta_rho.reshape(-1,1)
        N_L, N_H = gp_L.N, gp_H.N
        X_L, X_H = gp_L.X, gp_H.X
        p_L, p_H = gp_L.mean.p, gp_H.mean.p
        F_L, F_H = gp_L.Fmatrix, gp_H.Fmatrix              # shapes (N_L,p_L) and (N_H,p_H)
        F_LH = np.block([[F_L,  np.zeros((N_L, p_H))],     # shape (N_L+N_H,p_L+p_H)
                         [beta_rho*gp_L.mean.basis(X_H), F_H]])
        K_L = kernel_L.covariance(X_L, X_L)
        K_LH = kernel_L.covariance(X_L, X_H)
        K_H = kernel_H.covariance(X_H, X_H) 
        K_L_X_H = kernel_L.covariance(X_H, X_H)
        V_H = (beta_rho**2)*K_L_X_H + K_H + np.exp(zeta_H)*np.eye(N_H)
        V_LH = np.block([[K_L+np.exp(zeta_L)*np.eye(N_L), K_LH*beta_rho],
                         [beta_rho*K_LH.T,                V_H]]) + self.jitter * np.eye(N_L+N_H)
        L = cholesky(V_LH, lower=True, check_finite=False)    # shape (N_L+N_H,N_L+N_H), Lower triangular
        u_F = solve_triangular(L, F_LH, lower=True, check_finite=False)          # = L^-1 F
        u_z = solve_triangular(L, self.z_LH, lower=True, check_finite=False)     # = L^-1 z
        FVinvF = u_F.T @ u_F   # shape (p_L+p_H,p_L+p_H)
        beta_LH = np.linalg.solve(FVinvF, u_F.T @ u_z)  # MLE formula for betas, shape (p_L+p_H,1)
        omega = u_z - u_F @ beta_LH   # L^-1 (z - F @ beta)
        gp_L.mean.beta = beta_LH[0:p_L]      # Store MLE values 
        gp_H.mean.beta = beta_LH[p_L:]     
        self.Lmatrix = L
        grad = np.zeros(2+D_L+D_H+3)
        alpha_V = solve_triangular(L.T, omega, check_finite=False)  # shape (N,1)
        dV_dthetaL_LL = kernel_L.derivative_factor(X_L, X_L)   # shape (N_L,N_L,D_L)
        dV_dthetaL_LH = kernel_L.derivative_factor(X_L, X_H)   # shape (N_L,N_H,D_L)
        dV_dthetaL_HH = kernel_L.derivative_factor(X_H, X_H)   # shape (N_H,N_H,D_L)
        dV_dthetaH = kernel_H.derivative_factor(X_H, X_H)   # shape (N_H,N_H,D_H)
        # log Sigma2_L
        dV_LH = np.block([[K_L, K_LH*beta_rho], [beta_rho*K_LH.T, (beta_rho**2)*K_L_X_H]])
        term1 = alpha_V.T @ dV_LH @ alpha_V
        V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)     
        term2 = np.trace(V_inv_dV)
        grad[0] = -0.5 * (term1 - term2)
        # log Sigma2_H
        dV_LH = np.block([[np.zeros((N_L, N_L)), np.zeros((N_L, N_H))], [np.zeros((N_H, N_L)), K_H]])
        term1 = alpha_V.T @ dV_LH @ alpha_V
        V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)     
        term2 = np.trace(V_inv_dV)
        grad[1] = -0.5 * (term1 - term2)
        # Theta_L
        for d in range(D_L):
            dV1 = dV_dthetaL_LL[:,:,d] * K_L
            dV2 = dV_dthetaL_LH[:,:,d] * K_LH
            dV3 = dV_dthetaL_HH[:,:,d] * K_L_X_H
            dV_LH = np.block([[dV1, dV2*beta_rho], [beta_rho*dV2.T, (beta_rho**2)*dV3]])
            term1 = alpha_V.T @ dV_LH @ alpha_V
            V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)  # solves V X = dV_LH    
            term2 = np.trace(V_inv_dV)
            grad[2+d] = -0.5 * (term1 - term2)
        # Theta_H
        for d in range(D_H):
            dV = dV_dthetaH[:,:,d] * K_H
            dV_LH = np.block([[np.zeros((N_L, N_L)), np.zeros((N_L, N_H))], [np.zeros((N_H, N_L)), dV]])
            term1 = alpha_V.T @ dV_LH @ alpha_V
            V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)      
            term2 = np.trace(V_inv_dV)
            grad[2+D_L+d] = -0.5 * (term1 - term2)
        # Zeta_L
        dV_LH = np.block([[np.exp(zeta_L)*np.eye(N_L), np.zeros((N_L, N_H))],
                          [np.zeros((N_H, N_L)), np.zeros((N_H, N_H))]])
        term1 = alpha_V.T @ dV_LH @ alpha_V
        V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)     
        term2 = np.trace(V_inv_dV)
        grad[2+D_L+D_H] = -0.5 * (term1 - term2)
        # Zeta_H
        dV_LH = np.block([[np.zeros((N_L, N_L)), np.zeros((N_L, N_H))],
                          [np.zeros((N_H, N_L)), np.exp(zeta_H)*np.eye(N_H)]])
        term1 = alpha_V.T @ dV_LH @ alpha_V
        V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)     
        term2 = np.trace(V_inv_dV)
        grad[2+D_L+D_H+1] = -0.5 * (term1 - term2)
        # Beta_rho
        dV_LH = np.block([[np.zeros((N_L, N_L)), K_LH], 
                          [K_LH.T, 2*beta_rho*K_L_X_H]])
        term1 = alpha_V.T @ dV_LH @ alpha_V
        V_inv_dV = solve_triangular(L.T, solve_triangular(L, dV_LH, lower=True, check_finite=False), check_finite=False)     
        term2 = np.trace(V_inv_dV)
        dF_LH = np.block([[np.zeros((N_L, p_L)), np.zeros((N_L, p_H))], 
                          [gp_L.mean.basis(X_H), np.zeros((N_H, p_H))]])
        term3 = 2*(beta_LH.T @ dF_LH.T) @ alpha_V
        grad[-1] = -0.5 * (term1 - term2 + term3)
        return grad


    def run_minimize(self, x0, idx=None, fun=None, jac=None, bounds=None):
        """ Runs one instance of the minimize routine
        Args:
            x0 ((n_params,) array): Array containing the parameters to be optimized.
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
        grad_fun = self.grad_fun
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

        self.random_initial_points.append(initial_points_list)
        # Number of cores
        n_available = multiprocessing.cpu_count()    # Number of (logical) CPU cores available
        n_cores_used = min(n_cores, n_available)
        if n_cores_used < n_cores:
            print(f"Warning: {n_cores} cores were called, but only {n_cores_used} are used.")
        # Optimization
        if n_cores_used > 1:  # Parallel
            if verbose:
                print('Parallel optimization')
            results = Parallel(n_jobs=n_cores, backend="loky")(
                delayed(self.run_minimize)(x0, idx=i, fun=nlml_fun, jac=grad_fun, bounds=bounds) 
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
                res, msg = self.run_minimize(x0, idx=i, fun=nlml_fun, jac=grad_fun, bounds=bounds)
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
        """ Computes the predictive mean and variance with the co-kriging equations, obtained after Gaussian conditioning.  
        Args:
            X_test ((N_test,D) array): Input test set
            only_diag_cov (bool, optional): If True, computes only the diagonal of the covariance matrix prediction,
            useful when N_test is quite large (> 10^3). Defaults to False.
            Inv (bool, optional): If True, uses direct inversion of the covariance matrix, never do that in practice,
            this option is present only to present the analytical formula. Defaults to False.
        Returns: 
            mu_H ((N_test,1) array): HF predictive means at X_test.
            cov_H ((N_test,N_test) or (N_test,1) array): HF predictive covariances at X_test.
        """
        gp_L, gp_H = self.LF_GP, self.HF_GP
        assert gp_H.X.shape[1] == X_test.shape[1], f"The test input set must have the right dimension, got {X_test.shape[1]}, expected {self.X.shape[1]}."
        kernel_L, kernel_H = gp_L.kernel, gp_H.kernel
        beta_rho = self.scale.beta[0]
        N_L, N_H = gp_L.N, gp_H.N
        X_L, X_H = gp_L.X, gp_H.X
        p_L, p_H = gp_L.mean.p, gp_H.mean.p
        F_L, F_H = gp_L.Fmatrix, gp_H.Fmatrix              # shapes (N_L,p_L) and (N_H,p_H)
        F_LH = np.block([[F_L,  np.zeros((N_L, p_H))],     # shape (N_L+N_H,p_L+p_H)
                         [beta_rho*gp_L.mean.basis(X_H), F_H]])
        beta_L, beta_H = gp_L.mean.beta, gp_H.mean.beta    
        beta_LH = np.block([[beta_L],
                            [beta_H]])
        m_train = F_LH @ beta_LH   # shape (N_L+N_H,1)
        m_test = beta_rho*gp_L.mean.basis(X_test) @ beta_L + gp_H.mean.basis(X_test) @ beta_H  # shape (N_test,1)
        L = self.Lmatrix    # shape (N_L+N_H,N_L+N_H)
        V_LH_cross = np.block([[beta_rho*kernel_L.covariance(X_L, X_test)],        # shape (N_L+N_H,N_test)
                               [beta_rho**2 *kernel_L.covariance(X_H, X_test) + kernel_H.covariance(X_H, X_test)]])
        
        if only_diag_cov :   # shape (N_test,1)  # Only for stationary kernels (ok for RBF and Matérn)
            K_test = (beta_rho**2 * kernel_L.sigma2 + kernel_H.sigma2) * np.ones((X_test.shape[0],1))  
        else :               # shape (N_test,N_test)
            K_test = beta_rho**2 * kernel_L.covariance(X_test,X_test) + kernel_H.covariance(X_test,X_test)  
        if Inv : 
            K_inv = np.linalg.inv(L @ L.T)
            mu_H = m_test + V_LH_cross.T @ K_inv @ (self.z_LH - m_train)
            cov_H = K_test - V_LH_cross.T @ K_inv @ V_LH_cross           # shape (N_test,N_test)
        else : 
            alpha = cho_solve((L, True), self.z_LH - m_train, check_finite=False)
            mu_H = m_test + V_LH_cross.T @ alpha   # shape (N_test,1)
            u = solve_triangular(L, V_LH_cross, lower=True, check_finite=False)   # shape (N_L+N_H,N_test)
            if only_diag_cov :
                cov_H = K_test - np.sum(u**2, axis=0).reshape(-1,1)  # shape (N_test,1)
            else :
                cov_H = K_test - u.T @ u
        return mu_H, cov_H
