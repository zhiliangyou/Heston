from abc import ABC
import numpy as np
from scipy.stats import truncnorm, invgamma, norm
from tqdm import tqdm

class HestonCalibratorSimulator(ABC):

    def __init__(self, price_series, cost_of_carry: float = 0.03, mu_prior: float = 0.0, sigma_sq_mu_prior: float = 1.0, 
                 delta_t: float = 1.0, alpha_prior: float = 2.0, beta_prior: float = 0.005, p_prior: float = 2.0,
                 psi_prior: float = 0.0, theta_prior: float = 0.0, sigma_sq_theta_prior: float = 1.0,
                 kappa_prior: float = 0.0, sigma_sq_kappa_prior: float = 1.0, mu: float = 0.05,
                 kappa: float = 0.5, theta: float = 0.1, omega: float = 0.1, psi: float = 0,
                 *args, **kwargs):

        self.mu_prior = mu_prior
        self.sigma_sq_mu_prior = sigma_sq_mu_prior

        self.returns = np.diff(np.log(price_series))  # "Y" is "returns" here
        self.s0 = price_series[0]
        self.T = len(self.returns)
        self.delta_t = delta_t
        self.cost_of_carry = cost_of_carry

        # --- initialize prior parameters
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.p_prior = p_prior
        self.psi_prior = psi_prior
        self.theta_prior = theta_prior
        self.sigma_sq_theta_prior = sigma_sq_theta_prior
        self.kappa_prior = kappa_prior
        self.sigma_sq_kappa_prior = sigma_sq_kappa_prior
        self.alpha_star = self.T / 2 + self.alpha_prior
        # --- iniitalize posterior parameters
        self._mu = mu
        self._kappa = kappa
        self._theta = theta
        self._omega = omega
        self._psi = psi
        # --- placeholder for parameters' array
        self._all_params_array_full = None
        self._all_params_array_no_burnin = None
        self._params_dict = None

    @staticmethod
    def mu_star(psi, omega, kappa, theta, V, Y, Z, B, dt, mu_prior, sigma_sq_mu_prior):
        """ Posterior mean for the drift parameter"""
        numerator = sum((omega + psi ** 2) * (Y + 0.5 * V[:-2] * dt - Z * B) / (omega * V[:-2])) - \
                    sum(psi * (V[1:-1] - kappa * theta * dt - (1 - kappa * dt) * V[:-2]) / (omega * V[:-2])) \
                    + mu_prior / sigma_sq_mu_prior
        denominator = dt * sum((omega + psi ** 2) / (omega * V[:-2])) + 1 / sigma_sq_mu_prior
        return numerator / denominator

    @staticmethod
    def sigma_sq_star(psi, omega, V, dt, sigma_prior):
        """ Posterior variance for the drift parameter"""
        numerator = 1
        denominator = dt * sum((omega + psi ** 2) / (omega * V[:-2])) + 1 / (sigma_prior ** 2)
        return numerator / denominator

    @staticmethod
    def get_eps_s(V, Y, Z, B, mu, dt):
        return (Y - mu * dt + 0.5 * V[:-2] * dt - Z * B) / np.sqrt(V[:-2] * dt)

    @staticmethod
    def get_eps_v(V, dt, kappa, theta):
        return (V[1:-1] - kappa * theta * dt - (1 - kappa * dt) * V[:-2]) / np.sqrt(V[:-2] * dt)

    @classmethod
    def beta_star(cls, V, Y, Z, B, mu, dt, kappa, theta, beta_prior, p_prior, psi_prior):
        """ Posterior beta parameter for Omega which is
        used to parameterize the variance of variance and
        the correlation of the stock and variance processes"""
        eps_S = cls.get_eps_s(V, Y, Z, B, mu, dt)
        eps_V = cls.get_eps_v(V, dt, kappa, theta)
        result = beta_prior + 0.5 * sum(eps_V ** 2) + 0.5 * p_prior * psi_prior ** 2 - \
                 0.5 * ((p_prior * psi_prior + sum(eps_S * eps_V)) ** 2 / (p_prior + sum(eps_S ** 2)))
        return result

    @classmethod
    def psi_star(cls, Y, V, Z, B, mu, dt, kappa, theta, p_prior, psi_prior):
        """ Posterior mean parameter for psi which is also
        used to parameterize the variance of variance and
        the correlation of the stock and variance processes """
        eps_S = cls.get_eps_s(V, Y, Z, B, mu, dt)
        eps_V = cls.get_eps_v(V, dt, kappa, theta)
        result = (p_prior * psi_prior + sum(eps_S * eps_V)) / (p_prior + sum(eps_S ** 2))
        return result

    @classmethod
    def sigma_sq_psi_star(cls, Y, V, Z, B, mu, dt, p_prior, omega):
        """ Posterior variance parameter for psi which is used
        to parameterize the variance of variance and
        the correlation of the stock and variance processes """
        eps_S = cls.get_eps_s(V, Y, Z, B, mu, dt)
        result = omega / (p_prior + sum(eps_S ** 2))
        return result

    @staticmethod
    def theta_star(Y, V, Z, B, mu, dt, psi, kappa, omega, theta_prior, sigma_sq_theta_prior):
        """ Posterior mean parameter for the mean reversion parameter for
        the variance process """
        numerator = sum(kappa * (V[1:-1] - (1 - kappa * dt) * V[:-2]) / (omega * V[:-2])) - \
                    sum(psi * (Y - mu * dt + 0.5 * V[:-2] * dt - Z * B) * kappa / (omega * V[:-2]) +
                        theta_prior / sigma_sq_theta_prior)
        denominator = dt * sum(kappa ** 2 / (omega * V[:-2])) + 1 / sigma_sq_theta_prior
        theta = numerator / denominator
        return theta

    @staticmethod
    def sigma_sq_theta_star(V, dt, kappa, omega, sigma_sq_theta_prior):
        """ Posterior variance parameter for the mean reversion parameter for
        the variance process """
        denominator = dt * sum(kappa ** 2 / (omega * V[:-2])) + 1 / sigma_sq_theta_prior
        return 1 / denominator

    @staticmethod
    def kappa_star(Y, V, Z, B, mu, dt, psi, theta, omega, kappa_prior, sigma_sq_kappa_prior):
        """ Posterior mean parameter for the mean reversion rate parameter for
        the variance process """
        numerator = sum((theta - V[1:-1]) * (V[1:-1] - V[:-2]) / (omega * V[:-2])) - \
                    sum(psi * (Y - mu * dt + 0.5 * V[:-2] * dt - Z * B) * (theta - V[:-2]) / (omega * V[:-2])) + \
                    kappa_prior / sigma_sq_kappa_prior
        denominator = dt * sum((V[:-2] - theta) ** 2 / (omega * V[:-2])) + 1 / sigma_sq_kappa_prior
        return numerator / denominator

    @staticmethod
    def sigma_sq_kappa_star(V, dt, theta, omega, sigma_sq_kappa_prior):
        """ Posterior variance parameter for the mean reversion rate parameter for
        the variance process """
        denominator = dt * sum((V[:-2] - theta) ** 2 / (omega * V[:-2])) + 1 / sigma_sq_kappa_prior
        return 1 / denominator

    @staticmethod
    def mu_s_star(psi, omega, kappa, theta, V_t_minus_1, V_t, Y_t, mu, dt, mu_s, sigma_sq_s):
        """ Posterior mean for the jump size """
        numerator = ((omega + psi ** 2) * (Y_t + 0.5 * V_t_minus_1 * dt - mu * dt) / (omega * V_t_minus_1 * dt)) - \
                    (psi * (V_t - kappa * theta * dt - (1 - kappa * dt) * V_t_minus_1) / (omega * V_t_minus_1 * dt)) \
                    + mu_s / sigma_sq_s
        denominator = (omega + psi ** 2) / (omega * V_t_minus_1 * dt) + 1 / sigma_sq_s
        return numerator / denominator

    @staticmethod
    def sigma_sq_s_star(psi, omega, V_t_minus_1, dt, sigma_sq_s):
        """ Posterior variance for the jump size """
        denominator = (omega + psi ** 2) / (omega * V_t_minus_1 * dt) + 1 / sigma_sq_s
        return 1 / denominator

    @staticmethod
    def mu_m_s_star(S_0, sigma_sq_s, T, Z):
        numerator = sum(Z / sigma_sq_s)
        denominator = 1 / S_0 + T / sigma_sq_s
        return numerator / denominator

    @staticmethod
    def sigma_sq_m_s_star(S_0, sigma_sq_s, T):
        denominator = 1 / S_0 + T / sigma_sq_s
        return 1 / denominator

    @staticmethod
    def get_p_star(psi, omega, kappa, theta, V_t_minus_1, V_t, Y_t, Z_t, mu_drift, delta_t, lambda_d):
        A = ((omega + psi ** 2) * (
                Z_t ** 2 - 2 * Z_t * (Y_t - mu_drift * delta_t + 0.5 * V_t_minus_1 * delta_t)) + 2 * psi * (
                     V_t - kappa * theta * delta_t - (1 - kappa * delta_t) * V_t_minus_1) * Z_t) / (
                    omega * V_t_minus_1 * delta_t)
        denominator = (1 - lambda_d) * np.exp(0.5 * A) / lambda_d + 1
        return 1 / denominator

    @staticmethod
    def state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                       dt, mu, omega, psi):
        return (-1 / (2 * omega)) * (((omega + psi ** 2) * (
                0.5 * V_proposed_or_current * dt + Y_t_plus_1 - Z_t_plus_1 * B_t_plus_1 - mu * dt) ** 2) / (
                                             V_proposed_or_current * dt))

    @staticmethod
    def state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                       dt, mu, omega, psi, kappa, theta):
        return (-1 / (2 * omega)) * (
                (-2 * psi * (0.5 * V_proposed_or_current * dt + Y_t_plus_1 - Z_t_plus_1 * B_t_plus_1 -
                             mu * dt) * (
                         (kappa * dt - 1) * V_proposed_or_current - kappa * theta * dt + V_t_plus_1)) / (
                        V_proposed_or_current * dt))

    @staticmethod
    def state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta):
        return (-1 / (2 * omega)) * (
                ((kappa * dt - 1) * V_proposed_or_current - kappa * theta * dt + V_t_plus_1) ** 2 / (
                V_proposed_or_current * dt))

    @staticmethod
    def state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                       dt, mu, omega, psi, kappa, theta):
        return (-1 / (2 * omega)) * (
                -2 * psi * (Y_t - Z_t * B_t - mu * dt + 0.5 * V_t_minus_1 * dt) * (V_proposed_or_current -
                                                                                   kappa * theta * dt - (
                                                                                           1 - kappa * dt) * V_t_minus_1) / (
                        V_t_minus_1 * dt))

    @staticmethod
    def state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta):
        return (-1 / (2 * omega)) * (
                (V_proposed_or_current - kappa * theta * dt - (1 - kappa * dt) * V_t_minus_1) ** 2 / (V_t_minus_1 * dt))

    @classmethod
    def state_space_target_dist_t_0(cls, V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                    dt, mu, omega, psi, kappa, theta):
        """ Formula for the target distribution of the state space """
        multiplier = 1 / (V_proposed_or_current * dt)
        term_1 = cls.state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                                    dt, mu, omega, psi)
        term_2 = cls.state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                    B_t_plus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_3 = cls.state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta)
        return multiplier * np.exp(term_1 + term_2 + term_3)

    @classmethod
    def state_space_target_dist_t_1_to_T(cls, V_proposed_or_current, Y_t, Z_t, B_t, Y_t_plus_1, V_t_plus_1, V_t_minus_1,
                                         Z_t_plus_1, B_t_plus_1, dt, mu, omega, psi, kappa, theta):
        """ Formula for the target distribution of the state space """
        multiplier = 1 / (V_proposed_or_current * dt)
        term_1 = cls.state_space_target_dist_term_1(V_proposed_or_current, Y_t_plus_1, Z_t_plus_1, B_t_plus_1,
                                                    dt, mu, omega, psi)
        term_2 = cls.state_space_target_dist_term_2(V_proposed_or_current, Y_t_plus_1, V_t_plus_1, Z_t_plus_1,
                                                    B_t_plus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_3 = cls.state_space_target_dist_term_3(V_proposed_or_current, V_t_plus_1, dt, omega, kappa, theta)
        term_4 = cls.state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_5 = cls.state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta)
        return multiplier * np.exp(term_1 + term_2 + term_3 + term_4 + term_5)

    @classmethod
    def state_space_target_dist_t_T_plus_1(cls, V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1, dt,
                                           mu, omega, psi, kappa, theta):
        """ Formula for the target distribution of the state space """
        multiplier = 1 / (V_proposed_or_current * dt)
        term_4 = cls.state_space_target_dist_term_4(V_proposed_or_current, Y_t, Z_t, B_t, V_t_minus_1,
                                                    dt, mu, omega, psi, kappa, theta)
        term_5 = cls.state_space_target_dist_term_5(V_proposed_or_current, V_t_minus_1, dt, omega, kappa, theta)
        return multiplier * np.exp(term_4 + term_5)

    def calibrate(self, n_mcmc_steps=10000, burn_in=5000, rejection_rate=0.005):
        # ----- generate starting values for V using a truncated normal distribution
        #       (present-time as well as shifted backward and forward)
        V_t_array = np.array(truncnorm.rvs(a=0, b=np.inf, loc=0.0225, scale=0.005, size=self.T + 2))
        V_t_minus_1_array = np.roll(V_t_array, 1)
        V_t_minus_1_array[0] = 0
        V_t_plus_1_array = np.roll(V_t_array, -1)
        V_t_plus_1_array[-1] = 0

        # --- create a padded version of Y for computation purposes
        Y_t_array = np.append(0, np.append(self.returns, 0))
        Y_t_plus_1_array = np.roll(Y_t_array, -1)
        Y_t_plus_1_array[-1] = 0

        self._all_params_array_full = np.zeros((n_mcmc_steps, len(V_t_array) + 5))
        self._all_params_array_full[0, 0:5] = np.array([self._mu, self._kappa, self._theta, self._psi, self._omega])
        self._all_params_array_full[0, 5:] = V_t_array

        zero_array = np.zeros((len(self.returns),))

        omega_alpha = self.alpha_star
        for iter in tqdm(range(1, n_mcmc_steps)):

            # ------- 1. Gibbs' sampling of model parameters -------
            # ----- start with the initialized parameters and update them using MCMC
            # (a) drift
            mu_mean = self.mu_star(self._psi, self._omega, self._kappa, self._theta, V_t_array, self.returns,
                                   zero_array, zero_array, self.delta_t, self.mu_prior, self.sigma_sq_mu_prior)
            mu_variance = self.sigma_sq_star(self._psi, self._omega, V_t_array, self.delta_t, self.sigma_sq_mu_prior)
            self._mu = np.random.normal(mu_mean, np.sqrt(mu_variance))

            # (b) Omega
            omega_beta = self.beta_star(V_t_array, self.returns, zero_array, zero_array, self._mu, self.delta_t,
                                        self._kappa, self._theta, self.beta_prior, self.p_prior, self.psi_prior)
            self._omega = invgamma.rvs(omega_alpha, scale=omega_beta)

            # (c) psi
            psi_mean = self.psi_star(self.returns, V_t_array, zero_array, zero_array, self._mu, self.delta_t,
                                     self._kappa, self._theta, self.p_prior, self.psi_prior)
            psi_vola = np.sqrt(self.sigma_sq_psi_star(self.returns, V_t_array, zero_array, zero_array,
                                                      self._mu, self.delta_t, self.p_prior, self._omega))
            self._psi = np.random.normal(psi_mean, psi_vola)

            # (d) theta
            theta_mean = self.theta_star(self.returns, V_t_array, zero_array, zero_array, self._mu, self.delta_t,
                                         self._psi, self._kappa, self._omega, self.theta_prior,
                                         self.sigma_sq_theta_prior)
            theta_vola = np.sqrt(self.sigma_sq_theta_star(V_t_array, self.delta_t, self._kappa,
                                                          self._omega, self.sigma_sq_theta_prior))
            self._theta = truncnorm.rvs((0 - theta_mean) / theta_vola, (5 - theta_mean) / theta_vola, loc=theta_mean,
                                        scale=theta_vola)

            # (e) kappa
            kappa_mean = self.kappa_star(self.returns, V_t_array, zero_array, zero_array, self._mu, self.delta_t,
                                         self._psi, self._theta, self._omega, self.kappa_prior,
                                         self.sigma_sq_kappa_prior)
            kappa_vola = np.sqrt(self.sigma_sq_kappa_star(V_t_array, self.delta_t, self._theta,
                                                          self._omega, self.sigma_sq_kappa_prior))
            self._kappa = truncnorm.rvs((0 - kappa_mean) / kappa_vola, (5 - kappa_mean) / kappa_vola, loc=kappa_mean,
                                        scale=kappa_vola)

            # ------- 2. Metropolis-Hastings' sampling of variance paths -------
            Y_and_V_arrays = zip(Y_t_array, Y_t_plus_1_array, V_t_minus_1_array, V_t_array, V_t_plus_1_array)
            V_t_array_new = list()
            for t, (Y_t, Y_t_plus_1, V_t_minus_1, V_t, V_t_plus_1) in enumerate(Y_and_V_arrays):

                # ----- generate a proposal value
                V_proposal = np.random.normal(V_t, rejection_rate)

                # ----- get density of V at the previous and proposed values of V
                if t == 0:
                    V_density_at_curr = self.state_space_target_dist_t_0(V_t, Y_t_plus_1, V_t_plus_1, 0.0, 0.0,
                                                                         self.delta_t, self._mu, self._omega, self._psi,
                                                                         self._kappa, self._theta)
                    V_density_at_prop = self.state_space_target_dist_t_0(V_proposal, Y_t_plus_1, V_t_plus_1, 0.0, 0.0,
                                                                         self.delta_t, self._mu, self._omega, self._psi,
                                                                         self._kappa, self._theta)
                elif t != 0 and t <= len(self.returns):
                    V_density_at_curr = self.state_space_target_dist_t_1_to_T(V_t, Y_t, 0.0, 0.0, Y_t_plus_1,
                                                                              V_t_plus_1, V_t_minus_1, 0.0, 0.0,
                                                                              self.delta_t, self._mu, self._omega,
                                                                              self._psi, self._kappa, self._theta)
                    V_density_at_prop = self.state_space_target_dist_t_1_to_T(V_proposal, Y_t, 0.0, 0.0, Y_t_plus_1,
                                                                              V_t_plus_1, V_t_minus_1, 0.0, 0.0,
                                                                              self.delta_t, self._mu, self._omega,
                                                                              self._psi, self._kappa, self._theta)
                else:
                    V_density_at_curr = self.state_space_target_dist_t_T_plus_1(V_t, Y_t, 0.0, 0.0, V_t_minus_1,
                                                                                self.delta_t, self._mu, self._omega,
                                                                                self._psi, self._kappa, self._theta)
                    V_density_at_prop = self.state_space_target_dist_t_T_plus_1(V_proposal, Y_t, 0.0, 0.0, V_t_minus_1,
                                                                                self.delta_t, self._mu, self._omega,
                                                                                self._psi, self._kappa, self._theta)

                # ----- estimate an acceptance probability for a given variance value
                # corr_factor = norm.pdf(V_t, loc=V_proposal, scale=sigma_N) / norm.pdf(V_proposal, loc=V_t, scale=sigma_N)
                accept_prob = min(V_density_at_prop / V_density_at_curr, 1)
                u = np.random.uniform(0, 1)
                if u < accept_prob:
                    V_t = V_proposal
                V_t_array_new.append(V_t)
            # ----- save the updated values
            V_t_array = np.array(V_t_array_new)
            V_t_minus_1_array = np.roll(V_t_array, 1)
            V_t_minus_1_array[0] = 0
            V_t_plus_1_array = np.roll(V_t_array, -1)
            V_t_plus_1_array[-1] = 0
            self._all_params_array_full[iter, 0:5] = np.array([self._mu, self._kappa, self._theta,
                                                               self._psi, self._omega])
            self._all_params_array_full[iter, 5:] = V_t_array_new
        self._all_params_array_no_burnin = self._all_params_array_full[burn_in:, :]
        mu_final = np.mean(self._all_params_array_no_burnin[:, 0])
        kappa_final = np.mean(self._all_params_array_no_burnin[:, 1])
        theta_final = np.mean(self._all_params_array_no_burnin[:, 2])
        psi_final = np.mean(self._all_params_array_no_burnin[:, 3])
        omega_final = np.mean(self._all_params_array_no_burnin[:, 4])
        rho_final = np.sqrt(1 / (1 + omega_final / (psi_final ** 2)))
        volvol_final = psi_final / rho_final
        if volvol_final < 0:
            rho_final = -rho_final
            volvol_final = psi_final / rho_final
        self._params_dict = {"mu_final": mu_final, "kappa_final": kappa_final, "theta_final": theta_final,
                             "volvol_final": volvol_final, "rho_final": rho_final}

    @staticmethod
    def Variance_t_dt(Vt, dt, k, theta, psi, phiC=1.5):
        #step1
        m = theta + (Vt - theta) * np.exp( -k * dt )#assume 252 trading day
        s2 = ((Vt * psi ** 2) * np.exp( -k * dt ) / k) * (1 - np.exp( -k * dt )) + (theta * psi ** 2) * (
                    (1 - np.exp( -k * dt )) ** 2) / (2 * k)
        #step2
        phi = s2 / m ** 2#step 2

        #step4
        if phi <= phiC:
            # step3
            #4(a)
            b2 = (2 * phi ** (-1)) - 1 + ((2 * phi ** (-1)) ** 0.5) * ((2 * phi ** (-1)- 1) ** 0.5 )
            a = m / (1 + b2)
            #4(b)
            ####
            Zv = norm.ppf( np.random.rand() )
            #4(c)
            Vhat_t_dt = a * (Zv + b2 ** 0.5) ** 2  # Non central Chi square variable aproximate sufficiently big value of Vt

        #step 5
        elif phi > phiC:
            p = (phi - 1) / (phi + 1)
            beta = 2 / (m + m * phi)  # Function of Delta Dirac variable for sufficiently small value of Vt
            ######
            Uv = np.random.rand()

            if Uv <= p:
                Vhat_t_dt = 0

            elif Uv > p:
                Vhat_t_dt = (beta ** (-1)) * np.log( (1 - p) / (1 - Uv) )

        return Vhat_t_dt

    @classmethod
    def cal_t_dt(cls, Vt, Xt, dt, k, theta,psi, rho, gamma1, gamma2, phiC):
        Vhat_t_dt = cls.Variance_t_dt(Vt, dt, k, theta, psi, phiC)
        K0 = -1 * (rho * k * theta) * dt / psi
        K1 = gamma1 * dt * (-0.5 + (k * rho / psi)) - (rho / psi)
        K2 = gamma2 * dt * (-0.5 + (k * rho / psi)) + (rho / psi)
        K3 = gamma1 * dt * (1 - rho ** 2)
        K4 = gamma2 * dt * (1 - rho ** 2)
        ######
        Z = norm.ppf( np.random.rand() )

        X_t_dt_log = np.log(Xt)+ K0 + K1 * Vt + K2 * Vhat_t_dt + ((K3 * Vt + K4 * Vhat_t_dt) ** 0.5) * Z
        X_t_dt=np.exp(X_t_dt_log)

        return X_t_dt,Vhat_t_dt

    # QE discretization algorithm for the Variance Process {Vt+1}, yield one step ahead variance conditional on Vt
    def monte_carlo_qe(self, T, n_steps, n_paths, s0, v0, gamma1, gamma2, phiC):

        # Check if the calibrated params exist
        if self._params_dict is not None:
            mu = self._params_dict.get("mu_final")
            kappa = self._params_dict.get("kappa_final")
            theta = self._params_dict.get("theta_final")
            psi = self._params_dict.get("volvol_final")
            rho = self._params_dict.get("rho_final")
        else:
            raise ValueError('No calibrated params available')

        dt = T / n_steps

        S_t = np.full( (n_paths, n_steps + 1), fill_value = s0, dtype = np.float64 )
        V_t = np.full( (n_paths, n_steps + 1), fill_value = v0, dtype = np.float64 )

        for i in tqdm( range( len( S_t ) ), desc = "Simulating" ):
            for t in range( 1, n_steps + 1 ):
                S_t[ i, t ], V_t[ i, t ] = self.cal_t_dt( V_t[ i, t - 1 ], S_t[ i, t - 1 ], dt, kappa, theta, psi, rho, gamma1, gamma2, phiC )

        return S_t, V_t

    # Euler
    def monte_carlo_heston_euler_full_trunc(self, T, n_steps, n_paths, s0, v0):
        '''
        Simulate Heston model stock paths by Euler methods with full truncation, take max(v, 0) to avoid negative variance
        '''

        # Check if the calibrated params exist
        if self._params_dict is not None:
            mu = self._params_dict.get("mu_final")
            kappa = self._params_dict.get("kappa_final")
            theta = self._params_dict.get("theta_final")
            psi = self._params_dict.get("volvol_final")
            rho = self._params_dict.get("rho_final")
        else:
            raise ValueError('No calibrated params available')

        # dt = self.delta_t
        dt = T/n_steps
        dX1 = np.random.randn(n_paths, n_steps+1) * np.sqrt(dt)
        dX2 = np.random.randn(n_paths, n_steps+1) * np.sqrt(dt)

        X1 = np.cumsum(dX1, axis=1)
        X2 = np.cumsum(dX2, axis=1)

        W_s = X1
        W_v = rho*X1 + np.sqrt(1-rho**2)*X2

        dW_s = np.diff(W_s, axis=1)
        dW_v = np.diff(W_v, axis=1)

        S_t = np.full((n_paths, n_steps+1), fill_value=s0, dtype=np.float64)
        V_t = np.full((n_paths, n_steps+1), fill_value=v0, dtype=np.float64)

        for t in tqdm(range(1, n_steps+1), desc="Simulating"):
            S_t[:,t] = S_t[:,t-1] * np.exp((self.cost_of_carry - V_t[:,t-1] * 0.5) * dt + np.sqrt(V_t[:,t-1]) * dW_s[:,t-1])
            V_t[:,t] = np.maximum(V_t[:,t-1] + kappa * (theta - V_t[:,t-1]) * dt + psi * np.sqrt(V_t[:,t-1]) * dW_v[:,t-1], 0.)

        return S_t, V_t

    @property
    def all_params_array_full(self):
        return self._all_params_array_full

    @property
    def all_params_array_no_burnin(self):
        return self._all_params_array_no_burnin

    @property
    def params_dict(self):
        return self._params_dict