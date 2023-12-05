#%% package
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

#%% plot para

def plot_heston_param_dynamics( param_paths, burn_in_pos, title ):
    """
    Plot the posterior dynamics of parameters in the Heston model.

    Parameters:
    - heston_cal: A calibrated Heston model object with all_params_array_full attribute.
    - burn_in: The number of steps to be considered as burn-in.
    - offset: The offset from the burn-in period to start plotting.
    """

    fig, axes = plt.subplots( 3, 2, figsize = (10, 12) )

    axes[ 0, 0 ].plot( param_paths[ :, 0 ] )
    axes[ 0, 0 ].axvline( x = burn_in_pos, color = "red", linewidth = 3 )
    axes[ 0, 0 ].set_xlabel( "$\\mu$" )
    axes[ 0, 1 ].plot( param_paths[ :, 1 ] )
    axes[ 0, 1 ].axvline( x = burn_in_pos, color = "red", linewidth = 3 )
    axes[ 0, 1 ].set_xlabel( "$\\kappa$" )
    axes[ 1, 0 ].plot( param_paths[ :, 2 ] )
    axes[ 1, 0 ].axvline( x = burn_in_pos, color = "red", linewidth = 3 )
    axes[ 1, 0 ].set_xlabel( "$\\theta$" )
    axes[ 1, 1 ].plot( param_paths[ :, 3 ] )
    axes[ 1, 1 ].axvline( x = burn_in_pos, color = "red", linewidth = 3 )
    axes[ 1, 1 ].set_xlabel( "$\\eta$" )
    axes[ 2, 0 ].plot( param_paths[ :, 4 ] )
    axes[ 2, 0 ].axvline( x = burn_in_pos, color = "red", linewidth = 3 )
    axes[ 2, 0 ].set_xlabel( "$\\rho$" )
    axes[ 2, 1 ].remove()

    plt.suptitle( f'Posterior dynamics of {title} parameters in Heston model (burn-in cutoff in red)' )
    plt.subplots_adjust( wspace = None, hspace = 0.3 )
    plt.tight_layout( rect = [ 0, 0.03, 1, 0.95 ] )
    plt.show()



#%%# path plots

import matplotlib.pyplot as plt

def plot_paths(paths, title):
    """
    Plot multiple paths (e.g., stock or volatility paths).

    Parameters:
    - paths: A list or array of paths to plot.
    - title: The title of the plot.
    """
    fig, ax = plt.subplots(1, 1)
    for path in paths:
        ax.plot(path)
    ax.set_title(title)
    plt.show()

# 使用函数的例子
# plot_paths(S_t_qe, 'Stock Paths')
# plot_paths(V_t_qe, 'Volatility Paths')

#%% cal option price

def option_pricing(S_t,K,dt,r,n_steps):
    calls = []
    for k in K:
        P = np.mean(np.maximum(S_t - k,0))*np.exp(-r*dt*n_steps)
        calls.append(P)

    pricing = pd.DataFrame(dict(zip(K,calls)),index = ['Price'])
    return pricing


#%%qe euler convergence

def plot_convergence(Euler_pricing, QE_pricing, K):
    diff = []

    for i in range(len(Euler_pricing.columns)):
        diff.append(Euler_pricing.iloc[:, i] - QE_pricing.iloc[:, i])

    # 创建差异的 DataFrame
    comp = pd.DataFrame(dict(zip(K, diff)), index=['Price'])
    print(comp)

    # 绘制图表
    plt.figure(figsize=(13, 7))
    plt.plot(K, Euler_pricing.iloc[0, :], c='r', label='Euler')
    plt.plot(K, QE_pricing.iloc[0, :], c='c', label='QE')
    plt.ylabel('Price diff', fontsize=15)
    plt.xlabel('Strike', fontsize=15)
    plt.title('Convergence', fontsize=20)
    plt.legend()
    plt.show()

# 使用函数的例子
# plot_convergence(Euler_pricing, QE_pricing, K)

#%%b-s
def BS_CALL( S, K, T, r, sigma ):
    d1 = (np.log( S / K ) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt( T ))
    d2 = d1 - sigma * np.sqrt( T )
    call_price = S * norm.cdf( d1 ) - K * np.exp( -r * T ) * norm.cdf( d2 )
    return call_price

#%% implied vol

def implied_vol( opt_value, S, K, T, r, ):
    def objective_function( sigma ):
        return abs( BS_CALL( S, K, T, r, sigma ) - opt_value )



    result = minimize_scalar( objective_function,  bounds=(0.01,6), method = 'bounded' )

    return result.x

#%% plot implied vol

def plot_implied_volatility(K, imp_vol_Euler, imp_vol_QE, InitialP):
    plt.figure(figsize=(13, 7))

    # 绘制 Euler 隐含波动率
    plt.plot(K, imp_vol_Euler, c='b', label='Implied Volatility for Euler')
    plt.scatter(K, imp_vol_Euler, s=150, c='k', marker='1')

    # 绘制 QE 隐含波动率
    plt.plot(K, imp_vol_QE, c='m', label='Implied Volatility for QE')
    plt.scatter(K, imp_vol_QE, s=150, c='k', marker='x')

    plt.ylabel('Implied Volatility', fontsize=15)
    plt.xlabel('Strike', fontsize=15)
    plt.axvline(InitialP, color='orange', linestyle='--', label='Spot Price')
    plt.title('Implied Volatility Comparison', fontsize=20)
    plt.legend()
    plt.show()

#%% plot compare price



def plot_option_prices(K, BS_call_price, Euler_pricing_array, QE_pricing_array, True_Price_array=None):
    plt.figure(figsize=(12, 6))
    plt.title('Option prices for different K')

    plt.plot(K, BS_call_price, label='B-S')
    plt.plot(K, Euler_pricing_array, label='Euler_Heston')
    plt.plot(K, QE_pricing_array, label='QE_Heston')

    plt.scatter(K, Euler_pricing_array, c='k', marker='o')
    plt.scatter(K, QE_pricing_array, c='k', marker='o')
    plt.scatter(K, BS_call_price, c='r', marker='o')

    # 检查是否有额外的数组需要绘制
    if True_Price_array is not None:
        plt.plot(K, True_Price_array, label='True Price ')
        plt.scatter(K, True_Price_array, c='g', marker='x')

    plt.xlabel('Strike', fontsize=15)
    plt.ylabel('Call price', fontsize=15)
    plt.legend()
    plt.show()

