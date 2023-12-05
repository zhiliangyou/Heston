#!/usr/bin/env python
# coding: utf-8

#%% package
import os
import numpy as np
import json
import pandas as pd
from heston_calibrator_simulator import HestonCalibratorSimulator
import matplotlib.pyplot as plt
import heston_utils as hu


#%% read data

qqq = pd.read_csv('data\QQQ.csv')
cp=qqq['Close']
target_true_price=cp.iloc[-1]
cp=cp.iloc[:-1]

#%% paras
dt=1/252
K=np.arange(280,390,2)
InitialP=cp.iloc[0]
# mcmc para
r = 0.03
q = 0
n_mcmc_steps = 6000
burn_in = 3000
#qe para
T = 252/252 # this is maturity in year
n_steps = 252 # should be in day
n_paths = 1000
s0 = cp.iloc[0] #??? why set lastest price as s0
gamma1 = 0.5  # averaging factors for the discretivazion of Xt
gamma2 = 0.5  # averaging factors for the discretivazion of Xt
phiC = 1.5  # Threshold for the initiation of the two aproximate distribution of V(t+1 | Vt)
#b-s
year=1

#%%mcmc
qqq_heston = HestonCalibratorSimulator(cp, cost_of_carry=r-q, delta_t=1.)
qqq_heston.calibrate(n_mcmc_steps, burn_in)

#%% print para

print(qqq_heston.params_dict)


#%% plot para

offset=300
burn_in_pos = burn_in - offset
param_paths = qqq_heston.all_params_array_full[offset:, :]
hu.plot_heston_param_dynamics(param_paths, burn_in_pos, title='qqq_386')

#%% cal qe path

v0 = qqq_heston.params_dict.get('theta_final', 0.05)  # how can we get this? Now I am just using the longterm average as initial

all_params = qqq_heston.params_dict
mu = all_params.get( "mu_final" )
kappa = all_params.get( "kappa_final" )
theta = all_params.get( "theta_final" )
eta = all_params.get( "volvol_final" )
rho = all_params.get( "rho_final" )
print(f"{mu:.4f}, {kappa:.4f}, {theta:.4f}, {eta:.4f}, {rho:.4f}")

S_t_qe, V_t_qe = qqq_heston.monte_carlo_qe(T, n_steps, n_paths, s0, v0, gamma1, gamma2, phiC)

#%% plot qe path
hu.plot_paths(S_t_qe, 'Underlying Paths')
hu.plot_paths(V_t_qe, 'Volatility Paths')

#%% cal euler path


S_t_euler, V_t_euler = qqq_heston.monte_carlo_heston_euler_full_trunc(T, n_steps, n_paths, s0, v0)


#%% plot euler path


hu.plot_paths( S_t_euler, 'Underlying Paths' )
hu.plot_paths( V_t_euler, 'Volatility Paths' )



#%% cal option princing using qe and euler

sigma=theta**0.5

Euler_pricing=hu.option_pricing(S_t_euler,K,dt,r,n_steps)
QE_pricing=hu.option_pricing(S_t_qe,K,dt,r,n_steps)
BS_call_price = np.array([ hu.BS_CALL( InitialP, k, T, r, sigma ) for k in K ])

print(BS_call_price)
print(Euler_pricing)
print(QE_pricing)

# %% plot cal option princing using qe and euler

hu.plot_convergence(Euler_pricing, QE_pricing, K)

#%%implied vol





imp_vol = [ ]
imp_vol_Euler = np.array([hu.implied_vol(Euler_pricing.iloc[0, i], InitialP, K[i], year, r) for i in range(len(K))])
imp_vol_QE = np.array([hu.implied_vol(QE_pricing.iloc[0, i], InitialP, K[i], year, r) for i in range(len(K))])

#%% plot implied vol

hu.plot_implied_volatility(K, imp_vol_Euler, imp_vol_QE, InitialP)

#%% get true option data


import os

# 获取当前工作目录
current_path = os.getcwd()

# 初始和结束的文件名数字部分
start = 380000
end = 398000
step = 2000

# 构造文件名列表
filenames = [f'O_QQQ231229C00{num}.csv' for num in range(start, end + step, step)]

# 初始化一个空列表来存储 'Close' 列的最后一个值
close_values = []

# 遍历并读取每个文件
for filename in filenames:
    file_path = os.path.join(current_path, 'data', filename)
    try:
        df = pd.read_csv(file_path)
        # 获取 'Close' 列的最后一个值并添加到列表
        close_values.append(df['close'].iloc[-1])
    except FileNotFoundError:
        print(f'File not found: {filename}')
    except Exception as e:
        print(f'Error reading file {filename}: {e}')

# 将列表转换为数组
true_price_array = np.array(close_values)

print(true_price_array)


#%%  plot and compare price (b-s,qe,euler)


Euler_pricing_array = Euler_pricing.to_numpy().reshape(-1)
QE_pricing_array = QE_pricing.to_numpy().reshape(-1)

np.shape(QE_pricing_array)

hu.plot_option_prices(K, BS_call_price, Euler_pricing_array, QE_pricing_array)

#plot with true option price
K_last_10 = K[-10:]
BS_call_price_last_10 = BS_call_price[-10:]
Euler_pricing_array_last_10 = Euler_pricing_array[-10:]
QE_pricing_array_last_10 = QE_pricing_array[-10:]

hu.plot_option_prices(K_last_10, BS_call_price_last_10, Euler_pricing_array_last_10, QE_pricing_array_last_10,true_price_array)

#%% cal simulation error
errors_BS = BS_call_price_last_10 - true_price_array
errors_Euler = Euler_pricing_array_last_10 - true_price_array
errors_QE = QE_pricing_array_last_10 - true_price_array

mse_BS = round(np.mean((BS_call_price_last_10 - true_price_array)**2), 4)
mse_Euler = round(np.mean((Euler_pricing_array_last_10 - true_price_array)**2), 4)
mse_QE = round(np.mean((QE_pricing_array_last_10 - true_price_array)**2), 4)

mae_BS = round(np.mean(np.abs(BS_call_price_last_10 - true_price_array)), 4)
mae_Euler = round(np.mean(np.abs(Euler_pricing_array_last_10 - true_price_array)), 4)
mae_QE = round(np.mean(np.abs(QE_pricing_array_last_10 - true_price_array)), 4)

# 打印结果
print("MSE - Black-Scholes:", mse_BS)
print("MSE - Euler:", mse_Euler)
print("MSE - QE:", mse_QE)
print("MAE - Black-Scholes:", mae_BS)
print("MAE - Euler:", mae_Euler)
print("MAE - QE:", mae_QE)