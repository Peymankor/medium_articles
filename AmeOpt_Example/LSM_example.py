import numpy as np
import pandas as pd
from scipy.stats import norm

# We generate the prices in eight paths, according to the table of LSM paper
# eight paths

path_1 = np.array([1.00, 1.09, 1.08, 1.34])
path_2 = np.array([1.00, 1.16, 1.26, 1.54])
path_3 = np.array([1.00, 1.22, 1.07, 1.03])
path_4 = np.array([1.00, 0.93, 0.97, 0.92])
path_5 = np.array([1.00, 1.11, 1.56, 1.52])
path_6 = np.array([1.00, 0.76, 0.77, 0.90])
path_7 = np.array([1.00, 0.92, 0.84, 1.01])
path_8 = np.array([1.00, 0.88, 1.22, 1.34])

price_paths = np.array([path_1, path_2, path_3, path_4, 
            path_5, path_6, path_7,path_8])

S_Example = pd.DataFrame(data=price_paths,
                index=np.arange(1,8+1))
S_Example

def Val_Ame_Put_Opt(S, K, Nsteps, paths, r, T):

    dt = T/Nsteps
    df = np.exp(-r * dt)   
    
    exercise_payoff = np.maximum(K - S, 0)             
    
    cf = exercise_payoff.copy()
    cf[:] = 0
    cf.iloc[:,Nsteps] = exercise_payoff.iloc[:, Nsteps]
    

    for t in range(Nsteps-1,0,-1):

        table_t = pd.DataFrame({"Y":cf.iloc[:,t+1]*df, "X":S.iloc[:,t]})
        id_money_t = S[S.iloc[:, t] < K].index
        
        table_t_inmoney=table_t.loc[id_money_t]

    
        rg_t = np.polyfit(table_t_inmoney["X"], table_t_inmoney["Y"], 2)
   
        C_t = np.polyval(rg_t, S.loc[id_money_t].iloc[:,t])
    

        cf.loc[id_money_t,t] = np.where(exercise_payoff.loc[id_money_t,t] > C_t, 
            exercise_payoff.loc[id_money_t,t], 0)

        for tt in range(t, Nsteps):
            
            cf.loc[id_money_t,tt+1] = np.where(cf.loc[id_money_t,t] > 0, 
                0, cf.loc[id_money_t,tt+1])

    Sum_DCF = 0

    for t in range(Nsteps,0,-1):
    
        Sum_DCF = sum(cf.loc[:,t])*np.exp(-dt*r*t) + Sum_DCF

    Option_Value = Sum_DCF/paths

    # return both cashflow and the price of the option
    return cf, Option_Value

S = S_Example

# Strike Price
K_val = 1.1

# Number of exercise times until end of horiozon
Nsteps_val = 3

# Number of path
paths_val = 8

# Riskless Free rate
r_val = 0.06

# End of the Horizon
T_val = 3

CF_example,Value_Example = Val_Ame_Put_Opt(S=S_Example, K=K_val, \
                                           paths=paths_val \
                                           , r=r_val, T=T_val, \
                                           Nsteps=T_val)

print("The Cash Flow matrix resulting from LSM method")
print(CF_example)

print("The Value of American Put Option is:  %.4f" % Value_Example)