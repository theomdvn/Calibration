# %% [markdown]
# # Projet Calibration
# 
# ### Théo MIDAVAINE & Matthieu JULIEN

# %%
import math
import pandas as pd
import numpy as np
import random 
import copy

from scipy.stats import norm

import matplotlib.pyplot as plt

# %% [markdown]
# # I – Densités risque neutre 

# %% [markdown]
# 
# ### 1/ Sur les prix d’option données dans le tableau ci-après, calibrer une densité risque neutre en utilisant la formule de Breeden-Litzenberger et la technique de Shimko. Comparer avec une densité gaussienne. 

# %%
data = pd.DataFrame()
data["strikes"] = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
data["prices"] = [10.93, 9.55, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]

# %%
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def newton_raphson_implied_vol(row, S, r, initial_guess=0.2, T=1):     
    sigma = initial_guess     
    K = row['strikes']
    price_market = row['prices']  
    for _ in range(100):           
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price_model = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)  
        vega = S * np.sqrt(T) * norm.pdf(d1)  
        price_diff = price_market - price_model  
        
        if np.abs(price_diff) < 1e-8:               
            return sigma                  
        sigma += price_diff / vega  # Mise à jour de sigma
        
        if sigma < 0:               
            sigma = 0.001     
    return sigma

def quadratic_interpolation(x_interpolated, x_original, y_original):
    y_interpolated = []
    for x in x_interpolated:
        for i in range(len(x_original) - 2):
            if x_original[i] <= x <= x_original[i + 2]:
                x1, x2, x3 = x_original[i], x_original[i + 1], x_original[i + 2]
                y1, y2, y3 = y_original[i], y_original[i + 1], y_original[i + 2]
                break

        A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
        B = np.array([y1, y2, y3])
        a, b, c = np.linalg.solve(A, B)
        y = a*x**2 + b*x + c
        y_interpolated.append(y)

    return y_interpolated

# %% [markdown]
# Nelder-Mead function used as minimizing function that will be used throughout the project

# %%
def nelder_mead(cost_function, initial_guess,*args, tol=1e-8, max_iter=2000, alpha=1.0, beta=0.5, gamma=2.0):

    n = len(initial_guess)
    simplex = np.zeros((n + 1, n))
    simplex[0] = initial_guess

    for i in range(1, n + 1):
        simplex[i] = initial_guess + 0.1 * (i * np.eye(n)[i - 1])

    for _ in range(max_iter):
        values = np.array([cost_function(point,args) for point in simplex])

        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        centroid = np.mean(simplex[:-1], axis=0)

        reflection = centroid + alpha * (centroid - simplex[-1])
        reflection_value = cost_function(reflection,args)

        if values[0] <= reflection_value < values[-2]:
            simplex[-1] = reflection

        elif reflection_value < values[0]:
            expansion = centroid + gamma * (reflection - centroid)
            expansion_value = cost_function(expansion,args)

            if expansion_value < reflection_value:
                simplex[-1] = expansion
            else:
                simplex[-1] = reflection

        else:
            contraction = centroid + beta * (simplex[-1] - centroid)
            contraction_value = cost_function(contraction,args)

            if contraction_value < values[-1]:
                simplex[-1] = contraction
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])

        if np.linalg.norm(simplex[0] - simplex[-1]) < tol:
            break

    result = {'x': simplex[0], 'fun': values[0]}
    return result

# %%
T = 1 # On considère des calls 1 an
r = 0 # Risk free rate à 0
S0 = 100. # Spot price à T0 = 100

# %%
data["implied_vol"] = data.apply(lambda row: newton_raphson_implied_vol(row, S0, r, T=T), axis=1)
data.T

# %%
print('Prix théorique avec la volatilité implicite calculée par NewtonRaphson : ', round(black_scholes_call_price(S0, 95, T, r, data[data['strikes'] == 95]['implied_vol'][0]),2))
print('Prix de marché :', round(data[data['strikes'] == 95]['prices'][0],2))

# %% [markdown]
# Interpolation des volatilités implicites

# %%
x_original = np.array(data.index)
y_original = np.array(data["implied_vol"])

nb_pts = 100
x_interpolated = np.linspace(x_original.min(), x_original.max(), nb_pts)
vol_interpolated = quadratic_interpolation(x_interpolated, x_original, y_original)

plt.scatter(x_original, y_original, label='Volatilités implicites initiales')
plt.plot(x_interpolated, vol_interpolated, label='Volatilités implicites interpolées', linestyle='--')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Interpolation Quadratique de la Volatilité implicite')
plt.show()

# %% [markdown]
# Interpolation des strikes

# %%
x_original=data.index
y_original=data["strikes"]
nb_pts=100
x_interpolated=np.linspace(x_original.min(), x_original.max(), nb_pts)
strike_interpolated = quadratic_interpolation(x_interpolated, x_original, y_original)

plt.scatter(x_original, y_original, label='Original Data')
plt.plot(x_interpolated, strike_interpolated, label='Interpolated Curve', linestyle='--')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Interpolation quadratique du Strike')
plt.show()

# %% [markdown]
# Calcul des prix avec Black-Scholes avec les vol et strike interpolés

# %%
prices=[]

for i,j in zip(strike_interpolated,vol_interpolated):
    prices.append(black_scholes_call_price(S0,i,T,r,j))

df=pd.DataFrame([prices,strike_interpolated],index=["Price","Strike"]).T
plt.plot(np.array(df["Strike"]),np.array(df["Price"]))
plt.title("Evolution du prix du call en fonction du strike ")
plt.ylabel("Prix")
plt.xlabel("Strike")
plt.show()

# %%
def neutral_risk_density(strikes , r , S0 , T , implied_vols ) :
    density = np.zeros(len(strikes))
    k=df["Strike"][1]-df["Strike"][0]
    
    for i , K in enumerate (strikes) :
        C_plus_k = black_scholes_call_price (S0 , K + k , T , r , implied_vols[i] ) 
        C_minus_k = black_scholes_call_price (S0 , K - k , T , r , implied_vols[i] ) 
        C_actuel = black_scholes_call_price (S0 , K , T , r , implied_vols[i] )

        # Formule de Breeden - Litzenberger 
        density [ i ] = np.exp( r * T ) *(( C_plus_k - 2 * C_actuel + C_minus_k ) / k ** 2)

    return density

density_risque_neutre = neutral_risk_density (strike_interpolated , r , S0 , T , vol_interpolated )

# %%
def gaussian(x, mu, sigma, ratio):
    return ratio * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def objective(params, args):
    mu, sigma, ratio = params
    return np.sum((gaussian(args[0], mu, sigma, ratio) - args[1])**2)

initial_guess = [98., 5., 2.]
result = nelder_mead(objective,np.array(initial_guess),df["Strike"],density_risque_neutre)

fitted_mu, fitted_sigma, fitted_ratio = result['x']

# %%
plt.plot(df["Strike"], density_risque_neutre, label='Risk neutral Density')
plt.plot(df["Strike"], gaussian(df["Strike"], fitted_mu, fitted_sigma, fitted_ratio), label='Fitted Gaussian')
plt.title( "Comparaison densité Gaussienne et risque neutre")
plt.legend()
plt.show()

print("Fitted Mean:", fitted_mu)
print("Fitted Standard Deviation:", fitted_sigma)
print("Fitted Amplitude (Ratio):", fitted_ratio)

# %% [markdown]
# ### 2/ Vérifier, en faisant des tirages dans cette loi implicite, si l’on trouve un prix de modèle proche du prix de marché pour toutes les options ci-dessus. 

# %%
# Simulation Monte Carlo pour valoriser une option
def valorisation_MC(S, K, T, r, pas, sigma,nb_simulations):
    dt=T/pas
    paths = np.zeros((nb_simulations, pas + 1))   #on créé une matrice de taille nb_simulations x pas
    paths[:, 0] = S   #on remplit la première valeur de chaque ligne avec le spot
    
    for i in range(pas):
        paths[:, i + 1] = paths[:, i] *np.exp(dt*(r - 0.5*sigma**2) + sigma*np.sqrt(dt)* ((np.random.normal(fitted_mu,fitted_sigma,size=nb_simulations)-fitted_mu)/fitted_sigma))   #tirage dans notre loi implicite
        #pour chaque pas on calcule le prix future en utilisant le spot précédent
    option_payoffs = np.maximum(paths[:, -1] - K, 0)     #on prend ensuite le max entre notre dernier spot futur calculé - strike et 0
    option_value = np.exp(-r * T) * np.mean(option_payoffs)  # on calcule la moyenne des payoffs qu'on actualise
    
    return option_value

# %%
option_prices_mc = [valorisation_MC(S0, strike, T, r, 252, vol, 10000) for strike, vol in zip(data["strikes"], data["implied_vol"])]

fig, ax = plt.subplots()

index = np.arange(len(data["strikes"]))

rects1 = ax.scatter(index, data["prices"], label='Prix de marché',color='red')
rects2 = ax.plot(index, option_prices_mc, label='Prix Monte Carlo')

ax.set_xlabel('Strikes')
ax.set_ylabel('Prix')
ax.set_title('Prix de marché VS Prix générés avec la simulation de Monte Carlo')
ax.legend()
plt.show()


# %%
for strike,vol,market_price,mc_price in zip(data["strikes"],data["implied_vol"],data["prices"],option_prices_mc):
    print(f"Option : K = {strike}, Sigma = {vol:.2f}")
    print(f"Prix par simulation de Monte Carlo = {mc_price:.2f}")
    print(f"Market Price = {market_price:.2f}\nDifference = {abs(mc_price-market_price):.2f}\n")


# %% [markdown]
#  II – Interpolation et volatilité locale

# %% [markdown]
# En plus du tableau donné précédemment, on va utiliser pour cette partie les prix d’options suivants <br>
# On utilise la fonction d'interpolation quadratique, pour lisser la courbe

# %%
def interpolation(data,T):
    prices = []
    #interpolation of vol
    x_original = np.array(data.index)
    y_original = np.array(data["implied_vol"])
 
    nb_pts = 100
    x_interpolated = np.linspace(x_original.min(), x_original.max(), nb_pts)
    y_interpolated = quadratic_interpolation(x_interpolated, x_original, y_original)
 
    #interpolation of strikes
    x_original=data.index
    y_original=data["strikes"]
    x_interpolated=np.linspace(x_original.min(), x_original.max(), nb_pts)
    strike_interpolated = quadratic_interpolation(x_interpolated, x_original, y_original)
 
    #Calcul des prix avec la formule de black scholes en fonction des paramêtres interpolés
    for i,j in zip(strike_interpolated,y_interpolated):
        prices.append(black_scholes_call_price(S0,i,T,r,j))
   
    df=pd.DataFrame([prices,strike_interpolated,y_interpolated],index=["prices","strikes",'implied_vol']).T
    return df

# %%
data_12_months = {'strikes': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                'prices': [10.93, 9.55, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]}
data_9_months = {'strikes': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                 'prices':[11.79 ,8.95 ,8.07 ,7.03 ,6.18 ,6.04 ,5.76 ,5.50 ,5.50 ,5.39]}
data_6_months = {'strikes': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                    'prices': [10.71 ,8.28 ,6.91 ,6.36 ,5.29 ,5.07 ,4.76 ,4.47 ,4.35 ,4.14]}
data_3_months = {'strikes': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                 'prices': [8.67, 7.14, 5.98, 4.93, 4.09, 3.99, 3.43, 3.01, 2.72, 2.53]}

df_12_months = pd.DataFrame(data_12_months)
df_9_months = pd.DataFrame(data_9_months)
df_6_months = pd.DataFrame(data_6_months)
df_3_months = pd.DataFrame(data_3_months)

# On utilise l'algorithme de Newton Raphson pour trouver les volatilités implicites

df_12_months["implied_vol"] = df_12_months.apply(lambda row: newton_raphson_implied_vol(row, S0, r, 1), axis=1)
df_9_months["implied_vol"] = df_9_months.apply(lambda row: newton_raphson_implied_vol(row, S0, r, 9/12), axis=1)
df_6_months["implied_vol"] = df_6_months.apply(lambda row: newton_raphson_implied_vol(row, S0, r, 6/12), axis=1)
df_3_months["implied_vol"] = df_3_months.apply(lambda row: newton_raphson_implied_vol(row, S0, r, 3/12), axis=1)


df_12_months_interp = interpolation(df_12_months,1)
df_9_months_interp = interpolation(df_9_months,0.75)
df_6_months_interp = interpolation(df_6_months,0.5)
df_3_months_interp = interpolation(df_3_months,0.25)

df_12_months['maturity'] =  [1]*10
df_9_months['maturity'] = [0.75]*10
df_6_months['maturity'] = [0.5]*10
df_3_months['maturity'] = [0.25]*10

df_12_months_interp['maturity'] =  [1]*100
df_9_months_interp['maturity'] = [0.75]*100
df_6_months_interp['maturity'] = [0.5]*100
df_3_months_interp['maturity'] = [0.25]*100


df_concatenated = pd.concat([df_12_months,df_9_months, df_6_months, df_3_months])
df_concatenated_interp = pd.concat([df_12_months_interp,df_9_months_interp, df_6_months_interp, df_3_months_interp])

df_concatenated.reset_index(drop=True,inplace=True)
df_concatenated_interp.reset_index(drop=True,inplace=True)

# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(df_concatenated_interp['strikes'].values.reshape(4, -1),
                df_concatenated_interp['maturity'].values.reshape(4, -1),
                df_concatenated_interp['implied_vol'].values.reshape(4, -1),
                cmap='viridis', edgecolor='k', linewidth=0.5, alpha=0.7)

ax.set_title("Nappe de volatilité")
ax.set_xlabel('Strike')
ax.set_ylabel('Maturité')
ax.set_zlabel('Implied Vol')

plt.show()

# %% [markdown]
# ### 3/ Déterminez la nappe de volatilité correspondant à ces 40 options. Proposer un prix, le plus juste  possible, pour une option de strike 99.50 et de maturité 8 mois (justifier la méthode retenue). 
# 

# %% [markdown]
# Proposer un prix, le plus juste 
# possible, pour une option de strike 99.50 et de maturité 8 mois (justifier la méthode retenue). 

# %%
def quadratic_interpolation(x_interpolated, x_original, y_original):
    y_interpolated = []
    for x in x_interpolated:
        for i in range(len(x_original) - 2):
            if x_original[i] <= x <= x_original[i + 2]:
                x1, x2, x3 = x_original[i], x_original[i + 1], x_original[i + 2]
                y1, y2, y3 = y_original[i], y_original[i + 1], y_original[i + 2]
                break

        A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
        B = np.array([y1, y2, y3])
        a, b, c = np.linalg.solve(A, B)
        y = a*x**2 + b*x + c
        y_interpolated.append(y)

    return y_interpolated

# %% [markdown]
# On test l'interpolation quadratique avec une valeur que l'on connait pour estimer son erreur.<br>
# Ici avec un Call K=99 T=1

# %%
volatility_interpolated = quadratic_interpolation([99], data["strikes"], data["implied_vol"])

print('K = 99')
print(f"The interpolated volatility = {volatility_interpolated[0]:.4f}")
print(f"Implied vol = {float(data[data['strikes']==99]['implied_vol'].values):.4f}")

# %%
y_interpolated = quadratic_interpolation([8/12], [0.25, 0.5, 0.75, 1], list(df_concatenated['implied_vol']))
 
print('K = 99.50 and T = 8/12')
print(f"Implied vol interpolated : {y_interpolated[0]:.4f}")

# %% [markdown]
# We will use interp2d as a benchmark to test our functions

# %%
from scipy.interpolate import interp2d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

strike_prices = list(df_concatenated_interp[df_concatenated_interp['maturity']==1]['strikes'])
maturities = [0.5, 0.75]

implied_volatilities = [
    list(df_6_months_interp['implied_vol'].values),
    list(df_9_months_interp['implied_vol'].values)  
]

interp_func = interp2d(strike_prices, maturities, implied_volatilities, kind='linear')

x_target = 99.5
z_target = 8/12

implied_volatility_8_12 = interp_func(x_target, z_target)

print(f"Implied Volatility for maturity 8/12 and strike 99.5: {implied_volatility_8_12[0]:.4f}")

# %% [markdown]
# We remark that the error is too big, and the quadratic interpolation function is not adapted to fit both interpolated strikes and maturity, so we will use the bilinear interpolation which is more adapted to interpolate 2 parameters

# %%
def bilinear_interpolation(x, y, points):
# Fvoir sur https ://fr.wikipedia.org/wiki/Interpolation_bilin%C3%A9aire
    points = sorted(points)
    (x1 , y1 , q11), (_x1 , y2, q12), (x2, _y1 , q21), (_x2 , _y2 , q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError("points do not form a rectangle")
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError("(x, y) not within the rectangle ")

    return (q11 * (x2 - x) * (y2 - y) +
    q21 * (x - x1) * (y2 - y) +
    q12 * (x2 - x) * (y - y1) +
    q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)

# %%
x_val = 99
y_val = 1
z_val = bilinear_interpolation(x_val,y_val,[(98,0.75,float(df_9_months_interp[df_9_months_interp['strikes'] == 98]['implied_vol'].values)),
                                            (98,1,float(df_12_months_interp[df_12_months_interp['strikes'] == 98]['implied_vol'].values)),
                                        (100,0.75,float(df_9_months_interp[df_9_months_interp['strikes'] == 100]['implied_vol'].values)),
                                        (100,1,float(df_12_months_interp[df_12_months_interp['strikes'] == 100]['implied_vol'].values))])

# %%
true_z_val = float(df_12_months_interp[df_12_months_interp['strikes']==99]['implied_vol'].values)
print(f'Implied vol from the dataframe : {true_z_val:.4f}')
print(f'Bilinear interpolated implied vol : {z_val:.4f}')
print(f'The error between bilinear interpolation and true value is : {abs(true_z_val - z_val):.4f}')

# %%
t1 = float(df_concatenated_interp[(df_concatenated_interp['strikes'] == 99) & (df_concatenated_interp['maturity'] == 1)]['prices'].values)
print(f'Market Price: {t1:.4f}')
t2 = black_scholes_call_price(100,99,1,0,z_val)
print(f'BS Call price with bilinear interpolated implied vol : {t2:.4f}')
print(f'The error is {abs(t1-t2):.4f} or {abs((t1-t2)/t1):.4f}%')

# %% [markdown]
# We consider the error is acceptable so we will go with this method

# %%
x_val = 99.5
y_val = 8/12
z_995_val = bilinear_interpolation(x_val,y_val,[(99,0.75,float(df_9_months_interp[df_9_months_interp['strikes'] == 98]['implied_vol'].values)),
                                        (99,0.5,float(df_6_months_interp[df_6_months_interp['strikes'] == 98]['implied_vol'].values)),
                                        (100,0.75,float(df_9_months_interp[df_9_months_interp['strikes'] == 100]['implied_vol'].values)),
                                        (100,0.5,float(df_6_months_interp[df_6_months_interp['strikes'] == 100]['implied_vol'].values))
                                        ]                 
                               )

# %%
print(f'Implied vol from interp2d : {implied_volatility_8_12[0]:.4f}')
print(f'Bilinear interpolated implied vol : {z_995_val:.4f}')
print(f'The error between bilinear interpolation and true value is : {abs(implied_volatility_8_12[0] - z_995_val):.4f}')


prix_quad = black_scholes_call_price(100,99.5,8/12,0,y_interpolated[0])
print(f"Prix trouvé avec l'interpolation quadratique : {prix_quad:.4f}")
prix_bi = black_scholes_call_price(100,99.5,8/12,0,z_995_val)
print(f"Prix trouvé avec l'interpolation bilinéeaire : {prix_bi:.4f}")
prix_interp2d = black_scholes_call_price(100,99.5,8/12,0,implied_volatility_8_12[0])
print(f"Prix théorique avec interp2d: {prix_interp2d:.4f}")

# %% [markdown]
# ### 4/ Calibrer un modèle à volatilité locale de type SVI en suivant les étapes suivantes 

# %% [markdown]
# - écrire l’algorithme de valorisation d’une option avec le modèle SVI, par EDP d’une part et par Monte Carlo d’autre part et comparer les résultats pour plusieurs jeux de paramètres 

# %%
param = [0.1] * 5

def svi_model(x, param):
    a,b,rho,m,sigma = param
    variance = abs(a) + abs(b) * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
    return np.sqrt(variance)

def forward_price_calculation(S0, r, T):
    return S0 * np.exp(r*T)

def log_moneyness_calculation(strike, forward_price):
    return np.log(strike/forward_price)

# %%
def valorisation_EDP(S0,K,r,T,sigma,N,I,Smin,Smax):
    step=(Smax-Smin)/I
    dt=T/N
    S=np.arange(Smin,Smax+step,step)
    S_squared = np.array([x**2 for x in S])  
    grille = np.arange(0,T+dt,dt)
    alpha=pow(sigma,2)/(2*pow(step,2))*S_squared
    beta=(r*S)/(2*step)
    option_prices = np.zeros((I,1))
    A = np.zeros((I,I))

    for i in range(I):
        option_prices[i] = max(S[i]-K,0)
    for i in range(1,len(A)+1):
        A[i-1][i-1]=2*alpha[i]+r
        if i>1:
            A[i-1][i-2]=-alpha[i]+beta[i]

        if i<I:
            A[i-1][i]=-alpha[i]-beta[i]

    
    y=np.zeros((I,1))
    alpha1=pow(sigma,2)/(2*pow(step,2))*S_squared[1]
    beta1=(r*S[1])/(2*step)
    y[0]=0
    alphaI=pow(sigma,2)/(2*pow(step,2))*S_squared[I]
    betaI=(r*S[I])/(2*step)

    for n in range(0, N):
        y[I-1]=(-alphaI-betaI)*(Smax-K*np.exp(-r*grille[n]))
        option_prices= option_prices - dt * (A @ option_prices + y)
        
    return pd.DataFrame(option_prices,index=S[:-1]).loc[S0]

# %%
for strike,market_price,maturity in zip(df_12_months["strikes"],df_12_months["prices"], df_12_months["maturity"]):
    forward_price = forward_price_calculation(S0, r, maturity)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    sigma = svi_model(log_moneyness, param)  
    option_price_EDP_SVI = valorisation_EDP(S0,strike,r,maturity,sigma,10000,200,0,200)
    print(f"Prix par EDP pour K = {strike}: {option_price_EDP_SVI[0]:.2f}")
    print(f"Market Price = {market_price:.2f}, Difference = {abs(option_price_EDP_SVI[0]-market_price):.2f}")

# %%
def valorisation_MC_SVI(S0, param, r, nb_simulations, pas, strike, maturity):
    dt = maturity / pas
    paths = np.zeros((nb_simulations, pas + 1))
    paths[:, 0] = S0   

    for i in range(pas):
        forward_price = forward_price_calculation(S0, r, maturity)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol = svi_model(log_moneyness, param)  
        paths[:, i + 1] = paths[:, i] * np.exp((r - 0.5 * vol**2) * dt + np.sqrt(dt * vol**2) * np.random.normal(0, 1, size=nb_simulations))

    option_payoffs = np.maximum(paths[:, -1] - strike, 0)
    option_value = np.exp(-r * maturity) * np.mean(option_payoffs)  

    return option_value

# %% [markdown]
# - estimez les cinq paramètres permettant de réduire au plus l’erreur moyenne
# 

# %%
for strike,market_price,maturity in zip(df_12_months["strikes"],df_12_months["prices"], df_12_months["maturity"]):
    option_price_mc_SVI = valorisation_MC_SVI(S0, param, r, 1000, 252, strike, maturity)
    print(f"Prix par simulation de Monte Carlo pour K = {strike}: {option_price_mc_SVI:.2f}")
    print(f"Market Price = {market_price:.2f}, Difference = {abs(option_price_mc_SVI-market_price):.2f}")

# %%
def cost_function_SVI(param,args):
    strikes,vols_market,maturity=args
    vols_model = [] 
    for strike in strikes : 
        forward_price = forward_price_calculation(S0, r, maturity)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vols_model.append(svi_model(log_moneyness, param))
    return np.sum((vols_model - vols_market) ** 2)

# %%
def check_results_svi(results,data):
    calibrated_params = results
    pas=252
    diff_totale=0
    opt = []
    for strike,market_price,maturity in zip(data["strikes"],data["prices"], data["maturity"]):
        option_price_mc_SVI = valorisation_MC_SVI(S0, calibrated_params, r, 5000, pas, strike, maturity)
        print(f"Prix par simulation de Monte Carlo pour K = {strike}: {option_price_mc_SVI:.2f}")
        print(f"Market Price = {market_price:.2f}, Difference = {abs(option_price_mc_SVI-market_price):.2f}")
        opt.append(option_price_mc_SVI)
        diff_totale+=abs(option_price_mc_SVI-market_price)
    print(diff_totale)
    fig, ax = plt.subplots()

    rects1 = ax.plot(data["strikes"], data["prices"],'-o',label='Prix de marché',color='red')
    rects2 = ax.plot(data["strikes"], opt, label='Prix modèle (SVI)')
    #rects3 = ax.plot(data["strikes"], prices,'-', label='Prix EDP (SVI)')
 
    ax.set_xlabel('Strikes')
    ax.set_ylabel('Prices')
    ax.set_title('Graphique des prix obtenus')
    ax.legend()
    plt.show()

# %%
result_svi_12m = nelder_mead(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_12_months["strikes"],df_12_months["implied_vol"],1)['x']
check_results_svi(result_svi_12m,df_12_months) 

# %%
result_svi_12m

# %% [markdown]
# On teste notre EDP avec le modèle SVI calibré

# %%
prices_EDP = []
for strike,market_price,maturity in zip(df_12_months["strikes"],df_12_months["prices"], df_12_months["maturity"]):
    forward_price = forward_price_calculation(S0, r, maturity)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    sigma = svi_model(log_moneyness, result_svi_12m)  
    option_price_EDP_SVI = valorisation_EDP(S0,strike,r,maturity,sigma,1000,100,0,200)
    prices_EDP.append(option_price_EDP_SVI[0])
    print(f"Prix avec EDP pour K = {strike}: {option_price_EDP_SVI[0]:.2f}")
    print(f"Market Price = {market_price:.2f}, Difference = {abs(option_price_EDP_SVI[0]-market_price):.2f}")

# %%
plt.plot(df_12_months["strikes"],prices_EDP)

# %% [markdown]
# On teste la valorisation des options 12 mois et comparons à nos vraies valeurs de marché.
# Pour le set de paramètres nous utilisons celui déterminé dans la suite du projet, qui minimise l'erreur moyenne.

# %%
result_svi_9m = nelder_mead(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_9_months["strikes"],df_9_months["implied_vol"],0.75)['x']
check_results_svi(result_svi_9m,df_9_months) 

# %% [markdown]
# Notre algorithme de valorisation par Monte Carlo nous donne des prix proches

# %%
result_svi_6m = nelder_mead(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_6_months["strikes"],df_6_months["implied_vol"],0.5)['x']
result_svi_3m = nelder_mead(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_3_months["strikes"],df_3_months["implied_vol"],0.25)['x']

# %%
result_NM = [result_svi_12m , result_svi_9m , result_svi_6m , result_svi_3m ]

NM=pd.DataFrame(result_NM,index=["12 months","9 months","6 months","3 months"],columns=["a","b","rho","m","sigma"])
NM

# %%
vols_market_12m = df_12_months["implied_vol"]
vols_market_9m = df_9_months["implied_vol"]
vols_market_6m = df_6_months["implied_vol"]
vols_market_3m = df_3_months["implied_vol"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(10,10))
fig.suptitle('SVI Calibrated model for each maturity')

strikes_plot = np.linspace(95, 104, 10)
vols_plot_12m =[] 
for strike in strikes_plot : 
    forward_price = forward_price_calculation(S0, r, 1)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_12m.append(svi_model(log_moneyness, result_svi_12m))

ax1.plot(strikes_plot, vols_plot_12m, label='SVI Model')
ax1.scatter(strikes_plot, vols_market_12m, color='red', label='Market Data')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Implied Volatility')
ax1.set_title("12-months option")
ax1.legend()

vols_plot_9m =[] 
for strike in strikes_plot : 
    forward_price = forward_price_calculation(S0, r, 0.75)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_9m.append(svi_model(log_moneyness, result_svi_9m))

ax2.plot(strikes_plot, vols_plot_9m, label='SVI Model')
ax2.scatter(strikes_plot, vols_market_9m, color='red', label='Market Data')
ax2.set_xlabel('Strike')
ax2.set_ylabel('Implied Volatility')
ax2.set_title("9-months option")
ax2.legend()

vols_plot_6m =[] 
for strike in strikes_plot : 
    forward_price = forward_price_calculation(S0, r, 0.5)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_6m.append(svi_model(log_moneyness, result_svi_6m))

ax3.plot(strikes_plot, vols_plot_6m, label='SVI Model')
ax3.scatter(strikes_plot, vols_market_6m, color='red', label='Market Data')
ax3.set_xlabel('Strike')
ax3.set_ylabel('Implied Volatility')
ax3.set_title("6-months option")
ax3.legend()

vols_plot_3m =[] 
for strike in strikes_plot : 
    forward_price = forward_price_calculation(S0, r, 0.25)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_3m.append(svi_model(log_moneyness, result_svi_3m))

ax4.plot(strikes_plot, vols_plot_3m, label='SVI Model')
ax4.scatter(strikes_plot, vols_market_3m, color='red', label='Market Data')
ax4.set_xlabel('Strike')
ax4.set_ylabel('Implied Volatility')
ax4.set_title("3-months option")
ax4.legend()

plt.show()


# %% [markdown]
# - on fixe tous les paramètres aux valeurs estimées ci-avant (sauf a et b) et on va calibrer les deux 
# paramètres a et b pour chaque option : vous obtiendrez donc une nappe pour chacun de ces 
# deux paramètres et vous introduirez une contrainte de régularisation de ces deux nappes dans 
# votre problème d’optimisation. 

# %%
calibrated_params_svi = [result_svi_12m]+[result_svi_9m]+[result_svi_6m]+[result_svi_3m]

# %%
def svi_fixed(param, calibrated_params,x):
    a, b = param
    rho, m, sigma = calibrated_params[2:5]
    variance = abs(a) + abs(b) * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
    return np.sqrt(variance)
 
def cost_function_SVI_fixed(param, args):
    calibrated_params, strike, vols_market,maturity = args
    forward_price = forward_price_calculation(S0, r, maturity)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vol = svi_fixed(param, calibrated_params,log_moneyness)
    return np.sum((vol - vols_market) ** 2)


# %%
a_values_12m = []
a_values_9m = []
a_values_6m = []
a_values_3m = []
b_values_12m = []
b_values_9m = []
b_values_6m = []
b_values_3m = []

# %%
for k,sigma in zip(strikes_plot, df_12_months['implied_vol']):
    res_12m = nelder_mead(cost_function_SVI_fixed, np.array([0.1, 0.6]), calibrated_params_svi[0], k, sigma,1)['x']
    a_12m, b_12m = res_12m
    a_values_12m.append(a_12m)
    b_values_12m.append(b_12m)
for k,sigma in zip(strikes_plot, df_9_months['implied_vol']):
    res_9m = nelder_mead(cost_function_SVI_fixed, np.array([0.1, 0.6]), calibrated_params_svi[1], k, sigma,0.75)['x']
    a_9m, b_9m = res_9m
    a_values_9m.append(a_9m)
    b_values_9m.append(b_9m)

for k,sigma in zip(strikes_plot, df_6_months['implied_vol']):
    res_6m = nelder_mead(cost_function_SVI_fixed, np.array([0.1, 0.6]), calibrated_params_svi[2], k, sigma,0.5)['x']
    a_6m, b_6m = res_6m
    a_values_6m.append(a_6m)
    b_values_6m.append(b_6m)

for k,sigma in zip(strikes_plot, df_3_months['implied_vol']):
    res_3m = nelder_mead(cost_function_SVI_fixed, np.array([0.1, 0.6]), calibrated_params_svi[3], k, sigma,0.25)['x']
    a_3m, b_3m = res_3m
    a_values_3m.append(a_3m)
    b_values_3m.append(b_3m)

# %%
a_values = a_values_12m+a_values_9m+a_values_6m+a_values_3m
b_values = b_values_12m+b_values_9m+b_values_6m+b_values_3m
strikes = df_concatenated["strikes"]
maturities = df_concatenated['maturity']
vol_market = df_concatenated['implied_vol']

# %%
def surface(strikes,maturities,values):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(np.array(strikes).reshape(4,-1), 
                    np.array(maturities).reshape(4,-1),
                    np.array(values).reshape(4,-1), 
                    cmap='viridis', edgecolor='k', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('A values')

    plt.show()

# %%
surface(strikes,maturities,a_values)

# %%
surface(strikes,maturities,b_values)

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(10,10))
fig.suptitle('SVI Calibrated model for each maturity')

strikes_plot = np.linspace(95, 104, 10)
vols_plot_12m =[] 
for strike,a,b in zip(strikes_plot,a_values_12m,b_values_12m): 
    forward_price = forward_price_calculation(S0, r, 1)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_12m.append(svi_model(log_moneyness, list([a,b]) + list(result_svi_12m[2:5])))

ax1.plot(strikes_plot, vols_plot_12m, label='SVI Model')
ax1.scatter(strikes_plot, vols_market_12m, color='red', label='Market Data')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Implied Volatility')
ax1.set_title("12-months option")
ax1.legend()

vols_plot_9m =[] 
for strike,a,b  in zip(strikes_plot,a_values_9m,b_values_9m) : 
    forward_price = forward_price_calculation(S0, r, 0.75)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_9m.append(svi_model(log_moneyness,list([a,b]) + list(result_svi_9m[2:5])))

ax2.plot(strikes_plot, vols_plot_9m, label='SVI Model')
ax2.scatter(strikes_plot, vols_market_9m, color='red', label='Market Data')
ax2.set_xlabel('Strike')
ax2.set_ylabel('Implied Volatility')
ax2.set_title("9-months option")
ax2.legend()

vols_plot_6m =[] 
for strike,a,b in zip(strikes_plot,a_values_6m,b_values_6m)  : 
    forward_price = forward_price_calculation(S0, r, 0.5)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_6m.append(svi_model(log_moneyness, list([a,b]) + list(result_svi_6m[2:5])))

ax3.plot(strikes_plot, vols_plot_6m, label='SVI Model')
ax3.scatter(strikes_plot, vols_market_6m, color='red', label='Market Data')
ax3.set_xlabel('Strike')
ax3.set_ylabel('Implied Volatility')
ax3.set_title("6-months option")
ax3.legend()

vols_plot_3m =[] 
for strike,a,b in zip(strikes_plot,a_values_3m,b_values_3m)  : 
    forward_price = forward_price_calculation(S0, r, 0.25)
    log_moneyness = log_moneyness_calculation(strike, forward_price)
    vols_plot_3m.append(svi_model(log_moneyness, list([a,b]) + list(result_svi_3m[2:5])))

ax4.plot(strikes_plot, vols_plot_3m, label='SVI Model')
ax4.scatter(strikes_plot, vols_market_3m, color='red', label='Market Data')
ax4.set_xlabel('Strike')
ax4.set_ylabel('Implied Volatility')
ax4.set_title("3-months option")
ax4.legend()

plt.show()


# %%
def cost_function_reg_b(values,args):
    
    vol_market,lam,mu = args
    values_12m,values_9m,values_6m,values_3m = np.array_split(values,4)

    vol_model = []
    for strike,a,values in zip(strikes_plot,a_values_12m,values_12m)  : 
        forward_price = forward_price_calculation(S0, r, 1)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([a,values]) + list(result_svi_12m[2:5])))
    for strike,a,values in zip(strikes_plot,a_values_9m,values_9m)  : 
        forward_price = forward_price_calculation(S0, r, 0.75)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([a,values]) + list(result_svi_9m[2:5])))
    for strike,a,values in zip(strikes_plot,a_values_6m,values_6m)  : 
        forward_price = forward_price_calculation(S0, r, 0.5)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([a,values]) + list(result_svi_6m[2:5])))
    for strike,a,values in zip(strikes_plot,a_values_3m,values_3m)  : 
        forward_price = forward_price_calculation(S0, r, 0.25)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([a,values]) + list(result_svi_3m[2:5])))

    error_vol = np.sum((vol_market-vol_model)**2)

    values_surface = pd.DataFrame()
    values_surface.index = strikes_plot
    values_surface['12'] = values_12m
    values_surface['9'] = values_9m
    values_surface['6'] = values_6m
    values_surface['3'] = values_3m

    error_strike = 0
    for i in range(1,values_surface.shape[0]):
        error_strike += (lam * np.sum(((values_surface.iloc[i]) - values_surface.iloc[i-1])**2))
    error_maturity = 0
    for m in range(3,15,3):
        error_maturity+=(mu * np.sum((values_surface[f'{m}'] - values_surface[f'{m}'])**2))
    
    print(error_vol + error_strike + error_maturity)
    
    return error_vol + error_strike + error_maturity

# %% [markdown]
# Ajouter dans le nelder mead la condition stop si < 0.001 pour b et 0.125 pour a

# %%
result_reg_b = nelder_mead(cost_function_reg_b,np.array(b_values),vol_market,1,1)['x']

# %%
def cost_function_reg_a(values,args):
    
    vol_market,lam,mu = args
    values_12m,values_9m,values_6m,values_3m = np.array_split(values,4)

    vol_model = []
    for strike,values,b in zip(strikes_plot,values_12m,b_values_12m)  : 
        forward_price = forward_price_calculation(S0, r, 1)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([values,b]) + list(result_svi_12m[2:5])))
    for strike,values,b in zip(strikes_plot,values_9m,b_values_9m)  : 
        forward_price = forward_price_calculation(S0, r, 0.75)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([values,b]) + list(result_svi_9m[2:5])))
    for strike,values,b in zip(strikes_plot,values_6m,b_values_6m)  : 
        forward_price = forward_price_calculation(S0, r, 0.5)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([values,b]) + list(result_svi_6m[2:5])))
    for strike,values,b in zip(strikes_plot,values_3m,b_values_3m)  : 
        forward_price = forward_price_calculation(S0, r, 0.25)
        log_moneyness = log_moneyness_calculation(strike, forward_price)
        vol_model.append(svi_model(log_moneyness, list([values,b]) + list(result_svi_3m[2:5])))

    error_vol = np.sum((vol_market-vol_model)**2)

    values_surface = pd.DataFrame()
    values_surface.index = strikes_plot
    values_surface['12'] = values_12m
    values_surface['9'] = values_9m
    values_surface['6'] = values_6m
    values_surface['3'] = values_3m

    error_strike = 0
    for i in range(1,values_surface.shape[0]):
        error_strike += (lam * np.sum(((values_surface.iloc[i]) - values_surface.iloc[i-1])**2))
    error_maturity = 0
    for m in range(3,15,3):
        error_maturity+=(mu * np.sum((values_surface[f'{m}'] - values_surface[f'{m}'])**2))
    
    print(error_vol + error_strike + error_maturity)
    
    return error_vol + error_strike + error_maturity


# %%
result_reg_a = nelder_mead(cost_function_reg_a,np.array(a_values),vol_market,1,1)['x']

# %%
surface(strikes,maturities,a_values)

# %%
surface(strikes,maturities,result_reg_a)

# %%
surface(strikes,maturities,b_values)

# %%
surface(strikes,maturities,result_reg_b)

# %% [markdown]
# ### 5/ En utilisant les nappes obtenues pour le modèle SVI (à la dernière étape), proposer encore un prix, le plus juste possible, pour une option de strike 99.50 et de maturité 8 mois (justifier la méthode retenue).

# %%
df_concatenated['a_values'] = a_values
df_concatenated['b_values'] = b_values

# %%
x_val = 99
y_val = 1
z_val = bilinear_interpolation(x_val,y_val,[(98,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 98)]['a_values'].values)),
                                            (98,1,float(df_concatenated[(df_concatenated['maturity'] == 1) & (df_concatenated['strikes'] == 98)]['a_values'].values)),
                                        (99,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 99)]['a_values'].values)),
                                        (99,1,float(df_concatenated[(df_concatenated['maturity'] == 1) & (df_concatenated['strikes'] == 99)]['a_values'].values))
                                        ]                 
                               )
z2_val = bilinear_interpolation(x_val,y_val,[(98,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 98)]['b_values'].values)),
                                            (98,1,float(df_concatenated[(df_concatenated['maturity'] == 1) & (df_concatenated['strikes'] == 98)]['b_values'].values)),
                                        (99,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 99)]['b_values'].values)),
                                        (99,1,float(df_concatenated[(df_concatenated['maturity'] == 1) & (df_concatenated['strikes'] == 99)]['b_values'].values))
                                        ]                 
                               )

# %%
forward_price = forward_price_calculation(S0, r, 1)
log_moneyness = log_moneyness_calculation(99, forward_price)
test_imp = svi_model(log_moneyness, calibrated_params_svi[0])
price = black_scholes_call_price(100,99,1,0,test_imp)
price

# %%
float(df_concatenated[(df_concatenated['maturity'] == 1) & (df_concatenated['strikes'] == 99)]['prices'].values)

# %%
x_val = 99.5
y_val = 8/12
a_val = bilinear_interpolation(x_val,y_val,[(99,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 99)]['a_values'].values)),
                                            (99,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 99)]['a_values'].values)),
                                        (100,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 100)]['a_values'].values)),
                                        (100,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 100)]['a_values'].values))
                                        ]                 
                               )
b_val = bilinear_interpolation(x_val,y_val,[(99,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 99)]['b_values'].values)),
                                            (99,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 99)]['b_values'].values)),
                                        (100,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 100)]['b_values'].values)),
                                        (100,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 100)]['b_values'].values))
                                        ]                 
                               )

# %%
def linear_interpolation(x0, y0, x1, y1, x):
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x - x0)
    return y

# %%
rho_8m = linear_interpolation(6,calibrated_params_svi[2][2],9,calibrated_params_svi[1][2],8)
m_8m = linear_interpolation(6,calibrated_params_svi[2][3],9,calibrated_params_svi[1][3],8)
sigma_8m = linear_interpolation(6,calibrated_params_svi[2][4],9,calibrated_params_svi[1][4],8)

# %%
forward_price = forward_price_calculation(S0, r, 1)
log_moneyness = log_moneyness_calculation(99.5, forward_price)
imp_vol_svi_8 = svi_fixed([a_val,b_val],calibrated_params_svi[3],log_moneyness)
price_svi = black_scholes_call_price(100,99.5,8/12,0,imp_vol_svi_8)
print(f"Prix trouvé avec l'interpolation bilinéaire et param pour 9 mois : {price_svi:.4f}")

forward_price_int = forward_price_calculation(S0, r, 1)
log_moneyness_int = log_moneyness_calculation(99.5, forward_price_int)
imp_vol_svi_8_int = svi_fixed([a_val,b_val],[0,0,rho_8m,m_8m,sigma_8m],log_moneyness_int)
price_svi_inter = black_scholes_call_price(100,99.5,8/12,0,imp_vol_svi_8_int)
print(f"Prix trouvé avec l'interpolation bilinéaire de a et b et autres params interpolés pour 8 mois : {price_svi_inter:.4f}")

# %% [markdown]
# 4.7066 vrai prix

# %% [markdown]
# ### 6/ écrire l’algorithme de valorisation d’une option avec le modèle PDV, par Monte Carlo (les seuls rendements considérés seront ceux simulés entre la date courante et la maturité) 

# %%
def K1(t,param):
    return abs(param[3]) * np.exp(-param[3] * t)

def K2(t,param):
    return abs(param[3]) * np.exp(-param[4] * t)

def R(x):
    epsilon = 1e-7  # Une petite constante
    return (x[:,1] / (x[:,0] + epsilon)) - 1
def PDV_model(param,x,t):  
    rt = R(x)
    sigma  = param[0] + param[1]*(sum(([K1(T-i/t,param)*rt for i in range(t)]))) + param[2]*(np.sqrt(sum([K2(T-i/t,param)*(rt**2) for i in range(t)])))
    #print(sigma)
    return sigma

# %%
def valorisation_MC_PDV(S, param,r, nb_simulations, pas, strike, T):
    dt=T/pas

    paths = np.zeros((nb_simulations, pas + 1))
    paths[:, 0:2] = S 
        
    for i in range(1,pas):
        vol = PDV_model(param,paths[0:, i-1:i+1],pas)  #A chaque pas on calcule notre volatilité avec le modèle PDV
        paths[0:, i + 1] = paths[0:, i] * np.exp((r-0.5 * vol **2) * dt + vol * np.sqrt(dt) * np.random.normal(0,1,size=nb_simulations))
    option_payoffs = np.maximum(paths[0:, -1] - strike, 0)
    option_value = np.exp(-r * T) * np.mean(option_payoffs)  #payoff actualisé
    return option_value,paths

# %%
def cost_function_PDV(param,args):
    S0,r, nb_simulations, pas, strikes, maturity,market_prices = args
    dt=maturity/pas
    model_prices = []
    paths = np.zeros((nb_simulations, pas + 1))
    paths[:, 0:2] = S0 
    for strike in strikes:
        for i in range(1,pas):
            vol = PDV_model(param,paths[0:, i-1:i+1],pas)  #A chaque pas on calcule notre volatilité avec le modèle PDV
            paths[0:, i + 1] = paths[0:, i] * np.exp((r-0.5 * vol **2) * dt + vol * np.sqrt(dt) * np.random.normal(0,1,size=nb_simulations))
        option_payoffs = np.maximum(paths[0:, -1] - strike, 0)
        model_prices.append(np.exp(-r * maturity) * np.mean(option_payoffs))  #payoff actualisé
    
    print(param,np.sum((model_prices - market_prices)**2))

    return np.sum((model_prices - market_prices)**2)

# %% [markdown]
# On valorise nos options 1 an et les comparons à notre prix de marché

# %%
# Parameters for the PDV model
lambda_1 = 0.5 # Lambda 1 ne peut pas etre negatif
lambda_2 = 0.3
beta_0 = 0.1 
beta_1 = 0.2  
beta_2 = 0.3   
initial_params_PDV = [
beta_0,
beta_1, 
beta_2,  
lambda_1, 
lambda_2
]

# %%
def check_result_PDV(param,data,maturity):
    nb_simulations = 2000
    maturity=1
    pas=50
    opt = []
    total_error = 0
    for strike,price in zip(data["strikes"],data["prices"]):
        option_price,paths = valorisation_MC_PDV(S0, param,r, nb_simulations, pas, strike, maturity)
        print(f"Prix par simulation de Monte Carlo pour K = {strike} : {option_price:.2f}")
        opt.append(option_price)
        print(f"Market Price = {price:.2f}, Difference = {abs(option_price-price):.2f}")
        total_error+=abs(option_price-price)
    print(f'total error :{total_error:.2f}')
    fig, ax = plt.subplots()

    rects1 = ax.plot(data["strikes"], data["prices"], '-o',label='Market Prices',color='red')
    rects2 = ax.plot(data["strikes"], opt, label='Option Prices MC (PDV)')

    ax.set_xlabel('Strikes')
    ax.set_ylabel('Prices')
    ax.set_title('Market Prices vs Option Prices from Monte Carlo Simulations')
    ax.legend()
    plt.show()

# %%
check_result_PDV(initial_params_PDV,df_12_months,1)

# %%
result_PDV_12 = nelder_mead(cost_function_PDV,np.array(initial_params_PDV),S0,r,500, 5, df_12_months['strikes'], 0.75,df_12_months['prices'])['x']

# %%
check_result_PDV(result_PDV_12,df_12_months,1)

# %%
result_PDV_9 = nelder_mead(cost_function_PDV,np.array(initial_params_PDV),S0,r,2000, 100, df_9_months['strikes'], 0.75,df_9_months['prices'])['x']

# %%
result_PDV_6 = nelder_mead(cost_function_PDV,np.array(initial_params_PDV),S0,r, 10, 5, df_6_months['strikes'],0.5,df_6_months['prices'])['x']

# %%
check_result_PDV(result_PDV_6,df_6_months,0.5)

# %%
result_PDV_3 = nelder_mead(cost_function_PDV,np.array(initial_params_PDV),S0,r, 2000, 100, df_3_months['strikes'],0.25,df_3_months['prices'])['x']
result_PDV_3

# %%
check_result_PDV(result_PDV_3,df_3_months,0.25)

# %%
result_PDV = [result_PDV_12 , result_PDV_9 , result_PDV_6 , result_PDV_3 ]

PDV=pd.DataFrame(result_PDV,index=["12 months","9 months","6 months","3 months"],columns=["beta_0","beta_1","beta_2","lambda_1","lambda_2"])
PDV

# %%
def PDV_model_fixed(param,calibrated_params,x,t):
    #param = [beta_1, beta_2]
    #calibrated_params = [beta_0,beta_1,beta_2,lambda_1,lambda_2]
    rt = R(x)
    sigma  = calibrated_params[0] + param[0]*(sum(([K1(T-i/t,calibrated_params)*rt for i in range(t)]))) + param[1]*(np.sqrt(sum([K2(T-i/t,calibrated_params)*(rt**2) for i in range(t)])))
    return sigma
def cost_function_PDV_fixed(param,args):
    calibrated_params,S0,r, nb_simulations, pas, strikes, maturity,market_price = args
    dt=maturity/pas

    paths = np.zeros((nb_simulations, pas + 1))
    paths[:, 0:2] = S0 
    for i in range(1,pas):
        vol = PDV_model_fixed(param,calibrated_params,paths[0:, i-1:i+1],pas) 
        paths[0:, i + 1] = paths[0:, i] * np.exp((r-0.5 * vol **2) * dt + vol * np.sqrt(dt) * np.random.normal(0,1,size=nb_simulations))
    option_payoffs = np.maximum(paths[0:, -1] - strike, 0)
    model_price = (np.exp(-r * maturity) * np.mean(option_payoffs)) 
    #print(param,np.sum((model_price - market_price)**2))
    return np.sum((model_price - market_price)**2)

# %%
beta_1_values_12m = []
beta_1_values_9m = []
beta_1_values_6m = []
beta_1_values_3m = []
beta_2_values_12m = []
beta_2_values_9m = []
beta_2_values_6m = []
beta_2_values_3m = []

# %%
from tqdm import tqdm

# %%
for k,St in tqdm(zip(strikes_plot, df_12_months['prices'])):
    res_12m = nelder_mead(cost_function_PDV_fixed, np.array([0.1, 0.1]), result_PDV_12,S0,0,2000,50,k,1, St)['x']
    beta_1_12m, beta_2_12m = res_12m
    beta_1_values_12m.append(beta_1_12m)
    beta_2_values_12m.append(beta_2_12m)
for k,St in tqdm(zip(strikes_plot, df_9_months['prices'])):
    res_9m = nelder_mead(cost_function_PDV_fixed, np.array([0.1, 0.1]), result_PDV_9,S0,0,2000,50,k,0.75, St)['x']
    beta_1_9m, beta_2_9m = res_9m
    beta_1_values_9m.append(beta_1_9m)
    beta_2_values_9m.append(beta_2_9m)

for k,St in tqdm(zip(strikes_plot, df_6_months['prices'])):
    res_6m = nelder_mead(cost_function_PDV_fixed, np.array([0.1, 0.1]), result_PDV_6,S0,0,2000,50,k,0.5, St)['x']
    beta_1_6m, beta_2_6m = res_6m
    beta_1_values_6m.append(beta_1_6m)
    beta_2_values_6m.append(beta_2_6m)

for k,St in tqdm(zip(strikes_plot, df_3_months['prices'])):
    res_3m = nelder_mead(cost_function_PDV_fixed, np.array([0.1, 0.1]), result_PDV_3,S0,0,2000,50,k,0.25, St)['x']
    beta_1_3m, beta_2_3m = res_3m
    beta_1_values_3m.append(beta_1_3m)
    beta_2_values_3m.append(beta_2_3m)

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(10,10))
fig.suptitle('SVI Calibrated model for each maturity')

strikes_plot = np.linspace(95, 104, 10)
prices_plot_12m =[] 
for strike,beta_1,beta_2 in zip(strikes_plot,beta_1_values_12m,beta_2_values_12m): 
    st,p = (valorisation_MC_PDV(S0,list([result_PDV_12[0]]) + list([beta_1,beta_2]) + list(result_PDV_12[3:5]),0,2000,50,k,1))
    prices_plot_12m.append(st)
ax1.plot(strikes_plot, prices_plot_12m, label='PDV Model')
ax1.scatter(strikes_plot, df_12_months['prices'], color='red', label='Market Data')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Prices')
ax1.set_title("12-months option")
ax1.legend()

prices_plot_9m =[] 
for strike,beta_1,beta_2 in zip(strikes_plot,beta_1_values_9m,beta_2_values_9m): 
    st,p = (valorisation_MC_PDV(S0,list([result_PDV_9[0]]) + list([beta_1,beta_2]) + list(result_PDV_9[3:5]),0,2000,50,k,1))
    prices_plot_9m.append(st)
ax2.plot(strikes_plot, prices_plot_9m, label='PDV Model')
ax2.scatter(strikes_plot, df_9_months['prices'], color='red', label='Market Data')
ax2.set_xlabel('Strike')
ax2.set_ylabel('Prices')
ax2.set_title("9-months option")
ax2.legend()

prices_plot_6m =[] 
for strike,beta_1,beta_2 in zip(strikes_plot,beta_1_values_6m,beta_2_values_6m): 
    st,p = (valorisation_MC_PDV(S0,list([result_PDV_6[0]]) + list([beta_1,beta_2]) + list(result_PDV_6[3:5]),0,2000,50,k,1))
    prices_plot_6m.append(st)
ax3.plot(strikes_plot, prices_plot_6m, label='PDV Model')
ax3.scatter(strikes_plot, df_6_months['prices'], color='red', label='Market Data')
ax3.set_xlabel('Strike')
ax3.set_ylabel('Prices')
ax3.set_title("6-months option")
ax3.legend()

prices_plot_3m =[] 
for strike,beta_1,beta_2 in zip(strikes_plot,beta_1_values_3m,beta_2_values_3m): 
    st,p = (valorisation_MC_PDV(S0,list([result_PDV_3[0]]) + list([beta_1,beta_2]) + list(result_PDV_3[3:5]),0,2000,50,k,1))
    prices_plot_3m.append(st)
ax4.plot(strikes_plot, prices_plot_3m, label='PDV Model')
ax4.scatter(strikes_plot, df_3_months['prices'], color='red', label='Market Data')
ax4.set_xlabel('Strike')
ax4.set_ylabel('Prices')
ax4.set_title("3-months option")
ax4.legend()

plt.show()


# %%
beta_1_values = beta_1_values_12m+beta_1_values_9m+beta_1_values_6m+beta_1_values_3m
beta_2_values = beta_2_values_12m+beta_2_values_9m+beta_2_values_6m+beta_2_values_3m
strikes = df_concatenated["strikes"]
maturities = df_concatenated['maturity']
vol_market = df_concatenated['implied_vol']

# %%
surface(strikes,maturities,beta_1_values)

# %%
surface(strikes,maturities,beta_2_values)

# %%
def cost_function_reg_beta_1(values,args):
    
    market_prices,lam,mu = args
    values_12m,values_9m,values_6m,values_3m = np.array_split(values,4)

    price_model = []
    for k,beta_1,beta_2 in zip(strikes_plot,values_12m,beta_2_values_12m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_12[0]]) + list([beta_1,beta_2]) + list(result_PDV_12[3:5]),0,1000,100,k,1))
        price_model.append(st)
    for k,beta_1,beta_2 in zip(strikes_plot,values_9m,beta_2_values_9m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_9[0]]) + list([beta_1,beta_2]) + list(result_PDV_9[3:5]),0,1000,100,k,1))
        price_model.append(st)
    for k,beta_1,beta_2 in zip(strikes_plot,values_6m,beta_2_values_6m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_6[0]]) + list([beta_1,beta_2]) + list(result_PDV_6[3:5]),0,1000,100,k,1))
        price_model.append(st)
    for k,beta_1,beta_2 in zip(strikes_plot,values_3m,beta_2_values_3m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_3[0]]) + list([beta_1,beta_2]) + list(result_PDV_3[3:5]),0,1000,100,k,1))
        price_model.append(st)

    error_price = np.sum((market_prices-price_model)**2)
    values_surface = pd.DataFrame()
    values_surface.index = strikes_plot
    values_surface['12'] = values_12m
    values_surface['9'] = values_9m
    values_surface['6'] = values_6m
    values_surface['3'] = values_3m

    error_strike = 0
    for i in range(1,values_surface.shape[0]):
        error_strike += (lam * np.sum(((values_surface.iloc[i]) - values_surface.iloc[i-1])**2))
    error_maturity = 0
    for m in range(3,15,3):
        error_maturity+=(mu * np.sum((values_surface[f'{m}'] - values_surface[f'{m}'])**2))
    
    print(error_price + error_strike + error_maturity)
    
    return error_price + error_strike + error_maturity




# %%
result_reg_beta_1 = nelder_mead(cost_function_reg_beta_1,np.array(beta_1_values),df_concatenated['prices'],1,1)
result_reg_beta_1

# %%
surface(strikes,maturities,beta_1_values)

# %%
surface(strikes,maturities,result_reg_beta_1['x'])

# %%
def cost_function_reg_beta_2(values,args):
    
    market_prices,lam,mu = args
    values_12m,values_9m,values_6m,values_3m = np.array_split(values,4)

    price_model = []
    for k,beta_1,beta_2 in zip(strikes_plot,beta_1_values_12m,values_12m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_12[0]]) + list([beta_1,beta_2]) + list(result_PDV_12[3:5]),0,1000,100,k,1))
        price_model.append(st)
    for k,beta_1,beta_2 in zip(strikes_plot,beta_1_values_9m,values_9m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_9[0]]) + list([beta_1,beta_2]) + list(result_PDV_9[3:5]),0,1000,100,k,1))
        price_model.append(st)
    for k,beta_1,beta_2 in zip(strikes_plot,beta_1_values_6m,values_6m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_6[0]]) + list([beta_1,beta_2]) + list(result_PDV_6[3:5]),0,1000,100,k,1))
        price_model.append(st)
    for k,beta_1,beta_2 in zip(strikes_plot,beta_1_values_3m,values_3m): 
        st,p = (valorisation_MC_PDV(S0,list([result_PDV_3[0]]) + list([beta_1,beta_2]) + list(result_PDV_3[3:5]),0,1000,100,k,1))
        price_model.append(st)

    error_price = np.sum((market_prices-price_model)**2)
    values_surface = pd.DataFrame()
    values_surface.index = strikes_plot
    values_surface['12'] = values_12m
    values_surface['9'] = values_9m
    values_surface['6'] = values_6m
    values_surface['3'] = values_3m

    error_strike = 0
    for i in range(1,values_surface.shape[0]):
        error_strike += (lam * np.sum(((values_surface.iloc[i]) - values_surface.iloc[i-1])**2))
    error_maturity = 0
    for m in range(3,15,3):
        error_maturity+=(mu * np.sum((values_surface[f'{m}'] - values_surface[f'{m}'])**2))
    
    print(error_price + error_strike + error_maturity)
    
    return error_price + error_strike + error_maturity




# %%
result_reg_beta_2 = nelder_mead(cost_function_reg_beta_1,np.array(beta_2_values),df_concatenated['prices'],1,1)
result_reg_beta_2

# %%
surface(strikes,maturities,beta_2_values)

# %%
surface(strikes,maturities,result_reg_beta_2['x'])

# %% [markdown]
# ### 7/ Avec ce dernier modèle, proposer encore un prix, le plus juste possible, pour une option de strike 99.50 et de maturité 8 mois (justifier la méthode retenue)

# %%
df_concatenated

# %%
df_concatenated['beta_1'] = result_reg_beta_1['x']
df_concatenated['beta_2'] = result_reg_beta_2['x']

# %%
x_val = 99.5
y_val = 8/12
beta_1_val = bilinear_interpolation(x_val,y_val,[(99,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 99)]['beta_1'].values)),
                                            (99,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 99)]['beta_1'].values)),
                                        (100,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 100)]['beta_1'].values)),
                                        (100,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 100)]['beta_1'].values))
                                        ]                 
                               )
beta_2_val = bilinear_interpolation(x_val,y_val,[(99,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 99)]['beta_2'].values)),
                                            (99,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 99)]['beta_2'].values)),
                                        (100,0.5,float(df_concatenated[(df_concatenated['maturity'] == 0.5) & (df_concatenated['strikes'] == 100)]['beta_2'].values)),
                                        (100,0.75,float(df_concatenated[(df_concatenated['maturity'] == 0.75) & (df_concatenated['strikes'] == 100)]['beta_2'].values))
                                        ]                 
                               )

# %%
beta_0_8m = linear_interpolation(6,result_PDV_6[0],9,result_PDV_9[0],8)
lambda_1_8m = linear_interpolation(6,result_PDV_6[3],9,result_PDV_9[3],8)
lambda_2_8m = linear_interpolation(6,result_PDV_6[4],9,result_PDV_9[4],8)

# %%
price_pdv_9912_8 = valorisation_MC_PDV(S0,[beta_0_8m,beta_1_val,beta_2_val,lambda_1_8m,lambda_2_8m],0,500,5,99.5,8/12)
print(price_pdv_9912_8[0])

# %% [markdown]
# # III – Test d’algorithmes d’optimisation 
# 

# %% [markdown]
# ###  On veut comparer la performance de trois algorithmes d’optimisation : Nelder-Mead, le recuit simulé et l’essaim particulaire. Pour simplifier le problème on réduit sa dimension et on se place donc dans le cadre d’une estimation et non plus d’une calibration. Il s’agit d’estimer le modèle SVI (à cinq paramètres) qui minimise l’écart quadratique moyen entre prix de marché et prix de modèle pour les 40 options considérées. Trouver le couple de paramètres optimal avec Nelder-Mead puis avec recuit simulé et enfin avec essaim particulaire. Comparer les résultats obtenus : précision de l’estimation, rapidité de la convergence, stabilité de la solution.

# %%
def recuit_simule(f, x0,*args, q=0.2, c0=1e-2, alpha=0.99, max_iter=2000):
    # q = variance de la gaussienne
    # c0 = critère d'arret
    # alpha = taux d'acceptation

    for _ in range(max_iter):
        x_new = np.random.normal(x0, q)  # Tirez un nouveau x à partir d'une distribution q(x|xk)

        delta_f = f(x_new,args) - f(x0,args)

        """
        La probabilité que xk +1 soit accepé est :
                    1 si f (xk +1 ) ≤f (xk ) ;
                    exp(−f (xk +1 )−f (xk ) / ck) sinon;
        """
        if delta_f <= 0 or np.random.rand() < np.exp(-delta_f / c0):
            x0 = x_new

        """
        Parametre de controle initialement grand 
        (grand taux d’acceptation) puis
        décroissance lente : 
        ck +1 = α * ck 
        avec α = 0.99
        """
        c0 *= alpha 

    result = {'x': x0, 'fun': f(x0,args)}
    return result

# %%
result_svi_12m_recuit = recuit_simule(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_12_months["strikes"],df_12_months["implied_vol"],1)
result_svi_9m_recuit = recuit_simule(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_9_months["strikes"],df_9_months["implied_vol"],0.75)
result_svi_6m_recuit = recuit_simule(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_6_months["strikes"],df_6_months["implied_vol"],0.5)
result_svi_3m_recuit = recuit_simule(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_3_months["strikes"],df_3_months["implied_vol"],0.25)

# %%
result_concat_recuit= [result_svi_12m_recuit['x'] , result_svi_9m_recuit['x'] , result_svi_6m_recuit['x'] , result_svi_3m_recuit['x'] ]

res_recuit=pd.DataFrame(result_concat_recuit,index=["12 months","9 months","6 months","3 months"],columns=["a","b","rho","m","sigma"])
res_recuit

# %%
print(result_svi_12m_recuit)
print(result_svi_9m_recuit)
print(result_svi_6m_recuit)
print(result_svi_3m_recuit)

# %%
check_results_svi(result_svi_12m_recuit['x'],df_12_months)

# %%
check_results_svi(result_svi_9m_recuit['x'],df_9_months)

# %%
check_results_svi(result_svi_6m_recuit['x'],df_6_months)

# %%
check_results_svi(result_svi_3m_recuit['x'],df_3_months)

# %%
result_svi_6m = nelder_mead(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_6_months["strikes"],df_6_months["implied_vol"],0.5)['x']
result_svi_3m = nelder_mead(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_3_months["strikes"],df_3_months["implied_vol"],0.25)['x']

# %%
result_svi_9m_recuit = recuit_simule(cost_function_SVI,np.array([0.1,0.1,0.1,0.1,0.1]),df_9_months["strikes"],df_9_months["implied_vol"],0.75)

# %%
def PSO(cost_function, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,*args):
    bounds = np.array(problem_bounds)
    global_best_position = None
    global_best_value = float('inf')
   
    swarm = []
    for i in range(num_particles):
        position = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=bounds.shape[0])
        velocity = np.random.uniform(low=-1, high=1, size=bounds.shape[0])
        best_position = np.copy(position)
        best_value = float('inf')
        swarm.append([position,velocity,best_position,best_value])
       
    for iteration in range(max_iter):
        for particle in swarm:
            value = cost_function(particle[0],args)
            if value < particle[3]:
                particle[3] = value
                particle[2] = np.copy(particle[0])
            if value < global_best_value:
                global_best_value = value
                global_best_position = np.copy(particle[0])
 
        for particle in swarm:
            r_p = np.random.uniform(0, phi_p, size=particle[1].shape)
            r_g = np.random.uniform(0, phi_g, size=particle[1].shape)
            cognitive_component = r_p * (particle[2] - particle[0])
            social_component = r_g * (global_best_position - particle[0])
            particle[1] = w * particle[1] + cognitive_component + social_component
           
            particle[0] += particle[1]
            particle[0] = np.clip(particle[0],bounds[:,0],bounds[:,1])
           
    result = {'x': global_best_position, 'fun': global_best_value}
    return result
 
 

# %% [markdown]
# Test PSO pour le modèle SVI

# %%
# Define the bounds for the problem, for example: [[-10, 10], [-10, 10]] for a 2-dimensional problem
problem_bounds = np.array([[-1, 1], [-1, 1],[-1, 1], [-1, 1],[-1, 1]])

num_particles=30
w=0.7
phi_p=2.05
phi_g=2.05
max_iter=1000

# Initialize PSO with the objective function and bounds
result_12m_PSO_SVI = PSO(cost_function_SVI, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,df_12_months["strikes"],df_12_months["implied_vol"],1)
result_9m_PSO_SVI = PSO(cost_function_SVI, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,df_9_months["strikes"],df_9_months["implied_vol"],0.75)
result_6m_PSO_SVI = PSO(cost_function_SVI, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,df_6_months["strikes"],df_6_months["implied_vol"],0.5)
result_3m_PSO_SVI = PSO(cost_function_SVI, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,df_3_months["strikes"],df_3_months["implied_vol"],0.25)


print(result_12m_PSO_SVI)
print(result_9m_PSO_SVI)
print(result_6m_PSO_SVI)
print(result_3m_PSO_SVI)

# %%
param_PSO_SVI_concat = [result_12m_PSO_SVI['x'] , result_9m_PSO_SVI['x'] , result_6m_PSO_SVI['x'] , result_3m_PSO_SVI['x'] ]

PSO_SVI_param=pd.DataFrame(param_PSO_SVI_concat,index=["12 months","9 months","6 months","3 months"],columns=["a","b","rho","m","sigma"])
PSO_SVI_param

# %%
check_results_svi(result_12m_PSO_SVI['x'],df_12_months)

# %%
check_results_svi(result_9m_PSO_SVI['x'],df_9_months)

# %%
check_results_svi(result_6m_PSO_SVI['x'],df_6_months)

# %%
check_results_svi(result_3m_PSO_SVI['x'],df_3_months)

# %%
problem_bounds = np.array([[0, 2], [-2, 2],[-2, 2], [-2, 2],[-2, 2]])

num_particles=30
w=0.7
phi_p=0.5
phi_g=0.5
max_iter=1000


# %% [markdown]
# Test PSO pour PDV

# %%
result_12m_PDV = PSO(cost_function_PDV, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,S0,0,1000,100,df_12_months["strikes"],1,df_12_months["prices"])
result_9m_PDV = PSO(cost_function_PDV, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,S0,0,1000,100,df_9_months["strikes"],0.75,df_9_months["prices"])
result_6m_PDV = PSO(cost_function_PDV, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,S0,0,1000,100,df_6_months["strikes"],0.5,df_6_months["prices"])
result_3m_PDV = PSO(cost_function_PDV, problem_bounds,num_particles, w, phi_p, phi_g, max_iter,S0,0,1000,100,df_3_months["strikes"],0.25,df_3_months["prices"])

print(result_12m_PDV)
print(result_9m_PDV)
print(result_6m_PDV)
print(result_3m_PDV)

# %%
check_result_PDV(result_12m_PDV,df_12_months,1)


