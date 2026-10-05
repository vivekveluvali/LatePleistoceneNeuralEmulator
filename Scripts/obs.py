from run_config import PROJECT  # project root (LPNE/), found relative to run_config.py - no hard-coded paths
import pandas as pd
import numpy as np
import numpy.random as random
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

print('imports done')

DATAFOLDER = {
    # gas_age_calBP is in years BP -> divide by 1000 for a unified 'Age[kaBP]'
    # CH4_1s is a 1-sigma error -> Variance = CH4_1s**2
    'ch4_EDC': pd.read_csv(f"{PROJECT}/Data/Observations/edc-ch4-2008-noaa.txt", comment = "#", delim_whitespace = True)
        .assign(**{'Age[kaBP]': lambda d: d['gas_age_calBP'] / 1000, 'Variance': lambda d: d['CH4_1s'] ** 2})[['CH4_mean', 'gas_age_calBP', 'Age[kaBP]', 'Variance']],
    # no error column reported -> Variance = None
    'MgCa_Eld': pd.read_csv(f'{PROJECT}/Data/Observations/Elderfield_2012/datasets/181-1123_Mg_Ca.tab', comment = "#", delim_whitespace = True).drop_duplicates(subset='Age[kaBP]').reset_index(drop=True)
        .assign(Variance = None)[['Mg/Ca[mmol/mol](Normalized)', 'Age[kaBP]', 'Variance']],

    # no error column reported -> Variance = None
    'dO18_EDC': pd.read_csv(f"{PROJECT}/Data/Observations/EDC_d18O.tab", comment = "#", delim_whitespace = True).drop_duplicates(subset='Age[kaBP]').reset_index(drop=True)
        .assign(Variance = None)[['δ18O-O2[pmille]', 'Age[kaBP]', 'Variance']],
    # d18O_StdDev_pm is a stddev -> Variance = d18O_StdDev_pm**2
    'dO18_LR04': pd.read_csv(f"{PROJECT}/Data/Observations/Lisiecki_Raymo_2005/datasets/Global_stack_d18O.tab", comment = "#", delim_whitespace = True).drop_duplicates(subset='Age[kaBP]').reset_index(drop=True)
        .assign(**{'Variance': lambda d: d['d18O_StdDev_pm'] ** 2})[['d18O_stack[‰]', 'Age[kaBP]', 'Variance']],
    # no error column reported -> Variance = None
    'dO18_Eld': pd.read_csv(f'{PROJECT}/Data/Observations/Elderfield_2012/datasets/181-1123_d18O.tab', comment = "#", delim_whitespace = True)
        .assign(Variance = None)[['δ18O_H2O[‰SMOW]', 'Age[kaBP]', 'Variance']],

    # gas_ageBP is in years BP -> divide by 1000 for a unified 'Age[kaBP]'
    # no error column reported -> Variance = None
    'co2_EDC': pd.read_csv(f"{PROJECT}/Data/Observations/co2-2008-composite-EDC-noaa_Vodstock.txt", comment = "#", delim_whitespace = True).drop_duplicates(subset='gas_ageBP').reset_index(drop=True)
        .assign(**{'Age[kaBP]': lambda d: d['gas_ageBP'] / 1000, 'Variance': None})[['CO2', 'gas_ageBP', 'Age[kaBP]', 'Variance']],
    # pCO2StdDev[±] is a stddev -> Variance = pCO2StdDev[±]**2
    'co2_Hon': pd.read_csv(f"{PROJECT}/Data/Observations/Honisch2009_pCO2.tab", comment = "#", delim_whitespace = True).drop_duplicates(subset='Age[kaBP]').reset_index(drop=True)
        .assign(**{'Variance': lambda d: d['pCO2StdDev[±](uncertainty_Calculated)'] ** 2})[['pCO2water_SST_wet[µatm](constantAlkalinity_Calculated)', 'Age[kaBP]', 'Variance']],
    # no error column reported -> Variance = None
    'co2_Yam': pd.read_csv(f"{PROJECT}/Data/Observations/CO2_YamamotoEtAl2022.csv", comment = '#', delim_whitespace = True).drop_duplicates(subset='Age[kaBP]').reset_index(drop=True)
        .assign(Variance = None)[['Est_CO2_linear', 'Age[kaBP]', 'Variance']],

    # insolation is deterministic and does not have a variance
    'insol': pd.read_csv(f'{PROJECT}/Data/Observations/huybers06_65north_labeled.csv')
        .assign(Variance = None)[['ISI_thresh0Wm2', 'Age[kaBP]']],
}

AGE_COL, VAR_COL = 'Age[kaBP]', 'Variance'

NAMES = {
    'ch4_EDC': 'CH4_mean',
    'MgCa_Eld': 'Mg/Ca[mmol/mol](Normalized)',

    'dO18_EDC': 'δ18O-O2[pmille]',
    'dO18_LR04': 'd18O_stack[‰]',
    'dO18_Eld': 'δ18O_H2O[‰SMOW]',

    'co2_EDC': 'CO2',
    'co2_Hon': 'pCO2water_SST_wet[µatm](constantAlkalinity_Calculated)',
    'co2_Yam': 'Est_CO2_linear',

    'insol': 'ISI_thresh0Wm2',
}

CUTOFF = 4000 # in ka, originally in 10ka
print('variables established')

def select_data(name):
    if name in DATAFOLDER:
        return DATAFOLDER[name]
    else:
        print('\n Given name not present. Either Add new name or use: \n' \
        '- temp :  ch4_EDC, MgCa_Eld\n' \
        '- ice  :  dO18_EDC, dO18_LR04, dO18_Eld\n'\
        '- CO2  :  co2_EDC, co2_Hon, co2_Yam\n'\
        '- insol')
        return -1

def SM90(t,system,R,n=0):
    """
    Saltzman-Maasch 1990 model (SM90) from "A first-order global model of late Cenozoic climatic change";
    Late Pleistocene solution; 
    pages 320-321 

    t      - time 
    system - state vector
    R      - insolation forcing 
    n      - statistical forcing (e.g. normally distributed noise) 
    """

    X_t, Y_t, Z_t, R_t = system[0], system[1], system[2], R(CUTOFF-t)

    # random variable for noise inclusion if noise is non-yero
    b = random.choice([True, False])

    # prefactors for Late Pleistocene solution of the SM90 model 
    v = 0.2
    u = 0.6
    p = 1.0
    r = 0.9
    s = 1.0
    w = 0.5
    q = 2.5
    
    dX_dt = - X_t - Y_t - v*Z_t - u*R_t + b*n 
    dY_dt = - p*Z_t + r*Y_t + s*Z_t**2 - w*Y_t*Z_t - Z_t**2*Y_t + b*n 
    dZ_dt = - q*(X_t + Z_t) + b*n 

    return np.array([dX_dt, dY_dt, dZ_dt])

def create_SM90(dt = 1, system_0 = np.array([-1.0,0,1])):
    i = select_data('insol')
    R_Df = i[i[AGE_COL]<=CUTOFF].copy()
    #R_Df[AGE_COL] *= 1e-1 -- I don't think I need to use this since I'm staying in units of ka rather than 10ka
    R_Df = R_Df[::-1]

    # scaling insolation to mean of 0 and variance of 1 in accordance with SM90
    R_Scaler = StandardScaler().fit(R_Df[NAMES['insol']].values.reshape(-1, 1))
    R_Df[NAMES['insol']] = R_Scaler.transform(R_Df[NAMES['insol']].values.reshape(-1, 1))

    # interpolate forcing since solve_ivp uses adaptable time step
    R_forcing = interp1d(
        R_Df[AGE_COL],
        R_Df[NAMES['insol']],
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # simulation time range
    t = np.arange(0,CUTOFF,dt)

    # numerical solver
    sol = solve_ivp(SM90,t_span=[0,CUTOFF],y0=system_0,t_eval=t,args=(R_forcing,0))
    return pd.DataFrame({
        AGE_COL: sol.t,
        'X': sol.y[0, :],
        'Y': sol.y[1, :],
        'Z': sol.y[2, :],
        NAMES['insol']: R_forcing(sol.t),
    })

def gp_reg(source):
    data = select_data(source)
    x = np.array(data[AGE_COL])
    y = np.array(data[NAMES[source]])
    variance = data[VAR_COL]

    x_fine = np.arange(0, x.max() + 1, 1) # create grid with 1ka step size

    if variance.notna().all():
        variance = np.array(variance) / np.var(y) #normalize variance data
        kernel = Matern(length_scale=50, length_scale_bounds=(1, 1000), nu = 2.5)
        #+ WhiteKernel(noise_level = variance, noise_level_bounds=(variance*1e-1, variance*1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha = variance, n_restarts_optimizer=5, normalize_y=True)
    else:
        kernel = Matern(length_scale=50, length_scale_bounds=(1, 1000), nu = 2.5)\
        + WhiteKernel(noise_level_bounds=(1e-5, 1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gpr.fit(x.reshape(-1, 1), y)

    # Predict on the fine grid
    y_fine, y_std = gpr.predict(x_fine.reshape(-1, 1), return_std=True)
    return pd.DataFrame({AGE_COL: x_fine, NAMES[source]: y_fine, VAR_COL: y_std ** 2})

print('functions defined')

if __name__ == "__main__":
    # builds the GPR-interpolated observation table + SM90 numerical run and
    # writes them out; only runs when obs.py is executed directly, so that
    # importing DATAFOLDER/select_data/etc. elsewhere (e.g. standardplots.py)
    # doesn't trigger this every time
    outfile = select_data('insol')
    print('constructing Gaussian process regression')
    for name in NAMES:
        if name == 'insol':
            continue
        columns = gp_reg(name).rename(columns={VAR_COL: f'{VAR_COL}_{name}'})
        outfile = pd.merge(outfile, columns, on=AGE_COL, how='outer')
    outfile = outfile.sort_values(AGE_COL).reset_index(drop=True)

    print('GPR exported, now constructing SM90 trajectory')
    sm90_out = create_SM90()

    print('SM90 trajectory constructed, exporting data')
    sm90_out.to_csv(f'{PROJECT}/Data/SM90/sm90.csv')
    outfile.to_csv(f'{PROJECT}/Data/InterpolatedObs/gpr_{outfile[AGE_COL].max()}kaBP.csv')

