# Box Cox model for work distance

import numpy as np
import pandas as pd
import pysal as ps
import matplotlib.pyplot as plt

NSIMS = 1
NOBS = 2000
# Indicates whether to estimate Box-Cox transformation
BC = 0

# x_names = ["lic", "hhsize", "hhinc", "sex", "taz_avg_price", "autotrip", "activetrip", "t_long", "t_local"]
# x_names = ["lic", "hhsize", "hhinc", "sex", , "autotrip", "activetrip", "transpass"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "autotrip", "activetrip", "t_GO", "t_local", "t_PRESTO"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "autotrip", "activetrip", "t_long", "t_local"]
x_names = ["taz_avg_price"]
z_names = ["lic", "hhsize", "hhinc", "sex", "activetrip", "t_long", "t_local"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "activetrip", "t_long", "t_local", "taz_area_com"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "activetrip", "t_long", "t_local", "taz_area_park"]
y_name = "workdist"

beta_array = np.zeros((NSIMS, len(x_names) + 3))
std_err_array = np.zeros((NSIMS, len(x_names) + 3))

for i in range(1, NSIMS+1, 1):
    db = pd.read_csv('D:/PhD/Thesis/estimation/Spatial_SEM_Model/data/sample{0}.csv'.format(i), header=None)
    # Add the column names to the dataframe
    db.columns = ["uno", "sero", "householdid", "householdzone", "daycaredist", "facdist", "homedist", "shopdist", "othdist",
                  "schooldist", "workdist", "transtrip", "activetrip", "autotrip", "gotrip", "taxitrip", "ubertrip", "dwelltype",
                  "hhveh", "hhinc", "hhsize", "hhworkers", "hhstudents", "hhadults", "hhchildren", "hhtrips", "age", "sex",
                  "empstat", "occ", "studstat", "lic", "transpass", "unusualplaceofwork", "worktrips", "schooltrips", "taz_region",
                  "taz_area_m", "taz_area_h", "taz_mun", "taz_pd", "taz_avg_price", "taz_med_price", "taz_new_list", "taz_childcare",
                  "taz_school", "taz_shop_trips", "taz_pop", "taz_area_com", "taz_area_gov", "taz_area_res", "taz_area_ind", "taz_area_park"]
    ds_name = "sample{0}.csv".format(i)
    # Update average zone price variable to scale and remove nan (replace with average for region)
    db['taz_avg_price'] = db['taz_avg_price'] / 10**4
    db['taz_avg_price'].fillna(73.98787585, inplace=True)

    # PRESTO and GO are similar, so combine
    db['t_long'] = ((db['transpass'] == 1) | (db['transpass'] == 2) | (db['transpass'] == 6)).astype(int)
    db['t_local'] = ((db['transpass'] == 3) | (db['transpass'] == 5)).astype(int)

    ww = np.loadtxt('D:/PhD/Thesis/estimation/Spatial_SEM_Model/data/weight{0}.csv'.format(i), delimiter=',')
    ww = ((ww < 1000.0) & (ww > 0.0))*1
    # Update weights for same income class
    i = db.hhinc.values
    im = np.repeat(i, len(i)).reshape(-1, len(i))
    jm = im.T
    bm = (im == jm)*1
    np.fill_diagonal(bm, 0)
    ww = ww * bm
    # ww = ww / 10**3
    # ww = 1 / ww
    # ww[ww==np.inf] = 0
    # Remove zero elements
    lst_NW = db[db[y_name] == 0].index.tolist()
    ww = np.delete(ww, lst_NW, axis=0)
    ww = np.delete(ww, lst_NW, axis=1)
    db = db[db[y_name] > 0]

    y = db[y_name].values
    y = y[:, np.newaxis]
    # Convert distances in meters to km
    y = y / 10 ** 3
    x = db[x_names].values
    z = db[z_names].values

    w = ps.weights.util.full2W(ww)
    if BC == 1:
        mllag = ps.spreg.ML_Lag_BC(y, x, z, w, name_y=y_name, name_x=x_names, name_z1=z_names, name_ds=ds_name, LM=True)
    else:
        mllag = ps.spreg.ML_Lag(y, np.concatenate((x, z), axis=1), w, name_y=y_name, name_x=x_names+z_names, name_ds=ds_name)

    if NSIMS == 100:
        beta_array[i-1, :] = np.squeeze(mllag.betas)
        std_err_array[i-1, :] = np.squeeze(mllag.std_err)
    else:
        print(mllag.summary)

if NSIMS == 100:
    # Print the lists of beta parameters and standard errors for all simulations
    print(np.mean(beta_array, axis=0))
    print(np.mean(std_err_array, axis=0))

    beta_means, beta_std = tuple(np.mean(beta_array, axis=0)), tuple(np.std(beta_array, axis=0))
    std_err_means, std_err_std = tuple(np.mean(std_err_array, axis=0)), tuple(np.std(std_err_array, axis=0))

    ind = np.arange(len(beta_means))  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 10))
    rects1 = ax.bar(ind - width/3, beta_means, width, yerr=beta_std,
                    color='SkyBlue', label='Betas')
    rects2 = ax.bar(ind + width/3, std_err_means, width, yerr=std_err_std,
                    color='IndianRed', label='Standard Errors')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value')
    ax.set_title('Bootstrapped Mean and Standard Errors for 100 Samples')
    ax.set_xticks(ind)
    ax.set_xticklabels(("CONSTANT", "HHVEH", "HHSIZE", "HHINC", "SEX", "AVG PRICE", "AUTOTRIPS", "ACTIVETRIPS", "RHO", "LAMBDA"))
    ax.legend()
    plt.xticks(rotation=45)

    plt.savefig('boxcox.png')