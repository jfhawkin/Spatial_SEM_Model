# Box Cox model for work distance

import numpy as np
import pandas as pd
import pysal as ps
import matplotlib.pyplot as plt

NSIMS = 100
NOBS = 2000
# Indicates whether to estimate Box-Cox transformation
BC = True
# Indicates whether to calculate LM statistics
LM = False

# x_names = ["lic", "hhsize", "hhinc", "sex", "taz_avg_price", "autotrip", "activetrip", "t_long", "t_local"]
# x_names = ["lic", "hhsize", "hhinc", "sex", , "autotrip", "activetrip", "transpass"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "autotrip", "activetrip", "t_GO", "t_local", "t_PRESTO"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "autotrip", "activetrip", "t_long", "t_local"]
# x_names = ["taz_pop_dens", "taz_area_res"]
x_names = ["taz_pop_dens"]
# z_names = ["lic", "hhsize", "sex", "activetrip", "t_long", "t_local", "age"]
# z_names = ["hhsize", "activetrip", "veh_adult", "taz_area_com", "dwelltype_2", "dwelltype_3"]
# z_names = ["hhsize", "activetrip", "veh_adult", "dwelltype_2", "dwelltype_3", "occ_2", "occ_3", "occ_4", "occ_5"]
z_names = ["hhsize", "activetrip", "veh_adult", "taz_region_23"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "activetrip", "t_long", "t_local", "taz_area_com"]
# x_names = ["lic", "hhsize", "hhinc", "sex", "activetrip", "t_long", "t_local", "taz_area_park"]
y_name = "shopdist"


beta_array = np.zeros((NSIMS, len(x_names) + len(z_names) + 3))
std_err_array = np.zeros((NSIMS, len(x_names) + len(z_names) + 3))

for i in range(1, NSIMS+1, 1):
    df = pd.read_csv('D:/PhD/Thesis/estimation/Spatial_SEM_Model/data/sample{0}.csv'.format(i), header=None)
    # Add the column names to the dataframe
    df.columns = ["uno", "sero", "householdid", "householdzone", "daycaredist", "facdist", "homedist", "shopdist",
                  "othdist",
                  "schooldist", "workdist", "transtrip", "activetrip", "autotrip", "gotrip", "taxitrip", "ubertrip",
                  "dwelltype",
                  "hhveh", "hhinc", "hhsize", "hhworkers", "hhstudents", "hhadults", "hhchildren", "hhtrips", "age",
                  "sex",
                  "empstat", "occ", "studstat", "lic", "transpass", "unusualplaceofwork", "worktrips", "schooltrips",
                  "taz_region",
                  "taz_area_m", "taz_area_h", "taz_mun", "taz_pd", "taz_avg_price", "taz_med_price", "taz_new_list",
                  "taz_childcare",
                  "taz_school", "taz_shop_trips", "taz_pop", "taz_area_com", "taz_area_gov", "taz_area_res",
                  "taz_area_ind", "taz_area_park"]
    ds_name = "sample{0}.csv".format(i)

    # Update taz area columns
    df = df.drop(['taz_area_m', 'taz_area_h'], axis=1)
    df = df.reindex()
    df1 = pd.read_csv('taz_area.csv')
    df = df.merge(df1, left_on='householdzone', right_on='householdzone', how='left')

    # Update average zone price variable to scale and remove nan (replace with average for region)
    df['taz_avg_price'] = df['taz_avg_price'] / 10 ** 4
    df['taz_avg_price'].fillna(73.98787585, inplace=True)

    # Population density in persons per hectare
    df['taz_pop_dens'] = df['taz_pop'] / df['taz_area_h']
    # For the few records with no pop in taz table, assume average pop density
    row_mask = df.taz_pop_dens == 0
    df.loc[row_mask, 'taz_pop_dens'] = 60
    df['child'] = (df['hhchildren'] > 0) * 1

    # PRESTO and GO are similar, so combine
    df['t_long'] = ((df['transpass'] == 1) | (df['transpass'] == 2) | (df['transpass'] == 6)).astype(int)
    df['t_local'] = ((df['transpass'] == 3) | (df['transpass'] == 5)).astype(int)
    # Age categories
    #df['age_25'] = (df['age'] <=25).astype(int)
    df['age_26_64'] = ((df['age'] > 25) & (df['age'] < 65)).astype(int)
    df['age_65'] = (df['age'] >= 65).astype(int)
    # Get dummy variables
    # iv = df.dwelltype.values # Need to set before getting dummies because dummy function drops the original column
    df = pd.get_dummies(df, columns=['dwelltype', 'occ', 'studstat', 'empstat', 'taz_region', 'taz_pd'], drop_first=True)
    df['taz_region_23'] = (df['taz_region_2']==1 | (df['taz_region_3']==1)).astype(int)
    df['empstat_25'] = (df['empstat_2'] == 1 | (df['empstat_5'] == 1)).astype(int)
    # Vehicles per person variables
    df['veh_adult'] = df['hhveh'] / df['hhadults']
    df['veh_worker'] = df['hhveh'] / df['hhworkers']

    ww = np.loadtxt('D:/PhD/Thesis/estimation/Spatial_SEM_Model/data/weight{0}.csv'.format(i), delimiter=',')
    ww = ((ww < 1000.0) & (ww > 0.0))*1
    # Convert weights to inverse km
    # ww[ww==0] = 99999
    # ww = (ww / 10**3)**-1
    # np.fill_diagonal(ww, 0)
    # Update weights for same income class check
    # iv = df.hhinc.values
    # # Update weights for same hhchildren status
    # iv = df.child.values
    # im = np.repeat(iv, len(iv)).reshape(-1, len(iv))
    # jm = im.T
    # bm = (im == jm)*1
    # np.fill_diagonal(bm, 0)
    # ww = ww * bm

    # Remove zero elements
    lst_NW = df[df[y_name] == 0].index.tolist()
    ww = np.delete(ww, lst_NW, axis=0)
    ww = np.delete(ww, lst_NW, axis=1)
    df = df[df[y_name] > 0]

    y = df[y_name].values
    y = y[:, np.newaxis]
    # Convert distances in meters to km
    y = y / 10 ** 3
    x = df[x_names].values
    z = df[z_names].values

    w = ps.weights.util.full2W(ww)
    if BC:
        if LM:
            mllag = ps.spreg.ML_Lag_BC(y, x, z, w, name_y=y_name, name_x=x_names, name_z1=z_names, name_ds=ds_name, LM=True)
        else:
            mllag = ps.spreg.ML_Lag_BC(y, x, z, w, name_y=y_name, name_x=x_names, name_z1=z_names, name_ds=ds_name,
                                       LM=False)
    else:
        mllag = ps.spreg.ML_Lag(y, np.concatenate((x, z), axis=1), w, name_y=y_name, name_x=x_names+z_names, name_ds=ds_name)

    if NSIMS > 1:
        beta_array[i - 1, :] = np.squeeze(mllag.betas)
        std_err_array[i - 1, :] = np.squeeze(mllag.std_err)
        print("Finished regression run {0}".format(i))
    else:
        print(mllag.summary)

if NSIMS > 1:
    # Print the lists of beta parameters and standard errors for all simulations
    print(np.mean(beta_array, axis=0))
    print(np.mean(std_err_array, axis=0))

    beta_means, beta_std = tuple(np.mean(beta_array, axis=0)), tuple(np.std(beta_array, axis=0))
    std_err_means, std_err_std = tuple(np.mean(std_err_array, axis=0)), tuple(np.std(std_err_array, axis=0))

    ind = np.arange(len(beta_means))  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 10))
    rects1 = ax.bar(ind - width / 3, beta_means, width, yerr=beta_std,
                    color='SkyBlue', label='Betas')
    rects2 = ax.bar(ind + width / 3, std_err_means, width, yerr=std_err_std,
                    color='IndianRed', label='Standard Errors')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value')
    ax.set_title('Bootstrapped Mean and Standard Errors for 100 Samples')
    ax.set_xticks(ind)
    ax.set_xticklabels(("CONSTANT", "POP DENS", "HH SIZE", "ACTIVE TRIPS", "VEH RATE", "REGION 23", "RHO", "LAMBDA"))
    ax.legend()
    plt.xticks(rotation=45)

    plt.savefig('boxcox_shop.png')
