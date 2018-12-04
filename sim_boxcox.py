import numpy as np
import pandas as pd
import pysal as ps
import matplotlib.pyplot as plt

NSIMS = 100
NOBS = 2000

x_names = ["hhveh", "hhsize", "hhinc", "sex", "taz_avg_price", "autotrip", "activetrip"]

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
    y_name = "workdist"
    y = db[y_name].values
    y = y[:, np.newaxis]
    # Convert distances in meters to 10s of km
    y = y / 10**4
    # Update average zone price variable to scale and remove nan (replace with average for region)
    db['taz_avg_price'] = db['taz_avg_price'] / 10**4
    db['taz_avg_price'].fillna(73.98787585, inplace=True)
    x = db[x_names].values
    ww = np.loadtxt('D:/PhD/Thesis/estimation/Spatial_SEM_Model/data/weight{0}.csv'.format(i), delimiter=',')
    ww = ((ww < 1000.0) & (ww > 0.0))*1
    w = ps.weights.util.full2W(ww)
    mllag = ps.spreg.ML_Lag_BC(y,x,w,name_y=y_name,name_x=x_names,name_ds=ds_name)
    beta_array[i-1, :] = np.squeeze(mllag.betas)
    std_err_array[i-1, :] = np.squeeze(mllag.std_err)
    # print(mllag.summary)

# Print the lists of beta parameters and standard errors for all simulations
print(np.mean(beta_array, axis=0))
print(np.mean(std_err_array, axis=0))

beta_means, beta_std = tuple(np.mean(beta_array, axis=0)), tuple(np.std(beta_array, axis=0))
std_err_means, std_err_std = tuple(np.mean(std_err_array, axis=0)), tuple(np.std(std_err_array, axis=0))

ind = np.arange(len(beta_means))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, beta_means, width, yerr=beta_std,
                color='SkyBlue', label='Betas')
rects2 = ax.bar(ind + width/2, std_err_means, width, yerr=std_err_std,
                color='IndianRed', label='Standard Errors')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_title('Bootstrapped Mean and Standard Errors for 100 Samples')
ax.set_xticks(ind)
ax.set_xticklabels(("CONSTANT","HHVEH", "HHSIZE", "HHINC", "SEX", "AVG PRICE", "AUTOTRIPS", "ACTIVETRIPS", "RHO", "LAMBDA"))
ax.legend()
ax.xticks(rotation=45)

plt.savefig('boxcox.png')