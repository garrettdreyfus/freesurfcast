from netCDF4 import Dataset
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

def download_and_format(url,outputname,filetitle):

    dataset= Dataset(url)
    out = Dataset(outputname,mode="w",format='NETCDF4_CLASSIC')
    out.title = filetitle

    times = dataset.variables["time"][:]
    lon = dataset.variables["lon"][:]
    lat = dataset.variables["lat"][:]

    out.createDimension('lat',len(lat))
    out.createDimension('lon',len(lon))
    out.createDimension('time',len(times))

    lat = out.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lat[:] =  dataset.variables["lat"][:]
 

    lon = out.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees east'
    lon.long_name = 'longitude'
    lon[:] =  dataset.variables["lon"][:]

    time = out.createVariable('time', np.float32, ('time',))
    time.units = 'days since 1-1-1 00:00:0.0'
    time.long_name = 'time'
    time[:] =  dataset.variables["time"][:]


    htsgwsfc = out.createVariable('htsgwsfc', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    htsgwsfc.units = 'meters'
    htsgwsfc.long_name = 'surface significant height of combined wind waves and swell'
    htsgwsfc[:,:,:] =  dataset.variables["htsgwsfc"][:,:,:]

    perpwsfc = out.createVariable('perpwsfc', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    perpwsfc.units = 'seconds'
    perpwsfc.long_name = 'surface primary wave mean period'
    perpwsfc[:,:,:] =  dataset.variables["perpwsfc"][:,:,:]
 
    dirpwsfc = out.createVariable('dirpwsfc', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    dirpwsfc.units = 'degrees'
    dirpwsfc.long_name = 'surface primary wave direction'
    dirpwsfc[:,:,:] =  dataset.variables["dirpwsfc"][:,:,:]


    wdirsfc = out.createVariable('wdirsfc', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    wdirsfc.units = 'degrees'
    wdirsfc.long_name = 'surface wind direction'
    wdirsfc[:,:,:] =  dataset.variables["wdirsfc"][:,:,:]

    windsfc = out.createVariable('windsfc', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    windsfc.units = 'm/s'
    windsfc.long_name = 'surface wind speed'
    windsfc[:,:,:] =  dataset.variables["windsfc"][:,:,:]


    swell1 = out.createVariable('swell_1', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    swell1.units = 'm'
    swell1.long_name = '1 in sequence significant height of swell waves [m]'
    swell1[:,:,:] =  dataset.variables["swell_1"][:,:,:]


    swdir_1 = out.createVariable('swdir_1', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    swdir_1.units = 'm'
    swdir_1.long_name = '1 in sequence direction of swell waves'
    swdir_1[:,:,:] =  dataset.variables["swdir_1"][:,:,:]

    swell2 = out.createVariable('swell_2', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    swell2.units = 'm'
    swell2.long_name = '2 in sequence significant height of swell waves [m]'
    swell2[:,:,:] =  dataset.variables["swell_2"][:,:,:]


    swdir_2 = out.createVariable('swdir_2', np.float32, ('time','lat','lon'),fill_value=dataset.variables["htsgwsfc"]._FillValue)
    swdir_2.units = 'm'
    swdir_2.long_name = '2 in sequence direction of swell waves'
    swdir_2[:,:,:] =  dataset.variables["swdir_2"][:,:,:]



    out.close()


def nearest_not_nan(ds,lon,lat):
    nanmask = np.isnan(ds.htsgwsfc[0].values)
    X,Y = np.meshgrid(ds.lat.values,ds.lon.values)
    X[nanmask.T] = np.nan
    coord = np.nanargmin((X-lat)**2+(Y-lon)**2)
    coord = np.unravel_index(coord,X.shape)
    return coord



def display_forecast(ds,lon_surfspot,lat_surfspot):
    directions = ["↑","↗","→","↘","↓","↙","←","↖"]
    if lon_surfspot<0:
        lon_surfspot +=360
    coord = nearest_not_nan(ds,lon_surfspot,lat_surfspot)
    atspot = ds.isel(lon=coord[0],lat=coord[1])
    #atspot = atspot.resample(time="6H").mean()
    atspot["swell_1"] = np.round(atspot.swell_1*3.28084,1)
    atspot["swell_2"] = np.round(atspot.swell_2*3.28084,1)
    atspot["windsfc"] = np.round(atspot.windsfc*2.237,1)
    sw1arrow_i = np.round(atspot["swdir_1"].values/45+4).astype(int)%8
    sw2arrow_i = np.round(atspot["swdir_2"].values/45+4).astype(int)%8
    warrow_i = np.round(atspot["wdirsfc"].values/45+4).astype(int)%8

    df = pd.DataFrame(atspot.time)
    df = df[0].dt.tz_localize("US/Pacific")
 
    table = [["Time","Swell 1", "Swell 2", "Wind" ]]
    for i in range(len(atspot.time)):
        row = []
        timestr = "{}-{} at {}H".format(df[i].month, df[i].day, df[i].hour)
        row.append(timestr)
        row.append(str(directions[sw1arrow_i[i]]) + str(atspot.swell_1.values[i]))
        row.append(str(directions[sw2arrow_i[i]]) + str(atspot.swell_2.values[i]))
        row.append(str(directions[warrow_i[i]]) + str(atspot.windsfc.values[i]))
        table.append(row)
    table = np.asarray(table)
    print(tabulate(table, tablefmt='html'))




    #atspot = ds.interp(lon=lon_surfspot,lat=lat_surfspot)
    

#url="http://nomads.ncep.noaa.gov:80/dods/wave/gfswave/20240609/gfswave.wcoast.0p16_18z"
#download_and_format(url,"data/westcoast.nc","West Coast GFSwave model data")
ds = xr.open_dataset("data/westcoast.nc")
#ocean park
#display_forecast(ds,-118.488521,33.999145)
#no pass
display_forecast(ds,-124.085028,40.019993)
