## ---------------------------
##
## Script name: Multi_temporal_monitoring.py
##
## Purpose of script:
#                   1. Retrieve typical temporal patterns of vegetation indices (VIs) or satellite products using historical \
#                   data from multiple plots at different ages.
#                   2. Compare current VI values per plot with typical values based on the analysis of historical data

##
## Author: Liliana Castillo Villamor. Affiliation: Aberystwyth University

##
## Date Created: 03/05/2021
##
## Copyright (c) Liliana Castillo Villamor, 2021
## Email: lic42@aber.ac.uk
##
## ---------------------------
##
## Notes:
## the functions used in this script are kept in the file functions_multitemporal_monitoring.py
## Description of fields in the input files is presented in "input_files_description"
##
## ---------------------------
##
##
## --------------------------
#################################################################
## --------------------------
## To run this script on Docker
"""
docker run -i -t -v /Users/lilianacastillo/Documents/Data_Analysis:/data lccastillov/histogram_analysis_v2   \
python /data/Danper_ES/multitemporal_plots/Multi_temporal_monitoring.py
"""

## --------------------------
###### Import Functions #####

import time
import datetime
import random
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os
import warnings

from functions_multitemporal_monitoring import *


# Switch warnings off
warnings.simplefilter(action='ignore', category=FutureWarning)

## ---------------------------

## Declare variables
m_root='/data'
input_files_folder=m_root+'/Danper_ES/input_files'
s2_vi_list = ['NDVI8'] # List of VIs to be assessed

# Columns of dataframe that stores stats per day
metrics_day_columns_df = [ 'age', 'vi', 'count', 'period_mean', 'period_median', 'std', 'low_thres', \
                          'up_thres', 'IQR', 'Q1', 'Q3']
daily_stats_df_total = pd.DataFrame( columns=metrics_day_columns_df)

## ---------------------------

## Read csv with field info per plot  into a pandas dataframe (df_age)
plots_age_csv=input_files_folder+'/plots_age.csv'
df_age=pd.read_csv(plots_age_csv)

## ---------------------------
## Data cleaning

# Change fields data types
df_age['Yield']=df_age['Yield'].astype(float)
df_age['Month_Ini_Campana']=df_age['Month_Ini_Campana'].astype(int)
df_age['age']=df_age['age'].astype(float)
df_age['Ini_Campana'] = pd.to_datetime(df_age['Ini_Campana'])
df_age['Fin_Campana'] = pd.to_datetime(df_age['Fin_Campana'])
df_age['Image_date_dateformat'] = pd.to_datetime(df_age['Image_date'])

# Filter using harvest date. i.g. Only images acquired before harvest date
df_age=df_age[df_age['Image_date_dateformat']<=df_age['Fin_Campana']].copy()

# Delete cosecha stage as the VIs are shown as very low values and can affect the final averages
df_age=df_age[(df_age['Phenology']!= 'Harvest')&(df_age['Phenology']!= 'Pre-Harvest')]

#Exclude any plot older than 50 days presented as being in sprouting (Sprouting does not occur at such ages
drop_sprouting=df_age[(df_age['age']>60)&(df_age['Phenology']=='Sprouting')]
indexNames =drop_sprouting.index
df_age.drop(indexNames , inplace=True)

## ---------------------------
## Yield stats
yield_min=df_age['Yield'].min()
yield_max=df_age['Yield'].max()
yield_median=df_age['Yield'].median()
yield_mean=df_age['Yield'].mean()
yield_std=df_age['Yield'].std()
yield_count=df_age['Yield'].count()

# Define thresholds to classify yields
yield_low_threshold=yield_median-0*yield_std
yield_high_threshold=yield_median+0.5*yield_std
yield_high_threshold=yield_median-0.5*yield_std


###################

max_day = 140

# Maximum week to retrieve statistics of VIs.
# Identified as the latest week in which all plots have not been harvested yet
# This, as the harvest might start eralier or later sometimes, this is the week in which,
#for sure, al the plots are still in the filed (not harvested)

max_week_campaign_duration=20
min_vi = 0.3


for vi in s2_vi_list:


    hdf5_file = m_root + '/Danper_ES/input_files/SEN2_HistoMetrics' + vi + '.hdf5' #hdf file with VI stats per plot

    #csv file that stores the result of the union of field data and VI stats per plot
    csv_image_anomalies_vs_plots = \
        m_root + '/Danper_LaLibertad/Anomalies_interpretation/Sen2_' + vi + '_csv_image_anomalies_vs_plots_v2.csv'

    # Retrieve VI stats per plot from hdf5 file and merging them with field  data per plot
    df, = import_hdf5(vi, df_age,hdf5_file,csv_image_anomalies_vs_plots)

    #Maximum crop age to include in the stats
    max_day = 150

    df, = conditions_weeks(df) # Indicate the correspondent week for each day

    days_list = np.arange(1, max_day + 1)

    #############

    # As we are creating the typical pattern, we include only the yields that are not outliers
    df_nooutliers, = df_no_yield_outliers(df, yield_low_threshold, yield_high_threshold, min_vi= \
        min_vi, max_week_campaign_duration=max_week_campaign_duration)

    ############
    periodicity = 'daily'
    for day in days_list:
        # delete  outliers per week based on the mean VI values per plot (i.e. mean vi)
        column_name = 'mean_vi'
        indexNames, = delete_outliers_iqr_day(df, column_name, day)
        df.drop(indexNames, inplace=True)



    # Stats of daily data without outliers (from yield and mean_vi perspective)
    daily_stats_df_current, = daily_vi_statistics(df_nooutliers, max_day,vi,metrics_day_columns_df)


    #Append stats for current day to main stats dataframe
    daily_stats_df_total = daily_stats_df_total.append(daily_stats_df_current)

    # When less that 20 records per day, assign Null to period median
    daily_stats_df_total['period_median'] = \
        np.where(daily_stats_df_total['count'] < 5, np.nan, daily_stats_df_total['period_median'])


    metrics_df,=fill_gaps(metrics_df=daily_stats_df_total, periodicity='daily')

