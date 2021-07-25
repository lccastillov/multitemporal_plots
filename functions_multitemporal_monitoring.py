import pandas as pd
import numpy as np
def import_hdf5(VI, df_age, hdf5_file,csv_image_anomalies_vs_plots):
    import pandas as pd
    df = df_age
    print("list of columns", list(df.columns))

    # Read hdf file into pandas dataframe
    hdf = pd.HDFStore(hdf5_file, mode='r')

    # print("hdfkeys",hdf.keys()) #Shows all the keys within the hdf5 file

    df_mean_vi = hdf.get('/PlotStats/PlotMeanVal')
    df_norm_mean_vi = hdf.get('/NormalPlotStats/NormalMean')

    # Retrieve percentage of Lower and upper anomalies per plot per date (from hdf5 file)
    df_low_mean_vi = hdf.get('/LowPlotStats/low_Mean') #Percentage of low anomalies
    df_up_mean_vi = hdf.get('/UpPlotStats/up_Mean')#Percentage of high anomalies
    DF_numLowerAnomalies = hdf.get('/PlotStats/percLowAnomalies') #Number of pixels anomalous (low anomalies)
    DF_numUpperAnomalies = hdf.get('/PlotStats/percUpperAnomalies')#Number of pixels anomalous (highanomalies)

    # Create empty lists to store temporary lower and upper anomalies
    mean_vi_Array = []
    mean_normal_vi_array = []
    mean_low_vi_array = []
    mean_up_vi_array = []
    lowerAnomalies = []
    upperAnomalies = []

    # Loop to go through all the indexes of the df and retrieve the respective percentage of anomalies
    for index, row in df.iterrows():
        plot = row['IDLote']
        date = row['Image_date'].replace("-", "")

        meanNDVI = df_mean_vi.loc[int(float(plot))][str(date)]
        mean_normal_vi = df_norm_mean_vi.loc[int(float(plot))][str(date)]
        low_mean_VI = df_low_mean_vi.loc[int(float(plot))][str(date)]
        up_mean_VI = df_up_mean_vi.loc[int(float(plot))][str(date)]
        numLowerAnomalies = DF_numLowerAnomalies.loc[int(float(plot))][str(date)]
        numUpperAnomalies = DF_numUpperAnomalies.loc[int(float(plot))][str(date)]

        # print("mean",meanNDVI, "low",low_mean_VI, "up_mean_VI",up_mean_VI,'numLowerAnomalies',numLowerAnomalies,"numUpperAnomalies", numUpperAnomalies)

        # Show which of the plots are potentially clouded for specific dates
        # if meanNDVI == -999:
        # print("The plot", row['Codigo'], " is covered by clouds on ", row['Image_date'])

        # Append the current values to the lists of lower and upper anomalies respectively
        mean_vi_Array.append(meanNDVI)
        lowerAnomalies.append(numLowerAnomalies)
        upperAnomalies.append(numUpperAnomalies)
        mean_normal_vi_array.append(mean_normal_vi)
        mean_low_vi_array.append(low_mean_VI)
        mean_up_vi_array.append(up_mean_VI)

    # Create  new columns in the dataframe that will store the percentage of low and high anomalies
    mean_vi_colum_title = 'mean_' + VI
    df['mean_vi'] = mean_vi_Array
    df['LowerAnomalies'] = lowerAnomalies
    df['UpperAnomalies'] = upperAnomalies
    df['normalmean_vi'] = mean_normal_vi_array
    df['lowmean_vi'] = mean_low_vi_array
    df['upmean_vi'] = mean_up_vi_array

    # Remove the plots potentially clouded

    # Remove plots with very low VI values (clouded)
    df = df[df.mean_vi != -999]
    df.to_csv(csv_image_anomalies_vs_plots, index=False)
    return [df]



def daily_vi_statistics(df, max_day,vi,metrics_day_columns_df):
    import pandas as pd
    day_list = np.arange(1, max_day + 1)


    daily_stats_df = pd.DataFrame(index=day_list, \
                                  columns=metrics_day_columns_df)

    # Define indx
    daily_stats_df.index.name = 'age'

    #### First calculate exploratory statistics to remove outliers for each age (days)
    for day in day_list:
        # print("Estimating statistics for ", age, " days")

        df_current_day = df[(df.age == day)].copy()

        Q1 = df_current_day.normalmean_vi.quantile(0.25)

        Q3 = df_current_day['normalmean_vi'].quantile(0.75)

        IQR = Q3 - Q1
        min_thres = Q1 - 1.5 * IQR
        max_thres = Q3 + 1.5 * IQR
        mean = df_current_day['normalmean_vi'].mean()
        std = df_current_day['normalmean_vi'].std()
        count = df_current_day['normalmean_vi'].count()
        median = df_current_day['normalmean_vi'].median()

        df_non_outliers = df_current_day
        [(df_current_day.age >= min_thres) | (df_current_day.age <= max_thres)]
        # df_non_outliers=df_current_age[(df_current_age.age >= min_thres) & df_current_age(df_current_age.age <= max_thres)]

        daily_stats_df.loc[day]['age'] = day
        daily_stats_df.loc[day]['count'] = count
        daily_stats_df.loc[day]['period_mean'] = mean
        daily_stats_df.loc[day]['std'] = std
        daily_stats_df.loc[day]['low_thres'] = min_thres
        daily_stats_df.loc[day]['up_thres'] = max_thres
        daily_stats_df.loc[day]['IQR'] = IQR
        daily_stats_df.loc[day]['Q1'] = Q1
        daily_stats_df.loc[day]['Q3'] = Q3
        daily_stats_df.loc[day]['vi'] = vi
        daily_stats_df.loc[day]['period_median'] = median


    # print("statistics_df_all_ages 124 function",statistics_df_all_ages)
    # statistics_df_all_ages['age'] = statistics_df_all_ages['age'].astype(int)
    # If we want to record statistics into a csv file
    # statistics_df_all_ages.to_csv(m_root+'/Anomalies_interpretation/stats_'+s2_vi+'.csv', index=False)
    return [daily_stats_df]

def conditions_weeks (df):
    #Retrieve number of week
    #Check https://www.uaex.edu/publications/pdf/mp192/chapter-2.pdf
    def f(row):
        if row['age'] <= 7:
            val = 1
        elif row['age'] <= 14 :
            val = 2
        elif row['age'] <= 21:
            val = 3
        elif row['age'] <= 28:
            val = 4
        elif row['age'] <= 35:
            val = 5
        elif row['age'] <= 42:
            val = 6
        elif row['age'] <= 49:
            val = 7
        elif row['age'] <= 56:
            val = 8
        elif row['age'] <= 63:
            val = 9
        elif row['age'] <= 70:
            val = 10
        elif row['age'] <= 77:
            val = 11
        elif row['age'] <= 84:
            val = 12
        elif row['age'] <= 56:
            val = 13
        elif row['age'] <= 56:
            val = 14
        elif row['age'] <= 42:
            val = 6
        elif row['age'] <= 49:
            val = 7
        elif row['age'] <= 56:
            val = 8
        elif row['age'] <= 63:
            val = 9
        elif row['age'] <= 70:
            val = 10
        elif row['age'] <= 77:
            val = 11
        elif row['age'] <= 84:
            val = 12
        elif row['age'] <= 91:
            val = 13
        elif row['age'] <= 98:
            val = 14
        elif row['age'] <= 105:
            val = 15
        elif row['age'] <= 112:
            val = 16
        elif row['age'] <= 119:
            val = 17
        elif row['age'] <= 126:
            val = 18
        elif row['age'] <= 133:
            val = 19
        elif row['age'] <= 140:
            val = 20
        elif row['age'] <= 147:
            val = 21
        elif row['age'] <= 154:
            val = 22
        elif row['age'] <= 161:
            val = 23
        elif row['age'] <= 168:
            val = 24
        elif row['age'] <= 175:
            val = 25
        elif row['age'] <= 182:
            val = 26
        else:
            val = 26

        return val

    df['week_campaign']=df.apply(f, axis=1)
    return[df]


def df_no_yield_outliers(df, yield_low_threshold, yield_high_threshold, min_vi, max_week_campaign_duration):
    # This function drops the data associated to those plots which yield is an outlier based on the interquartile \
    # range outlier criteria
    import numpy as np
    # create emtpy df
    df["yield_type"] = ""

    df["yield_type"] = np.where(df['Yield'] <= yield_low_threshold, \
                                "low yield", df['yield_type']).copy()
    df["yield_type"] = np.where(df['Yield'] > yield_high_threshold, \
                                "high_yield", df['yield_type'])

    # Only high yields
    df = df[df['yield_type'] == 'high_yield'].copy()

    # Delete weird values (lower then NDvi 0.3 in weeks highern than 21)
    # this max week refers to the maximum expected week before harvesing

    drop_outliers = df[(df['week_campaign'] >= max_week_campaign_duration) & (df['mean_vi'] < min_vi)]
    indexNames = drop_outliers.index
    df.drop(indexNames, inplace=True)
    return [df]

def delete_outliers_iqr_day (df, column_name, day):
    current_day_df=df[df['age']==day]
    Q1 = current_day_df[column_name].quantile(0.25)

    Q3 = current_day_df[column_name].quantile(0.75)

    IQR = Q3 - Q1
    upper_outlier_threshold=Q3 + 1.5 * IQR
    lower_outlier_threshold=Q1-1.5*IQR

    df_drop = df[((df[column_name] <= (Q1 - 1.5 * IQR))&(df['week_campaign'] == day))|\
                 ((df[column_name] >= (Q3 + 1.5 * IQR))&(df['week_campaign'] == day))]
    indexNames =df_drop.index

    return [indexNames]

def fill_gaps(metrics_df, periodicity):
    #Fills gaps using the mean
    if periodicity == "weekly":
        metrics_df = metrics_df.set_index('week_campaign').copy()
    elif periodicity == "daily":
        metrics_df = metrics_df.set_index('age').copy()

    # Filling using mean or median
    # Creating a column in the dataframe
    # instead of : df['NewCol']=0, we use
    # df = df.assign(NewCol=default_value)
    # to avoid pandas warning.

    # fill_mean is the column with the filled values
    metrics_df = metrics_df.assign(fill_mean=metrics_df.period_median.fillna(metrics_df.period_median. \
                                                                             rolling(11, min_periods=1, ).mean()))
    #min_periods: Minimum number of observations in window required to have a value
    return[metrics_df]


def std_approach_thresholds(metrics_df):

    metrics_df['up_threshold'] = metrics_df['period_median_filter'] + metrics_df['std']
    metrics_df['low_threshold'] = metrics_df['period_median_filter'] - metrics_df['std']

    # More extreme thresholds
    metrics_df['top_threshold'] = metrics_df['period_median_filter'] + 2.5 * metrics_df['std']
    metrics_df['bottom_threshold'] = metrics_df['period_median_filter'] - 2.5 * metrics_df['std']

    return [metrics_df]

def plot_pattern_thresholds (metrics_df,vi):


    #def plot_pattern_thresholds(x, y_median, y_median_filter, extreme_thres1, extreme_thres2, normal_thres1, normal_thres2, df_current):
    import matplotlib.pyplot as plt
    import seaborn as sns
    metrics_df.reset_index(inplace=True)

    plt.figure(figsize=(15, 6))
    plt.title(vi+ " change over time")
    plt.plot(metrics_df.age,metrics_df.period_median_filter, color='blue',label="Baseline")

    #sns.scatterplot(x, y_median, data=weekly_stats_df, s=35)
    plt.plot(metrics_df.age, metrics_df.up_threshold, color='green')
    plt.plot(metrics_df.age, metrics_df.low_threshold, color='green')
    plt.plot(metrics_df.age,metrics_df.top_threshold, color='orange')
    plt.plot(metrics_df.age,metrics_df.bottom_threshold, color='orange')
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Median NDVI/plot', fontsize=14)
    plt.legend()
    ### TO plot points
    #plt.scatter(df_current['week_campaign'], df_current['mean_vi'], data=df_current, s=35)
    plt.show()

