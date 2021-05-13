import folium
from folium import plugins
from IPython.display import clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

FIGSIZE = np.array([10, 5])

descent_dict = {'A': 'Other Asian',
                'B': 'Black',
                'C': 'Chinese',
                'D': 'Cambodian',
                'F': 'Filipino',
                'G': 'Guamanian',
                'H': 'Hispanic/Latin/Mexican',
                'I': 'American Indian/Alaskan Native',
                'J': 'Japanese',
                'K': 'Korean',
                'L': 'Laotian',
                'O': 'Other',
                'P': 'Pacific Islander',
                'S': 'Samoan',
                'U': 'Hawaiian',
                'V': 'Vietnamese',
                'W': 'White',
                'X': 'Unknown',
                'Z': 'Asian Indian'}


def format_mil_string(x):
    """ format_mil_string - format military time integer to a 'HH:MM'
                            string.
    @param x: integer between 0-2359.
    @return string of format 'HH:MM'
    """
    x = str(x).zfill(4)  # left pad if less than 4 digits
    minutes = x[-2:]
    if len(x) == 3:
        hour = x[:1]
    else:
        hour = x[:2]
    return hour + ':' + minutes


def split_date(df, column, dt_range):
    """ split_date - allows us to create a new column on the dataframe
                     for month, year, and day.
    @param df - dataframe to get the data and add the column to.
    @param column - column name in df of datetime format
    column dt_range - string indicating what date type we want:
                        'YR' - year, 'MTH' - month, 'DAY' - day
    """
    if dt_range == 'YR':
        df[column + ' ' + dt_range] = df[column].dt.year
    if dt_range == 'MTH':
        df[column + ' ' + dt_range] = df[column].dt.month
    if dt_range == 'DAY':
        df[column + ' ' + dt_range] = df[column].dt.day
    return df


def create_crime_group_df(df, value):
    """ get_codes_from_desc - extract the list of codes whose description matches
                                the substring of the regular expression.
    @param df - dataframe with column 'Crm Cd Desc'
    @param value - string that can be used as a regular expression.
    @return df - a dataframe with columns ['Crm Cd', 'Crm Cd Desc'] where
                 each row crime description matches the regex 'value'.
    """
    # surround value in a regular expression so we can use it as a substring
    regex = rf'.*{value}.*'
    df = df[df['Crm Cd Desc'].str.match(regex) == True].reset_index()
    return df


def get_crime_subset(df, value, crime_dict):
    """ get_crime_subset - Filter the dataframe to only have the records matching 
                            specific crime codes.
    @param df - dataframe with column 'Crm Cd Desc'
    @param value - string of category of crime requested.
    @param crime_dict - dictionary of crimes {crime_category: dataframe of crime 
                                                                codes in that category}
    @return df - crime dataframe where each row has crime codes in the 
                 crime_dict[value] dataframe.
    """
    crimes_subset_mask = df['Crm Cd'].isin(crime_dict[value]['Crm Cd'])
    return df[crimes_subset_mask]


def replace_descent_code(x):
    x = str(x)
    if not x or x == '' or x == 'nan' or x == '-':
        return 'Unknown'
    return descent_dict[x]


def remove_prefix(x):
    """ remove_prefix - some cities have a prefix showing there are in LA.
                        eg. 'Los Angeles city - Tujunga'
                        Use this function to remove it.
    @param x - string of the cityname
    @return string without the prefix.
    """
    prefix = 'Los Angeles city - '
    if prefix in x:
        return x[len(prefix):]


def trim_df(df, column, cutoff):
    """ trim_df - Takes a dataframe grouped by some time frame and trims off
                  the last value if it's incomplete (if last date is not the 
                  latest available date).
    @param df - dataframe containing a datetime column 'column'
    @param column - the name of the datetime columnn
    @param cutoff - the latest available date in our crime data (in datetime 
                    format)
    @return trimmed_df - dataframe of the same format as df.
    """
    end = df[column].tolist()[-1]
    # check if complete
    if (end - cutoff).days != 0:
        # drop end data
        trimmed_df = df[df[column] != end]
    return trimmed_df


def plot_area_range(start, end, area_df, area_dict):
    plt.figure(figsize=FIGSIZE)
    for i in range(start, end):
        plt.plot(area_df[i], label=area_dict[i])
    plt.legend()
    plt.xticks(rotation=70)
    plt.show()


def plot_map(crime_data, crime_dict, crime_label, law_enforcement_df, num_records, police, sheriff=False, year=0):
    """ plot_map - Plots a sample of the locations of the crime type (red) as well as 
                   the locations of the police departments (blue).
    @param crime_data - full crime dataframe
    @param crime_label - category of crime to plot. String type.
    @param law_enforcement_df - dataframe of police/sheriff station locations (LAT/LON).
    @param police - boolean on whether to show police stations or not
    @param sheriff - boolean on whether to show sheriff stations or not
    @param num_records - a sample of the total data to read plot (Suggested <= 10K).
    @param year
    """

    # Get the GeoJSON info for drawing the border of LA County
    link = 'https://raw.githubusercontent.com/ritvikmath/StarbucksStoreScraping/master/laMap.geojson'
    f = requests.get(link)

    la_area = f.text

    # use the two lines below instead if we want to make the map bigger or smaller
    #f = folium.Figure(width=1250, height=750)
    #laMap = folium.Map(location=[34.0522,-118.2437], tiles='Stamen Toner', zoom_start=9).add_to(f)
    laMap = folium.Map(location=[34.0522, -118.2437],
                       tiles='Stamen Toner', zoom_start=9)

    # year is by far the best filter for reducing records. Do this first to make the dataframe much smaller
    # if a specific year is requested, filter for only that year
    if year > 0:
        year_mask = crime_data['DATE OCC'].dt.year.astype(int) == year
        year_subset_df = crime_data[year_mask]

    # get only the crime records associated with this category of crime
    crime_subset_df = get_crime_subset(year_subset_df, crime_label, crime_dict)

    # avoid plotting invalid records
    valid_records_mask = pd.eval(
        'crime_subset_df.LAT != 0 & crime_subset_df.LON != 0')
    crime_subset_df = crime_subset_df[valid_records_mask]

    # due to resource restrictions, sample the data rather than plot all the data
    if len(crime_subset_df) >= num_records:
        crime_subset_df = crime_subset_df.sample(num_records)

    # add the shape of LA County to the map
    folium.GeoJson(la_area).add_to(laMap)

    # for each row in the homicide dataset, plot the corresponding latitude and longitude on the map
    for i, row in crime_subset_df.iterrows():
        folium.CircleMarker((row.LAT, row.LON),
                            radius=2,
                            weight=1,
                            color='red',
                            fill_color='red',
                            fill_opacity=.5,
                            prefer_canvas=True).add_to(laMap)
    # for each row in police_stations dataset, plot the corresponding latitude and longitude on the map
    for i, row in law_enforcement_df.iterrows():
        fill = 'blue'
        # if this coordinate is a police station
        if row['cat3'] == 'Police Stations':
            # do we want to show police stations?
            if police:
                fill = 'blue'
            else:
                continue
        # if this coordinate is a sheriff station
        elif row['cat3'] == 'Sheriff Stations':
            # do we want to show sheriff stations?
            if sheriff:
                fill = 'fuchsia'
            else:
                continue
        # add this coordinate to the map
        folium.CircleMarker((row.Y, row.X),
                            radius=6,
                            weight=3,
                            fill=True,
                            fill_color=fill,
                            color='white',
                            fill_opacity=1).add_to(laMap)
    # add a heatmap to the plot
    laMap.add_child(plugins.HeatMap(data=crime_subset_df[['LAT', 'LON']].to_numpy(),
                                    radius=10,
                                    blur=8))

    display(laMap)

    # save the map as an html if you want to reference it later
    # laMap.save('LA_Crime_Map_'+crime_label+'.html')


def plot_func(category, year, police, sheriff, crime_data, crime_dict, law_stations, n_samples):
    """ plot_func - handler for the widgets to call our plotting function.
    @param category - category widget object
    @param year - year widget object
    @param police - police widget object
    @param sheriff - sheriff widget object
    @param crime_data - full crime dataset
    @param crime_dict - dictionary map of crime category to crime codes
    @param law_stations - law enforcement dataframe of coordinates
    @param n_samples - number of records to map each time (for performance reasons)
    """
    clear_output()
    print('Crunching data. Please be patient.')
    plot_map(crime_data,
             crime_dict,
             category,
             law_stations,
             n_samples,
             police,
             sheriff,
             year=year)


def plot_crimes_by_count(crimes_df, crime_dict, plot_type, latest_date, grouping='Tri-Annually'):
    """ plot_crimes_by_count - plot a dataframe of crimes based on whether
                                those codes can be found in crime_dict. Plot
                                is grouped by date based on 'grouping'.
    @param crimes_df - dataframe with columns ['DATE OCC', 'Crm Cd'] 
    @param crime_dict - dictionary of all the crime types we want to plot.
                        key = crime category string, value = dataframe with
                        column 'Crm Cd'
    @param plot_type - type of plot we want ('Line', 'Hist')
    @param grouping - frequency we want to group by ['Monthly', 'Tri-Annually', 'Bi-Annually', 'Yearly']
    """
    # make a map of grouping->frequency. Use END of periods (the 'S' part of 'AS')
    frequency_dict = {'Monthly': '1MS', 'Tri-Annually': '3MS',
                      'Bi-Annually': '6MS', 'Yearly': 'AS'}
    fig = plt.figure(figsize=FIGSIZE)

    # plot each crime type one by one
    for key, value in crime_dict.items():
        # get a subset of the crime based user requested crime type
        crime_by_code = crimes_df[crimes_df['Crm Cd'].isin(value['Crm Cd'])]
        # get counts for each date range in this crime subset
        crime_by_date_code = crime_by_code.groupby([pd.Grouper(key='DATE OCC',
                                                               freq=frequency_dict[grouping],
                                                               closed='left')]).size()\
                                                                               .reset_index()\
                                                                               .rename(columns={0: 'Counts'})
        bins = crime_by_date_code['DATE OCC'].tolist()
        # avoid plotting the last row if that date range is incomplete
        crimes_date_code_trimmed = trim_df(crime_by_date_code,
                                           'DATE OCC',
                                           latest_date)
        if plot_type == 'Line':
            plt.plot(crimes_date_code_trimmed.set_index('DATE OCC'), label=key)
        if plot_type == 'Hist':
            plt.hist(crimes_date_code_trimmed['DATE OCC'],
                     weights=crimes_date_code_trimmed['Counts'],
                     bins=bins,
                     alpha=0.5,
                     align='left',
                     label=key)
        # monthly will have too many bins and be unreadable
        if grouping == 'Monthly':
            bins = bins[0::6]  # use every 6th xtick/bin
            plt.xticks(bins[:-1], rotation=80)
            plt.tight_layout()
        else:
            plt.xticks(bins[:-1], rotation=80)
        plt.legend()

    plt.grid(True, linestyle='-', linewidth=0.5, zorder=0)
    plt.ylabel('Number of incidents')
    plt.xlabel('Date of incident')
    plt.show()


def call_category_plot(categories, grouping, plot_type, crime_data, crime_dict, output1, latest_date):
    """ call_category_plot - handler for the widgets to call our plotting function.
    @param categories - categories widget object
    @param grouping - grouping widget object
    @param plot_type - plot_type widget object
    """
    crime_list = categories
    output1.clear_output()
    # don't try and plot if no categories selected
    if not crime_list:
        return None
    # create a dictionary subset with the selected crime_category/crime_codes only
    dict_subset = {c: crime_dict[c] for c in crime_list}
    # create a dataframe with the selected crimes only
    subset_df = crime_data.iloc[0:0, :].copy()
    for i in crime_list:
        subset_df = subset_df.append(
            get_crime_subset(crime_data, i, dict_subset))
    # draw and output the dataframe subset
    with output1:
        print('Crunching data. Please be patient.')
        plot_crimes_by_count(subset_df, dict_subset,
                             plot_type, latest_date, grouping,)


def violin_plot(df, year, title, sex_descent):
    """ violin_plot - easy way to plot a violin plot using a dataframe.
    @param df - dataframe containing a subset of the crime_data dataframe.
    @param year - the year of crime we wish to plot
    @param title - the name of the crime category such as 'ASSAULT' 
                   (if 'ALL Crime', plots all crime)
    @param sex_descent - string indicating what we are plotting count/age against. 
                         If 'Vict Sex', plot sex, if 'Vict Descent', plot descent.
    """
    # get a subset based on the year
    mask = (df['DATE OCC YR'] == year)
    # get a subset of the data based on victim age (if applicable)
    if sex_descent == 'Vict Age':
        mask = (mask & (df['Vict Age'] >= 0))
    df = df[mask]

    scale = 1
    # Allow setting different size figure if we graph all crime
    if title == 'ALL Crime':
        scale = 1

    figure_size = scale*FIGSIZE

    # plot the violin
    plt.figure(figsize=figure_size+[0, 5])
    sns.violinplot(data=df,
                   y=sex_descent,
                   x='Vict Age',
                   scale='width',
                   orient='h',
                   cut=0)
    plt.title(label='Age and ' + sex_descent + ' distribution for '+title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # some descents are very small, plot the large ones in one count plot and
    # the rest in another
    if sex_descent == 'Vict Descent':
        # get the large population rows
        large_pops = ['Hispanic/Latin/Mexican',
                      'Unknown', 'White', 'Black', 'Other']
        larger_pops_df = df[df['Vict Descent'].isin(large_pops)]
        # plot
        plt.figure(figsize=figure_size)
        plt.title(label=sex_descent +
                  ' distribution for ' + title + ' (Part 1)')
        sns.countplot(data=larger_pops_df,
                      y=sex_descent)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        # next plot will be everything else
        df = df[~df['Vict Descent'].isin(large_pops)]
        title += ' (Part 2)'

    # plot the count plot
    plt.figure(figsize=figure_size)
    plt.title(label=sex_descent + ' distribution for ' + title)
    sns.countplot(data=df,
                  y=sex_descent)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def call_violin_plot(category, year, sex_descent, crime_data, crime_dict, output2):
    """ call_violin_plot - handler for the widgets to call our plotting function.
    @param category - category widget object
    @param year - year widget object
    @param sex_descent - sex_descent widget object
    """
    output2.clear_output()
    # create a dataframe with the selected crimes only
    if category == 'ALL Crime':
        subset_df = crime_data
    else:
        subset_df = get_crime_subset(crime_data, category, crime_dict)
    with output2:
        print('Crunching data. Please be patient.')
        violin_plot(subset_df, year, category, sex_descent)
