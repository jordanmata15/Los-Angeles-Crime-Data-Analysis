import folium
from folium import plugins
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

FIGSIZE = np.array([10, 5])

# map of descent codes to group name
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

# map general crime types to regex descriptions (some crimes are in multiple categories)
crime_desc_map = {
    'ANIMAL_CRIME':     'ANIMALS.*|.*BEASTIALITY',
    'ASSAULT':          'ASSAULT',
    'BURGLARY':         'BURGLARY.*|.*PROWLER.*|.*SHOPLIFTING',
    'DISRUPTION':       'DISTURB.*|.*FAILURE TO DISPERSE.*|.*DISRUPT' +
                        '.*|.*BLOCKING DOOR INDUCTION CENTER',
    'MINOR_UNDERAGE':   'CHILD.*|.*CHLD.*|.*DRUGS, TO A MINOR',
    'WEAPON':           'SHOTS FIRED.*|.*FIREARM.*|.*WEAPON',
    'FRAUD':            'EXTORTION.*|.*BRIBERY.*|.*FALSE POLICE REPORT.*|.*FRAUD' +
                        '.*|.*COUNTERFEIT.*|.*UNAUTHORIZED.*|.*BUNCO',
    'HOMICIDE':         'HOMICIDE.*|.*MANSLAUGHTER',
    'KIDNAPPING':       'FALSE IMPRISONMENT.*|.*KIDNAP.*|.*HUMAN TRAFFICKING' +
                        '.*|.*CHILD STEALING',
    'ROBBERY':          'ROBBERY.*|.*PURSE SNATCHING.*|.*PICKPOCKET.*|.*DRUNK ROLL',
    'SEXUAL_CRIME':     'RAPE.*|.*LEWD.*|.*ABORTION.*|.*PEEPING TOM' +
                        '.*|.*ORAL COPULATION.*|.*PIMPING.*|.*SEX' +
                        '.*|.*INDECENT EXPOSURE.*|.*PANDERING.*|.*CHILD PORNOGRAPHY',
    'STALKING':         'STALKING.*|.*RESTRAINING ORDER',
    'THEFT':            'THEFT.*|.*STOLEN.*|.*TILL TAP',
    'THREATS':          'THREAT.*|.*BOMB SCARE',
    'VANDALISM':        'DUMPING.*|.*ARSON.*|.*THROWING OBJECT AT MOVING VEHICLE' +
                        '.*|.*VANDALISM.*|.*TELEPHONE PROPERTY - DAMAGE' +
                        '.*|.*TRAIN WRECKING.*|.*TRESPASSING',
    'VIOLENT':          'BATTERY.*|.*RESISTING ARREST.*|.*LYNCHING.*|.*INCITING A RIOT',
    'VEHICLE':          'VEHICLE.*|.*DRIVE.*|.*DRIVING.*|.*FAILURE TO YIELD',
    'OTHER':            'MISCELLANEOUS.*|.*CONSPIRACY.*|.*CONTRIBUTING' +
                        '.*|.*DOCUMENT WORTHLESS.*|.*CONTEMPT OF COURT' +
                        '.*|.*BIGAMY.*|.*VIOLATION OF COURT ORDER'}

# descriptions that involve a gun
gun = ["HAND GUN", "SEMI-AUTOMATIC PISTOL", "UNKNOWN FIREARM",
       "REVOLVER", "OTHER FIREARM", "SHOTGUN",
       "ASSAULT WEAPON/UZI/AK47/ETC", "SEMI-AUTOMATIC RIFLE",
       "STARTER PISTOL/REVOLVER", "SAWED OFF RIFLE/SHOTGUN",
       "AIR PISTOL/REVOLVER/RIFLE/BB GUN"]

# descriptions that involve a blade
blade = ["OTHER KNIFE", "KNIFE WITH BLADE 6INCHES OR LESS",
         "UNKNOWN TYPE CUTTING INSTRUMENT",
         "KITCHEN KNIFE", "OTHER CUTTING INSTRUMENT",
         "KNIFE WITH BLADE OVER 6 INCHES IN LENGTH",
         "FOLDING KNIFE", "MACHETE", "SCISSORS", "SWORD"]

# description for location types
street = ["STREET", "SIDEWALK", "PARKING LOT", "ALLEY",
          "PARKING UNDERGROUND/BUILDING", "DRIVEWAY"]
house = ["SINGLE FAMILY DWELLING", "MULTI-UNIT DWELLING (APARTMENT, DUPLEX, ETC)",
         "YARD (RESIDENTIAL/BUSINESS)", "OTHER RESIDENCE", "PORCH, RESIDENTIAL",
         "CONDOMINIUM/TOWNHOUSE"]
market = ["OTHER BUSINESS", "RESTAURANT/FAST FOOD", "DEPARTMENT STORE", "MARKET",
          "OTHER STORE", "DRUG STORE", "DIY CENTER (LOWE'S,HOME DEPOT,OSH,CONTRACTORS WAREHOUSE)"]
vehicle = ["VEHICLE, PASSENGER/TRUCK", "GARAGE/CARPORT", "GAS STATION"]


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
    @return df a dataframe with the added column appended
    """
    if dt_range == 'YR':
        df[column + ' ' + dt_range] = df[column].dt.year
    if dt_range == 'MTH':
        df[column + ' ' + dt_range] = df[column].dt.month
    if dt_range == 'DAY':
        df[column + ' ' + dt_range] = df[column].dt.day
    return df


def create_crime_group_df(df, value):
    """ create_crime_group_df - extract the list of codes whose description matches
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
    """ replace_descent_code - 
    @param x - string type of code ('X', 'F', etc.)
    @return string of the expanded out name ('Unknown', 'Filipino', etc.)
    """
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


def plot_area_range(area_df):
    """ plot_area_range - plots the crime of each general area
                          by year.
    @param area_df - dataframe indexed by station # and year.
    """
    area_map = {1: 'Hollywood',
                2: 'South County',
                3: 'Western SFV',
                4: 'Central LA',
                5: 'Western Beach',
                6: 'Eastern SFV'}
    plt.figure(figsize=FIGSIZE)
    # for each station group
    for i in range(1, 7):
        plt.plot(area_df[i], label=area_map[i])
    plt.xlabel('Year')
    plt.ylabel('Crime Count')
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
                       tiles='cartodbpositron', zoom_start=9)

    # year is by far the best filter for reducing records. Do this first to make the dataframe much smaller
    # if a specific year is requested, filter for only that year
    if year > 0:
        year_mask = crime_data['DATE OCC YR'].astype(int) == year
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
    @param latest_date - datetime of the latest date of our crime_data dataframe
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
    @param crime_data - crime dataframe
    @param crime_dict - dictionary of {category_name:crime_code_df}
    @param output1 - widget output object. To direct what is to be output to what cell.
    @param latest_date - datetime of the latest date of our crime_data dataframe
    """
    crime_list = categories
    # don't try and plot if no categories selected
    if not crime_list:
        return None
    # create a dictionary subset with the selected crime_category/crime_codes only
    dict_subset = {c: crime_dict[c] for c in crime_list}
    # create a dataframe with the selected crimes only
    subset_df = crime_data.iloc[0:0, :].copy()
    for i in crime_list:
        subset_df = pd.concat(
            [subset_df, get_crime_subset(crime_data, i, dict_subset)])
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
    # if title == 'ALL Crime':
    #    scale = 1

    figure_size = scale*FIGSIZE

    # plot the violin
    plt.figure(figsize=figure_size+[0, 5])
    ax = sns.violinplot(data=df,
                        y=sex_descent,
                        x='Vict Age',
                        scale='width',
                        orient='h',
                        cut=0)
    ax.tick_params(top=True, labeltop=True)
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
    @param crime_data - crime dataframe
    @param crime_dict - dictionary of {category_name:crime_code_df}
    @param output2 - widget output object. To direct what is to be output to what cell.
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


def remove_negative(x):
    """ remove_negative - changes negative or nan timedelta type to 0. 
    @param x - timedelta type.
    @return timedelta type.
    """
    if x.days < 0 or x.seconds < 0:
        return pd.Timedelta(days=0, hours=0)
    else:
        return x


def plot_time(df, name):
    """ plot_time - plotting function for displaying the time plot.
    @param df - crime dataframe subset, of only the crime we want to show.
    @param name - Category of crime to plot.
    """
    hour_list = [t.hour for t in df["TIME OCC"]]
    numbers = [x for x in range(0, 24)]
    labels = map(lambda x: str(x), numbers)
    labels = ["12 AM"] + [str(x)+" AM" for x in range(1, 12)] + \
        ["12 PM"]+[str(x)+" PM" for x in range(1, 12)]

    _, ax = plt.subplots(figsize=(10, 5))
    plt.xticks(numbers, labels, rotation=45)
    plt.title(f"Crime: ({name}) occurance frequency during the day")
    plt.xlim(0, 24)
    _ = plt.hist(hour_list)
    ax.axvspan(18, 23, facecolor='black', alpha=0.25)
    ax.axvspan(0, 6, facecolor='black', alpha=0.25)
    plt.show()


def regplot_plotter(category1, category2, df, output3):
    """ regplot_plotter - plotting function for displaying
                          the regplot.
    @param category1 - string category of crime to plot ('All Categories', 'ASSAULT', etc)
    @param category1 -string category of crime to plot ('ALL CRIME', 'ASSAULT', etc)
    @param df - crime dataframe
    @param output3 - widget output object. To direct what is to be output to what cell.
    """
    output3.clear_output()
    print('Crunching Data')
    _, ax = plt.subplots(figsize=FIGSIZE)
    ax = sns.regplot(data=df,
                     x=df.index,
                     y=category1,
                     label=category1)
    # if we have a second category, plot it
    if category2:
        ax = sns.regplot(data=df,
                         x=df.index,
                         y=category2,
                         label=category2)
    ax.set(ylabel='# of Crimes', xlabel='Year')
    ax.set_title(f"crime trends")
    ax.legend()
    with output3:
        plt.show()


def time_plotter(cat, df, output):
    """ time_plotter - handler for the plot widget.
    @param cat - string category of crime to plot ('ALL CRIME', 'ASSAULT', etc)
    @param df - crime dataframe
    @param output - widget output object. To direct what is to be output to what cell.
    """
    output.clear_output()
    if cat != 'ALL CRIME':
        # get the subset if we want only a subset
        mask = df['Crime Category'] == cat
        df = df[mask]
    with output:
        print('Crunching data. Please be patient. ALL CRIME will take some time.')
        plot_time(df, cat)


def plot_pie(crime_df, crime, category):
    """ weapon_desc - Classify a weapon by the description.
    @param x - String description of the weapon type.
    @return string ('GUN', 'BLADE' 'OTHER')
    """
    df = crime_df[crime_df["Crime Category"] == crime]
    df = df.groupby(category)[[category]].count()

    i = [0]

    def absolute_value(val):
        a = df.iloc[i[0] % len(df), i[0]//len(df)]
        i[0] += 1
        return a
    _ = df.plot.pie(y=category, figsize=(7, 7), autopct=absolute_value)


def weapon_desc(x):
    """ weapon_desc - Classify a weapon by the description.
    @param x - String description of the weapon type.
    @return string ('GUN', 'BLADE' 'OTHER')
    """
    if x in gun:
        return "GUN"
    elif x in blade:
        return "BLADE"
    else:
        return "ALL OTHER"


def age_group(age):
    """ age_group - return the string version of an age grouping.
    @param age - a numeric value between 0-inf
    @return string
    """
    if age <= 12:
        return "0-12"
    if age > 12 & age <= 19:
        return "12-19"
    if age > 19 & age <= 30:
        return "19-30"
    if age > 30 & age <= 40:
        return "30-40"
    if age > 40 & age <= 55:
        return "40-55"
    if age > 55:
        return "55+"


def area_group(area):
    """ area_group - return the area grouping code we defined based on the 
                     station location name.
    @param area - the name of the station location name
    @return numeric value between 1-6
    """
    if area in ['Hollywood', "Wilshire"]:
        return 1
    if area in ["Southwest", "Southeast", "Harbor"]:
        return 2
    if area in ["West Valley", "Devonshire",  "Topanga"]:
        return 3
    if area in ['Olympic', 'Rampart', '77th Street', 'Central', 'Foothill', 'Hollenbeck', 'Newton']:
        return 4
    if area in ["West LA", "Pacific",  "Wilshire"]:
        return 5
    if area in ["Mission", "N Hollywood", "Van Nuys", "Northeast"]:
        return 6


def plot_weekly_crime(crime_df, crime_category):
    df = crime_df
    if crime_category != 'ALL CRIME':
        df = crime_df[crime_df["Crime Category"] == crime_category]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(x=df["Day Occurred"].value_counts().index,
                y=df["Day Occurred"].value_counts(),
                color="#0066cc")

    ax.set_title("Total Crimes occured by Day of the Week")
    ax.set_xticklabels(["Monday", "Tuesday", "Wednesday",
                       "Thursday", "Friday", "Saturday", "Sunday"], fontsize=9)
    ax.set_ylabel("Total Crimes")

    # barplot values visualization
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2.,
                p.get_height(),
                '%d' % round(int(p.get_height()), -2),
                fontsize=9,
                color='black',
                ha='center',
                va='bottom')

    sns.despine()


def plot_monthly_crime(df, crime_category):
    crime = df
    if crime_category != 'ALL CRIME':
        crime = df[(df["Crime Category"] == crime_category)]

    mon_occurred = [d.month for d in crime["DATE OCC"]]
    
    pd.options.mode.chained_assignment = None
    crime["Month Occurred"] = np.array(mon_occurred)
    pd.options.mode.chained_assignment = 'warn'

    fig, ax = plt.subplots(figsize=(10, 6))

    # aggregating crimes permonth
    sns.barplot(x=crime["Month Occurred"].value_counts().index,
                y=crime["Month Occurred"].value_counts(),
                color="#0066cc")

    ax.set_title("Total Crimes Occured by Month")
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=12)
    ax.set_ylabel("Total Crimes")

    # barplot values visualization
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2.,
                p.get_height(),
                '%d' % round(int(p.get_height())),
                fontsize=9,
                color='black',
                ha='center',
                va='bottom')

    sns.despine()


def premis_category(premis):
    """ premis_category - maps premis name to category #. For ML purposes.
    @param premis - string of the description
    @return int between 1-5
    """
    if premis in street:
        return 1
    if premis in house:
        return 2
    if premis in market:
        return 3
    if premis in vehicle:
        return 4
    else:
        return 5


##############################
##### MACHINE LEARNING #######
##############################

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.         
    title:         Title for the heatmap. Default is None.
    '''
    # font = {'weight' : 'bold',
    #         'size'   : 23}

    # plt.rc('font', **font)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in np.concatenate(
            [cf[0, :]/np.sum(cf[0]), cf[1, :]/np.sum(cf[1])])]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(
        group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x != '']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value != '']
        if row != []:
            values.append(row)
    df = pd.DataFrame(data=values, columns=header)
    return df


def knn_model(neighbours=False, label="zone", plot_k_values=True, optimize=False, plot_confusion_matrix=True, test_size=False, X=[], y=[]):
    ''' 
    SPLITING DATA INTO TESTING AND TRAINING SETS
    To change the splitting ratio, change the test_size value in train_test_split() block
    '''

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    default_test_size = test_size if test_size else 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=default_test_size, random_state=33)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ''' LOOPING THROUGH DIFFERENT NUMBERS OF CLUSTERS TO FIND THE OPTIMAL K-VALUE (number of clusters)'''
    # centroids to be calculated later

    loop_count = 22

    error = []
    from sklearn.neighbors import KNeighborsClassifier
    # Calculating error for K values between 1 and 20
    if optimize:
        for i in range(1, loop_count):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train.values.ravel())
            pred_i = knn.predict(X_test)

        # computing the mean error between predicted and actual values
            error.append(np.mean(pred_i != y_test[label].values))

        if plot_k_values:
            plt.figure(figsize=(loop_count * .6, loop_count * .3))
            plt.plot(range(1, loop_count), error, color='black', linestyle='dashed', marker='o',
                     markerfacecolor='grey', markersize=10)
            plt.xticks(np.arange(1, loop_count, 1))
            plt.title('Error Rate K Value')
            plt.xlabel('K Value')
            plt.ylabel('Mean Error')

    '''FITTING THE MODEL USING OPITMAL K VALUE'''

    k = neighbours if neighbours else error.index(min(error)) + 1
    # print("optimal K value is ", k, "\n")

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train.values.ravel())

    ''' TESTING THE MODEL '''

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    from sklearn.metrics import accuracy_score
    kval_df = 0
    # kval_df = pd.DataFrame(zip(np.arange(1,len(error),1), error)).rename(columns = {0:"k-value", 1: "mse"}).set_index("k-value", drop = True)
    accuracy_score = round(accuracy_score(y_test, y_pred) * 100, 2)
    report_df = report_to_df(classification_report(y_test, y_pred))

    # if plot_confusion_matrix:
    #     make_confusion_matrix(cm, figsize=(8,6), cbar=False,  sum_stats = True,categories = np.sort(y_test[label].unique()))

    joblib.dump(classifier, 'LA_CRIME_MODEL.sav')

    return kval_df, accuracy_score, report_df
