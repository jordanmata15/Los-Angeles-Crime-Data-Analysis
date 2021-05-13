# GENERAL INFO
The project requires a few non standard libraries. The following are required:
+ folium
+ ipywidgets
+ missingno
<br><br>

# Columns

+ **DR_NO**<br>
Division of Records Number: Official file number made up of a 2 digit year, area ID, and 5 digits

+ **Date Rptd**<br>
Date the crime was reported in MM/DD/YYYY format.

+ **DATE OCC**<br>
Date the crime occurred in MM/DD/YYYY format.

+ **TIME OCC**<br>
In 24 hour military time.

+ **AREA**<br>
The LAPD has 21 Community Police Stations referred to as Geographic Areas within the department. These Geographic Areas are sequentially numbered from 1-21.

+ **AREA NAME**<br>
The 21 Geographic Areas or Patrol Divisions are also given a name designation that references a landmark or the surrounding community that it is responsible for. For example 77th Street Division is located at the intersection of South Broadway and 77th Street, serving neighborhoods in South Los Angeles.

+ **Rptd Dist No**<br>
A four-digit code that represents a sub-area within a Geographic Area. All crime records reference the "RD" that it occurred in for statistical comparisons.

+ **Part 1-2**<br>
Undefined per dataset website.

+ **Crm Cd**<br>
Indicates the crime committed. (Same as Crime Code 1)

+ **Crm Cd Desc**<br>
Defines the Crime Code provided.

+ **Mocodes**<br>
Modus Operandi: Activities associated with the suspect in commission of the crime.See attached PDF for list of MO Codes in numerical order.

+ **Vict Age**<br>
Two character numeric.

+ **Vict Sex**<br>
    + F - Female
    + M - Male
    + X - Unknown
    + H - Undefined in dataset website

+ **Vict Descent**<br>
Descent Code: 
    + A - Other Asian 
    + B - Black 
    + C - Chinese
    + D - Cambodian
    + F - Filipino
    + G - Guamanian
    + H - Hispanic/Latin/Mexican
    + I - American Indian/Alaskan Native
    + J - Japanese
    + K - Korean
    + L - Laotian
    + O - Other
    + P - Pacific Islander
    + S - Samoan
    + U - Hawaiian
    + V - Vietnamese
    + W - White
    + X - Unknown
    + Z - Asian Indian

+ **Premis Cd**<br>
The type of structure, vehicle, or location where the crime took place.

+ **Premise Desc**<br>
Defines the Premise Code provided.

+ **Weapon Used Cd**<br>
The type of weapon used in the crime.

+ **Weapon Desc**<br>
Defines the Weapon Used Code provided.

+ **Status**<br>
Status of the case. (IC is the default)

+ **Status Desc**<br>
Defines the Status Code provided.

+ **Crm Cd 1**<br>
Indicates the crime committed. Crime Code 1 is the primary and most serious one. Crime Code 2, 3, and 4 are respectively less serious offenses. Lower crime class numbers are more serious.

+ **Crm Cd 2**<br>
May contain a code for an additional crime, less serious than Crime Code 1.

+ **Crm Cd 3**<br>
May contain a code for an additional crime, less serious than Crime Code 1.

+ **Crm Cd 4**<br>
May contain a code for an additional crime, less serious than Crime Code 1.

+ **LOCATION**<br>
Street address of crime incident rounded to the nearest hundred block to maintain anonymity.

+ **Cross Street**<br>
Cross Street of rounded Address

+ **LAT**<br>
Latitude coordinate

+ **LON**<br>
Longtitude coordinate
<br><br>

# REFERENCES

### LA Crime datasets:
#### Main pages
[LA Crime Dataset 1](https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z)<br>
[LA Crime Dataset 2](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8)
#### Direct downloads
[LA Crime Dataset 1 CSV](https://data.lacity.org/api/views/2nrs-mtv8/rows.csv)<br>
[LA Crime Dataset 2 CSV](https://data.lacity.org/api/views/63jg-8b9z/rows.csv)

### LAPD stations latitudes/longitudes:<br>
[LAPD Stations Dataset Page](https://geohub.lacity.org/datasets/1dd3271db7bd44f28285041058ac4612_0/data?geometry=-119.196%2C33.816%2C-117.614%2C34.214)

### LA Sheriff Station references:
[About Us Page](https://www.lasd.org/about_us.html)<br>
[Jurisdiction Info](http://shq.lasdnews.net/content/uoa/EPC/LASD_Jurisdiction.pdf)<br>
[Sheriff Stations Dataset Page](https://geohub.lacity.org/datasets/lacounty::sheriff-and-police-stations/data?geometry=-160.268%2C22.722%2C-59.018%2C47.492)

### 2016 Census data by age group and ethnicity (general areas).
The census data is from 2016 as it's the most recent census data we could find that also includes population by general area.<br>
[2016 Census Data CSV](https://data.lacounty.gov/resource/ai64-dnh8.csv)