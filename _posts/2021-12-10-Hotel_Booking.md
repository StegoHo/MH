layout: page
title: Hotel Booking Cancellation Prediction
description: Hotel Booking Cancellation Prediction
image: 





### Acknowledgements

The data is originally from the article [**Hotel Booking Demand Datasets**](https://www.sciencedirect.com/science/article/pii/S2352340918315191), written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019.

The data was downloaded and cleaned by Thomas Mock and Antoine Bichat for [#TidyTuesday during the week of February 11th, 2020](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md). 

The kaggle page can be find [here](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), where the explanation of the data is.

### About this file
This data set contains a single file which compares various booking information between two hotels: a city hotel and a resort hotel. A binary value "is_cancelled" is in the dataset indicating if the booking was canceled or not. With the other features such as type of hotel, country code, previous records, number of people staying etc, we could build a predictive model to estimate the probability of cancel rate on given information.

### Summary
1. Data and Packages


2. EDA & Preprocessing

    * Feature engineering
          a. Transforming month data
          b. Re-categorization for agent, country, company column
    * Drop missing value
          a. drop rows that contain missing value of child and company
    * Eliminate Error Data or Outliers
          a. expecting staying duration should be larger than 0
          b. adr should be larger or equal to 0
          c. repeated guest but has no previous records or new guest but has previous record
    * Create New Feature: 
          a. room_changed: assigned room type not equal to reserved room type
          b. expect_duration: total days staying in the hotel
    * Get Dummies for category data
    * Drop features that are not using
    
3. Train Test Split


4. Normalization


5. Modeling (Logistic Regression, kNN, Decision Trees, Random Forest, Artificial Neural Network)
    * Adjust hyper parameter or set up grid search parameter
    * Fit train data
    * Predict Test data
    * Compare accuracy with different parameter or threshold
    * Choose the best parameter or threshold

#### 1.Data and Packages


```python
import time
import winsound
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
```


```python
df = pd.read_csv('hotel_bookings.csv', parse_dates=['reservation_status_date'])
```

#### 2.EDA & Preprocessing


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 119390 entries, 0 to 119389
    Data columns (total 32 columns):
     #   Column                          Non-Null Count   Dtype         
    ---  ------                          --------------   -----         
     0   hotel                           119390 non-null  object        
     1   is_canceled                     119390 non-null  int64         
     2   lead_time                       119390 non-null  int64         
     3   arrival_date_year               119390 non-null  int64         
     4   arrival_date_month              119390 non-null  object        
     5   arrival_date_week_number        119390 non-null  int64         
     6   arrival_date_day_of_month       119390 non-null  int64         
     7   stays_in_weekend_nights         119390 non-null  int64         
     8   stays_in_week_nights            119390 non-null  int64         
     9   adults                          119390 non-null  int64         
     10  children                        119386 non-null  float64       
     11  babies                          119390 non-null  int64         
     12  meal                            119390 non-null  object        
     13  country                         118902 non-null  object        
     14  market_segment                  119390 non-null  object        
     15  distribution_channel            119390 non-null  object        
     16  is_repeated_guest               119390 non-null  int64         
     17  previous_cancellations          119390 non-null  int64         
     18  previous_bookings_not_canceled  119390 non-null  int64         
     19  reserved_room_type              119390 non-null  object        
     20  assigned_room_type              119390 non-null  object        
     21  booking_changes                 119390 non-null  int64         
     22  deposit_type                    119390 non-null  object        
     23  agent                           103050 non-null  float64       
     24  company                         6797 non-null    float64       
     25  days_in_waiting_list            119390 non-null  int64         
     26  customer_type                   119390 non-null  object        
     27  adr                             119390 non-null  float64       
     28  required_car_parking_spaces     119390 non-null  int64         
     29  total_of_special_requests       119390 non-null  int64         
     30  reservation_status              119390 non-null  object        
     31  reservation_status_date         119390 non-null  datetime64[ns]
    dtypes: datetime64[ns](1), float64(4), int64(16), object(11)
    memory usage: 29.1+ MB



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Online TA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2015-07-03</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119386.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>103050.000000</td>
      <td>6797.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.370416</td>
      <td>104.011416</td>
      <td>2016.156554</td>
      <td>27.165173</td>
      <td>15.798241</td>
      <td>0.927599</td>
      <td>2.500302</td>
      <td>1.856403</td>
      <td>0.103890</td>
      <td>0.007949</td>
      <td>0.031912</td>
      <td>0.087118</td>
      <td>0.137097</td>
      <td>0.221124</td>
      <td>86.693382</td>
      <td>189.266735</td>
      <td>2.321149</td>
      <td>101.831122</td>
      <td>0.062518</td>
      <td>0.571363</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.482918</td>
      <td>106.863097</td>
      <td>0.707476</td>
      <td>13.605138</td>
      <td>8.780829</td>
      <td>0.998613</td>
      <td>1.908286</td>
      <td>0.579261</td>
      <td>0.398561</td>
      <td>0.097436</td>
      <td>0.175767</td>
      <td>0.844336</td>
      <td>1.497437</td>
      <td>0.652306</td>
      <td>110.774548</td>
      <td>131.655015</td>
      <td>17.594721</td>
      <td>50.535790</td>
      <td>0.245291</td>
      <td>0.792798</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>-6.380000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>2016.000000</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>69.290000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>69.000000</td>
      <td>2016.000000</td>
      <td>28.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>179.000000</td>
      <td>0.000000</td>
      <td>94.575000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>160.000000</td>
      <td>2017.000000</td>
      <td>38.000000</td>
      <td>23.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>229.000000</td>
      <td>270.000000</td>
      <td>0.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>737.000000</td>
      <td>2017.000000</td>
      <td>53.000000</td>
      <td>31.000000</td>
      <td>19.000000</td>
      <td>50.000000</td>
      <td>55.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>72.000000</td>
      <td>21.000000</td>
      <td>535.000000</td>
      <td>543.000000</td>
      <td>391.000000</td>
      <td>5400.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



###### Check missing data in the dataset. 


```python
df.isna().sum()
```




    hotel                                  0
    is_canceled                            0
    lead_time                              0
    arrival_date_year                      0
    arrival_date_month                     0
    arrival_date_week_number               0
    arrival_date_day_of_month              0
    stays_in_weekend_nights                0
    stays_in_week_nights                   0
    adults                                 0
    children                               4
    babies                                 0
    meal                                   0
    country                              488
    market_segment                         0
    distribution_channel                   0
    is_repeated_guest                      0
    previous_cancellations                 0
    previous_bookings_not_canceled         0
    reserved_room_type                     0
    assigned_room_type                     0
    booking_changes                        0
    deposit_type                           0
    agent                              16340
    company                           112593
    days_in_waiting_list                   0
    customer_type                          0
    adr                                    0
    required_car_parking_spaces            0
    total_of_special_requests              0
    reservation_status                     0
    reservation_status_date                0
    dtype: int64



4 missing data for children column and 488 for country column. I would drop the row that contains the missing one in the two columns since we have a rather large dataset.

For agent and company column, I believe the missing data actually has meaning. No agent and company could indicate that the booking is made by individuals. This should be checked later.


```python
#drop na
df.dropna(subset=['children', 'country'], axis=0, inplace=True)
```

###### Feature engineering

Transforming month data to numerical one.


```python
month_tran = {'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12,
              'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6}
#map the transformation
df['arrival_date_month'] = df['arrival_date_month'].map(month_tran)
```

Adding up the stays to total duration. For every transaction that has 0 night booked should be taken off from the dataset, since it could be errors.


```python
#adding two column
df['expect_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
print('number of row has 0 night booked:', df[df['expect_duration']<=0].shape[0])
df = df[df['expect_duration']>0]
```

    number of row has 0 night booked: 701


Room changed column shows whether reserved and assigned room are same or not. After getting the feature, I try to do an EDA on the types of room and change. If the variation isn't obvious, I could drop the columns to reduce the dimension.


```python
#create new column that assigned is not equal to reserved
df['room_changed'] = np.where(df['assigned_room_type'] != df['reserved_room_type'], 1, 0)
```


```python
df['assigned_room_type'].value_counts()
```




    A    73619
    D    25056
    E     7701
    F     3709
    G     2520
    C     2331
    B     2147
    H      704
    I      217
    K      192
    L        1
    Name: assigned_room_type, dtype: int64




```python
df['reserved_room_type'].value_counts()
```




    A    85084
    D    19080
    E     6459
    F     2875
    G     2065
    B     1109
    C      922
    H      597
    L        6
    Name: reserved_room_type, dtype: int64




```python
#EDA through the room type
tt = df.groupby('assigned_room_type')['is_canceled'].agg({'sum', 'count'}).reset_index()
tt['per'] = tt['sum'] / tt['count']

sns.barplot(data=tt, x='assigned_room_type', y='per', color=sns.color_palette()[0])
plt.show()

tt = df.groupby('reserved_room_type')['is_canceled'].agg({'sum', 'count'}).reset_index()
tt['per'] = tt['sum'] / tt['count']

sns.barplot(data=tt, x='reserved_room_type', y='per', color=sns.color_palette()[0])
plt.show()

tt = df.groupby('room_changed')['is_canceled'].agg({'sum', 'count'}).reset_index()
tt['per'] = tt['sum'] / tt['count']

sns.barplot(data=tt, x='room_changed', y='per', color=sns.color_palette()[0])
plt.show()
```


![png](assets/images/output_24_0.png)



![png](assets/images/output_24_1.png)



![png](assets/images/output_24_2.png)


Surprisingly, if room is changed, the cancel rate is actually drastically lower. It could be explained that usually the change of room means upgrading in room (normal rooms are overbooked). Hence, customers wouldn't cancel for the upgraded room.

Since I obtain the variation from the two columns and the two columns seem to have less information, I would drop the two columns also for reducing the dimensions. However, their information is recorded in new room_changed column.

Outliers

We notice from the describe function above that the adr function contains negative values, which doesn't make sense. ADR stands for Average Daily Rate which measures the average rental revenue earned for an occupied room per day. For every booking, the value shouldn't be negative. However, it could be equal to 0 due to some promotions.


```python
print('number of row contains Negative ADR:', df[df['adr']<0].shape[0])
df = df[df['adr']>=0]
```

    number of row contains Negative ADR: 1


Data Explanation from kaggle
* is_repeated_guest: Value indicating if the booking name was from a repeated guest (1) or not (0)
* previous_cancellations: Number of previous bookings that were cancelled by the customer prior to the current booking
* previous_bookings_not_canceled: Number of previous bookings not cancelled by the customer prior to the current booking

If there are records for previous booking, the customer should be repeated guest. Hence, I will delete the error rows.


```python
err1 = df[(df['previous_cancellations']+df['previous_bookings_not_canceled']==0)&(df['is_repeated_guest']==1)].index
err2 = df[(df['previous_cancellations']+df['previous_bookings_not_canceled']>0)&(df['is_repeated_guest']==0)].index

print('number of row of erros:', len(err1|err2))
df = df[~df.index.isin(err1|err2)]
```

    number of row of erros: 6309


Check possibly categorical data.


```python
#num of unique for diffrent categorical data
cat_col = ['hotel', 'customer_type', 'meal', 'country', 'market_segment', 
           'distribution_channel', 'reserved_room_type', 'assigned_room_type', 
           'deposit_type', 'agent', 'company', 'reservation_status']
for i in cat_col:
    print(i, df[i].nunique())
```

    hotel 2
    customer_type 4
    meal 5
    country 177
    market_segment 7
    distribution_channel 5
    reserved_room_type 9
    assigned_room_type 10
    deposit_type 3
    agent 332
    company 344
    reservation_status 3


The missing values in agent and company columns actually have meaning as I mentioning above. Also, I'm not creating 332 and 344 more dummy columns testing which company and agent has higher cancel rate because most of the unique values have less data and lots of data is missing. Instead, I would try to group the values by their counts and find out whether the missing value impacts the prediction. That is, do booking individually (no assist from agent and company) affect the cancel rate? 


```python
#Missing values are factors?
for i in ['agent', 'company']:
    print(i)
    print('\tNA percentage:', round(df[df[i].isna()]['is_canceled'].sum() / df[df[i].isna()].shape[0], 3))
    print('\tpercentage:', round(df[~df[i].isna()]['is_canceled'].sum() / df[~df[i].isna()].shape[0], 3))
```

    agent
    	NA percentage: 0.241
    	percentage: 0.361
    company
    	NA percentage: 0.356
    	percentage: 0.156



```python
df['agent'].value_counts(normalize=True).head(10)
```




    9.0      0.325190
    240.0    0.139658
    1.0      0.050818
    14.0     0.036999
    7.0      0.035946
    6.0      0.030229
    250.0    0.028908
    241.0    0.017411
    28.0     0.016864
    8.0      0.015109
    Name: agent, dtype: float64




```python
df['company'].value_counts(normalize=True).head(10)
```




    40.0     0.148746
    223.0    0.110459
    45.0     0.039101
    153.0    0.033724
    67.0     0.032258
    174.0    0.023949
    219.0    0.022483
    281.0    0.022320
    233.0    0.018573
    405.0    0.018573
    Name: company, dtype: float64



We could see that if the cancel rate decreases if there's no agent assisting and increases if it is booked by a company. The changes are around 0.12~0.2, not a drastical change but something I would take in consideration. Hence, I would change the two columns to whether assisted by agent and whether booked by company.


```python
#re-categorize the column
def cat_agent(x):
    if x == 9:
        return '9'
    elif x == 240:
        return '240'
    elif pd.isna(x):
        return 'NA'
    else:
        return 'Other'
    
def cat_company(x):
    if x == 40:
        return '40'
    elif x == 223:
        return '223'
    elif pd.isna(x):
        return 'NA'
    else:
        return 'Other'
    
df['agent'] = df['agent'].apply(cat_agent)
df['company'] = df['company'].apply(cat_company)
```

For the country column, jugding from its value counts, I will keep the top 3 values and leave other values as other.


```python
df['country'].value_counts(normalize=True).head(10)
```




    PRT    0.377300
    GBR    0.106491
    FRA    0.092790
    ESP    0.075970
    DEU    0.064860
    ITA    0.033266
    IRL    0.029467
    BEL    0.020691
    BRA    0.019654
    NLD    0.018706
    Name: country, dtype: float64




```python
##re-categorize the column
def cat_country(x):
    if x == 'PRT':
        return 'PRT'
    elif x == 'GBR':
        return 'GBR'
    elif x == 'FRA':
        return 'FRA'
    else:
        return 'Other'
    
df['country'] = df['country'].apply(cat_country)
```

Time series EDA


```python
#EDA on time series
tt = df.groupby('arrival_date_month')['is_canceled'].agg({'sum', 'count'}).reset_index()
tt['per'] = tt['sum'] / tt['count']

sns.barplot(data=tt, x='arrival_date_month', y='per', color=sns.color_palette()[0])
plt.show()

tt = df.groupby('arrival_date_year')['is_canceled'].agg({'sum', 'count'}).reset_index()
tt['per'] = tt['sum'] / tt['count']

sns.barplot(data=tt, x='arrival_date_year', y='per', color=sns.color_palette()[0])
plt.show()
```


![png](assets/images/output_42_0.png)



![png](assets/images/output_42_1.png)



```python
df['arrival'] = df[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']].astype(str).agg(lambda x: '-'.join(x), axis=1)
df['arrival'] = pd.to_datetime(df['arrival'])
tt = df.set_index('arrival').resample('D')['is_canceled'].agg({'sum', 'count'}).reset_index()
tt['per'] = tt['sum'] / tt['count']


plt.subplots(figsize=(15, 7))
sns.lineplot(data=tt, x='arrival', y='per')
plt.xticks(rotation=85)
plt.show()
```


![png](assets/images/output_43_0.png)


As the charts above showing, cancel rate is increasing in each year. This could be attributed to the policy change or people getting used to the policy. Hence, we are putting the year column in our model.

Although the month doesn't show great variation, there should be some holidays month influencing the cancel rate. Taking summer vacation for example, the booking rate should be higher than the other months. However, the cancel should also fluctuate with the rising booking rate. Hence, I'm still putting in the months into the model.

Dummy Variables


```python
cat_col = ['arrival_date_year', 'arrival_date_month', 'hotel', 'customer_type', 
           'meal', 'country', 'market_segment', 
           'distribution_channel',
           'deposit_type', 'agent', 'company']
dff = pd.get_dummies(df, columns=cat_col, drop_first=True)
```

Drop not using variables 


```python
dff.drop(axis=1, 
         labels=['reservation_status', 'reservation_status_date', 'arrival_date_week_number',
                 'arrival_date_day_of_month', 'arrival', 'reserved_room_type', 
                 'assigned_room_type'], 
         inplace=True)
```


```python
print(dff.corr()[['is_canceled']].sort_values('is_canceled').to_string())
```

                                    is_canceled
    room_changed                      -0.232639
    total_of_special_requests         -0.214309
    required_car_parking_spaces       -0.187531
    country_Other                     -0.161092
    customer_type_Transient-Party     -0.152783
    market_segment_Direct             -0.141391
    distribution_channel_Direct       -0.137844
    booking_changes                   -0.131693
    hotel_Resort Hotel                -0.111783
    country_GBR                       -0.108361
    agent_NA                          -0.086303
    market_segment_Corporate          -0.074105
    company_Other                     -0.067788
    is_repeated_guest                 -0.065704
    company_40                        -0.050827
    previous_bookings_not_canceled    -0.050415
    market_segment_Offline TA/TO      -0.047007
    agent_Other                       -0.043221
    market_segment_Complementary      -0.038040
    customer_type_Group               -0.034279
    babies                            -0.030033
    meal_HB                           -0.029380
    meal_Undefined                    -0.028406
    arrival_date_month_11             -0.022693
    arrival_date_month_2              -0.022426
    arrival_date_month_12             -0.020284
    arrival_date_month_9              -0.019831
    arrival_date_month_3              -0.019115
    distribution_channel_GDS          -0.012812
    arrival_date_month_10             -0.010414
    deposit_type_Refundable           -0.009432
    arrival_date_month_8              -0.002870
    distribution_channel_Undefined    -0.002171
    arrival_date_month_7              -0.001750
    stays_in_weekend_nights            0.004652
    previous_cancellations             0.005706
    arrival_date_year_2016             0.006831
    children                           0.018992
    meal_SC                            0.019096
    expect_duration                    0.026501
    stays_in_week_nights               0.033058
    meal_FB                            0.033872
    arrival_date_month_5               0.036217
    arrival_date_month_4               0.039039
    market_segment_Online TA           0.039714
    arrival_date_month_6               0.046243
    days_in_waiting_list               0.047290
    adults                             0.059534
    adr                                0.070881
    arrival_date_year_2017             0.072406
    agent_9                            0.085553
    company_NA                         0.095878
    distribution_channel_TA/TO         0.162538
    market_segment_Groups              0.171803
    customer_type_Transient            0.182620
    lead_time                          0.246521
    country_PRT                        0.297388
    deposit_type_Non Refund            0.448828
    is_canceled                        1.000000



```python
dff.shape
```




    (111887, 59)



#### 3.Train test split


```python
#define the x and y
y = dff['is_canceled']
X = dff.drop(axis=1, labels='is_canceled')


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)
```

#### 4.Normalization


```python
#Using mean and std from training data to perform standardization
X_train_means = X_train.mean()
X_train_std = X_train.std()

X_train = (X_train - X_train_means)/X_train_std
X_test = (X_test - X_train_means)/X_train_std
```

#### Modeling
1. Logistic Regression
2. kNN
3. Decision Trees
4. Random Forest
5. Artificial Neural Network


```python
#Log Reg
t1 = time.time()

log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train,y_train)

t2 = time.time()
log_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % log_elapsed)
```

    Elapsed time is 0.624297 seconds.



```python
# Store predicted probability for class 1
y_pred_prob = log_reg.predict_proba(X_test)[:,1]

# make class predictions for the testing set
y_pred_class = log_reg.predict(X_test)
```


```python
# Create list of values for loop to iterate over
threshold = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# Create empty lists to store metric values created within loop
recall = []
FPR = []
FNR = []
F1 = []
Accuracy = []

# Start loop
for i in threshold:
    
    # Create class assignments given threshold value
    y_pred_class = binarize([y_pred_prob],threshold=i)[0]
    
    # Create Metrics
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    recall_value = metrics.recall_score(y_test, y_pred_class).round(3)
    fpr_value = (confusion[0,1] / (confusion[0,1] + confusion[0,0]) ).round(2)
    fnr_value = (confusion[1,0] / (confusion[1,0] + confusion[1,1]) ).round(2)
    f1_value = metrics.f1_score(y_test, y_pred_class).round(2)
    accuracy_value = metrics.accuracy_score(y_test, y_pred_class).round(2)
    
    # Append lists
    recall.append(recall_value)
    FPR.append(fpr_value)
    FNR.append(fnr_value)
    F1.append(f1_value)
    Accuracy.append(accuracy_value)

# Create dataframe
result = pd.DataFrame({"threshold":threshold,
                       "recall":recall,
                       "FPR":FPR,
                       "FNR":FNR,
                       "F1_Score": F1,
                       "Accuracy": Accuracy
                      })

# Let's look at our dataframe
result #Maybe choose with low False Negative
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>threshold</th>
      <th>recall</th>
      <th>FPR</th>
      <th>FNR</th>
      <th>F1_Score</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.000</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.51</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>0.972</td>
      <td>0.52</td>
      <td>0.03</td>
      <td>0.66</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
      <td>0.928</td>
      <td>0.35</td>
      <td>0.07</td>
      <td>0.72</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3</td>
      <td>0.863</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>0.75</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4</td>
      <td>0.770</td>
      <td>0.16</td>
      <td>0.23</td>
      <td>0.75</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.5</td>
      <td>0.651</td>
      <td>0.10</td>
      <td>0.35</td>
      <td>0.71</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>0.526</td>
      <td>0.06</td>
      <td>0.47</td>
      <td>0.64</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7</td>
      <td>0.414</td>
      <td>0.03</td>
      <td>0.59</td>
      <td>0.56</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8</td>
      <td>0.348</td>
      <td>0.02</td>
      <td>0.65</td>
      <td>0.50</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.9</td>
      <td>0.297</td>
      <td>0.00</td>
      <td>0.70</td>
      <td>0.46</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>



#### KNN


```python
# kNN
# Train a classifier for different values of k
results = []
for k in range(3, 17, 2):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    confusion = metrics.confusion_matrix(y_test,y_pred_class)
    results.append({'k':k,
                    'accuracy':metrics.accuracy_score(y_test,knn.predict(X_test)),
                    'fnr':(confusion[1,0] / (confusion[1,0] + confusion[1,1])).round(2)
                   })

# Convert results to Pandas dataframe
#winsound.Beep(2000, 1500)
results = pd.DataFrame(results)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>accuracy</th>
      <th>fnr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.847991</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.847276</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0.848527</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0.846632</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>0.845453</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>0.843558</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>0.842807</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select K

t1 = time.time()

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)

t2 = time.time()
knn_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % knn_elapsed)
```

    Elapsed time is 0.012965 seconds.


#### Decision Tree


```python
param_grid = { 'max_features': ['auto', 'sqrt'],
               'max_depth': [5, 10, 15, 20],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4]}

t1 = time.time()

clf_dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(estimator = clf_dt, param_grid = param_grid, 
                          cv = 3, n_jobs = 3, verbose = 2)
dt_grid.fit(X_train, y_train)

t2 = time.time()
dt_grid_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % dt_grid_elapsed)
```

    Fitting 3 folds for each of 72 candidates, totalling 216 fits
    Elapsed time is 11.805789 seconds.



```python
# Best paramete set
print('Best parameters found:\n', dt_grid.best_params_)

# All results
means = dt_grid.cv_results_['mean_test_score']
stds = dt_grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, dt_grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
```

    Best parameters found:
     {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.713 (+/-0.014) for {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.792 (+/-0.006) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.801 (+/-0.020) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.785 (+/-0.044) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.801 (+/-0.029) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.785 (+/-0.012) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.795 (+/-0.029) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.766 (+/-0.028) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.766 (+/-0.028) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.802 (+/-0.010) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.792 (+/-0.006) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.801 (+/-0.020) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.785 (+/-0.044) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.801 (+/-0.029) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.785 (+/-0.012) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.795 (+/-0.029) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.766 (+/-0.028) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.766 (+/-0.028) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.802 (+/-0.010) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.833 (+/-0.008) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.834 (+/-0.016) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.837 (+/-0.009) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.832 (+/-0.009) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.825 (+/-0.018) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.832 (+/-0.009) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.841 (+/-0.004) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.841 (+/-0.004) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.829 (+/-0.001) for {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.833 (+/-0.008) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.834 (+/-0.016) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.837 (+/-0.009) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.832 (+/-0.009) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.825 (+/-0.018) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.832 (+/-0.009) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.841 (+/-0.004) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.841 (+/-0.004) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.829 (+/-0.001) for {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.843 (+/-0.006) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.845 (+/-0.001) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.845 (+/-0.008) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.834 (+/-0.010) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.840 (+/-0.006) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.843 (+/-0.004) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.844 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.844 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.842 (+/-0.008) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10}
    0.843 (+/-0.006) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}
    0.845 (+/-0.001) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5}
    0.845 (+/-0.008) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10}
    0.834 (+/-0.010) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}
    0.840 (+/-0.006) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}
    0.843 (+/-0.004) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10}
    0.844 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2}
    0.844 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5}
    0.842 (+/-0.008) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10}



```python
# instantiate classification tree
t1 = time.time()

clf_dt = DecisionTreeClassifier(max_depth=20, max_features='auto', min_samples_leaf=1, 
                                min_samples_split=10, random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

t2 = time.time()
dt_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % dt_elapsed)
```

    Elapsed time is 0.119881 seconds.



```python
clf_dt_pred_prob = clf_dt.predict_proba(X_test)[:,1]

recall = []
FPR = []
FNR = []
F1 = []
Accuracy = []

for i in threshold:
    
    # Create class assignments given threshold value
    y_pred_class = binarize([clf_dt_pred_prob],threshold=i)[0]
    
    # Create Metrics
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    recall_value = metrics.recall_score(y_test, y_pred_class).round(3)
    fpr_value = (confusion[0,1] / (confusion[0,1] + confusion[0,0]) ).round(2)
    fnr_value = (confusion[1,0] / (confusion[1,0] + confusion[1,1]) ).round(2)
    f1_value = metrics.f1_score(y_test, y_pred_class).round(2)
    accuracy_value = metrics.accuracy_score(y_test, y_pred_class).round(2)
    
    # Append lists
    recall.append(recall_value)
    FPR.append(fpr_value)
    FNR.append(fnr_value)
    F1.append(f1_value)
    Accuracy.append(accuracy_value)

# Create dataframe
result = pd.DataFrame({"threshold":threshold,
                       "recall":recall,
                       "FPR":FPR,
                       "FNR":FNR,
                       "F1_Score": F1,
                       "Accuracy": Accuracy
                      })

# Let's look at our dataframe
result #Maybe choose with low False Negative
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>threshold</th>
      <th>recall</th>
      <th>FPR</th>
      <th>FNR</th>
      <th>F1_Score</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.957</td>
      <td>0.41</td>
      <td>0.04</td>
      <td>0.70</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>0.945</td>
      <td>0.34</td>
      <td>0.06</td>
      <td>0.73</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
      <td>0.912</td>
      <td>0.27</td>
      <td>0.09</td>
      <td>0.76</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3</td>
      <td>0.864</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.77</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4</td>
      <td>0.805</td>
      <td>0.13</td>
      <td>0.20</td>
      <td>0.78</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.5</td>
      <td>0.757</td>
      <td>0.10</td>
      <td>0.24</td>
      <td>0.78</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>0.697</td>
      <td>0.07</td>
      <td>0.30</td>
      <td>0.76</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7</td>
      <td>0.617</td>
      <td>0.05</td>
      <td>0.38</td>
      <td>0.72</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8</td>
      <td>0.539</td>
      <td>0.03</td>
      <td>0.46</td>
      <td>0.67</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.9</td>
      <td>0.437</td>
      <td>0.02</td>
      <td>0.56</td>
      <td>0.59</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>



#### Random Forest


```python
param_grid = {'n_estimators': [200, 400],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10, 20, 30],
               'min_samples_split': [2, 4],
               'min_samples_leaf': [1, 2]}
```


```python
t1 = time.time()

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = 3, verbose = 2)
rf_grid.fit(X_train, y_train)

t2 = time.time()
rf_grid_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % rf_grid_elapsed)
```

    Fitting 3 folds for each of 48 candidates, totalling 144 fits
    Elapsed time is 1084.483737 seconds.



```python
# Best paramete set
print('Best parameters found:\n', rf_grid.best_params_)

# All results
means = rf_grid.cv_results_['mean_test_score']
stds = rf_grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rf_grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
```

    Best parameters found:
     {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.849 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.849 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
    0.848 (+/-0.001) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 400}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}
    0.849 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.849 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
    0.848 (+/-0.001) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 400}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    0.848 (+/-0.002) for {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}
    0.878 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.878 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    0.877 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
    0.878 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 400}
    0.875 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
    0.875 (+/-0.003) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
    0.875 (+/-0.002) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    0.875 (+/-0.003) for {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}
    0.878 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.878 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    0.877 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
    0.878 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 400}
    0.875 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
    0.875 (+/-0.003) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
    0.875 (+/-0.002) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    0.875 (+/-0.003) for {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}
    0.883 (+/-0.003) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.883 (+/-0.002) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    0.882 (+/-0.003) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
    0.882 (+/-0.003) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 400}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}
    0.883 (+/-0.003) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    0.883 (+/-0.002) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    0.882 (+/-0.003) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
    0.882 (+/-0.003) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 400}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 200}
    0.879 (+/-0.002) for {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}



```python
t1 = time.time()

rf = RandomForestClassifier(max_depth=30, max_features='auto', min_samples_leaf=1, 
                            min_samples_split=2, n_estimators=200, random_state=42)

rf.fit(X_train,y_train)

y_pred_prob = rf.predict_proba(X_test)[:,1]
y_pred_class = binarize([y_pred_prob],threshold=0.5)[0]
confusion=confusion_matrix(y_test,y_pred_class)

t2 = time.time()
rf_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % rf_elapsed)
```

    Elapsed time is 18.546071 seconds.



```python
rf_pred_prob = rf.predict_proba(X_test)[:,1]

recall = []
FPR = []
FNR = []
F1 = []
Accuracy = []

for i in threshold:
    
    # Create class assignments given threshold value
    y_pred_class = binarize([rf_pred_prob],threshold=i)[0]
    
    # Create Metrics
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    recall_value = metrics.recall_score(y_test, y_pred_class).round(3)
    fpr_value = (confusion[0,1] / (confusion[0,1] + confusion[0,0]) ).round(2)
    fnr_value = (confusion[1,0] / (confusion[1,0] + confusion[1,1]) ).round(2)
    f1_value = metrics.f1_score(y_test, y_pred_class).round(2)
    accuracy_value = metrics.accuracy_score(y_test, y_pred_class).round(2)
    
    # Append lists
    recall.append(recall_value)
    FPR.append(fpr_value)
    FNR.append(fnr_value)
    F1.append(f1_value)
    Accuracy.append(accuracy_value)

# Create dataframe
result = pd.DataFrame({"threshold":threshold,
                       "recall":recall,
                       "FPR":FPR,
                       "FNR":FNR,
                       "F1_Score": F1,
                       "Accuracy": Accuracy
                      })

# Let's look at our dataframe
result #Maybe choose with low False Negative
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>threshold</th>
      <th>recall</th>
      <th>FPR</th>
      <th>FNR</th>
      <th>F1_Score</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.999</td>
      <td>0.82</td>
      <td>0.00</td>
      <td>0.56</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>0.982</td>
      <td>0.35</td>
      <td>0.02</td>
      <td>0.74</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
      <td>0.952</td>
      <td>0.23</td>
      <td>0.05</td>
      <td>0.80</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3</td>
      <td>0.907</td>
      <td>0.15</td>
      <td>0.09</td>
      <td>0.83</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4</td>
      <td>0.856</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.84</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.5</td>
      <td>0.800</td>
      <td>0.07</td>
      <td>0.20</td>
      <td>0.83</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>0.738</td>
      <td>0.05</td>
      <td>0.26</td>
      <td>0.81</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7</td>
      <td>0.664</td>
      <td>0.03</td>
      <td>0.34</td>
      <td>0.78</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8</td>
      <td>0.564</td>
      <td>0.01</td>
      <td>0.44</td>
      <td>0.71</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.9</td>
      <td>0.445</td>
      <td>0.00</td>
      <td>0.56</td>
      <td>0.61</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>



##### Neural Network


```python
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (20,20,20,20)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
```


```python
t1 = time.time()

nn = MLPClassifier(max_iter=1000)
clf = GridSearchCV(nn, parameter_space, n_jobs=3, cv=3)
clf.fit(X_train,y_train)

t2 = time.time()
rf_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % rf_elapsed)
#winsound.Beep(2000, 1500)
```

    Elapsed time is 8810.405328 seconds.



```python
# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
```

    Best parameters found:
     {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.857 (+/-0.002) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.843 (+/-0.006) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
    0.857 (+/-0.005) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.844 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.863 (+/-0.000) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.856 (+/-0.003) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'adam'}
    0.862 (+/-0.003) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.856 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.863 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.859 (+/-0.003) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
    0.863 (+/-0.003) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.859 (+/-0.002) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.863 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.863 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'adam'}
    0.864 (+/-0.001) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.864 (+/-0.002) for {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.858 (+/-0.005) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.849 (+/-0.009) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
    0.859 (+/-0.000) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.850 (+/-0.008) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.861 (+/-0.004) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.859 (+/-0.003) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'adam'}
    0.860 (+/-0.001) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.860 (+/-0.003) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.861 (+/-0.003) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.858 (+/-0.003) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
    0.862 (+/-0.000) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.858 (+/-0.003) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
    0.861 (+/-0.004) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'sgd'}
    0.864 (+/-0.003) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'constant', 'solver': 'adam'}
    0.862 (+/-0.002) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    0.862 (+/-0.002) for {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (20, 20, 20, 20), 'learning_rate': 'adaptive', 'solver': 'adam'}



```python
t1 = time.time()

nn_model = MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20),
                         activation='tanh', 
                         alpha=0.05,
                         learning_rate='adaptive',
                         solver='adam',
                         max_iter=1000)
nn_model.fit(X_train,y_train)

t2 = time.time()
nn_elapsed = t2 - t1
print('Elapsed time is %f seconds.' % nn_elapsed)
```

    Elapsed time is 90.541268 seconds.



```python
nn_y_pred_prob = nn_model.predict_proba(X_test)[:,1]

recall = []
FPR = []
FNR = []
F1 = []
Accuracy = []

for i in threshold:
    
    # Create class assignments given threshold value
    y_pred_class = binarize([nn_y_pred_prob],threshold=i)[0]
    
    # Create Metrics
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    recall_value = metrics.recall_score(y_test, y_pred_class).round(3)
    fpr_value = (confusion[0,1] / (confusion[0,1] + confusion[0,0]) ).round(2)
    fnr_value = (confusion[1,0] / (confusion[1,0] + confusion[1,1]) ).round(2)
    f1_value = metrics.f1_score(y_test, y_pred_class).round(2)
    accuracy_value = metrics.accuracy_score(y_test, y_pred_class).round(2)
    
    # Append lists
    recall.append(recall_value)
    FPR.append(fpr_value)
    FNR.append(fnr_value)
    F1.append(f1_value)
    Accuracy.append(accuracy_value)

# Create dataframe
result = pd.DataFrame({"threshold":threshold,
                       "recall":recall,
                       "FPR":FPR,
                       "FNR":FNR,
                       "F1_Score": F1,
                       "Accuracy": Accuracy
                      })

# Let's look at our dataframe
result #Maybe choose with low False Negative
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>threshold</th>
      <th>recall</th>
      <th>FPR</th>
      <th>FNR</th>
      <th>F1_Score</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.000</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.51</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>0.974</td>
      <td>0.36</td>
      <td>0.03</td>
      <td>0.73</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
      <td>0.938</td>
      <td>0.26</td>
      <td>0.06</td>
      <td>0.77</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3</td>
      <td>0.895</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>0.80</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.4</td>
      <td>0.842</td>
      <td>0.13</td>
      <td>0.16</td>
      <td>0.80</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.5</td>
      <td>0.778</td>
      <td>0.09</td>
      <td>0.22</td>
      <td>0.80</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>0.714</td>
      <td>0.06</td>
      <td>0.29</td>
      <td>0.78</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.7</td>
      <td>0.638</td>
      <td>0.04</td>
      <td>0.36</td>
      <td>0.75</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8</td>
      <td>0.539</td>
      <td>0.02</td>
      <td>0.46</td>
      <td>0.69</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.9</td>
      <td>0.427</td>
      <td>0.01</td>
      <td>0.57</td>
      <td>0.59</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>



#### Threshold Selection
Since in the case false negative and positive won't matter much, I would choose the threshold based on the highest accuracy. If there is a tie in accuracy, I choose the one with lower FNR and FPR.


#### Comparison

| Models      | Time        | threshold/K | Accuracy    |
| ----------- | ----------- | ----------- | ----------- |
| Logistic Regression          | 0.62        | 0.4         | 0.82        |
| KNN         | 0.01        | 7           | 0.85        |
| Decision Trees      | 0.12        | 0.4         | 0.85        |
| Random Forest    | 18.55       | 0.5         | 0.89        |
| Neural Network     | 90.54      | 0.4         | 0.86        |


#### The winning method

I would choose **Random Forest** over other model because it has the highest accuracy score and an affordable time. Since all of the model achieve at least 80% of accuracy, I believe my data preprocessing and data itself are fine. Also, I believe that if running with any boosted random tree, learning from the misclassified data (ex:xgboost), the accuracy score could even reach higher.

From the model, we could predict whether a customer is cancelling his or her booking or not. In order to improve our business, we could reach out to the customer and ask for improvement directly. Are they cancelling for personal reason or are they disatisfied by our platform? By improving ourselves and decreasing the cancel rate, the company could actually profit more.
