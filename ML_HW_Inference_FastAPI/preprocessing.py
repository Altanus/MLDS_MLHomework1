import pandas as pd
import re
import pickle
import sklearn
import joblib

# loading imputer and model from pickles
model = pickle.load(open("model.pkl", "rb"))
train_mileage_median, train_engine_median, train_maxpower_median, train_seats_median, train_torque_new_median, \
train_torque_max_median = pickle.load(open("filling_na_medians.pkl", "rb"))

# taking brand from name
def taking_brand_from_name(df, column):
    df['brand'] = df[column].apply(lambda x: re.search(r'\w+', x)[0])
    df.drop('name', axis=1, inplace=True)


# func for cleaning torque column
def torque_dp(x):
    #     print(x)
    if pd.isna(x):
        return None, None

    x = str(x).replace(',', '.').lower()
    values = re.findall(r'\d+\.*\,*\d*', str(x))
    measure = re.findall(r'kgm|KGM|rpm|Nm|RPM|nm|NM', str(x))

    if len(values) == 1 & len(measure) == 1:
        if measure[0] == 'kgm':
            fin_value_nm = 9.80655 * float(values[0])
        elif measure[0] == 'nm':
            fin_value_nm = float(values[0])
            fin_value_rpm = None
        elif measure[0] == 'rpm':
            fin_value_rpm = float(values[0])

    elif len(values) > len(measure) == 1:
        fin_value_nm = values[0]
        fin_value_rpm = values[-1]

    elif len(values) == 2 & len(measure) == 2:
        if measure[0] == 'nm':
            fin_value_nm = values[0]
        elif measure[0] == 'kgm':
            fin_value_nm = 9.80655 * float(values[0])
        if measure[1] == 'rpm':
            fin_value_rpm = float(values[1])

    elif (len(values) == 3) & (len(measure) == 2):
        if measure[0] == 'nm':
            fin_value_nm = values[0]
        elif measure[0] == 'kgm':
            fin_value_nm = 9.80655 * float(values[0])
        if measure[1] == 'rpm':
            fin_value_rpm = float(values[-1])

    elif (len(values) == 2) & (len(measure) == 0):
        fin_value_nm = values[0]
        fin_value_rpm = float(values[-1])

    elif (len(values) == 3) & (len(measure) == 0 or len(measure) == 3):
        fin_value_nm = values[0]
        fin_value_rpm = float(values[-1])

    else:
        return None, None

    return fin_value_nm, fin_value_rpm


# func for cleaning garbage from numercis which are in strings format contains useless info
def cleaning(x):
    if pd.isna(x):
        return x
    else:
        cleaned = re.search(r'\d+\.?\d+?', str(x))
        if pd.isna(cleaned):
            return None
        if cleaned is None:
            return None
        else:
            return float(cleaned.group())

# Binarizing variable seats
def binarize_seats (x):
    if x < 5:
        return 'less than 5'
    if x == 5 :
        return '5'
    if x == 6 or x == 7:
        return '6-7'
    elif x > 7:
        return 'more than 7'
    else:
        print(f'wow{x}')

# Get and forming scaler
categorial = ['object']
numerics = ['float64', 'int']
scaler = joblib.load('scaler.save')

# making test as train
train_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
       'max_torque_rpm', 'year_x2', 'km_driven_x2', 'power_by_litr', 'seats_5',
       'seats_6-7', 'seats_less than 5', 'seats_more than 7', 'fuel_CNG',
       'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Dealer',
       'seller_type_Individual', 'seller_type_Trustmark Dealer',
       'transmission_Automatic', 'transmission_Manual', 'brand_Ambassador',
       'brand_Audi', 'brand_BMW', 'brand_Chevrolet', 'brand_Daewoo',
       'brand_Datsun', 'brand_Fiat', 'brand_Force', 'brand_Ford',
       'brand_Honda', 'brand_Hyundai', 'brand_Isuzu', 'brand_Jaguar',
       'brand_Jeep', 'brand_Kia', 'brand_Land', 'brand_Lexus', 'brand_MG',
       'brand_Mahindra', 'brand_Maruti', 'brand_Mercedes', 'brand_Mitsubishi',
       'brand_Nissan', 'brand_Peugeot', 'brand_Renault', 'brand_Skoda',
       'brand_Tata', 'brand_Toyota', 'brand_Volkswagen', 'brand_Volvo',
       'owner_binarized_First Owner', 'owner_binarized_Second Owner',
       'owner_binarized_Third & Above Owner']

# function to compare columns of test and train
def comparison_test_train(train_cols, test):
    for i in train_cols:
        if i not in test.columns:
            test[i] = 0
    for i in test.columns:
        if i not in train_cols:
            #         X_test['other'] = 1
            test.drop(i, axis=1, inplace=True)

    # делаю проверку, что нет никаких столбцов из трейна, которых нет в тесте и наоборот
    for i in train_cols:
        if i not in test.columns:
            print(f'Problem train cols not in test! {i}')
    for i in test.columns:
        if i not in train_cols:
            print(f'Problem test cols not in train! {i}')

    return test.reindex(columns=train_columns)