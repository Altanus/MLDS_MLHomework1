import numpy as np
from typing import List
import pandas as pd
from fastapi import FastAPI, Request, Form, File, UploadFile
from pandas import DataFrame
from pydantic import BaseModel
import preprocessing
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


@app.get('/root')
async def root(request: Request, message='Download File'):
    return templates.TemplateResponse('index.html',
                                      {'request': request,
                                       'message': message})


class Item(BaseModel):
    name: str = "Maruti Swift Dzire VDI"
    year: int = 2014
    selling_price: int = None
    km_driven: int = 70000
    fuel: str = 'Petrol'
    seller_type: str = 'Individual'
    transmission: str = 'Manual'
    owner: str = 'First Owner'
    mileage: str = '19.6 kmpl'
    engine: str = '1248 CC'
    max_power: str = '81.8 bhp'
    torque: str = '170Nm@ 2800rpm'
    seats: float = 5.0


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    one_item_df: DataFrame = pd.DataFrame(list(item.dict().values()))
    one_item_df = one_item_df.T
    one_item_df = one_item_df.set_axis(list(item.dict().keys()), axis='columns')

    # Cleaning numeric columns
    one_item_df['mileage'] = one_item_df['mileage'].apply(lambda x: preprocessing.cleaning(x))
    one_item_df['engine'] = one_item_df['engine'].apply(lambda x: preprocessing.cleaning(x))
    one_item_df['max_power'] = one_item_df['max_power'].apply(lambda x: preprocessing.cleaning(x))

    # Cleaning torque column
    one_item_df['max_torque_rpm'] = one_item_df['torque'].apply(lambda x: preprocessing.torque_dp(x)[1])
    one_item_df['torque'] = one_item_df['torque'].apply(lambda x: preprocessing.torque_dp(x)[0])

    # setting right datatypes
    print(one_item_df.dtypes)
    for col in ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats',
                'max_torque_rpm']:
        one_item_df[col] = pd.to_numeric(np.float(one_item_df[col]))

    # taking brand
    preprocessing.taking_brand_from_name(one_item_df, 'name')

    # filling NA
    one_item_df['mileage'].fillna(preprocessing.train_mileage_median, inplace=True)
    one_item_df['engine'].fillna(preprocessing.train_engine_median, inplace=True)
    one_item_df['max_power'].fillna(preprocessing.train_maxpower_median, inplace=True)
    one_item_df['torque'].fillna(preprocessing.train_torque_new_median, inplace=True)
    one_item_df['max_torque_rpm'].fillna(preprocessing.train_torque_max_median, inplace=True)
    one_item_df['seats'].fillna(preprocessing.train_seats_median, inplace=True)

    # new features
    one_item_df['year_x2'] = one_item_df['year'] ** 2
    one_item_df['km_driven_x2'] = one_item_df['km_driven'] ** 2
    one_item_df['power_by_litr'] = one_item_df['max_power'] / one_item_df['engine']
    one_item_df['seats'] = one_item_df['seats'].apply(lambda x: preprocessing.binarize_seats(x))

    # dropping target as of no use
    one_item_df.drop('selling_price', axis=1, inplace=True)

    # scaling numeric features
    num_cols = one_item_df.select_dtypes(preprocessing.numerics).columns
    one_item_df[num_cols] = preprocessing.scaler.transform(one_item_df[num_cols])

    # OHE
    cat_cols = one_item_df.select_dtypes(preprocessing.categorial).columns
    one_item_df = pd.get_dummies(one_item_df, columns=cat_cols)

    # making similar to train
    one_item_df = preprocessing.comparison_test_train(preprocessing.train_columns, one_item_df)

    # making prediction
    prediction = preprocessing.model.predict(one_item_df)
    prediction = np.exp(prediction[0])
    return prediction


@app.post('/predict_items')
async def upload_base(request: Request,
                      name: str = Form(...),
                      db_file: UploadFile = File(...)):
    file_name = '_'.join(name.split()) + '.csv'
    save_path = f'static/Bases/{file_name}'
    with open(save_path, 'wb') as f:
        for line in db_file.file:
            f.write(line)
    items_df = pd.read_csv(f'static/Bases/{file_name}')

    # Cleaning numeric columns
    items_df['mileage'] = items_df['mileage'].apply(lambda x: preprocessing.cleaning(x))
    items_df['engine'] = items_df['engine'].apply(lambda x: preprocessing.cleaning(x))
    items_df['max_power'] = items_df['max_power'].apply(lambda x: preprocessing.cleaning(x))

    # Cleaning torque column
    items_df['max_torque_rpm'] = items_df['torque'].apply(lambda x: preprocessing.torque_dp(x)[1])
    items_df['torque'] = items_df['torque'].apply(lambda x: preprocessing.torque_dp(x)[0])

    # setting right datatypes
    for col in ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats',
                'max_torque_rpm']:
        items_df[col] = pd.to_numeric(items_df[col])

    # taking brand
    preprocessing.taking_brand_from_name(items_df, 'name')

    # filling NA
    items_df['mileage'].fillna(preprocessing.train_mileage_median, inplace=True)
    items_df['engine'].fillna(preprocessing.train_engine_median, inplace=True)
    items_df['max_power'].fillna(preprocessing.train_maxpower_median, inplace=True)
    items_df['torque'].fillna(preprocessing.train_torque_new_median, inplace=True)
    items_df['max_torque_rpm'].fillna(preprocessing.train_torque_max_median, inplace=True)
    items_df['seats'].fillna(preprocessing.train_seats_median, inplace=True)

    # new features
    items_df['year_x2'] = items_df['year'] ** 2
    items_df['km_driven_x2'] = items_df['km_driven'] ** 2
    items_df['power_by_litr'] = items_df['max_power'] / items_df['engine']
    items_df['seats'] = items_df['seats'].apply(lambda x: preprocessing.binarize_seats(x))

    # dropping target as of no use
    items_df.drop('selling_price', axis=1, inplace=True)

    # scaling numeric features
    num_cols = items_df.select_dtypes(preprocessing.numerics).columns
    items_df[num_cols] = preprocessing.scaler.transform(items_df[num_cols])

    # OHE
    cat_cols = items_df.select_dtypes(preprocessing.categorial).columns
    one_item_df = pd.get_dummies(items_df, columns=cat_cols)

    # making similar to train
    one_item_df = preprocessing.comparison_test_train(preprocessing.train_columns, one_item_df)

    # making prediction
    predictions = preprocessing.model.predict(one_item_df)
    predictions = np.exp(predictions)
    print(predictions)
    return list(predictions)
