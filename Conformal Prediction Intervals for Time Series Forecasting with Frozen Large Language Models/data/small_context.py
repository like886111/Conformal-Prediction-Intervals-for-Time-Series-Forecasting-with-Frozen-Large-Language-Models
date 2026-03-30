import darts.datasets
import pandas as pd
import numpy as np
import torch

dataset_names = [
    'AirPassengersDataset', 
    'AusBeerDataset', 
    'AustralianTourismDataset', 
    'ETTh1Dataset', 
    'ETTh2Dataset', 
    'ETTm1Dataset', 
    'ETTm2Dataset', 
    'ElectricityDataset', 
    'EnergyDataset', 
    'ExchangeRateDataset', 
    'GasRateCO2Dataset', 
    'HeartRateDataset', 
    'ILINetDataset', 
    'IceCreamHeaterDataset', 
    'MonthlyMilkDataset', 
    'MonthlyMilkIncompleteDataset', 
    'SunspotsDataset', 
    'TaylorDataset', 
    'TemperatureDataset',
    'TrafficDataset', 
    'USGasolineDataset', 
    'UberTLCDataset', 
    'WeatherDataset', 
    'WineDataset', 
    'WoolyDataset',
]

def get_descriptions(w_references=False):
    descriptions = []
    for dsname in dataset_names:
        d = getattr(darts.datasets,dsname)().__doc__
        
        if w_references:
            descriptions.append(d)
            continue

        lines = []
        for l in d.split("\n"):
            if l.strip().startswith("References"):
                break
            if l.strip().startswith("Source"):
                break
            if l.strip().startswith("Obtained"):
                break
            lines.append(l)
        
        d = " ".join([x.strip() for x in lines]).strip()

        descriptions.append(d)

    return dict(zip(dataset_names,descriptions))

def get_dataset(dsname):
    darts_ds = getattr(darts.datasets,dsname)().load()
    if dsname=='GasRateCO2Dataset':
        darts_ds = darts_ds[darts_ds.columns[1]]
    #series = darts_ds.pd_series()
    series = pd.Series(data=darts_ds.values().flatten(), index=darts_ds.time_index)
    if dsname == 'SunspotsDataset':
        series = series.iloc[::4]
    if dsname =='HeartRateDataset':
        series = series.iloc[::2]
    return series

def add_noise(series, noise_level=0.1):

    std_dev = np.std(series)  
    noise = np.random.normal(0, noise_level * std_dev, len(series))
    noisy_series = series + noise
    return noisy_series

def add_def_noise(series, noise_level=0.1, noise_type='gaussian'):
    
    std_dev = np.std(series)  

    if noise_type == 'gaussian':
        
        noise = np.random.normal(0, noise_level * std_dev, len(series))
    elif noise_type == 'uniform':
        
        noise = np.random.uniform(-noise_level * std_dev, noise_level * std_dev, len(series))
    elif noise_type == 'laplace':
        
        noise = np.random.laplace(0, noise_level * std_dev / np.sqrt(2), len(series))
    elif noise_type == 'beta':
        
        a, b = 2, 5  
        noise = np.random.beta(a, b, len(series)) * (2 * noise_level * std_dev) - noise_level * std_dev
    elif noise_type == 'geometric':
        
        p = 0.5  
        noise = np.random.geometric(p, len(series)) * noise_level * std_dev
    elif noise_type == 'gamma':
        shape = 2.0  
        scale = noise_level * std_dev  
        noise = np.random.gamma(shape, scale, len(series)) - np.mean(np.random.gamma(shape, scale, len(series)))  
    else:
        raise ValueError("please in :'gaussian', 'uniform', 'laplace', 'beta', 'geometric',  'gamma'")

    noisy_series = series + noise
    return noisy_series


def get_datasets(n=-1, testfrac=0.2, predict_steps=None, noise=False, noise_level=0.1, noise_type='gaussian'):
    """
    predict_steps: if not None, test set has exactly this many points (train = series[:-predict_steps], test = series[-predict_steps:]).
    Ensures fixed horizon for paper experiments (e.g. 30).
    """
    datasets = [
        'AirPassengersDataset', #144 29
        'AusBeerDataset', #168 43
        'GasRateCO2Dataset', # multivariate #500 60
        'MonthlyMilkDataset', #144 34
        'SunspotsDataset', # very big, need to subsample? #  141
        'WineDataset', #159 36
        'WoolyDataset', #336 24
        'HeartRateDataset',# also subsample. # 90
    ]
    names, datas = [], []
    for i, dsname in enumerate(datasets):
        series = get_dataset(dsname)
        # if reverse:
        #     series = series[::-1]
        if predict_steps is not None:
            if len(series) <= predict_steps:
                continue
            splitpoint = len(series) - predict_steps
        else:
            splitpoint = int(len(series) * (1 - testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level, noise_type)
        names.append(dsname)
        datas.append((train, test))
        if n != -1 and len(names) >= n:
            break
    return dict(zip(names, datas))

def get_memorization_datasets(n=-1,testfrac=0.15, predict_steps=30, noise=False, noise_level=0.1,noise_type = 'gaussian'):
    datasets = [
        'IstanbulTraffic', #267
        'TSMCStock', #247
        'TurkeyPower' #365
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_datasets/memorization/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

# def get_datasets(n=-1,testfrac=0.2):
#     datasets = [
#         'AirPassengersDataset',
#         'AusBeerDataset',
#         'GasRateCO2Dataset', # multivariate
#         'MonthlyMilkDataset',
#         'SunspotsDataset', #very big, need to subsample?
#         'WineDataset',
#         'WoolyDataset',
#         'HeartRateDataset',
#     ]
#     datas = []
#     for i,dsname in enumerate(datasets):
#         series = get_dataset(dsname)
#         splitpoint = int(len(series)*(1-testfrac))
        
#         train = series.iloc[:splitpoint]
#         test = series.iloc[splitpoint:]
#         datas.append((train,test))
#         if i+1==n:
#             break
#     return dict(zip(datasets,datas))

# def get_memorization_datasets(n=-1,testfrac=0.15, predict_steps=30):
#     datasets = [
#         'IstanbulTraffic',
#         'TSMCStock',
#         'TurkeyPower'
#     ]
#     datas = []
#     for i,dsname in enumerate(datasets):
#         with open(f"datasets/memorization/{dsname}.csv") as f:
#             series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
#             # treat as float
#             series = series.astype(float)
#             series = pd.Series(series)
#         if predict_steps is not None:
#             splitpoint = len(series)-predict_steps
#         else:    
#             splitpoint = int(len(series)*(1-testfrac))
#         train = series.iloc[:splitpoint]
#         test = series.iloc[splitpoint:]
#         datas.append((train,test))
#         if i+1==n:
#             break
#     return dict(zip(datasets,datas))

def get_ETTh1_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.1,noise_type='gaussian'):
    datasets = [
        'ETTh1_1',
        'ETTh1_2',
        'ETTh1_3',
        'ETTh1_4',
        'ETTh1_5',
        'ETTh1_6',
        'ETTh1_7',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_datasets/informer_600/ETTh1/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_ETTh2_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.1,noise_type='gaussian'):
    datasets = [
        'ETTh2_1',
        'ETTh2_2',
        'ETTh2_3',
        'ETTh2_4',
        'ETTh2_5',
        'ETTh2_6',
        'ETTh2_7',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_datasets/informer_600/ETTh2/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_ETTm1_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.1,noise_type='gaussian'):
    datasets = [
        'ETTm1_1',
        'ETTm1_2',
        'ETTm1_3',
        'ETTm1_4',
        'ETTm1_5',
        'ETTm1_6',
        'ETTm1_7',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_datasets/informer_600/ETTm1/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))


def get_ETTm2_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.1,noise_type='gaussian'):
    datasets = [
        'ETTm2_1',
        'ETTm2_2',
        'ETTm2_3',
        #'ETTm2_4',
        #'ETTm2_5',
        #'ETTm2_6',
        #'ETTm2_7',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_datasets/informer_600/ETTm2/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_exchange_rate_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.1,noise_type='gaussian'):
    datasets = [
        'exchange_rate_1',
        'exchange_rate_2',
        'exchange_rate_3',
        'exchange_rate_4',
        'exchange_rate_5',
        'exchange_rate_6',
        'exchange_rate_7',
        'exchange_rate_8',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_data/informer_600/exchange_rate/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_national_illness_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.1,noise_type='gaussian'):
    datasets = [
        'national_illness_1',
        'national_illness_2',
        'national_illness_3',
        'national_illness_4',
        'national_illness_5',
        'national_illness_6',
        'national_illness_7',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"TS_datasets/informer_600/national_illness/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level,noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))
