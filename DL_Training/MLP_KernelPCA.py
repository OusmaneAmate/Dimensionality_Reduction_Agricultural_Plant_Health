#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# # Load data

# In[ ]:


DATASETS_FILE_NAMES = {
    "Carotte": {
        "x": "combined_daily_meteo.csv",
        "y": "carrot_no_sensitive_data.csv",
        "d": "field_distance.txt"
    },
    "Laitue": {
        "x": "combined_daily_meteo.csv",
        "y": "lettuce_no_sensitive_data.csv",
        "d": "field_distance.txt"
    },
    "Oignon": {
        "x": "combined_daily_meteo.csv",
        "y": "onion_no_sensitive_data.csv",
        "d": "field_distance.txt"
    }
}

DATASETS = {}
for name in DATASETS_FILE_NAMES:
    DATASETS[name] = {}
    for k, v in DATASETS_FILE_NAMES[name].items():
        if k == "d":
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}", header=None)
        else:
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}")


# # Reduce dimensions

# In[ ]:


def simplify_x(x: pd.DataFrame) -> pd.DataFrame:
    """Remove redundant columns."""
    KEPT_COLUMNS = (
        "FarmID",
        "Date",
        "Day_avg_Temp_C",
        "Day_max_Temp_C",
        "Day_min_Temp_C",
        "Day_avg_RH",
        "Day_max_RH",
        "Day_min_RH",
        "Day_sum_Rain",
        "Night_avg_Temp_C",
        "Night_max_Temp_C",
        "Night_min_Temp_C",
    )
    copy = x.copy()
    copy.head()
    return copy


simplify_x(DATASETS["Carotte"]["x"]).head(10)


# # Classification

# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer, precision_score
import numpy as np


DATASETS = {}
for name in DATASETS_FILE_NAMES:
    DATASETS[name] = {}
    for k, v in DATASETS_FILE_NAMES[name].items():
        if k == "d":
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}", header=None)
        else:
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}")


def normalize(sub_df):
    min_max_scaler = preprocessing.MinMaxScaler()
    sub_df_scaled = min_max_scaler.fit_transform(sub_df)
    return(pd.DataFrame(sub_df_scaled, index=sub_df.index, columns=sub_df.columns))


def preprocess_data_classification(crop, obs_df, meteo_df):
    obs_df.rename(columns={'SampleDate':'Date'}, inplace=True)
    if crop == 'Oignon':
        # Determine whether the plant is affected by a disease or not.
        obs_df.loc[obs_df['cote_b_squamosa'] >= 1, 'cote_b_squamosa'] = 1
        obs_df.loc[obs_df['cote_p_destructor'] >= 1, 'cote_p_destructor'] = 1 
        obs_df.loc[obs_df['cote_s_vesicarium'] >= 1, 'cote_s_vesicarium'] = 1 
        unique_sample_date = obs_df['Date'].unique()
        unique_sample_date = meteo_df[meteo_df['Date'].isin(unique_sample_date)]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_df = combined_df.get('cote_b_squamosa')
        combined_df = combined_df.drop(['cote_b_squamosa', 'cote_p_destructor', 'cote_s_vesicarium', 'Bulb_onions_date'], axis=1)
    elif crop == 'Laitue':
        obs_df.loc[obs_df['cote_b_lactucae'] >= 1, 'cote_b_lactucae'] = 1
        obs_df.loc[obs_df['incidence_sclerotinia'] >= 1, 'incidence_sclerotinia'] = 1 
        obs_df.loc[obs_df['incidence_b_cinerea'] >= 1, 'incidence_b_cinerea'] = 1 
        unique_sample_date = obs_df['Date'].unique()
        unique_sample_date = meteo_df[meteo_df['Date'].isin(unique_sample_date)]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_df = combined_df.get('cote_b_lactucae')
        combined_df = combined_df.drop(['cote_b_lactucae', 'incidence_sclerotinia', 'incidence_b_cinerea', 'Pommaison_lettuce_date'], axis=1)
    elif crop == 'Carotte':
        print(obs_df[['cote_c_carotae','incidence_a_dauci','incidence_s_sclerotiorum']])
        obs_df = obs_df.drop(obs_df[obs_df['FarmID'] == 0].index)
        obs_df.loc[obs_df['cote_c_carotae'] >= 1, 'cote_c_carotae'] = 1
        obs_df.loc[obs_df['incidence_s_sclerotiorum'] >= 1, 'incidence_s_sclerotiorum'] = 1 
        obs_df.loc[obs_df['incidence_a_dauci'] >= 1, 'incidence_a_dauci'] = 1 
        unique_sample_date = obs_df['Date'].unique()
        unique_sample_date = meteo_df[meteo_df['Date'].isin(unique_sample_date)]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_df = combined_df.get('cote_c_carotae')
        combined_df = combined_df.drop(['cote_c_carotae', 'incidence_s_sclerotiorum', 'incidence_a_dauci'], axis=1)
    return combined_df, label_df


# In[ ]:


def NNModel(x, y):
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # KernelPCA 
    kpca = KernelPCA(n_components=22)
    x_kpca = kpca.fit_transform(x_scaled)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x_kpca, y, test_size=0.2, shuffle=True)

    # Model
    model = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),  
        Dense(32, activation='relu'),  
        Dense(16, activation='relu'),  
        Dense(8, activation='relu'),   
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=0)

    y_pred = (model.predict(x_test).flatten() >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, prec, f1

# Training
N_RUNS = 10
models = [("NN", NNModel)]

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)  # Normalisation

    acc_scores = {name: [] for name, _ in models}
    prec_scores = {name: [] for name, _ in models}
    f_scores = {name: [] for name, _ in models}

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        for name, model in models:
            a, p, f = model(x, y)
            acc_scores[name].append(a)
            prec_scores[name].append(p)
            f_scores[name].append(f)
            print(f"    {name}: accuracy = {a:.4f}, precision = {p:.4f}, f1 = {f:.4f}")

    print(f"\n📊 Moyennes des {N_RUNS} runs pour {crop} :")
    for name in acc_scores:
        acc_mean = np.mean(acc_scores[name])
        prec_mean = np.mean(prec_scores[name])
        f1_mean = np.mean(f_scores[name])
        print(f"    ▶️ {name} : Accuracy = {acc_mean:.4f}, Precision = {prec_mean:.4f}, F1-score = {f1_mean:.4f}")


# # Regression

# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np


# In[ ]:


DATASETS = {}
for name in DATASETS_FILE_NAMES:
    DATASETS[name] = {}
    for k, v in DATASETS_FILE_NAMES[name].items():
        if k == "d":
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}", header=None)
        else:
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}")


def normalize(sub_df):
    min_max_scaler = preprocessing.MinMaxScaler()
    sub_df_scaled = min_max_scaler.fit_transform(sub_df)
    return(pd.DataFrame(sub_df_scaled, index=sub_df.index, columns=sub_df.columns))


def preprocess_data_regression(crop, obs_df, meteo_df):
    obs_df.rename(columns={'SampleDate':'Date'}, inplace=True)
    if crop == 'Oignon':
        unique_sample_date = obs_df['Date'].unique()
        unique_sample_date = meteo_df[meteo_df['Date'].isin(unique_sample_date)]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_df = combined_df.get('cote_b_squamosa')
        combined_df = combined_df.drop(['cote_b_squamosa', 'cote_p_destructor', 'cote_s_vesicarium', 'Bulb_onions_date'], axis=1)
    elif crop == 'Laitue':
        unique_sample_date = obs_df['Date'].unique()
        unique_sample_date = meteo_df[meteo_df['Date'].isin(unique_sample_date)]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_df = combined_df.get('cote_b_lactucae')
        combined_df = combined_df.drop(['cote_b_lactucae', 'incidence_sclerotinia', 'incidence_b_cinerea', 'Pommaison_lettuce_date'], axis=1)
    elif crop == 'Carotte':
        obs_df = obs_df.drop(obs_df[obs_df['FarmID'] == 0].index)
        unique_sample_date = obs_df['Date'].unique()
        unique_sample_date = meteo_df[meteo_df['Date'].isin(unique_sample_date)]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_df = combined_df.get('cote_c_carotae')
        combined_df = combined_df.drop(['cote_c_carotae', 'incidence_s_sclerotiorum', 'incidence_a_dauci'], axis=1)
    return combined_df, label_df


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS,Isomap

import numpy as np
import pandas as pd

def NNModel(x, y):
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    # Convert to numpy
    x_train = x_train.to_numpy() if hasattr(x_train, 'to_numpy') else x_train
    x_test = x_test.to_numpy() if hasattr(x_test, 'to_numpy') else x_test
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Define model with 4 hidden layers
    model = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),  
        Dense(32, activation='relu'),   
        Dense(16, activation='relu'),   
        Dense(8, activation='relu'),    
        Dense(1)  
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    history = model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)

    y_pred = model.predict(x_test).flatten()
    r = r2_score(y_test, y_pred)
    return r

N_RUNS = 10
models = [("NN", NNModel)]

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_regression(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)


    kpca = KernelPCA(n_components=22)
    x_kpca = kpca.fit_transform(x_scaled)

    print(f"📉 Kernel PCA : {x.shape[1]} ➝ {x_kpca.shape[1]} dimensions (fixées ) pour {crop}")

    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)

    r2_scores = {name: [] for name, _ in models}

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        for name, model in models:
            r = model(pd.DataFrame(x_kpca), y)
            r2_scores[name].append(r)
            print(f"    {name}: R² = {r:.4f}")

    # Moyenne finale
    print(f"\n📊 Moyenne des {N_RUNS} runs pour {crop} :")
    for name in r2_scores:
        r2_mean = np.mean(r2_scores[name])
        print(f"    ▶️ {name} : R² moyen = {r2_mean:.4f}")

