#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras import callbacks
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt


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
        obs_df.loc[obs_df['incidence_b_cinerea'] >= 1, 'incidence_b_cinerea'] = 1 *
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


def LSTMModel(x, y):
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    x_ = x_scaled.values if hasattr(x_scaled, 'values') else x_scaled
    y = y.values if hasattr(y, 'values') else y

    
    x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, shuffle=True)

    
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(1, x.shape[1]), kernel_regularizer=regularizers.l1(0.0001)),
        tf.keras.layers.LSTM(16, activation='relu', return_sequences=False, kernel_regularizer=regularizers.l1(0.0001)),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
            x_train, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
    )

    
    y_pred = model.predict(x_test)
    y_pred = [0 if y < 0.5 else 1 for y in y_pred.flatten()]

    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Train and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, len(history.history['loss']), 10))  # ✅ ici
    plt.grid(True)
    plt.savefig('Loss_curve')
    plt.show()

    
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Train and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, len(history.history['accuracy']), 10))  # ✅ ici
    plt.grid(True)
    plt.savefig('Accuracy_curve')
    plt.show()

    return accuracy, precision, f1


N_RUNS = 5  
models = (
    ("LSTM", LSTMModel),
)

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
    x = normalize(x)

   
    acc_list, prec_list, f1_list = [], [], []

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        for name, model in models:
            a, p, f = model(x, y)
            print(f"    {name} (run {run + 1}): accuracy= {a:.4f}, precision={p:.4f}, f1={f:.4f}")
            acc_list.append(a)
            prec_list.append(p)
            f1_list.append(f)

    
    acc_mean = np.mean(acc_list)
    prec_mean = np.mean(prec_list)
    f1_mean = np.mean(f1_list)

    print(f"\n📊 Moyenne des {N_RUNS} runs pour {crop} :")
    print(f"    ▶️ Accuracy  = {acc_mean:.4f}")
    print(f"    ▶️ Precision = {prec_mean:.4f}")
    print(f"    ▶️ F1-score  = {f1_mean:.4f}")


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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def LSTMRegressionModel(x, y, run_number=None):
    
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x = x_scaled
    y = y.values if hasattr(y, 'values') else y

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1, x_train.shape[2])),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)  
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    
    if run_number is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Train and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(np.arange(0, len(history.history['loss']), 10))
        plt.grid(True)
        plt.savefig("LSTM_reg_loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    y_pred = model.predict(x_test).flatten()

    return r2_score(y_test, y_pred)


N_RUNS = 10
models = (("LSTM_Regression", LSTMRegressionModel),)

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_regression(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
    x = normalize(x)
    y = y.values if hasattr(y, 'values') else y

    r2_list = []

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        for name, model in models:
            r2 = model(x, y, run_number=run + 1)
            r2_list.append(r2)
            print(f"    {name} (run {run + 1}): R² = {r2:.4f}")

    print(f"\n📊 Moyenne des {N_RUNS} runs pour {crop} :")
    print(f"    ▶️ R² moyen = {np.mean(r2_list):.4f}")

