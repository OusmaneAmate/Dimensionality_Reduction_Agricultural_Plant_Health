#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from tensorflow.keras import layers, regularizers


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


def LSTMClassificationModel(x, y,run_number=None):
    
    kpca = KernelPCA(n_components=22)
    x_kpca = kpca.fit_transform(x_std)
    print(f"🧪 KernelPCA: {x.shape[1]} ➝ {x_kpca.shape[1]} dimensions conservées")

    
    scaler_minmax = MinMaxScaler()
    x_scaled = scaler_minmax.fit_transform(x_kpca)

    
    y = y.values if hasattr(y, 'values') else y
    y = (y >= 0.5).astype(int)  

    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=True)

    
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

   
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1, x_train.shape[2])),
        tf.keras.layers.LSTM(64, return_sequences=True,kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(16, activation='tanh', kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(8, activation='tanh', kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    
    history = model.fit(
        x_train, y_train,
        epochs=100,
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
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title("Train and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(np.arange(0, len(history.history['accuracy']), 10)) 
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    y_pred = (model.predict(x_test).flatten() >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, prec, f1

N_RUNS = 5
for run in range(N_RUNS):
    acc_list, prec_list, f1_list = [], [], []
    print(f"   ▶️ Run {run + 1}/{N_RUNS}")
    acc, prec, f1 = LSTMClassificationModel(x, y, run_number=run+1)
    acc_list.append(acc)
    prec_list.append(prec)
    f1_list.append(f1)
    print(f"    LSTM (run {run+1}): Accuracy = {acc:.4f}, Precision = {prec:.4f}, F1 = {f1:.4f}")


for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    acc_list, prec_list, f1_list = [], [], []

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        acc, prec, f1 = LSTMClassificationModel(x, y, run_number=run+1)
        acc_list.append(acc)
        prec_list.append(prec)
        f1_list.append(f1)
        print(f"    LSTM (run {run+1}): Accuracy = {acc:.4f}, Precision = {prec:.4f}, F1 = {f1:.4f}")

    print(f"\n📊 Moyenne des {N_RUNS} runs pour {crop} :")
    print(f"    ▶️ Accuracy  = {np.mean(acc_list):.4f}")
    print(f"    ▶️ Precision = {np.mean(prec_list):.4f}")
    print(f"    ▶️ F1-score  = {np.mean(f1_list):.4f}")

