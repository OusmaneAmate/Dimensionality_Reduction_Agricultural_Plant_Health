#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd

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


# # Classification

# In[14]:


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
    obs_df.rename(columns={'SampleDate': 'Date'}, inplace=True)

    # Colonnes météo communes
    meteo_columns = [
        'Quot_more_30_Temp_C',
        'Rolling_Quot_more_30_Temp_C_3D',
        'Night_between_70_95_RH',
        'Night_more_0_Rain',
        'Quot_sum_Rain'
    ]

    # Colonnes spécifiques par culture
    crop_columns = {
        'Oignon': ['LivingLeavesNum_onions', 'DeadLeavesNum_onions', 'FeuillageCouche_onions'],
        'Carotte': ['GreenLeavesNum_carrots', 'carrot_stage'],
        'Laitue': []
    }

    if crop == 'Oignon':
        obs_df.loc[obs_df['cote_b_squamosa'] >= 1, 'cote_b_squamosa'] = 1
        obs_df.loc[obs_df['cote_p_destructor'] >= 1, 'cote_p_destructor'] = 1 
        obs_df.loc[obs_df['cote_s_vesicarium'] >= 1, 'cote_s_vesicarium'] = 1 
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_col = 'cote_b_squamosa'
        drop_cols = ['cote_p_destructor', 'cote_s_vesicarium', 'Bulb_onions_date']

    elif crop == 'Laitue':
        obs_df.loc[obs_df['cote_b_lactucae'] >= 1, 'cote_b_lactucae'] = 1
        obs_df.loc[obs_df['incidence_sclerotinia'] >= 1, 'incidence_sclerotinia'] = 1 
        obs_df.loc[obs_df['incidence_b_cinerea'] >= 1, 'incidence_b_cinerea'] = 1 
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_col = 'cote_b_lactucae'
        drop_cols = ['incidence_sclerotinia', 'incidence_b_cinerea', 'Pommaison_lettuce_date']

    elif crop == 'Carotte':
        obs_df = obs_df.drop(obs_df[obs_df['FarmID'] == 0].index)
        obs_df.loc[obs_df['cote_c_carotae'] >= 1, 'cote_c_carotae'] = 1
        obs_df.loc[obs_df['incidence_s_sclerotiorum'] >= 1, 'incidence_s_sclerotiorum'] = 1 
        obs_df.loc[obs_df['incidence_a_dauci'] >= 1, 'incidence_a_dauci'] = 1 
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_col = 'cote_c_carotae'
        drop_cols = ['incidence_s_sclerotiorum', 'incidence_a_dauci']

    else:
        raise ValueError("Culture non reconnue")

    
    selected_columns = crop_columns[crop] + meteo_columns
    final_df = combined_df[selected_columns].copy()
    labels = combined_df[label_col].copy()

    return final_df, labels


# # RNN-GRU Classification

# # Without Dimensionality reduction

# In[18]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


torch.manual_seed(32)
np.random.seed(32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Paramètres du modèle ===
seq_length = 7
hidden_dim = 32
n_layers = 2
dropout = 0.2
batch_size = 64
num_epochs = 100
learning_rate = 0.001
n_runs = 10
patience = 10


def create_sequences(X, y, ids_dates, seq_length):
    X_seq, y_seq, ids_dates_seq = [], [], []
    unique_ids = np.unique(ids_dates[:, 0])
    for uid in unique_ids:
        mask = ids_dates[:, 0] == uid
        X_id, y_id, ids_id = X[mask], y[mask], ids_dates[mask]
        for i in range(len(X_id) - seq_length + 1):
            X_seq.append(X_id[i:i+seq_length])
            y_seq.append(y_id[i+seq_length-1])
            ids_dates_seq.append(ids_id[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq), np.array(ids_dates_seq)

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RNNGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(RNNGRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_mlp = MLP([input_dim, hidden_dim])
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embed_mlp(x.view(-1, x.size(-1)))
        x = x.view(batch_size, -1, self.hidden_dim)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        out, _ = self.gru1(x, h0)
        out, _ = self.gru2(out, h0)
        return self.sigmoid(self.fc(out[:, -1, :]))


DATASETS = {}
for name in DATASETS_FILE_NAMES:
    DATASETS[name] = {}
    for k, v in DATASETS_FILE_NAMES[name].items():
        if k == "d":
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}", header=None)
        else:
            DATASETS[name][k] = pd.read_csv(f"data/{name}/{v}")

# === Entraînement par culture ===
for crop in ['Oignon', 'Laitue', 'Carotte']:
    print(f"\n🌾 Culture : {crop}")
    f1_scores = []

    for run in range(n_runs):
        print(f"\n▶️ Run {run+1}/{n_runs}")
        x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
        x_scaled = StandardScaler().fit_transform(x)
        y = y.values.astype(float)

        X_seq, y_seq, ids_dates_seq = create_sequences(x_scaled, y, np.zeros((len(y), 2)), seq_length)
        X_train, X_test, y_train, y_test, _, _ = train_test_split(X_seq, y_seq, ids_dates_seq, test_size=0.2, random_state=run)

        train_loader = DataLoader(TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=False)

        model = RNNGRUClassifier(
            input_dim=X_train.shape[2], hidden_dim=hidden_dim,
            output_dim=1, n_layers=n_layers, dropout=dropout
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            model.eval()
            val_loss = 0
            preds_val, trues_val = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    preds_val.extend(outputs.cpu().numpy())
                    trues_val.extend(targets.cpu().numpy())

            val_losses.append(val_loss / len(test_loader))

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{crop} – Run {run+1} – Loss")
        plt.xlabel("Epochs"); plt.ylabel("BCELoss")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        f1 = f1_score(np.round(trues_val), np.round(preds_val))
        f1_scores.append(f1)
        print(f"🎯 F1-score run {run+1} : {f1:.4f}")

    print(f"\n📈 Moyenne du F1-score pour {crop} sur {n_runs} runs : {np.mean(f1_scores):.4f}")


# # Machine learning

# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def DecisionTreeModel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    clf = DecisionTreeClassifier()
    param_dist = {
        'criterion': ['gini'],
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [5, 10, 20],
        'splitter': ['best']
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=-1, cv=5, scoring=scoring, refit='accuracy')
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision, f1


def kNNModel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    clf = KNeighborsClassifier()
    param_dist = {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform'],
        'metric' : ['euclidean'],
        'leaf_size' : [int(x) for x in np.linspace(start=1, stop= 50, num= 10)]
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=-1, cv=5, scoring=scoring, refit='accuracy')
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision, f1


def RandomForestModel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    clf = RandomForestClassifier()
    param_dist = {
        'n_estimators': [10, 20, 30, 40, 50],
        'criterion': ['gini'],
        'max_features' : ['log2', 'sqrt'],
        'max_depth' : [1,2,3,4,5],
        'min_samples_split' : [2, 5],
        'min_samples_leaf' : [ 2, 5],
        'bootstrap': [True, False]
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=-1, cv=5, scoring=scoring, refit='accuracy')
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision, f1


# In[9]:


N_RUNS = 5  
models = (
    ("DT", DecisionTreeModel),
    ("k-NN", kNNModel),
    ("RF", RandomForestModel),
)

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    # Standardisation
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Normalisation de y
    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)

    
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
           
             
             print(f"    {name}: accuracy= {a:.4}, precision={p:.4}, f1={f:.4}")
    
    print(f"\n📊 Moyennes des {N_RUNS} runs pour {crop} :")
    for name in acc_scores:
        acc_mean = np.mean(acc_scores[name])
        prec_mean = np.mean(prec_scores[name])
        f1_mean = np.mean(f_scores[name])
        print(f"    ▶️ {name} : Accuracy = {acc_mean:.4f}, Precision = {prec_mean:.4f}, F1-score = {f1_mean:.4f}")


# # Neural network

# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def NNModel(x, y, show_plot=True):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=True)

    # Model
    model = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Fit model
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    
    y_pred = (model.predict(x_test).flatten() >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xticks(np.arange(0, len(history.history['loss']), 10))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xticks(np.arange(0, len(history.history['accuracy']), 10))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Train and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return acc, prec, f1

N_RUNS = 10
models = [("NN", NNModel)]

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)  

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


# # LSTM

# In[17]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np

def LSTMClassificationModel(x, y, show_plot=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, f1_score
    import tensorflow as tf
    import matplotlib.pyplot as plt

   
    scaler_std = StandardScaler()
    x_std = scaler_std.fit_transform(x)

    
    y = y.values if hasattr(y, 'values') else y
    y = (y >= 0.5).astype(int)

   
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, shuffle=True)

    
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

   
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1, x_train.shape[2])),
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(16, activation='tanh', kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(8, activation='tanh', kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,  verbose=1)

    
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',       
    factor=0.5,               
    patience=5,              
    min_lr=1e-6,              
    verbose=1
)
    
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[reduce_lr, early_stop],
    )

    # 10. Évaluation
    y_pred = (model.predict(x_test).flatten() >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.xticks(np.arange(0, len(history.history['loss']), 10))  # ✅ ici
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Train and Validation Accuracy")
        plt.xticks(np.arange(0, len(history.history['accuracy']), 10))  # ✅ ici
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return acc, prec, f1
    
N_RUNS = 10
for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    acc_list, prec_list, f1_list = [], [], []

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        acc, prec, f1 = LSTMClassificationModel(x, y, show_plot=True)
        acc_list.append(acc)
        prec_list.append(prec)
        f1_list.append(f1)
        print(f"    LSTM (run {run+1}): Accuracy = {acc:.4f}, Precision = {prec:.4f}, F1 = {f1:.4f}")

    
    print(f"\n📊 Moyenne des {N_RUNS} runs pour {crop} :")
    print(f"    ▶️ Accuracy  = {np.mean(acc_list):.4f}")
    print(f"    ▶️ Precision = {np.mean(prec_list):.4f}")
    print(f"    ▶️ F1-score  = {np.mean(f1_list):.4f}")


# # RNN GRU

# In[18]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Fix random seed
torch.manual_seed(32)
np.random.seed(32)



#Define the GRU classifier model
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


def train_gru_classification_model(x, y, n_runs=10):
    acc_list, prec_list, f1_list = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for run in range(n_runs):
        print(f"▶️ Run {run+1}/{n_runs}")
        
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x = x_scaled
        y = y.values if hasattr(y, 'values') else y
        y = (y / max(y)).astype(np.float32)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        x_train = torch.tensor(x_train[:, np.newaxis, :], dtype=torch.float32)
        x_test = torch.tensor(x_test[:, np.newaxis, :], dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        model = GRUClassifier(input_dim=x.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        model.train()

        patience = 5
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(100):
            epoch_loss = 0
            all_preds, all_trues = [], []

            
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                all_preds += outputs.detach().cpu().numpy().flatten().tolist()
                all_trues += targets.cpu().numpy().flatten().tolist()

           
            preds_bin = (np.array(all_preds) >= 0.5).astype(int)
            acc_epoch = accuracy_score(all_trues, preds_bin)

            train_losses.append(epoch_loss / len(train_loader))
            train_accuracies.append(acc_epoch)

            
            model.eval()
            val_loss = 0
            val_preds, val_trues = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_preds += outputs.cpu().numpy().flatten().tolist()
                    val_trues += targets.cpu().numpy().flatten().tolist()

            val_preds_bin = (np.array(val_preds) >= 0.5).astype(int)
            val_acc = accuracy_score(val_trues, val_preds_bin)

            val_losses.append(val_loss / len(test_loader))
            val_accuracies.append(val_acc)

            
            acc_list.append(val_acc)
            prec_list.append(precision_score(val_trues, val_preds_bin))
            f1_list.append(f1_score(val_trues, val_preds_bin))

            
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
               epochs_no_improve += 1
               if epochs_no_improve >= patience:
                   print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
                   break


            
            if best_model_state is not None:
                 model.load_state_dict(best_model_state)

            if run == n_runs - 1:
                plt.figure(figsize=(10, 4))
                plt.plot(train_losses, label="Train Loss")
                plt.plot(val_losses, label="Validation Loss")
                plt.xticks(np.arange(0, len(train_losses), 10))
                plt.title("Train and Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
 
                plt.figure(figsize=(10, 4))
                plt.plot(train_accuracies, label="Train Accuracy")
                plt.plot(val_accuracies, label="Validation Accuracy")
                plt.xticks(np.arange(0, len(train_accuracies), 10))
                plt.title(" Train and Validation Accuracy")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show
                
    return np.mean(acc_list), np.mean(prec_list), np.mean(f1_list)


# In[19]:


for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
    acc, prec, f1 = train_gru_classification_model(x, y, n_runs=10)
    print(f"📊 Moyennes pour {crop} après 10 runs :")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    Precision = {prec:.4f}")
    print(f"    F1-score  = {f1:.4f}")


# # Regression

# In[23]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
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

def preprocess_data_regression(crop, obs_df, meteo_df):
    obs_df.rename(columns={'SampleDate': 'Date'}, inplace=True)

    meteo_columns = [
        'Quot_more_30_Temp_C',
        'Rolling_Quot_more_30_Temp_C_3D',
        'Night_between_70_95_RH',
        'Night_more_0_Rain',
        'Quot_sum_Rain'
    ]

    
    crop_columns = {
        'Oignon': ['LivingLeavesNum_onions', 'DeadLeavesNum_onions', 'FeuillageCouche_onions'],
        'Carotte': ['GreenLeavesNum_carrots', 'carrot_stage'],
        'Laitue': []
    }

    if crop == 'Oignon':
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_col = 'cote_b_squamosa'
        drop_cols = ['cote_p_destructor', 'cote_s_vesicarium', 'Bulb_onions_date']

    elif crop == 'Laitue':
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_col = 'cote_b_lactucae'
        drop_cols = ['incidence_sclerotinia', 'incidence_b_cinerea', 'Pommaison_lettuce_date']

    elif crop == 'Carotte':
        obs_df = obs_df[obs_df['FarmID'] != 0]
        combined_df = obs_df.merge(meteo_df, on=['FarmID', 'Date'])
        label_col = 'cote_c_carotae'
        drop_cols = ['incidence_s_sclerotiorum', 'incidence_a_dauci']

    else:
        raise ValueError("Culture non reconnue")

    
    y = combined_df[label_col].copy()

    
    ids_dates = combined_df[['FarmID', 'Date']].copy()

   
    combined_df.drop(columns=drop_cols + [label_col], inplace=True)

    
    selected_columns = crop_columns[crop] + meteo_columns
    x = combined_df[selected_columns].copy()

    return x, y, ids_dates


# In[21]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


def DecisionTreeModel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    clf = DecisionTreeRegressor()
    param_dist = {
        'criterion': ['poisson'],
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [5, 10, 20],
        'splitter': ['best']
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=-1, cv=5, scoring=scoring, refit='accuracy')
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    y_pred = clf.predict(x_test)
    return r2_score(y_test, y_pred)


def kNNModel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    clf = KNeighborsRegressor()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return r2_score(y_test, y_pred)


def RandomForestModel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    clf = RandomForestRegressor()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return r2_score(y_test, y_pred)


# In[22]:


N_RUNS = 5 
models = (
    ("DT", DecisionTreeModel),
    ("k-NN", kNNModel),
    ("RF", RandomForestModel),
)

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_regression(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])

    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

   
    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)

    
    r2_scores = {name: [] for name, _ in models}

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        for name, model in models:
            r = model(x, y)
            r2_scores[name].append(r)
            print(f"    {name}: R² = {r:.4f}")

    
    print(f"\n📊 Moyennes des {N_RUNS} runs pour {crop} :")
    for name in r2_scores:
        r2_mean = np.mean(r2_scores[name])
        print(f"    ▶️ {name} : R² moyen = {r2_mean:.4f}")


# # Neural network

# In[27]:


from sklearn.metrics import mean_absolute_error, r2_score

def NNRegressionModel(x, y, show_plot=True):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=True)

    
    model = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
        Dense(1)  
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    
    y_pred = model.predict(x_test).flatten()

    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xticks(np.arange(0, len(history.history['loss']), 10))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return r2, mae


# In[28]:


N_RUNS = 10
models = [("NN", NNRegressionModel)]

for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_regression(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)  

    r2_scores = {name: [] for name, _ in models}
    mae_scores = {name: [] for name, _ in models}

    for run in range(N_RUNS):
        print(f"   ▶️ Run {run + 1}/{N_RUNS}")
        for name, model in models:
            r2, mae = model(x, y)
            r2_scores[name].append(r2)
            mae_scores[name].append(mae)
            print(f"    {name}: R² = {r2:.4f}, MAE = {mae:.4f}")

    print(f"\n📊 Moyennes des {N_RUNS} runs pour {crop} :")
    for name in r2_scores:
        r2_mean = np.mean(r2_scores[name])
        mae_mean = np.mean(mae_scores[name])
        print(f"    ▶️ {name} : R² = {r2_mean:.4f}, MAE = {mae_mean:.4f}")


# # LSTM

# In[29]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

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
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(16, activation='tanh', kernel_regularizer=regularizers.l1(0.0002)),
        tf.keras.layers.Dense(8, activation='tanh', kernel_regularizer=regularizers.l1(0.0002)),
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


# # RNN GRU

# #  Botcast

# In[24]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(32)
np.random.seed(32)


seq_length = 7
hidden_dim = 32
n_layers = 2
dropout = 0.2
batch_size = 32
num_epochs = 100
learning_rate = 0.001
l1_lambda = 1e-4
n_runs = 10
patience = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def create_sequences(X, y, ids_dates, seq_length):
    X_seq, y_seq, ids_dates_seq = [], [], []
    unique_ids = np.unique(ids_dates[:, 0])
    for uid in unique_ids:
        mask = ids_dates[:, 0] == uid
        X_id, y_id, ids_id = X[mask], y[mask], ids_dates[mask]
        for i in range(len(X_id) - seq_length + 1):
            X_seq.append(X_id[i:i+seq_length])
            y_seq.append(y_id[i+seq_length-1])
            ids_dates_seq.append(ids_id[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq), np.array(ids_dates_seq)


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class RNNGRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(RNNGRURegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_mlp = MLP([input_dim, hidden_dim])
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.embed_mlp(x.view(-1, x.size(-1)))
        x = x.view(batch_size, -1, self.hidden_dim)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        out, _ = self.gru1(x, h0)
        out, _ = self.gru2(out, h0)
        return self.fc(out[:, -1, :])


for crop in ['Oignon', 'Laitue', 'Carotte']:
    print(f"\n🌾 Culture : {crop}")
    r2_scores = []

    for run in range(n_runs):
        print(f"\n▶️ Run {run+1}/{n_runs}")
        x, y, ids_dates = preprocess_data_regression(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
        x_scaled = StandardScaler().fit_transform(x)
        y = y.values / max(y.values)
        ids_dates = ids_dates.values

        X_seq, y_seq, ids_dates_seq = create_sequences(x_scaled, y, ids_dates, seq_length)
        X_train, X_test, y_train, y_test, _, _ = train_test_split(X_seq, y_seq, ids_dates_seq, test_size=0.2, random_state=run)

        train_loader = DataLoader(TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=False)

        model = RNNGRURegressor(
            input_dim=X_train.shape[2], hidden_dim=hidden_dim,
            output_dim=1, n_layers=n_layers, dropout=dropout
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, val_losses = [], []
        train_mae, val_mae = [], []
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            preds, trues = [], []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                preds.extend(outputs.detach().cpu().numpy().flatten())
                trues.extend(targets.cpu().numpy().flatten())

            train_losses.append(epoch_loss / len(train_loader))
            train_mae.append(mean_absolute_error(trues, preds))

            model.eval()
            val_loss, preds_val, trues_val = 0, [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    preds_val.extend(outputs.cpu().numpy().flatten())
                    trues_val.extend(targets.cpu().numpy().flatten())

            val_losses.append(val_loss / len(test_loader))
            val_mae.append(mean_absolute_error(trues_val, preds_val))

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{crop} – Run {run+1} – Loss")
        plt.xlabel("Epochs"); plt.ylabel("MSE")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        r2 = r2_score(trues_val, preds_val)
        r2_scores.append(r2)
        print(f"🌟 R² run {run+1} : {r2:.4f}")

    print(f"\n📊 Moyenne du R² pour {crop} sur {n_runs} runs : {np.mean(r2_scores):.4f}")


# In[ ]:




