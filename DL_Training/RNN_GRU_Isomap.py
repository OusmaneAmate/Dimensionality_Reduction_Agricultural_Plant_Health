#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap


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
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# Define the training and evaluation function

def train_gru_classification_model(x, y, n_runs=10):
    acc_list, prec_list, f1_list = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for run in range(n_runs):
        print(f"▶️ Run {run+1}/{n_runs}")
        
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        
        Iso = Isomap(n_components=22)
        x_isom = Iso.fit_transform(x_scaled)
        
    
        
        y = y.values if hasattr(y, 'values') else y
        y = (y / max(y)).astype(np.float32)

        x_train, x_test, y_train, y_test = train_test_split(x_isom, y, test_size=0.2, shuffle=True)

        x_train = torch.tensor(x_train[:, np.newaxis, :], dtype=torch.float32)
        x_test = torch.tensor(x_test[:, np.newaxis, :], dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        model = GRUClassifier(input_dim=x_isom.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        model.train()
        patience = 10
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(100):
            epoch_loss = 0
            all_preds, all_trues = [], []

            # --- Training ---
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

            # 🟩 Accuracy training
            preds_bin = (np.array(all_preds) >= 0.5).astype(int)
            acc_epoch = accuracy_score(all_trues, preds_bin)

            train_losses.append(epoch_loss / len(train_loader))
            train_accuracies.append(acc_epoch)

            # --- Validation (test) ---
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

            # Early Stopping check
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break

            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                

            
            acc_list.append(val_acc)
            prec_list.append(precision_score(val_trues, val_preds_bin))
            f1_list.append(f1_score(val_trues, val_preds_bin))

            if run == n_runs - 1:
                plt.figure(figsize=(10, 4))
                plt.plot(train_losses, label="Train Loss")
                plt.plot(val_losses, label="Validation Loss")
                plt.title("Train and Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.xticks(np.arange(0, len(train_losses), 10))
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
 
                plt.figure(figsize=(10, 4))
                plt.plot(train_accuracies, label="Train Accuracy")
                plt.plot(val_accuracies, label="Validation Accuracy")
                plt.title("Train and Validation Accuracy")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.xticks(np.arange(0, len(train_accuracies), 10))
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show
                
    return np.mean(acc_list), np.mean(prec_list), np.mean(f1_list)


for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_classification(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
    acc, prec, f1 = train_gru_classification_model(x, y, n_runs=10)
    print(f"📊 Moyennes pour {crop} après 10 runs :")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    Precision = {prec:.4f}")
    print(f"    F1-score  = {f1:.4f}")


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


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

# Fix seed
torch.manual_seed(32)
np.random.seed(32)

# === GRU Regressor Model ===
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# === Training Function ===
def train_gru_regression_model(x, y, n_runs=10):
    r2_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for run in range(n_runs):
        print(f"\n▶️ Run {run+1}/{n_runs}")

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # Isomap
        Iso = Isomap(n_components=22)
        X_iso = Iso.fit_transform(x_scaled)
        print(f"🧪 PCA: {x.shape[1]} ➝ {X_iso.shape[1]} dimensions conservées")

        x_train, x_test, y_train, y_test = train_test_split(X_iso, y, test_size=0.2, shuffle=True)

        x_train = torch.tensor(x_train[:, np.newaxis, :], dtype=torch.float32)
        x_test = torch.tensor(x_test[:, np.newaxis, :], dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        model = GRURegressor(input_dim=X_iso.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, val_losses = [], []
        train_mae, val_mae = [], []

        for epoch in range(100):
            model.train()
            epoch_loss = 0
            all_preds_train, all_trues_train = [], []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                all_preds_train.extend(outputs.detach().cpu().numpy().flatten())
                all_trues_train.extend(targets.cpu().numpy().flatten())

            train_losses.append(epoch_loss / len(train_loader))
            train_mae.append(mean_absolute_error(all_trues_train, all_preds_train))

            # --- Validation ---
            model.eval()
            val_loss = 0
            val_preds, val_trues = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_trues.extend(targets.cpu().numpy().flatten())

            val_losses.append(val_loss / len(test_loader))
            val_mae.append(mean_absolute_error(val_trues, val_preds))

        
        r2_list.append(r2_score(val_trues, val_preds))

       
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.title(f"Courbe de Loss - Run {run + 1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(train_mae, label="Train MAE")
        plt.plot(val_mae, label="Val MAE")
        plt.title(f"Courbe de MAE - Run {run + 1}")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return np.mean(r2_list)


for crop in DATASETS:
    print(f"\n🌾 Culture : {crop}")
    x, y = preprocess_data_regression(crop, DATASETS[crop]["y"], DATASETS[crop]["x"])
    y = y.values if hasattr(y, 'values') else y
    y = y / max(y)  
    r2 = train_gru_regression_model(x, y, n_runs=10)
    print(f"📊 Moyenne du R² pour {crop} : {r2:.4f}")

