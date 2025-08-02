import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import joblib

# Load datasets
train_data = pd.read_csv("Security AI/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("Security AI/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv")

# Drop unnecessary columns
train_data.drop(columns=['id', 'attack_cat'], inplace=True)
test_data.drop(columns=['id', 'attack_cat'], inplace=True)

# Define important features
important_features = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", 
    "sinpkt", "dinpkt", "spkts_per_dur", "dpkts_per_dur"
]

# Replace invalid values
train_data.replace({'-': np.nan, 'http': np.nan, 'ftp': np.nan, 'tcp': np.nan}, inplace=True)
test_data.replace({'-': np.nan, 'http': np.nan, 'ftp': np.nan, 'tcp': np.nan}, inplace=True)
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Label encode categorical features
proto_encoder = LabelEncoder()
state_encoder = LabelEncoder()

train_data['proto'] = train_data['proto'].apply(lambda x: x if x in ['tcp', 'udp'] else 'other')
test_data['proto'] = test_data['proto'].apply(lambda x: x if x in ['tcp', 'udp'] else 'other')
proto_encoder.fit(['tcp', 'udp', 'other'])
train_data['proto'] = proto_encoder.transform(train_data['proto'])
test_data['proto'] = proto_encoder.transform(test_data['proto'])

train_data['state'] = train_data['state'].apply(lambda x: x if x in ['SYN', 'FIN', '-'] else '-')
test_data['state'] = test_data['state'].apply(lambda x: x if x in ['SYN', 'FIN', '-'] else '-')
state_encoder.fit(['SYN', 'FIN', '-'])
train_data['state'] = state_encoder.transform(train_data['state'])
test_data['state'] = state_encoder.transform(test_data['state'])

# Derived features
train_data['spkts_per_dur'] = train_data['spkts'] / (train_data['dur'] + 1e-6)
test_data['spkts_per_dur'] = test_data['spkts'] / (test_data['dur'] + 1e-6)
train_data['dpkts_per_dur'] = train_data['dpkts'] / (train_data['dur'] + 1e-6)
test_data['dpkts_per_dur'] = test_data['dpkts'] / (test_data['dur'] + 1e-6)

# Normalize
scaler = MinMaxScaler()
train_data[important_features] = scaler.fit_transform(train_data[important_features])
test_data[important_features] = scaler.transform(test_data[important_features])

# Prepare training data (normal only)
X_train = train_data[train_data['label'] == 0][important_features].values
y_train = train_data[train_data['label'] == 0]['label'].values
X_test = test_data[important_features].values
y_test = test_data['label'].values

# Autoencoder hyperparameters
input_dim = X_train.shape[1]
latent_dim = 2
batch_size = 256
epochs = 25
learning_rate = 0.00005

# Build autoencoder
input_layer = Input(shape=(input_dim,), name="Input")
encoder = Dense(64, activation='sigmoid', name="Encoder_1")(input_layer)
encoder = Dropout(0.2)(encoder)
encoder = Dense(32, activation='sigmoid', name="Encoder_2")(encoder)
encoder = Dropout(0.2)(encoder)
latent_layer = Dense(latent_dim, activation='sigmoid', name="Latent_Space")(encoder)

decoder = Dense(32, activation='sigmoid', name="Decoder_1")(latent_layer)
decoder = Dropout(0.2)(decoder)
decoder = Dense(64, activation='sigmoid', name="Decoder_2")(decoder)
decoder = Dropout(0.2)(decoder)
output_layer = Dense(input_dim, activation='sigmoid', name="Output")(decoder)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

# Train with K-Fold and collect histories
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
histories = []

for train_idx, val_idx in kf.split(X_train):
    print(f"Training fold {fold}...")
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]

    history = autoencoder.fit(
        X_fold_train, X_fold_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_fold_val, X_fold_val),
        verbose=0
    )

    histories.append(history.history)
    fold += 1

# Save model and encoders
autoencoder.save('trained_autoencoder.h5')
joblib.dump(scaler, 'trained_scaler.save')
joblib.dump(proto_encoder, 'proto_encoder.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(important_features, 'important_features.pkl')

print("Autoencoder trained and saved successfully.")

# === Plot training and validation loss for each fold ===
plt.figure(figsize=(14, 6))
for i, hist in enumerate(histories):
    plt.plot(hist['loss'], label=f'Train Loss Fold {i+1}')
    plt.plot(hist['val_loss'], label=f'Val Loss Fold {i+1}', linestyle='--')
plt.title("Training & Validation Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot average loss across all folds ===
avg_train_loss = np.mean([h['loss'] for h in histories], axis=0)
avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(avg_train_loss, label='Average Training Loss')
plt.plot(avg_val_loss, label='Average Validation Loss', linestyle='--')
plt.title("Average Training & Validation Loss (5-Fold)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
