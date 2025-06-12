import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Medical Data Generator (from your code)
class MedicalDataGenerator:
    def __init__(self, patient_id, num_records=50000):
        self.patient_id = patient_id
        self.num_records = num_records
        self.vitals = ['HR', 'PP', 'BP_S', 'BP_D', 'RESP', 'SpO2']
        self.base_values = {
            'HR': np.random.normal(75, 5),
            'PP': np.random.normal(40, 5),
            'BP_S': np.random.normal(120, 5),
            'BP_D': np.random.normal(80, 5),
            'RESP': np.random.normal(16, 2),
            'SpO2': np.random.normal(98, 0.5)
        }

    def generate_normal_data(self):
        data = []
        previous = {v: val for v, val in self.base_values.items()}
        for _ in range(self.num_records):
            record = {}
            for v in self.vitals:
                new_val = 0.8 * previous[v] + 0.2 * np.random.normal(self.base_values[v], 1.5)
                record[v] = max(new_val, 0)
                previous[v] = new_val
            data.append(record)
        return pd.DataFrame(data)

    def inject_events(self, df, num_events=20, duration_range=(20, 50)):
        df = df.copy()
        self.event_indices = []
        for _ in range(num_events):
            duration = np.random.randint(*duration_range)
            start = np.random.randint(1000, len(df) - 1000)
            self.event_indices.extend(range(start, start + duration))
            affected = np.random.choice(self.vitals, np.random.randint(2, 4), replace=False)
            for v in affected:
                peak = np.random.uniform(5, 10)
                ramp_up = np.linspace(0, peak, (duration + 1) // 2)
                ramp_down = np.linspace(peak, 0, duration // 2)
                pattern = np.concatenate([ramp_up, ramp_down])
                df.loc[start:start + duration - 1, v] += pattern + np.random.normal(0, 0.5, duration)
        return df

    def inject_anomalies(self, df, anomaly_rate=0.15):
        df = df.copy()
        num = max(int(len(df) * anomaly_rate), 100)
        self.anomaly_indices = set()
        types = ['spike', 'drop', 'noise', 'freeze', 'drift']
        probs = [0.3, 0.3, 0.2, 0.1, 0.1]
        for _ in range(num):
            t = np.random.choice(types, p=probs)
            idx = np.random.randint(100, len(df) - 100)
            self.anomaly_indices.add(idx)
            v = np.random.choice(self.vitals)
            if t == 'spike':
                df.loc[idx, v] += np.random.normal(15, 3)
            elif t == 'drop':
                df.loc[idx, v] -= np.random.normal(12, 2)
            elif t == 'noise':
                df.loc[idx, v] += np.random.normal(0, 5)
            elif t == 'freeze':
                dur = np.random.randint(5, 10)
                df.loc[idx:idx + dur, v] = df.loc[idx, v]
            elif t == 'drift':
                dur = np.random.randint(10, 20)
                slope = np.random.uniform(-0.5, 0.5)
                df.loc[idx:idx + dur, v] += np.arange(dur + 1) * slope
        return df

# Generate data
generator = MedicalDataGenerator(patient_id=101)
normal_data = generator.generate_normal_data()
train_data, test_data = train_test_split(normal_data, test_size=0.3, shuffle=False)
test_data = test_data.reset_index(drop=True)
test_data = generator.inject_events(test_data)
test_data = generator.inject_anomalies(test_data)

# Save test data for inference
test_data.to_csv("test_data.csv", index=False)
with open("event_indices.pkl", "wb") as f:
    pickle.dump(generator.event_indices, f)
with open("anomaly_indices.pkl", "wb") as f:
    pickle.dump(generator.anomaly_indices, f)

# Train autoencoder
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

input_layer = Input(shape=(scaled_train.shape[1],))
x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01))(input_layer)
x = Dropout(0.4)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
bottleneck = Dense(16)(x)
x = Dense(32, activation='relu')(bottleneck)
x = Dropout(0.3)(x)
output = Dense(scaled_train.shape[1], activation='sigmoid')(x)

autoencoder = Model(input_layer, output)
autoencoder.compile(Adam(0.001), 'mse')

noise = np.random.normal(0, 0.05, scaled_train.shape)
noisy_train = np.clip(scaled_train + noise, 0, 1)

autoencoder.fit(
    noisy_train, scaled_train,
    epochs=100,
    batch_size=512,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# Save autoencoder
autoencoder.save("autoencoder.h5")

# Calculate threshold
val_pred = autoencoder.predict(scaled_train, verbose=0)
val_mse = np.mean(np.square(scaled_train - val_pred), axis=1)
threshold = np.mean(val_mse) + 2 * np.std(val_mse)
if threshold <= 0:
    threshold = np.percentile(val_mse, 95)

# Save scaler and threshold
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold, f)

# Train classifier
reconstructions = autoencoder.predict(scaled_test, verbose=0)
mse = np.mean(np.square(scaled_test - reconstructions), axis=1)
anomalies = np.where(mse > threshold)[0]
X, y = [], []

for idx in anomalies:
    try:
        errors = np.abs(scaled_test[idx] - reconstructions[idx])
        start = max(0, idx - 20)
        window = scaled_test[start:idx + 1]
        temp_features = [
            np.mean(window, 0),
            np.std(window, 0),
            np.median(window, 0),
            np.ptp(window, 0)
        ]
        features = np.concatenate([errors, *temp_features, [mse[idx]]])
        X.append(features)
        y.append(1 if idx in generator.event_indices else 0)
    except:
        continue

X, y = np.array(X), np.array(y)
X, y = SMOTE(random_state=42).fit_resample(X, y)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l1_l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(Adam(0.0005), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=64, validation_split=0.2, verbose=1)

# Save classifier
model.save("classifier.h5")

print("Models and artifacts saved successfully.")