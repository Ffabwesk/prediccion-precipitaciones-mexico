import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset_maestro.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# Split Temporal estricto (Rúbrica)
train_set = df[df['fecha'].dt.year <= 2019]
test_set = df[df['fecha'].dt.year >= 2020]

# Variables incluyendo el Rango Térmico que creamos
X_cols = ['MINIMA', 'MEDIA', 'MAXIMA', 'rango_termico', 'LATITUD', 'LONGITUD', 
          'SolarRad', 'SO_2', 'CO', 'NOx', 'PM_010', 'radiacion_lag1']
y_col = 'lluvia_binaria'

X_train_raw = train_set[X_cols]
y_train_raw = train_set[y_col]
X_test_raw = test_set[X_cols]
y_test_raw = test_set[y_col]

scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_raw)
X_test_final = scaler.transform(X_test_raw)

# Guardar con nombres estándar
np.save('data_x_train.npy', X_train_final)
np.save('data_x_test.npy', X_test_final)
np.save('data_y_train.npy', y_train_raw.values)
np.save('data_y_test.npy', y_test_raw.values)