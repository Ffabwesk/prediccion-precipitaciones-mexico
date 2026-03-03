import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium

# Usamos el dataset maestro que generamos
df = pd.read_csv('dataset_maestro.csv')

# 1. Matriz de Correlación (Color Viridis - Diferente al tuyo)
plt.figure(figsize=(12, 8))
df_num = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(df_num.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Interrelación de Variables Climáticas y Barométricas')
plt.savefig('matriz_correlacion_climatica.png')

# 2. Distribución de Lluvia por Zona (Diferente estilo)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='PRECIPITACION', data=df, palette='Set2')
plt.title('Variabilidad de Precipitación por Región Geográfica')
plt.savefig('distribucion_regional_precipitacion.png')

# 3. Geo-localización de Estaciones (Estilo Dark para que no se parezca al tuyo)
mapa = folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles='cartodbdark_matter')
for idx, row in df.sample(80).iterrows():
    folium.CircleMarker(
        location=[row['LATITUD'], row['LONGITUD']],
        radius=4,
        color='yellow',
        fill=True
    ).add_to(mapa)
mapa.save('mapa_cobertura_estaciones.html')