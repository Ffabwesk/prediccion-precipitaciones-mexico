# 🌧️ Evaluación Predictiva de Precipitaciones Pluviales en México

## 📋 Descripción del Proyecto
Este proyecto final se centra en la evaluación predictiva de precipitaciones pluviales a escala nacional. Para ello, se integraron y analizaron datos climáticos de CONAGUA (clima y lluvia) junto con registros de radiación solar (SSM/UNAM & CeMIE-Sol) y emisiones contaminantes (SEMARNAT). El objetivo principal es predecir la precipitación diaria (binarizada >0 mm) y analizar cómo la radiación solar y las partículas contaminantes influyen en estos fenómenos meteorológicos.

## 🗂️ Estructura del Repositorio
El proyecto está organizado de la siguiente manera:

* **`/python`**: Contiene los scripts principales del proyecto.
  * `analisis_exploratorio_regional.py`: Script con el EDA (Análisis Exploratorio de Datos) nacional, correlaciones y preprocesamiento.
  * `procesamiento_datos_temporales.py`: Integración de datasets (CONAGUA, radiación, SEMARNAT), manejo de gaps mediante interpolación y feature engineering.
  * `evaluacion_modelos_predictivos.py`: Entrenamiento de modelos con split temporal (train hasta 2019, test 2020+).
* **`/imagenes`**: Contiene las visualizaciones generadas, incluyendo distribución regional, matrices de correlación climática, comparativas de rendimiento (F1) y matrices de confusión.
* **`dataset_maestro.csv`**: Dataset integrado y preprocesado utilizado para el entrenamiento de los modelos.

## 🛠️ Modelos Implementados y Tecnologías
Para este análisis se programó en **Python** (Pandas, Scikit-learn, TensorFlow/XGBoost), implementando y evaluando los siguientes modelos:
1. Regresión Logística
2. Random Forest
3. Gradient Boosting (XGBoost)
4. Redes Neuronales LSTM (para el análisis de series temporales)

Los modelos fueron evaluados utilizando matrices de confusión, Accuracy, F1-Score y AUC-ROC, prestando especial atención a la importancia de las variables (feature importance) como la radiación y la concentración de PM por estado.
