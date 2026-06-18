---

## Dataset

**Fuente:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

**Archivo utilizado:** `bank.csv` — versión reducida con 4.521 registros y 17 columnas, seleccionada específicamente para el uso con SVM dado su costo computacional.

**Variable objetivo:** `y` — ¿el cliente suscribió un depósito a plazo? (yes/no)

**Desbalance de clases:** 88.5% no suscribe / 11.5% suscribe

---

## Tecnologías Utilizadas

- Python 3.10
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Proceso del Análisis

### 1. Limpieza de Datos
- Eliminación de duplicados
- Conversión de variables binarias yes/no a 1/0
- Identificación y tratamiento de valores `unknown` disfrazados de nulos
- Imputación con moda en columnas `job`, `education` y `contact`
- Análisis y documentación de outliers
- Eliminación de `duration` por data leakage

### 2. Feature Engineering
- Transformación de `pdays` en dos variables:
  - `contacted_before`: variable binaria (1 = fue contactado antes, 0 = nunca)
  - `pdays`: reemplazo de -1 por 0

### 3. Análisis Exploratorio (EDA)
- Distribución del target y análisis del desbalance de clases
- Tasa de suscripción por variable categórica
- Mapa de calor de correlaciones entre variables numéricas
- Distribuciones de variables numéricas para selección de escaladores

### 4. Preprocesamiento
- `StandardScaler`: variable `age` (distribución aproximadamente normal)
- `RobustScaler`: `balance`, `campaign`, `pdays`, `previous`, `day` (distribuciones sesgadas con outliers)
- `OrdinalEncoder`: `education` (orden natural confirmado por EDA)
- `OneHotEncoder`: `job`, `marital`, `contact`, `month`, `poutcome`
- `passthrough`: variables binarias ya codificadas

### 5. Modelado
- Decision Tree Classifier (base + GridSearchCV)
- Support Vector Machine — SVC (base + GridSearchCV)
- Métrica de optimización: F1-score (por desbalance de clases)

### 6. Evaluación
- Accuracy, Precision, Recall, F1-score
- Matriz de confusión
- Curva ROC y AUC

---

## Resultados

| Modelo | Accuracy | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|---|
| Decision Tree Base | 0.8155 | 0.2353 | 0.2692 | 0.2511 | — |
| Decision Tree Tuneado | 0.8729 | 0.4225 | 0.2885 | 0.3429 | 0.621 |
| SVM Base | 0.8398 | 0.2929 | 0.2788 | 0.2857 | — |
| SVM Tuneado | 0.7901 | 0.2737 | 0.5000 | 0.3537 | 0.699 |

**Modelo seleccionado: SVM Tuneado**

Mejor F1-score (0.3537) y Recall (0.5000) — detecta la mitad de los clientes 
que van a suscribir, con un AUC de 0.699. En el contexto bancario, maximizar 
el Recall es prioritario porque el costo de perder un cliente potencial supera 
el costo de una llamada innecesaria.

---

## Hallazgos Clave de Negocio

- Clientes con historial de suscripción exitosa tienen 6 veces más probabilidad 
  de volver a suscribir
- Octubre, diciembre y marzo son los meses con mayor tasa de conversión (>40%)
- Mayo concentra la mayor cantidad de llamadas pero tiene la peor tasa de 
  conversión (6.7%)
- Jubilados y estudiantes son los segmentos con mayor propensión a suscribir
- Más de 3 contactos por cliente tiene rendimientos decrecientes

---

## Consideraciones Éticas

**Transparencia:** todos los pasos del análisis están documentados y son 
reproducibles. El notebook incluye justificación de cada decisión tomada.

**Sesgo:** el dataset presenta desbalance severo (88.5%/11.5%) que fue tratado 
con `class_weight='balanced'` en ambos modelos. Se documentaron las limitaciones 
del modelo en la detección de la clase minoritaria.

**Vigencia:** los datos corresponden a 2008-2010. El modelo debe reentrenarse 
con datos recientes antes de desplegarse en producción.

---

## Cómo Ejecutar el Proyecto

```bash
# Clonar el repositorio
git clone <url-del-repositorio>

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Ejecutar el notebook
jupyter notebook Bank_Marketing_Clasificacion.ipynb
```

---

## Autor
Karen Herrera
Bootcamp Data Science — Módulo 2  
Instructor: Jesús Ortiz
