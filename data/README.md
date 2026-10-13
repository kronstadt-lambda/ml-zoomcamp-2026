# Data Reference — ML Zoomcamp 2026

> Referencia de estudio para los datasets usados durante ML Zoomcamp 2026.
>
> Este README está pensado para consultar rápidamente **qué representa cada dataset, qué significa cada feature, cuál es el target, qué valores faltantes existen y qué relaciones merece la pena investigar** durante EDA, feature engineering, modelado y evaluación.
>
> **Importante:** los datasets son controlados y contienen relaciones introducidas deliberadamente. Las correlaciones y dependencias descritas aquí sirven para entender el comportamiento de los datos, pero **no deben interpretarse como relaciones causales del mundo real**.

---

# Estructura sugerida

```text
data/
├── README.md
│
├── module02/
│   └── raw/
│       └── car_fuel_efficiency_2026.csv
│
└── module03/
    └── raw/
        └── course_lead_scoring_2026.csv
```

Los CSV pueden mantenerse fuera de Git si se descargan desde la fuente oficial del curso.

---

# 1. Car Fuel Efficiency 2026

## Objetivo general

Dataset del dominio **automotriz** para una tarea de **regresión**.

El objetivo es estudiar cómo las características de un vehículo se relacionan con su eficiencia de combustible y construir un modelo capaz de predecir:

```text
fuel_efficiency_mpg
```

El dataset contiene:

```text
10,000 filas
11 columnas totales
```

El target tiene aproximadamente:

```text
media     = 29.9977 mpg
std       = 2.9302
mínimo    = 19.8 mpg
máximo    = 41.2 mpg
```

---

## Target

### `fuel_efficiency_mpg`

Eficiencia de combustible del vehículo expresada en millas por galón.

Tipo:

```text
numérico continuo
```

Problema:

```text
regresión
```

### Relaciones esperadas con el target

La construcción del dataset introduce relaciones plausibles como:

```text
vehículos más nuevos
        ↑
mayor eficiencia

mayor displacement
        ↓
menor eficiencia

mayor horsepower
        ↓
menor eficiencia

mayor peso
        ↓
menor eficiencia

Hybrid
        ↑
mayor eficiencia

All-wheel drive
        ↓
menor eficiencia
```

Estas relaciones son útiles como **hipótesis iniciales para EDA**, no como conclusiones que deban asumirse sin verificar.

---

# Data Dictionary — Car Fuel Efficiency

| Feature | Tipo | Significado | Observaciones |
|---|---|---|---|
| `model_year` | Numérica | Año/modelo del vehículo | Rango esperado: 1975–2024 |
| `origin` | Categórica | Región amplia de fabricación | USA, Europe, Asia |
| `fuel_type` | Categórica | Tipo principal de combustible/powertrain | Gasoline, Diesel, Hybrid |
| `drivetrain` | Categórica | Tipo de ruedas motrices | Front-wheel, Rear-wheel, All-wheel |
| `num_doors` | Numérica discreta | Número de puertas | 2–5 |
| `engine_displacement` | Numérica | Cilindrada del motor en cc | Relacionada con año, origen y drivetrain |
| `num_cylinders` | Numérica discreta | Número de cilindros | Relacionada con displacement |
| `horsepower` | Numérica | Potencia del motor | Principalmente relacionada con displacement y cilindros |
| `vehicle_weight` | Numérica | Peso del vehículo en libras | Relacionado con motor, cilindros, puertas y drivetrain |
| `acceleration` | Numérica | Tiempo de aceleración | Mayor peso tiende a empeorarla; mayor potencia ayuda |
| `fuel_efficiency_mpg` | Numérica continua | **Target** | Eficiencia de combustible |

---

# Distribución de variables categóricas

## `origin`

```text
USA       4202
Asia      2994
Europe    2804
```

No existe una categoría extremadamente dominante.

---

## `fuel_type`

```text
Gasoline    6913
Diesel      2053
Hybrid      1034
```

Hay un claro predominio de vehículos a gasolina.

Durante EDA conviene observar si:

```text
fuel_type
    ↓
produce diferencias claras en fuel_efficiency_mpg
```

---

## `drivetrain`

```text
Front-wheel drive    5519
Rear-wheel drive     2455
All-wheel drive      2026
```

---

# Variables numéricas — referencia rápida

| Feature | Media | Desviación estándar | Missing |
|---|---:|---:|---:|
| `model_year` | 1999.55 | 14.21 | 0% |
| `num_doors` | 3.50 | 0.95 | 0% |
| `engine_displacement` | 2268.81 | 183.45 | 0% |
| `num_cylinders` | 6.04 | 0.54 | 0% |
| `horsepower` | 254.45 | 22.84 | **8.77%** |
| `vehicle_weight` | 4286.75 | 266.21 | 0% |
| `acceleration` | 18.89 | 1.28 | **2.64%** |

---

# Missing Values — Car Dataset

Las variables con missing values son:

```text
horsepower      ≈ 8.77%
acceleration    ≈ 2.64%
```

Esto es importante porque la ausencia de datos **no es necesariamente completamente aleatoria**.

En particular:

- `horsepower` presenta mayor probabilidad de faltar en vehículos anteriores a 1985.
- `acceleration` presenta missingness condicionado por `origin = Europe`.

Por eso, antes de hacer una imputación automática, conviene estudiar:

```python
df.groupby("model_year")["horsepower"].apply(lambda s: s.isna().mean())
```

o comparar:

```text
missing horsepower
vs
model_year
```

y:

```text
missing acceleration
vs
origin
```

Esto puede ayudarte a reconocer patrones de missingness y evitar tratar todos los NaN como si surgieran al azar.

---

# Relaciones internas importantes

El dataset contiene relaciones relevantes entre features.

## `engine_displacement` ↔ `num_cylinders`

Correlación observada:

```text
≈ 0.660
```

Interpretación:

> Motores con mayor desplazamiento tienden a tener más cilindros.

---

## `engine_displacement` ↔ `horsepower`

Correlación observada:

```text
≈ 0.761
```

Relación fuerte.

Esto puede generar:

```text
información redundante
+
multicolinealidad
```

especialmente relevante para modelos lineales.

---

## `num_cylinders` ↔ `horsepower`

Correlación:

```text
≈ 0.602
```

---

## `engine_displacement` ↔ `vehicle_weight`

Correlación:

```text
≈ 0.686
```

Los vehículos con motores mayores tienden a ser más pesados.

---

## `num_cylinders` ↔ `vehicle_weight`

Correlación:

```text
≈ 0.524
```

---

## `num_doors` ↔ `vehicle_weight`

Correlación:

```text
≈ 0.180
```

Relación relativamente débil.

---

# Preguntas útiles para EDA — Car Dataset

Antes de modelar, intentar responder:

```text
¿fuel_efficiency_mpg cambia claramente según fuel_type?

¿qué relación existe entre vehicle_weight y fuel_efficiency_mpg?

¿horsepower y engine_displacement contienen información demasiado similar?

¿model_year mejora la predicción incluso después de considerar motor y peso?

¿el efecto de fuel_type cambia según model_year?

¿la ausencia de horsepower aporta información por sí misma?

¿conviene crear una variable vehicle_age?

¿existen interacciones importantes entre drivetrain y vehicle_weight?
```

---

# Posibles ideas de Feature Engineering

No asumir que serán útiles: deben validarse.

## Edad aproximada del vehículo

```python
vehicle_age = reference_year - model_year
```

Puede ser más interpretable que utilizar directamente el año.

---

## Power-to-weight ratio

```python
power_to_weight = horsepower / vehicle_weight
```

Puede capturar una relación física más útil que ambas variables por separado.

---

## Displacement por cilindro

```python
displacement_per_cylinder = (
    engine_displacement / num_cylinders
)
```

---

## Missing indicators

```python
horsepower_missing = horsepower.isna().astype(int)
```

Puede ser útil porque el missingness de `horsepower` tiene estructura.

---

# Riesgos durante el modelado

## Multicolinealidad

Estas variables están relacionadas:

```text
engine_displacement
num_cylinders
horsepower
vehicle_weight
```

En regresión lineal esto merece especial atención.

Árboles y boosting manejan este tipo de redundancia de manera diferente, por lo que la importancia de las features puede cambiar según el modelo.

---

## Leakage

No construir features utilizando información del target.

Por ejemplo:

```text
fuel_efficiency_mpg
```

solo debe utilizarse como `y`.

---

---

# 2. Course Lead Scoring 2026

## Objetivo general

Dataset del dominio **marketing / funnel** para una tarea de **clasificación**.

Representa prospectos interesados en un curso.

El objetivo consiste en predecir:

```text
converted
```

es decir:

```text
1 → el lead convierte
0 → el lead no convierte
```

El dataset contiene:

```text
5,000 filas
9 columnas totales
```

Distribución del target:

```text
converted = 1    2883
converted = 0    2117
```

Proporción positiva:

```text
≈ 57.66%
```

Existe cierto desbalance, pero no es un caso extremo.

---

# Target

## `converted`

Variable binaria que representa si un prospecto terminó convirtiéndose.

Tipo:

```text
binario
```

Problema:

```text
clasificación
```

---

# Data Dictionary — Course Lead Scoring

| Feature | Tipo | Significado | Observaciones |
|---|---|---|---|
| `lead_source` | Categórica | Cómo conoció el prospecto el curso | paid_ads, social_media, organic_search, referral, events |
| `industry` | Categórica | Industria del prospecto | technology, healthcare, retail, finance, education, manufacturing |
| `employment_status` | Categórica | Situación laboral | employed, self_employed, unemployed, student |
| `location` | Categórica | Región geográfica amplia | north_america, europe, asia, south_america, africa |
| `annual_income` | Numérica | Ingreso anual estimado | Relacionado con industria, empleo y ubicación |
| `number_of_courses_viewed` | Numérica discreta | Número de cursos vistos | Medida de interés inicial |
| `interaction_count` | Numérica discreta | Número de interacciones medibles | Relacionada con engagement |
| `lead_score` | Numérica | Score previo a conversión | Resume señales observables de engagement e ingreso |
| `converted` | Binaria | **Target** | 1 = convierte; 0 = no convierte |

---

# Distribuciones categóricas

## `lead_source`

```text
organic_search    1251
paid_ads          1140
social_media      1055
referral           790
events             616
```

Missing:

```text
≈ 2.96%
```

---

## `industry`

Valores principales:

```text
technology    1173
retail         925
healthcare     840
finance        720
education      639
```

Missing:

```text
≈ 4.80%
```

---

## `employment_status`

```text
employed         2633
student           760
self_employed     715
unemployed        698
```

Missing:

```text
≈ 3.88%
```

---

## `location`

```text
north_america    1371
europe           1149
asia             1052
south_america     749
africa            471
```

Missing:

```text
≈ 4.16%
```

---

# Variables numéricas — referencia rápida

| Feature | Media | Desviación estándar | Missing |
|---|---:|---:|---:|
| `annual_income` | 54,533.21 | 21,245.66 | **7.38%** |
| `number_of_courses_viewed` | 2.09 | 1.19 | 0% |
| `interaction_count` | 4.47 | 2.44 | 0% |
| `lead_score` | 0.495 | 0.199 | **0.70%** |

---

# Missing Values — Lead Dataset

Missingness:

```text
lead_source          2.96%
industry             4.80%
employment_status    3.88%
location              4.16%
annual_income         7.38%
lead_score            0.70%
```

Hay patrones importantes:

### `annual_income`

Tiene mayor probabilidad de faltar para:

```text
employment_status = student
employment_status = unemployed
```

Por lo tanto:

```text
annual_income is missing
```

puede contener información predictiva por sí misma.

Una posible feature:

```python
income_missing = df["annual_income"].isna().astype(int)
```

Debe validarse antes de conservarla.

### `lead_score`

Parte de sus missing values está asociada a:

```text
lead_source = events
```

Nuevamente, el missingness merece investigarse y no simplemente eliminarse.

---

# Relaciones internas importantes

## `number_of_courses_viewed` ↔ `interaction_count`

Correlación:

```text
≈ 0.722
```

Interpretación:

> Quienes ven más cursos tienden a registrar más interacciones.

---

## `number_of_courses_viewed` ↔ `lead_score`

Correlación:

```text
≈ 0.769
```

---

## `interaction_count` ↔ `lead_score`

Correlación:

```text
≈ 0.929
```

Esta relación es extremadamente fuerte.

Esto merece especial atención porque:

```text
interaction_count
+
lead_score
```

pueden contener información altamente redundante.

Para un modelo lineal, conviene estudiar multicolinealidad.

Para árboles/boosting, ambas pueden seguir siendo útiles, pero sus importancias pueden repartirse entre sí.

---

## `annual_income` ↔ `lead_score`

Correlación:

```text
≈ 0.231
```

Relación positiva pero relativamente moderada.

---

# Dependencias conceptuales del funnel

El dataset sigue aproximadamente esta historia:

```text
lead_source
industry
employment_status
location
       ↓
características del prospecto
       ↓
annual_income
number_of_courses_viewed
       ↓
interaction_count
       ↓
lead_score
       ↓
converted
```

No significa que todas estas relaciones sean causales en el mundo real.

Sirve para entender que existen **features más cercanas al momento de conversión** y otras más tempranas.

Esto es especialmente útil cuando analices:

```text
feature importance
leakage
disponibilidad temporal de features
```

---

# Una pregunta muy importante: ¿`lead_score` debería usarse?

`lead_score` es una feature válida dentro del dataset, pero conceptualmente merece una evaluación especial.

Preguntas:

```text
¿este score existiría realmente antes de conocer converted?

¿cómo se calcularía en producción?

¿está disponible en el momento exacto de hacer la predicción?

¿usar lead_score hace redundantes otras variables de engagement?
```

En este dataset está definido como un score **pre-conversión**, por lo que conceptualmente puede utilizarse.

Sin embargo, este tipo de pregunta es exactamente la que debe hacerse un ML Engineer para detectar posibles casos de **target leakage** en datasets reales.

---

# Preguntas útiles para EDA — Lead Dataset

```text
¿qué lead_source tiene mayor tasa de conversión?

¿cómo cambia converted según employment_status?

¿interaction_count tiene una relación monotónica con conversion?

¿lead_score domina demasiado al resto de features?

¿podemos mantener buen rendimiento sin lead_score?

¿annual_income agrega señal una vez incluido lead_score?

¿los missing values tienen relación con converted?

¿lead_source interactúa con engagement?

¿number_of_courses_viewed e interaction_count son redundantes?
```

---

# Posibles ideas de Feature Engineering

Estas son hipótesis; deben validarse experimentalmente.

## Engagement per viewed course

```python
engagement_per_course = (
    interaction_count /
    number_of_courses_viewed.clip(lower=1)
)
```

---

## Income missing indicator

```python
income_missing = annual_income.isna().astype(int)
```

---

## Engagement buckets

Ejemplo:

```text
0–2 interactions      → low
3–6                   → medium
7+                    → high
```

Solo tiene sentido si mejora interpretación o rendimiento.

---

## Agrupaciones del source

Puede investigarse si ciertos canales tienen comportamiento parecido.

No agrupar antes de observar evidencia.

---

# Feature Engineering: estrategia recomendada

Para ambos datasets:

```text
1. Baseline con variables originales
        ↓
2. Validación
        ↓
3. Añadir una feature nueva
        ↓
4. Reentrenar bajo las mismas condiciones
        ↓
5. Comparar validation metric
        ↓
6. Mantener solo si aporta valor
```

No evaluar una feature usando únicamente training performance.

---

# Feature Selection: preguntas clave

Al analizar una feature:

```text
¿aporta señal nueva?

¿es redundante?

¿está disponible en producción?

¿contiene información del futuro?

¿tiene demasiados missing values?

¿es estable en el tiempo?

¿es fácil de calcular?

¿su costo de cálculo vale la mejora obtenida?
```

---

# Herramientas útiles durante el análisis

## Correlation

Útil especialmente para features numéricas.

Pero recuerda:

```text
correlación ≠ causalidad
```

y una correlación baja no implica necesariamente que una feature sea inútil para un modelo no lineal.

---

## Permutation Importance

Pregunta:

> ¿Cuánto empeora el modelo si destruyo la información de esta feature?

Útil para comparar la dependencia real del modelo respecto de cada feature.

---

## Ablation

Ejemplo:

```text
modelo completo
AUC = 0.89

sin lead_score
AUC = 0.81

sin annual_income
AUC = 0.887
```

Permite estudiar qué features realmente aportan.

---

## SHAP

Puede ayudar a estudiar:

```text
importancia global
+
explicaciones individuales
```

Pero una alta importancia SHAP no demuestra causalidad.

---

# Comparar modelos antes de concluir sobre las features

La utilidad de una feature depende también del modelo.

Ejemplo:

```text
Linear Regression / Logistic Regression
    ↓
puede beneficiarse de transformaciones explícitas
e interacciones diseñadas manualmente

Tree models
    ↓
pueden descubrir relaciones no lineales
e interacciones automáticamente

Boosting
    ↓
puede aprovechar relaciones complejas
sin necesitar tantas transformaciones manuales
```

Por eso una feature que mejora mucho un modelo lineal puede aportar poco a XGBoost, y viceversa.

---

# Checklist antes de entrenar

```text
[ ] Identifiqué correctamente el target
[ ] Separé variables numéricas y categóricas
[ ] Revisé missing values
[ ] Revisé la distribución del target
[ ] Revisé cardinalidad de categóricas
[ ] Revisé posibles relaciones entre features
[ ] Revisé posibles features redundantes
[ ] Pensé en posibles casos de leakage
[ ] Definí train / validation / test
[ ] Construí primero un baseline
```

---

# Checklist para evaluar una nueva feature

```text
[ ] Tiene sentido dentro del dominio
[ ] Puede calcularse sin utilizar el target
[ ] Estaría disponible en producción
[ ] No utiliza información futura
[ ] Se calcula de la misma forma en train e inference
[ ] Comparé performance antes/después
[ ] Revisé estabilidad en validation
[ ] Revisé si duplica información existente
[ ] Consideré su costo de cálculo/mantenimiento
```

---

# Recordatorio

La finalidad de estos datasets no es simplemente:

```text
obtener el mejor score posible
```

sino aprender a razonar sobre:

```text
datos
↓
features
↓
target
↓
modelo
↓
validación
↓
decisión
```

Una buena feature no es necesariamente la más compleja.

Es aquella que:

```text
representa información útil
+
está disponible cuando necesitamos predecir
+
mejora de forma reproducible el modelo
+
puede mantenerse en producción
```
