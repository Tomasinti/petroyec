import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# ── Configuración de la página ───────────────────────────────
st.set_page_config(
    page_title='Predicción de Calidad de Hidrocarburos',
    page_icon='🛢️',
    layout='wide'
)

# ── Arquitectura de la red neuronal (debe coincidir con notebook 03) ──
class RedNeuronalCrudo(nn.Module):
    def __init__(self, input_dim):
        super(RedNeuronalCrudo, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.red(x).squeeze()


# ── Cargar modelos ───────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    # Ruta base relativa a la ubicación de este archivo
    base = os.path.join(os.path.dirname(__file__), '..', 'data')

    gb     = joblib.load(os.path.join(base, 'modelo_gb_dulce_agrio.pkl'))
    scaler = joblib.load(os.path.join(base, 'scaler_dulce_agrio.pkl'))

    modelo_nn = RedNeuronalCrudo(input_dim=8)
    modelo_nn.load_state_dict(torch.load(
        os.path.join(base, 'modelo_nn.pth'),
        map_location=torch.device('cpu')
    ))
    modelo_nn.eval()
    scaler_nn = joblib.load(os.path.join(base, 'scaler_nn.pkl'))

    return gb, scaler, modelo_nn, scaler_nn

# ── Header ───────────────────────────────────────────────────
st.title('🛢️ Predicción de Calidad de Hidrocarburos')
st.markdown('''
Ingresá las propiedades físico-químicas de una muestra de crudo y el sistema
predecirá si es **dulce (sweet)** o **agrio (sour)** según su contenido de azufre,
sin necesidad de medirlo directamente.

> Modelos entrenados con **9.069 muestras reales** del U.S. Department of Energy — Bureau of Mines.
''')

st.divider()

# ── Panel de entrada ─────────────────────────────────────────
st.subheader('📋 Propiedades de la muestra')

col1, col2, col3, col4 = st.columns(4)

with col1:
    api = st.number_input(
        'Gravedad API (°API)',
        min_value=0.0, max_value=70.0, value=35.0, step=0.1,
        help='Rango típico: 10–60°. Liviano >31.1°, Mediano 22.3–31.1°, Pesado <22.3°'
    )
    nitrogeno = st.number_input(
        'Nitrógeno (% peso)',
        min_value=0.0, max_value=1.0, value=0.05, step=0.001,
        format='%.3f',
        help='Contenido de nitrógeno en el crudo. Rango típico: 0–0.5%'
    )

with col2:
    viscosidad = st.number_input(
        'Viscosidad a 100°F (SUS)',
        min_value=0.0, max_value=2000.0, value=45.0, step=1.0,
        help='Viscosidad Saybolt Universal a 100°F. Crudos livianos: 30–60 SUS'
    )
    residuo_carbono = st.number_input(
        'Residuo de carbono (% peso)',
        min_value=0.0, max_value=20.0, value=1.5, step=0.1,
        help='Residuo Conradson/Ramsbottom. Indica calidad del crudo.'
    )

with col3:
    punto_fluidez = st.number_input(
        'Punto de fluidez (°F)',
        min_value=-50.0, max_value=150.0, value=10.0, step=1.0,
        help='Temperatura mínima a la que el crudo fluye.'
    )
    vol_gasolina = st.number_input(
        'Vol. gasolina ligera (%)',
        min_value=0.0, max_value=50.0, value=8.0, step=0.1,
        help='Volumen % de fracción de gasolina ligera en destilación.'
    )

with col4:
    vol_nafta = st.number_input(
        'Vol. gasolina/nafta (%)',
        min_value=0.0, max_value=80.0, value=28.0, step=0.1,
        help='Volumen % de fracción gasolina y nafta en destilación.'
    )
    vol_residuo = st.number_input(
        'Vol. residuo (%)',
        min_value=0.0, max_value=80.0, value=24.0, step=0.1,
        help='Volumen % de residuo tras destilación al vacío.'
    )

st.divider()

# ── Clasificación de tipo de crudo ───────────────────────────
if api > 31.1:
    tipo_crudo = '🔵 Liviano'
    color_tipo = 'blue'
elif api >= 22.3:
    tipo_crudo = '🟢 Mediano'
    color_tipo = 'green'
else:
    tipo_crudo = '🟠 Pesado'
    color_tipo = 'orange'

# ── Predicción ───────────────────────────────────────────────
if st.button('🔍 Predecir calidad', type='primary', use_container_width=True):

    features = np.array([[
        api, nitrogeno, viscosidad, punto_fluidez,
        residuo_carbono, vol_gasolina, vol_nafta, vol_residuo
    ]])

    # Gradient Boosting
    X_gb = scaler_gb.transform(features)
    proba_gb  = gb.predict_proba(X_gb)[0][1]
    pred_gb   = 'Agrio (sour)' if proba_gb >= 0.5 else 'Dulce (sweet)'

    # Red Neuronal
    X_nn = scaler_nn.transform(features)
    X_tensor = torch.FloatTensor(X_nn)
    with torch.no_grad():
        proba_nn = modelo_nn(X_tensor).item()
    pred_nn = 'Agrio (sour)' if proba_nn >= 0.5 else 'Dulce (sweet)'

    # ── Resultados ───────────────────────────────────────────
    st.subheader('📊 Resultados')

    col_tipo, col_gb, col_nn = st.columns(3)

    with col_tipo:
        st.metric('Tipo de crudo', tipo_crudo)
        st.caption('Clasificación por gravedad API (norma estándar)')

    with col_gb:
        emoji = '🟢' if pred_gb == 'Dulce (sweet)' else '🔴'
        st.metric(
            'Gradient Boosting',
            f'{emoji} {pred_gb}',
            f'Confianza: {max(proba_gb, 1-proba_gb)*100:.1f}%'
        )
        st.caption('AUC-ROC: 0.935 — modelo principal')

    with col_nn:
        emoji_nn = '🟢' if pred_nn == 'Dulce (sweet)' else '🔴'
        st.metric(
            'Red Neuronal (PyTorch)',
            f'{emoji_nn} {pred_nn}',
            f'Confianza: {max(proba_nn, 1-proba_nn)*100:.1f}%'
        )
        st.caption('AUC-ROC: 0.932 — modelo de comparación')

    st.divider()

    # ── Interpretación ───────────────────────────────────────
    st.subheader('📖 Interpretación')

    if pred_gb == 'Dulce (sweet)':
        st.success('''
        **Clasificación estimada: Crudo dulce (Sweet)**

        Basándose en las propiedades físico-químicas suministradas, los modelos 
        predicen que el contenido de azufre probablemente sea inferior al umbral 
        industrial de 0.5% en peso.

        *Esta clasificación fue inferida mediante modelos de aprendizaje automático 
        entrenados con datos históricos. No sustituye una determinación analítica 
        directa del contenido de azufre.*
        ''')
    else:
        st.warning('''
        **Clasificación estimada: Crudo agrio (Sour)**

        Basándose en las propiedades físico-químicas suministradas, los modelos 
        predicen que el contenido de azufre probablemente sea igual o superior 
        al umbral industrial de 0.5% en peso.

        *Esta clasificación fue inferida mediante modelos de aprendizaje automático 
        entrenados con datos históricos. No sustituye una determinación analítica 
        directa del contenido de azufre.*
        ''')

    if pred_gb != pred_nn:
        st.info('⚠️ Los modelos no coinciden en este caso — muestra en zona de incertidumbre. Se recomienda análisis de azufre directo.')

    # ── Referencia ───────────────────────────────────────────
    with st.expander('ℹ️ Sobre los modelos'):
        st.markdown('''
        **Gradient Boosting:** entrenado con 200 estimadores, profundidad 4, 
        learning rate 0.1. Features: propiedades físicas sin azufre.

        **Red Neuronal:** 3 capas ocultas (64→32→16 neuronas), BatchNorm, 
        Dropout, 80 épocas. Implementada en PyTorch.

        Ambos modelos fueron entrenados con **8.842 muestras reales** 
        del DOE Bureau of Mines y evaluados en un conjunto de test separado 
        del 20% (1.769 muestras).
        ''')

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption('Tomás Malafiej — Licenciatura en Ciencia de Datos, UCASAL Argentina | '
           'Datos: U.S. DOE Bureau of Mines Crude Oil Analysis Database')