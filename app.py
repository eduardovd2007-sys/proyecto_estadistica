import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import google.generativeai as genai
import time

# ================================
# 🔑 CONFIGURACIÓN GEMINI
# ================================
# Usamos gemini-1.5-flash por ser el más estable en la versión gratuita
MODELO_IA = 'gemini-2.5-flash-lite' 

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(MODELO_IA)
else:
    st.error("⚠️ No se encontró la API KEY. Agrégala en .streamlit/secrets.toml")

st.set_page_config(page_title="Estadística Pro AI", layout="wide")
st.title("📊 App de Estadística con IA")
st.write("Proyecto Final - Probabilidad y Estadística")

# ================================
# 📂 CARGA Y PROCESAMIENTO
# ================================
st.header("📂 Gestión de Datos")
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
else:
    st.info("💡 Usando datos de ejemplo (Distribución Normal)")
    data = np.random.normal(50, 10, 100)
    df = pd.DataFrame(data, columns=["valores"])

# --- Conversión Inteligente de Encuestas ---
mapa_respuestas = {
    "muy bajo": 1, "bajo": 2, "medio": 3, "alto": 4, "muy alto": 5,
    "nunca": 1, "a veces": 2, "frecuente": 3, "siempre": 4,
    "no": 0, "sí": 1, "si": 1
}

def limpiar_datos(df_input):
    temp_df = df_input.copy()
    for col in temp_df.columns:
        # Intentar convertir a número directamente
        if temp_df[col].dtype == object:
            # Aplicar mapa de encuestas si es texto
            temp_df[col] = temp_df[col].astype(str).str.lower().str.strip().map(mapa_respuestas).fillna(temp_df[col])
            # Intentar forzar a numérico después del mapeo
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    return temp_df

df_final = limpiar_datos(df)
df_numeric = df_final.select_dtypes(include=[np.number])

if df_numeric.empty:
    st.error("❌ No hay datos numéricos para analizar.")
    st.stop()

st.dataframe(df_final.head(), use_container_width=True)

# ================================
# 📊 VISUALIZACIÓN
# ================================
st.header("📊 Análisis Descriptivo")
col_seleccionada = st.selectbox("Selecciona la variable", df_numeric.columns)

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots()
    sns.histplot(df_numeric[col_seleccionada], kde=True, ax=ax, color="#4A90E2")
    st.pyplot(fig)

with c2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df_numeric[col_seleccionada], ax=ax2, color="#F5A623")
    st.pyplot(fig2)

# Estadísticas Rápidas
media_calc = df_numeric[col_seleccionada].mean()
std_calc = df_numeric[col_seleccionada].std()
n_calc = len(df_numeric[col_seleccionada].dropna())

st.write(f"**Media:** {media_calc:.2f} | **Desviación Estándar:** {std_calc:.2f} | **n:** {n_calc}")

# ================================
# 📈 PRUEBA DE HIPÓTESIS
# ================================
st.header("📈 Prueba de Hipótesis (Z)")

with st.container():
    k1, k2, k3 = st.columns(3)
    mu_muestral = k1.number_input("Media Muestral (x̄)", value=float(media_calc))
    mu_h0 = k2.number_input("Media Hipotética (μ₀)", value=50.0)
    sigma = k3.number_input("Desviación Estándar (σ)", value=float(std_calc))

    k4, k5, k6 = st.columns(3)
    n_input = k4.number_input("Tamaño Muestra (n)", min_value=1, value=int(n_calc))
    alpha = k5.slider("Significancia (α)", 0.01, 0.10, 0.05)
    tipo_test = k6.selectbox("Hipótesis", ["bilateral", "derecha", "izquierda"])

if st.button("🚀 Ejecutar Prueba Z"):
    # Cálculo de Z
    z_stat = (mu_muestral - mu_h0) / (sigma / np.sqrt(n_input))
    
    # Cálculo de p-value y crítico
    if tipo_test == "bilateral":
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))
        z_crit = norm.ppf(1 - alpha/2)
    elif tipo_test == "derecha":
        p_val = 1 - norm.cdf(z_stat)
        z_crit = norm.ppf(1 - alpha)
    else:
        p_val = norm.cdf(z_stat)
        z_crit = norm.ppf(alpha)

    # Guardar en estado de sesión
    st.session_state.stats = {
        "z": z_stat, "p": p_val, "z_crit": z_crit, 
        "decision": "Rechazar H0" if p_val < alpha else "No rechazar H0"
    }
    # Limpiar respuesta de IA previa al cambiar datos
    if "respuesta_ia" in st.session_state:
        del st.session_state.respuesta_ia

# Mostrar resultados estadísticos
if "stats" in st.session_state:
    res = st.session_state.stats
    st.divider()
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Estadístico Z", f"{res['z']:.4f}")
        st.metric("P-Value", f"{res['p']:.5f}")
        if res['p'] < alpha:
            st.error(f"Resultado: {res['decision']}")
        else:
            st.success(f"Resultado: {res['decision']}")
    
    with m2:
        x = np.linspace(-4, 4, 500)
        y = norm.pdf(x)
        fig_z, ax_z = plt.subplots(figsize=(7, 3))
        ax_z.plot(x, y, 'k')
        
        # Regiones de rechazo
        if tipo_test == "bilateral":
            ax_z.fill_between(x, y, where=(x <= -res['z_crit']) | (x >= res['z_crit']), color='red', alpha=0.3)
        elif tipo_test == "derecha":
            ax_z.fill_between(x, y, where=(x >= res['z_crit']), color='red', alpha=0.3)
        else:
            ax_z.fill_between(x, y, where=(x <= res['z_crit']), color='red', alpha=0.3)
            
        ax_z.axvline(res['z'], color='blue', linestyle='--', label=f"Z={res['z']:.2f}")
        ax_z.legend()
        st.pyplot(fig_z)

# ================================
# 🤖 ASISTENTE IA
# ================================
st.divider()
st.subheader("🤖 Consultoría Estadística con IA")

if st.button("🧠 Obtener Interpretación Profunda"):
    if "stats" in st.session_state:
        res = st.session_state.stats
        with st.spinner("La IA está analizando tus resultados..."):
            prompt = f"""
            Actúa como un experto en estadística (PhD). 
            Analiza estos resultados de una prueba Z:
            - Variable: {col_seleccionada}
            - x̄: {mu_muestral}, μ₀: {mu_h0}, σ: {sigma}, n: {n_input}
            - α: {alpha}, Tipo: {tipo_test}
            - Z calculado: {res['z']:.4f}, p-value: {res['p']:.5f}
            - Decisión: {res['decision']}
            
            Estructura tu respuesta en español:
            1. Significado del P-value en este caso.
            2. Interpretación para un cliente no técnico.
            3. Conclusión sobre si el efecto es práctico o solo estadístico.
            4. Recomendación técnica (¿se cumplen los supuestos?).
            """
            try:
                response = model.generate_content(prompt)
                st.session_state.respuesta_ia = response.text
            except Exception as e:
                st.error(f"Error de Cuota o Conexión: {e}. Intenta en unos minutos.")
    else:
        st.warning("⚠️ Primero ejecuta la Prueba Z arriba.")

# Mostrar respuesta con persistencia
if "respuesta_ia" in st.session_state:
    st.markdown("---")
    st.markdown("### 📝 Análisis del Experto AI")
    st.info(st.session_state.respuesta_ia)