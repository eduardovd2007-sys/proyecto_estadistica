import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import google.generativeai as genai

# ================================
# CONFIGURACIÓN DE GEMINI
# ================================
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("⚠️ No se encontró la GEMINI_API_KEY en los secretos de Streamlit.")

st.title("📊 App de Estadística con IA")
st.write("Proyecto final - Probabilidad y Estadística")

# ================================
# CARGA DE DATOS
# ================================
st.header("📂 Carga de datos")
archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.write("Datos cargados:")
    st.dataframe(df)
else:
    st.info("Generando datos de ejemplo...")
    data = np.random.normal(50, 10, 100)
    df = pd.DataFrame(data, columns=["valores"])
    st.dataframe(df)

# ================================
# VISUALIZACIÓN
# ================================
st.header("📊 Visualización de datos")
col = st.selectbox("Selecciona variable para analizar", df.columns)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df[col], ax=ax2)
    st.pyplot(fig2)

# ================================
# ✅ ANÁLISIS AUTOMÁTICO (NUEVO)
# ================================
st.subheader("📌 Análisis de la distribución")

media = df[col].mean()
mediana = df[col].median()

st.write(f"Media: {media:.2f}")
st.write(f"Mediana: {mediana:.2f}")

if abs(media - mediana) < 1:
    st.write("✔ La distribución parece aproximadamente normal")
else:
    st.write("⚠ La distribución puede tener sesgo")

# Outliers
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1

outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]

if len(outliers) > 0:
    st.write(f"⚠ Hay {len(outliers)} outliers detectados")
else:
    st.write("✔ No se detectaron outliers")

# ================================
# PRUEBA Z
# ================================
st.header("📈 Prueba de hipótesis (Z)")

c1, c2, c3 = st.columns(3)
media_muestral = c1.number_input("Media muestral", value=float(df[col].mean()))
media_hipotetica = c2.number_input("Media hipotética (H0)", value=50.0)
sigma = c3.number_input("Desviación estándar (σ)", value=float(df[col].std()))

c4, c5 = st.columns(2)
n = c4.number_input("Tamaño de muestra (n)", min_value=1, value=len(df))
alpha = c5.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05)

tipo = st.selectbox("Tipo de prueba", ["bilateral", "izquierda", "derecha"])

if st.button("🚀 Ejecutar análisis"):
    z = (media_muestral - media_hipotetica) / (sigma / np.sqrt(n))

    if tipo == "bilateral":
        p = 2 * (1 - norm.cdf(abs(z)))
        z_crit = norm.ppf(1 - alpha/2)
    elif tipo == "derecha":
        p = 1 - norm.cdf(z)
        z_crit = norm.ppf(1 - alpha)
    else:
        p = norm.cdf(z)
        z_crit = norm.ppf(alpha)

    st.session_state.z = z
    st.session_state.p = p
    st.session_state.z_crit = z_crit
    st.session_state.decision = "Rechazar H0" if p < alpha else "No rechazar H0"

# ================================
# RESULTADOS
# ================================
if "z" in st.session_state:

    z = st.session_state.z
    p = st.session_state.p
    decision = st.session_state.decision
    z_crit = st.session_state.z_crit

    st.subheader("📊 Resultados Estadísticos")
    st.write(f"**Z:** {z:.4f} | **p-value:** {p:.5f}")

    if p < alpha:
        st.error(f"❌ {decision}")
    else:
        st.success(f"✅ {decision}")

    # Gráfica
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(x, y)

    if tipo == "bilateral":
        ax3.fill_between(x, y, where=(x <= -z_crit) | (x >= z_crit), alpha=0.3)
    elif tipo == "derecha":
        ax3.fill_between(x, y, where=(x >= z_crit), alpha=0.3)
    else:
        ax3.fill_between(x, y, where=(x <= z_crit), alpha=0.3)

    ax3.axvline(z, linestyle='--', label=f'Z ({z:.2f})')
    ax3.legend()

    st.pyplot(fig3)

    st.write("### 🧠 Interpretación automática")
    if p < alpha:
        st.write("Se rechaza H0. Existe evidencia suficiente.")
    else:
        st.write("No se rechaza H0. No hay evidencia suficiente.")

    # ================================
    # IA GEMINI
    # ================================
    st.divider()
    st.subheader("🤖 Asistente IA")

    if st.button("🧠 Analizar con IA"):

        with st.spinner("Analizando..."):

            prompt = f"""
Se realizó una prueba Z con los siguientes datos:

Media muestral: {media_muestral}
Media hipotética: {media_hipotetica}
n: {n}
sigma: {sigma}
alpha: {alpha}
tipo: {tipo}
Z: {z}
p-value: {p}

¿Se rechaza H0? Explica en términos simples y da una conclusión práctica.
"""

            try:
                response = model.generate_content(prompt)
                st.session_state.respuesta_ia = response.text

            except Exception as e:
                st.error(f"Error con la IA: {e}")

    # Mostrar IA
    if "respuesta_ia" in st.session_state:
        st.subheader("🤖 Respuesta de la IA")
        st.write(st.session_state.respuesta_ia)

        # ================================
# ✅ COMPARACIÓN CON IA
# ================================
st.subheader("📊 Comparación con la IA")

if "decision" in st.session_state and "respuesta_ia" in st.session_state:

    decision = st.session_state.decision.lower()
    respuesta = st.session_state.respuesta_ia.lower()

    st.write(f"📌 Tu decisión: {st.session_state.decision}")

    if "no se rechaza" in respuesta and "no rechazar" in decision:
        st.success("🤖 La IA coincide con tu resultado")

    elif "rechaza" in respuesta and "rechazar" in decision:
        st.success("🤖 La IA coincide con tu resultado")

    else:
        st.warning("⚠ La IA podría diferir, revisa interpretación")