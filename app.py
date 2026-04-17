import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import google.generativeai as genai

# ========================
# CONFIGURACIÓN
# ========================
st.set_page_config(page_title="App Estadística", layout="centered")

# ========================
# API GEMINI
# ========================
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("⚠️ Falta la API KEY de Gemini")

# ========================
# TÍTULO
# ========================
st.title("📊 App Estadística con IA")
st.write("Aplicación para análisis de datos, prueba Z e interpretación con IA")

st.divider()

# ========================
# CARGA DE DATOS
# ========================
st.header("📂 Carga de datos")

archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
else:
    data = np.random.normal(50, 10, 100)
    df = pd.DataFrame(data, columns=["valores"])

st.dataframe(df)

st.divider()

# ========================
# VISUALIZACIÓN
# ========================
st.header("📊 Visualización")

col = st.selectbox("Variable", df.columns)

fig, ax = plt.subplots()
sns.histplot(df[col], kde=True, ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.boxplot(x=df[col], ax=ax2)
st.pyplot(fig2)

st.divider()

# ========================
# ANÁLISIS DISTRIBUCIÓN
# ========================
st.header("🧠 Análisis de la distribución")

media = df[col].mean()
mediana = df[col].median()

st.write(f"Media: {media:.2f}")
st.write(f"Mediana: {mediana:.2f}")

if abs(media - mediana) < 1:
    sesgo = "Distribución aproximadamente normal"
elif media > mediana:
    sesgo = "Sesgo positivo"
else:
    sesgo = "Sesgo negativo"

# RESPUESTAS
st.write("¿La distribución parece normal?")
st.write("Sí" if abs(media - mediana) < 1 else "No completamente")

st.write("¿Hay sesgo?")
st.write(sesgo)

st.write("¿Hay outliers?")
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1

outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
st.write("Sí hay" if len(outliers) > 0 else "No hay")

st.divider()

# ========================
# PRUEBA Z
# ========================
st.header("📈 Prueba Z")

media_muestral = st.number_input("Media muestral", value=float(media))
media_hipotetica = st.number_input("Media H0", value=50.0)
sigma = st.number_input("Sigma", value=float(df[col].std()))
n = st.number_input("n", value=len(df))
alpha = st.slider("Alpha", 0.01, 0.10, 0.05)

tipo = st.selectbox("Tipo", ["bilateral", "izquierda", "derecha"])

# HIPÓTESIS
st.subheader("Hipótesis")

if tipo == "bilateral":
    st.write("H0: μ = valor")
    st.write("H1: μ ≠ valor")
elif tipo == "derecha":
    st.write("H0: μ ≤ valor")
    st.write("H1: μ > valor")
else:
    st.write("H0: μ ≥ valor")
    st.write("H1: μ < valor")

# SUPUESTOS
st.subheader("Supuestos")

st.write("• n ≥ 30")
st.write("• Varianza conocida")
st.write("• Datos aproximadamente normales")

if st.button("Ejecutar"):

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

    decision = "Rechazar H0" if p < alpha else "No rechazar H0"

    st.session_state.z = z
    st.session_state.p = p
    st.session_state.decision = decision
    st.session_state.z_crit = z_crit

# ========================
# RESULTADOS
# ========================
if "z" in st.session_state:

    z = st.session_state.z
    p = st.session_state.p
    decision = st.session_state.decision
    z_crit = st.session_state.z_crit

    st.subheader("Resultados")

    st.write(f"Z: {z:.4f}")
    st.write(f"p-value: {p:.5f}")
    st.write(f"Decisión: {decision}")

    # INTERPRETACIÓN
    st.subheader("Interpretación")

    if p < alpha:
        st.write("Se rechaza H0. Hay evidencia suficiente.")
    else:
        st.write("No se rechaza H0. No hay evidencia suficiente.")

    # GRÁFICA
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)

    fig3, ax3 = plt.subplots()
    ax3.plot(x, y)

    if tipo == "bilateral":
        ax3.fill_between(x, y, where=(x <= -z_crit) | (x >= z_crit), alpha=0.3)
    elif tipo == "derecha":
        ax3.fill_between(x, y, where=(x >= z_crit), alpha=0.3)
    else:
        ax3.fill_between(x, y, where=(x <= z_crit), alpha=0.3)

    ax3.axvline(z, color='red')
    st.pyplot(fig3)

    st.divider()

    # ========================
    # IA
    # ========================
    st.header("🤖 Asistente IA")

    if st.button("Analizar con IA"):
        with st.spinner("Pensando..."):
            try:
                prompt = f"""
Se realizó una prueba Z:

media={media_muestral}
H0={media_hipotetica}
n={n}
sigma={sigma}
alpha={alpha}
Z={z}
p={p}
tipo={tipo}

Explica decisión, interpretación y conclusión práctica.
"""
                response = model.generate_content(prompt)
                st.session_state.respuesta_ia = response.text

            except Exception as e:
                st.error(e)

    if "respuesta_ia" in st.session_state:
        st.write(st.session_state.respuesta_ia)

        # COMPARACIÓN
        st.subheader("Comparación con IA")

        st.write(f"Decisión estadística: {decision}")

        if "rechaza" in st.session_state.respuesta_ia.lower():
            st.write("La IA sugiere rechazar H0")
        else:
            st.write("La IA sugiere no rechazar H0")

else:
    st.warning("Ejecuta la prueba primero")