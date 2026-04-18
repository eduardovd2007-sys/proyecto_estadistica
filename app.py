import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import google.generativeai as genai

# ================================
# ⚙️ CONFIGURACIÓN DE PÁGINA Y API
# ================================
st.set_page_config(
    page_title="DataSight AI | Análisis Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ocultar el menú de Streamlit por defecto para un look más limpio
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

MODELO_IA = 'gemini-2.5-flash-lite'

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(MODELO_IA)
else:
    st.sidebar.error("⚠️ Falta API KEY en .streamlit/secrets.toml")

# ================================
# 🗂️ SIDEBAR: CONTROLES PRINCIPALES
# ================================
with st.sidebar:
    st.title("⚙️ Panel de Control")
    st.markdown("---")
    
    st.subheader("1. Carga de Datos")
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"], help="Formatos aceptados: .csv")
    
    # Cargar datos o usar ejemplo
    if archivo is not None:
        df = pd.read_csv(archivo)
        st.success("¡Datos cargados con éxito!")
    else:
        st.info("Demo activada: Usando datos sintéticos normales.")
        data = np.random.normal(50, 10, 1000)
        df = pd.DataFrame(data, columns=["valores_demo"])
    
    # Limpieza rápida (Tu lógica original optimizada)
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        st.error("❌ El archivo no contiene columnas numéricas.")
        st.stop()
        
    st.markdown("---")
    st.subheader("2. Selección de Variable")
    col_seleccionada = st.selectbox("Variable a analizar:", df_numeric.columns)

# ================================
# 🚀 PANTALLA PRINCIPAL
# ================================
st.title("📊 DataSight AI Dashboard")
st.markdown("Plataforma interactiva para análisis descriptivo y pruebas de hipótesis guiadas por Inteligencia Artificial.")
st.markdown("---")

# --- SECCIÓN 1: KPI'S Y ESTADÍSTICA DESCRIPTIVA ---
st.header("1. Exploración de la Distribución")

# Cálculos rápidos
media_calc = df_numeric[col_seleccionada].mean()
mediana_calc = df_numeric[col_seleccionada].median()
std_calc = df_numeric[col_seleccionada].std()
n_calc = len(df_numeric[col_seleccionada].dropna())

# Mostrar KPIs en tarjetas
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Media (x̄)", value=f"{media_calc:.2f}")
kpi2.metric(label="Mediana", value=f"{mediana_calc:.2f}")
kpi3.metric(label="Desviación Est. (σ)", value=f"{std_calc:.2f}")
kpi4.metric(label="Muestra (n)", value=f"{n_calc}")

# Detección rápida de normalidad visual
sesgo_msg = "Aproximadamente Normal" if abs(media_calc - mediana_calc) < (std_calc * 0.1) else "Posible Sesgo"
st.caption(f"💡 *Inspección rápida:* La relación entre media y mediana sugiere una distribución **{sesgo_msg}**.")

# Gráficas Interactivas con Plotly
g1, g2 = st.columns(2)

with g1:
    fig_hist = px.histogram(
        df_numeric, x=col_seleccionada, 
        marginal="box", # Añade un pequeño boxplot arriba
        title=f"Distribución de {col_seleccionada}",
        color_discrete_sequence=['#4A90E2'],
        opacity=0.8
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

with g2:
    fig_box = px.box(
        df_numeric, y=col_seleccionada, 
        title=f"Rango Intercuartílico y Outliers",
        color_discrete_sequence=['#50E3C2']
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ================================
# 📈 SECCIÓN 2: PRUEBA DE HIPÓTESIS
# ================================
st.markdown("---")
st.header("2. Laboratorio de Pruebas Z")

# Contenedor con estilo para el formulario
with st.container(border=True):
    st.markdown("#### Configuración de la Prueba")
    p1, p2, p3, p4 = st.columns(4)
    
    mu_muestral = p1.number_input("Media Muestral", value=float(media_calc), format="%.2f")
    mu_h0 = p2.number_input("Media Hipotética (μ₀)", value=50.0, format="%.2f")
    sigma = p3.number_input("Desviación Poblacional", value=float(std_calc), format="%.2f")
    alpha = p4.selectbox("Significancia (α)", [0.01, 0.05, 0.10], index=1)
    
    p5, p6, p7 = st.columns([1,1,2])
    n_input = p5.number_input("N", min_value=1, value=int(n_calc))
    tipo_test = p6.selectbox("Tipo de Cola", ["bilateral", "derecha", "izquierda"])
    
    # Botón principal de cálculo
    if p7.button("⚡ Ejecutar Prueba Estadística", type="primary", use_container_width=True):
        z_stat = (mu_muestral - mu_h0) / (sigma / np.sqrt(n_input))
        
        if tipo_test == "bilateral":
            p_val = 2 * (1 - norm.cdf(abs(z_stat)))
            z_crit = norm.ppf(1 - alpha/2)
        elif tipo_test == "derecha":
            p_val = 1 - norm.cdf(z_stat)
            z_crit = norm.ppf(1 - alpha)
        else:
            p_val = norm.cdf(z_stat)
            z_crit = norm.ppf(alpha)

        # Guardar en memoria
        st.session_state.stats = {
            "z": z_stat, "p": p_val, "z_crit": abs(z_crit), 
            "decision": "Rechazar Hipótesis Nula (H0)" if p_val < alpha else "No rechazar Hipótesis Nula (H0)",
            "rechazo_booleano": p_val < alpha
        }
        if "respuesta_ia" in st.session_state: del st.session_state.respuesta_ia

# Mostrar Resultados de la Prueba
if "stats" in st.session_state:
    res = st.session_state.stats
    
    res1, res2 = st.columns([1, 2])
    
    with res1:
        st.markdown("#### Resultados")
        st.metric("Estadístico Z", f"{res['z']:.4f}")
        st.metric("Valor P (p-value)", f"{res['p']:.5f}")
        
        if res['rechazo_booleano']:
            st.error(f"⚠️ **Conclusión:** {res['decision']}")
        else:
            st.success(f"✅ **Conclusión:** {res['decision']}")
            
    with res2:
        st.markdown("#### Región de Rechazo")
        x = np.linspace(-4, 4, 500)
        y = norm.pdf(x)
        
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Distribución Normal', line=dict(color='white')))
        
        # Sombreado interactivo según Plotly
        # (Lógica simplificada para visualización rápida)
        fig_z.add_vline(x=res['z'], line_dash="dash", line_color="cyan", annotation_text=f"Z={res['z']:.2f}", annotation_position="top right")
        
        fig_z.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20), height=250
        )
        st.plotly_chart(fig_z, use_container_width=True)

# ================================
# 🤖 SECCIÓN 3: EXPERTO IA (GEMINI)
# ================================
if "stats" in st.session_state:
    st.markdown("---")
    st.header("3. Consultoría Algorítmica AI")
    
    if st.button("🧠 Generar Reporte de Interpretabilidad AI", icon="✨"):
        with st.status("Iniciando análisis profundo...", expanded=True) as status:
            st.write("🔍 Contextualizando parámetros estadísticos...")
            
            res = st.session_state.stats
            prompt = f"""
            Actúa como un Consultor Senior en Ciencia de Datos. 
            Analiza estos resultados de una prueba Z:
            - Variable: {col_seleccionada}
            - x̄: {mu_muestral}, μ₀: {mu_h0}, σ: {sigma}, n: {n_input}
            - α: {alpha}, Tipo: {tipo_test}
            - Z calculado: {res['z']:.4f}, p-value: {res['p']:.5f}
            - Decisión preliminar: {res['decision']}
            
            Estructura tu respuesta en formato Markdown profesional con estos encabezados:
            ### 🎯 Significado del P-value
            (Explica el p-value en este contexto específico)
            ### 🗣️ Traducción para Negocios
            (Explica qué significa esto si la variable fuera una métrica de negocio, en lenguaje muy sencillo)
            ### ⚖️ Efecto Práctico vs Estadístico
            (Diferencia si este resultado importa en la realidad)
            ### 🔬 Evaluación de Supuestos
            (Menciona si es seguro usar Z con n={n_input})
            """
            st.write("🌐 Consultando a Gemini 2.0 Flash...")
            try:
                response = model.generate_content(prompt)
                st.session_state.respuesta_ia = response.text
                status.update(label="Análisis completado", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"Error en la API: {e}", state="error")
                st.error("No se pudo conectar con la IA. Revisa tus cuotas o conexión.")

    # Mostrar la respuesta bonita
    if "respuesta_ia" in st.session_state:
        st.info("Este reporte fue generado por Inteligencia Artificial y debe ser validado por un experto humano.", icon="ℹ️")
        st.markdown(st.session_state.respuesta_ia)