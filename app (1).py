import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Dashboard Balanza Comercial Perú",
    page_icon="🇵🇪",
    layout="wide"
)

# 2. Carga optimizada de datos (Punto obligatorio de la rúbrica - Formato Parquet)
@st.cache_data
def cargar_datos():
    url_drive = "https://drive.google.com/uc?export=download&id=1ITvcXTg8o5wFT4yeXiXkc0PZawZqnGcl"
    # Carga la base final exportada
    df = pd.read_parquet(url_drive)
    return df

try:
    df_final = cargar_datos()
except Exception as e:
    st.error(f"Por favor, asegúrate de que el archivo 'url_drive' esté en la misma carpeta: {e}")
    st.stop()

# =========================================================================
# BARRA LATERAL (SIDEBAR): FILTROS DINÁMICOS
# =========================================================================
st.sidebar.header("🎛️ Filtros del Tablero")

# --- FILTRO 1: Flujo Comercial (Se aplica a TODO el dashboard) ---
if 'flowDesc' in df_final.columns:
    flujos_disponibles = df_final['flowDesc'].unique()
    flujo_seleccionado = st.sidebar.multiselect("Flujo Comercial", flujos_disponibles, default=list(flujos_disponibles))
    df_base_flujo = df_final[df_final['flowDesc'].isin(flujo_seleccionado)]
else:
    df_base_flujo = df_final.copy()

# --- FILTRO 2: Año de Análisis (Solo afecta a KPIs, Clustering y Mapa) ---
if 'cmdYear' in df_base_flujo.columns:
    años_disponibles = sorted(df_base_flujo['cmdYear'].unique())
    año_seleccionado = st.sidebar.selectbox("Selecciona el Año de Análisis", años_disponibles, index=len(años_disponibles)-1)
    df_filtrado = df_base_flujo[df_base_flujo['cmdYear'] == año_seleccionado]
else:
    df_filtrado = df_base_flujo.copy()


# =========================================================================
# CUERPO PRINCIPAL DEL DASHBOARD
# =========================================================================
st.title("🇵🇪 Diagnóstico, Predicción y Segmentación Comercial del Perú")
st.markdown("---")

# KPIs en la parte superior (Usan el año seleccionado)
col1, col2 = st.columns(2)
with col1:
    if 'primaryValue' in df_filtrado.columns:
        total_valor = df_filtrado['primaryValue'].sum()
        st.metric(label=f"Monto Total Transaccionado ({año_seleccionado}) (USD)", value=f"${total_valor:,.2f}")
with col2:
    if 'partnerDesc' in df_filtrado.columns:
        socios_activos = df_filtrado['partnerDesc'].nunique()
        st.metric(label=f"Socios Comerciales Activos ({año_seleccionado})", value=socios_activos)

st.markdown("---")

# Organización por Pestañas (Tabs) para limpieza visual
tab1, tab2, tab3 = st.tabs(["📊 Tendencias Históricas", "🤖 Clustering (ML)", "🗺️ Concentración Geográfica"])

with tab1:
    st.subheader("Evolución de los Flujos Comerciales")
    # Para ver la tendencia, usamos 'df_base_flujo' (Muestra TODOS los años del histórico)
    if 'refPeriodId' in df_base_flujo.columns and 'primaryValue' in df_base_flujo.columns:
        # Agrupamos por periodo completo para limpiar la gráfica histórica
        df_temp = df_base_flujo.groupby(['refPeriodId', 'flowDesc'])['primaryValue'].sum().reset_index()
        df_temp = df_temp.sort_values(by='refPeriodId') # Asegura el orden cronológico
        
        fig_temporal = px.line(df_temp, x='refPeriodId', y='primaryValue', color='flowDesc',
                               title="Evolución Histórica Comercial del Perú",
                               labels={'primaryValue': 'Valor Comercial (USD)', 'refPeriodId': 'Periodo'})
        st.plotly_chart(fig_temporal, use_container_width=True)
    else:
        st.warning("No se encontraron las columnas temporales necesarias ('refPeriodId' y 'primaryValue') en la base de datos.")

with tab2:
    st.subheader("Segmentación Estructural (Autoencoder + K-Means)")
    # Reconstrucción del scatter plot de clústeres usando el espacio latente (Usa el año filtrado)
    x_col = 'pc1' if 'pc1' in df_filtrado.columns else (df_filtrado.select_dtypes(include=['number']).columns[0] if len(df_filtrado.select_dtypes(include=['number']).columns) > 0 else None)
    y_col = 'pc2' if 'pc2' in df_filtrado.columns else (df_filtrado.select_dtypes(include=['number']).columns[1] if len(df_filtrado.select_dtypes(include=['number']).columns) > 1 else None)
    cluster_col = 'cluster' if 'cluster' in df_filtrado.columns else ('Cluster' if 'Cluster' in df_filtrado.columns else None)

    if x_col and y_col and cluster_col:
        fig_clusters = px.scatter(df_filtrado, x=x_col, y=y_col, color=df_filtrado[cluster_col].astype(str),
                                  title=f"Estructura de Clústeres en el Espacio Latente ({año_seleccionado})",
                                  labels={cluster_col: 'Clúster'})
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.warning("Para visualizar el mapa de clústeres, asegúrate de que las columnas del espacio latente y las etiquetas del clúster estén integradas en 'Base_Final_Grupo_1.parquet'")

with tab3:
    st.subheader("Distribución Geográfica del Comercio Exterior")
    # Reconstrucción del mapa mundial interactivo (Usa el año filtrado)
    iso_col = 'partnerISO' if 'partnerISO' in df_filtrado.columns else ('partnerCode' if 'partnerCode' in df_filtrado.columns else None)

    if iso_col and 'primaryValue' in df_filtrado.columns:
        df_mapa_data = df_filtrado.groupby([iso_col, 'partnerDesc'])['primaryValue'].sum().reset_index()
        fig_mapa = px.choropleth(df_mapa_data, locations=iso_col, color="primaryValue",
                                 hover_name="partnerDesc", color_continuous_scale=px.colors.sequential.Plasma,
                                 title=f"Concentración Global de Socios de Perú ({año_seleccionado})")
        st.plotly_chart(fig_mapa, use_container_width=True)
    else:
        st.warning("No se encontraron variables geográficas válidas en la base final para pintar el mapa.")
