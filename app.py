import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Dashboard Balanza Comercial Perú",
    page_icon="🇵🇪",
    layout="wide"
)

# 2. Carga optimizada de datos (Formato Parquet)
@st.cache_data
def cargar_datos():
    url_drive = "https://drive.google.com/uc?export=download&id=1ITvcXTg8o5wFT4yeXiXkc0PZawZqnGcl"
    df = pd.read_parquet(url_drive)
    
    # Reducción drástica de tipos de datos para liberar RAM de inmediato
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'object':
            df[col] = df[col].astype('category')
            
    return df

try:
    df_final = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar los datos desde Google Drive: {e}")
    st.stop()

# Mapeo de columnas fijas reales
col_anio = 'refYear'
col_flujo = 'tradeFlow'
col_valor = 'primaryValue'
col_pais = 'partnerDesc'

# =========================================================================
# BARRA LATERAL (SIDEBAR): FILTROS DINÁMICOS
# =========================================================================
st.sidebar.header("🎛️ Filtros del Tablero")

df_base_flujo = df_final.copy()
df_filtrado = df_final.copy()
año_seleccionado = "Todos"

if col_flujo in df_final.columns:
    flujos_disponibles = df_final[col_flujo].dropna().unique()
    flujo_seleccionado = st.sidebar.multiselect("Flujo Comercial", list(flujos_disponibles), default=list(flujos_disponibles))
    df_base_flujo = df_final[df_final[col_flujo].isin(flujo_seleccionado)]

if col_anio in df_base_flujo.columns:
    años_disponibles = sorted(df_base_flujo[col_anio].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("Selecciona el Año de Análisis", años_disponibles, index=len(años_disponibles)-1)
    df_filtrado = df_base_flujo[df_base_flujo[col_anio] == año_seleccionado]

# =========================================================================
# CUERPO PRINCIPAL DEL DASHBOARD
# =========================================================================
st.title("🇵🇪 Diagnóstico, Predicción y Segmentación Comercial del Perú")
st.markdown("---")

# KPIs principales
col1, col2 = st.columns(2)
with col1:
    if col_valor in df_filtrado.columns:
        total_valor = float(df_filtrado[col_valor].sum())
        st.metric(label=f"Monto Total Transaccionado ({año_seleccionado}) (USD)", value=f"${total_valor:,.2f}")
with col2:
    if col_pais in df_filtrado.columns:
        socios_activos = df_filtrado[col_pais].nunique()
        st.metric(label=f"Socios Comerciales Activos ({año_seleccionado})", value=socios_activos)

st.markdown("---")

# Organización por Pestañas
tab1, tab2, tab3 = st.tabs(["📊 Tendencias Históricas", "🤖 Clustering (ML)", "🗺️ Concentración Geográfica"])

with tab1:
    st.subheader("Evolución de los Flujos Comerciales")
    if col_anio in df_base_flujo.columns and col_valor in df_base_flujo.columns:
        # AGRUPACIÓN ULTRA-LIGERA: Plotly solo procesará un par de filas por año, no miles.
        df_temp = df_base_flujo.groupby([col_anio, col_flujo])[col_valor].sum().reset_index()
        df_temp = df_temp.sort_values(by=col_anio)
        
        fig_temporal = px.line(
            df_temp, x=col_anio, y=col_valor, color=col_flujo,
            title="Evolución Histórica Comercial del Perú",
            labels={col_valor: 'Valor Comercial (USD)', col_anio: 'Año'}
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

with tab2:
    st.subheader("Segmentación Estructural (Autoencoder + K-Means)")
    x_col, y_col, cluster_col = 'pc1', 'pc2', 'cluster'

    if x_col in df_filtrado.columns and y_col in df_filtrado.columns and cluster_col in df_filtrado.columns:
        # RESTRICCIÓN DE PUNTOS: Limitamos a máximo 5,000 puntos para que la RAM no explote al renderizar el scatter
        df_scatter = df_filtrado if len(df_filtrado) < 5000 else df_filtrado.sample(5000, random_state=42)
        
        fig_clusters = px.scatter(
            df_scatter, x=x_col, y=y_col, color=df_scatter[cluster_col].astype(str),
            title=f"Estructura de Clústeres en el Espacio Latente ({año_seleccionado})",
            labels={cluster_col: 'Clúster'}
        )
        fig_clusters.update_traces(marker=dict(size=5)) # Puntos más pequeños para optimizar rendimiento visual
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.warning("Las variables 'pc1', 'pc2' o 'cluster' no están disponibles.")

with tab3:
    st.subheader("Distribución Geográfica del Comercio Exterior")
    iso_col = 'partnerISO'
    if iso_col in df_filtrado.columns and col_valor in df_filtrado.columns:
        # AGRUPACIÓN GEOGRÁFICA MÁXIMA
        df_mapa_data = df_filtrado.groupby([iso_col, col_pais])[col_valor].sum().reset_index()
        
        fig_mapa = px.choropleth(
            df_mapa_data, locations=iso_col, color=col_valor,
            hover_name=col_pais, color_continuous_scale=px.colors.sequential.Plasma,
            title=f"Concentración Global de Socios de Perú ({año_seleccionado})"
        )
        st.plotly_chart(fig_mapa, use_container_width=True)
