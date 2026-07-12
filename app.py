import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Dashboard Balanza Comercial Perú",
    page_icon="🇵🇪",
    layout="wide"
)

# 2. Carga optimizada de datos con tipos de datos más ligeros
@st.cache_data
def cargar_datos():
    url_drive = "https://drive.google.com/uc?export=download&id=1ITvcXTg8o5wFT4yeXiXkc0PZawZqnGcl"
    df = pd.read_parquet(url_drive)
    
    # Reducimos drásticamente el uso de memoria optimizando los tipos de datos
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

# =========================================================================
# IDENTIFICACIÓN DE COLUMNAS
# =========================================================================
col_anio = 'cmdYear' if 'cmdYear' in df_final.columns else ('Year' if 'Year' in df_final.columns else ('refPeriodId' if 'refPeriodId' in df_final.columns else None))
col_flujo = 'flowDesc' if 'flowDesc' in df_final.columns else ('flowCode' if 'flowCode' in df_final.columns else None)
col_valor = 'primaryValue' if 'primaryValue' in df_final.columns else ('TradeValue' if 'TradeValue' in df_final.columns else None)
col_pais = 'partnerDesc' if 'partnerDesc' in df_final.columns else ('partnerISO' if 'partnerISO' in df_final.columns else None)

# =========================================================================
# BARRA LATERAL (SIDEBAR): FILTROS
# =========================================================================
st.sidebar.header("🎛️ Filtros del Tablero")

df_base_flujo = df_final.copy()
df_filtrado = df_final.copy()
año_seleccionado = "Todos"

# Filtro 1: Flujo Comercial
if col_flujo and col_flujo in df_final.columns:
    flujos_disponibles = df_final[col_flujo].dropna().unique()
    flujo_seleccionado = st.sidebar.multiselect("Flujo Comercial", list(flujos_disponibles), default=list(flujos_disponibles))
    df_base_flujo = df_final[df_final[col_flujo].isin(flujo_seleccionado)]

# Filtro 2: Año de Análisis
if col_anio and col_anio in df_base_flujo.columns:
    años_disponibles = sorted(df_base_flujo[col_anio].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("Selecciona el Año de Análisis", años_disponibles, index=len(años_disponibles)-1)
    df_filtrado = df_base_flujo[df_base_flujo[col_anio] == año_seleccionado]

# =========================================================================
# CUERPO PRINCIPAL DEL DASHBOARD
# =========================================================================
st.title("🇵🇪 Diagnóstico, Predicción y Segmentación Comercial del Perú")
st.markdown("---")

# KPIs
col1, col2 = st.columns(2)
with col1:
    if col_valor and col_valor in df_filtrado.columns:
        total_valor = float(df_filtrado[col_valor].sum())
        st.metric(label=f"Monto Total Transaccionado ({año_seleccionado}) (USD)", value=f"${total_valor:,.2f}")
with col2:
    if col_pais and col_pais in df_filtrado.columns:
        socios_activos = df_filtrado[col_pais].nunique()
        st.metric(label=f"Socios Comerciales Activos ({año_seleccionado})", value=socios_activos)

st.markdown("---")

# Pestañas
tab1, tab2, tab3 = st.tabs(["📊 Tendencias Históricas", "🤖 Clustering (ML)", "🗺️ Concentración Geográfica"])

with tab1:
    st.subheader("Evolución de los Flujos Comerciales")
    if col_anio and col_valor:
        columnas_agrupacion = [col_anio, col_flujo] if col_flujo else [col_anio]
        df_temp = df_base_flujo.groupby(columnas_agrupacion)[col_valor].sum().reset_index()
        df_temp = df_temp.sort_values(by=col_anio)
        
        fig_temporal = px.line(
            df_temp, x=col_anio, y=col_valor, color=col_flujo if col_flujo else None,
            title="Evolución Histórica Comercial del Perú",
            labels={col_valor: 'Valor Comercial (USD)', col_anio: 'Año'}
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

with tab2:
    st.subheader("Segmentación Estructural (Autoencoder + K-Means)")
    x_col = 'pc1' if 'pc1' in df_filtrado.columns else (df_filtrado.select_dtypes(include=['number']).columns[0] if len(df_filtrado.select_dtypes(include=['number']).columns) > 0 else None)
    y_col = 'pc2' if 'pc2' in df_filtrado.columns else (df_filtrado.select_dtypes(include=['number']).columns[1] if len(df_filtrado.select_dtypes(include=['number']).columns) > 1 else None)
    cluster_col = 'cluster' if 'cluster' in df_filtrado.columns else ('Cluster' if 'Cluster' in df_filtrado.columns else None)

    if x_col and y_col and cluster_col:
        # Muestreamos datos si es muy pesado para el scatter plot
        df_scatter = df_filtrado if len(df_filtrado) < 50000 else df_filtrado.sample(50000, random_state=42)
        fig_clusters = px.scatter(df_scatter, x=x_col, y=y_col, color=df_scatter[cluster_col].astype(str),
                                  title=f"Estructura de Clústeres en el Espacio Latente ({año_seleccionado})",
                                  labels={cluster_col: 'Clúster'})
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.warning("Asegúrate de que las columnas de clústeres estén integradas en la base.")

with tab3:
    st.subheader("Distribución Geográfica del Comercio Exterior")
    iso_col = 'partnerISO' if 'partnerISO' in df_filtrado.columns else ('partnerCode' if 'partnerCode' in df_filtrado.columns else None)
    if iso_col and col_valor:
        hover_p = col_pais if col_pais else iso_col
        df_mapa_data = df_filtrado.groupby([iso_col, hover_p])[col_valor].sum().reset_index()
        fig_mapa = px.choropleth(df_mapa_data, locations=iso_col, color=col_valor,
                                 hover_name=hover_p, color_continuous_scale=px.colors.sequential.Plasma,
                                 title=f"Concentración Global de Socios de Perú ({año_seleccionado})")
        st.plotly_chart(fig_mapa, use_container_width=True)
