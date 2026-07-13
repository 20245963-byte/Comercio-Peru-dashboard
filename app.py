import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Dashboard Balanza Comercial Perú",
    page_icon="🇵🇪",
    layout="wide"
)

# 2. Carga ULTRA-OPTIMIZADA
@st.cache_data
def cargar_datos_ultra_light():
    url_drive = "https://drive.google.com/uc?export=download&id=1ITvcXTg8o5wFT4yeXiXkc0PZawZqnGcl"
    df_raw = pd.read_parquet(url_drive)

    df_raw = df_raw.rename(columns={
        'refYear': 'Año',
        'tradeFlow': 'Flujo',
        'primaryValue': 'Valor',
        'partnerDesc': 'Pais',
        'partnerISO': 'ISO'
    })

    columnas_base = ['Año', 'Flujo', 'Pais', 'ISO', 'cluster', 'pc1', 'pc2', 'Valor']
    return df_raw[[c for c in columnas_base if c in df_raw.columns]].copy()

try:
    df_final = cargar_datos_ultra_light()
except Exception as e:
    st.error(f"Error al procesar los datos optimizados: {e}")
    st.stop()

# =========================================================================
# BARRA LATERAL (SIDEBAR): FILTROS INDEPENDIENTES
# =========================================================================
st.sidebar.header("🎛️ Filtros del Tablero")

df_base_flujo = df_final.copy()
df_filtrado = df_final.copy()
año_seleccionado = "Todos"

if 'Flujo' in df_final.columns:
    flujos_disponibles = df_final['Flujo'].dropna().unique()
    flujo_seleccionado = st.sidebar.multiselect("Flujo Comercial", list(flujos_disponibles), default=list(flujos_disponibles))
    df_base_flujo = df_final[df_final['Flujo'].isin(flujo_seleccionado)]

if 'Año' in df_base_flujo.columns:
    años_disponibles = sorted(df_base_flujo['Año'].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("Selecciona el Año de Análisis", años_disponibles, index=len(años_disponibles)-1)
    df_filtrado = df_base_flujo[df_base_flujo['Año'] == año_seleccionado]

# =========================================================================
# CUERPO PRINCIPAL DEL DASHBOARD
# =========================================================================
st.title("🇵🇪 Diagnóstico, Predicción y Segmentación Comercial del Perú")
st.markdown("---")

# KPIs 100% REALES DIRECTOS DE TU BASE LIMPIA
col1, col2 = st.columns(2)
with col1:
    if 'Valor' in df_filtrado.columns:
        total_valor = float(df_filtrado['Valor'].sum())
        st.metric(label=f"Monto Total Transaccionado ({año_seleccionado}) (USD)", value=f"${total_valor:,.2f}")
with col2:
    if 'Pais' in df_filtrado.columns:
        socios_activos = df_filtrado['Pais'].nunique()
        st.metric(label=f"Socios Comerciales Activos ({año_seleccionado})", value=socios_activos)

st.markdown("---")

# Organización por Pestañas (Tabs)
tab1, tab2, tab3 = st.tabs(["📊 Tendencias Históricas", "🤖 Clustering (ML)", "🗺️ Concentración Geográfica"])

with tab1:
    st.subheader("Evolución de los Flujos Comerciales")
    if 'Año' in df_base_flujo.columns and 'Valor' in df_base_flujo.columns:
        # Agrupación veloz para la línea de tiempo histórica
        df_temp = df_base_flujo.groupby(['Año', 'Flujo'])['Valor'].sum().reset_index()
        df_temp = df_temp.sort_values(by='Año')

        fig_temporal = px.line(
            df_temp, x='Año', y='Valor', color='Flujo',
            title="Evolución Histórica Comercial del Perú",
            labels={'Valor': 'Valor Comercial (USD)', 'Año': 'Año'}
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

with tab2:
    st.subheader("Segmentación Estructural (Autoencoder + K-Means)")
    if 'pc1' in df_filtrado.columns and 'pc2' in df_filtrado.columns and 'cluster' in df_filtrado.columns:
        
        # Muestreo interno de seguridad para evitar caídas de RAM en años pesados (ej. 2022)
        if len(df_filtrado) > 4000:
            df_scatter = df_filtrado.sample(4000, random_state=42)
        else:
            df_scatter = df_filtrado.copy()

        fig_clusters = px.scatter(
            df_scatter, x='pc1', y='pc2', color=df_scatter['cluster'].astype(str),
            title=f"Estructura de Clústeres en el Espacio Latente ({año_seleccionado})",
            labels={'cluster': 'Clúster'}
        )
        fig_clusters.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.warning("⚠️ Variables de clustering ('pc1', 'pc2', 'cluster') no encontradas en el archivo Parquet.")

with tab3:
    st.subheader("Distribución Geográfica del Comercio Exterior")
    if 'ISO' in df_filtrado.columns and 'Valor' in df_filtrado.columns:
        # Agrupación veloz por país para que el mapa renderice al instante
        df_mapa_data = df_filtrado.groupby(['ISO', 'Pais'])['Valor'].sum().reset_index()

        fig_mapa = px.choropleth(
            df_mapa_data, locations='ISO', color='Valor',
            hover_name='Pais', color_continuous_scale=px.colors.sequential.Plasma,
            title=f"Concentración Global de Socios de Perú ({año_seleccionado})"
        )
        st.plotly_chart(fig_mapa, use_container_width=True)
