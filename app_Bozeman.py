import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr


@st.cache_resource

def load_data():
    df = pd.read_csv('Bozeman_V1.csv')
    numeric_df=df.select_dtypes(['float','int'])
    numeric_cols=numeric_df.columns

    text_df=df.select_dtypes(['object'])
    text_cols=text_df.columns
    return df, numeric_cols, text_cols
st.set_page_config(
    page_title="Bozeman Dashboard",
    layout="wide",  # Esto hace que use el ancho completo
    initial_sidebar_state="expanded"
)

def load_data_logistic():
    # Cambia el nombre del archivo por el del dataset real para regresión logística
    df = pd.read_csv('Bozeman_V2.csv')
    numeric_df = df.select_dtypes(['float','int'])
    numeric_cols = numeric_df.columns.tolist()

    text_df = df.select_dtypes(['object'])
    text_cols = text_df.columns.tolist()
    return df, numeric_cols, text_cols
#en la view descripcion se muestra una descripcion general de la variable seleccionadad
#que significa, que tipos de valores tiene, etc 
#en la view regresion lineal simple se hace una regresion lineal simple entre dos variables, una dependiente y una independiente, y se muestra con un heatmap o una grafica de dispersión
#en la view regresion lineal multiple se hace una regresion lineal multiple entre varias variables, una dependiente y varias independientes y se muestra con un heatmap o una grafica de dispersión
#en la view regresion logistica se hace una regresion logistica entre dos variables, una dependiente y una independiente y se muestra con un heatmap o una grafica de dispersión
def load_data_variable():
    # Cambia el nombre del archivo por el del dataset real para regresión logística
    df = pd.read_csv('diccionario.csv')
    numeric_df = df.select_dtypes(['float','int'])
    numeric_cols = numeric_df.columns.tolist()

    text_df = df.select_dtypes(['object'])
    text_cols = text_df.columns.tolist()
    return df, numeric_cols, text_cols

df, numeric_cols, text_cols = load_data()

st.sidebar.title("DASHBOARD")
st.sidebar.header("Panel de selección")


view = st.sidebar.selectbox(
    label="Opciones", 
    options=['bienvenida', 'Analisis Univariado', 'regresion lineal simple', 'regresion lineal multiple', 'regresion logistica'], 
    index=0
)

if view == 'bienvenida':
    st.markdown("<h1 style='text-align: center; color: #000000;'>Dashboard para Analisis de Datos</h1>", unsafe_allow_html=True)
    st.write(
        "Bienvenido al dashboard de análisis de datos de Bozeman, Montana. "
        "Aquí podrás explorar diferentes aspectos relacionados con el turismo en esta hermosa ciudad. "
        "Los datos han sido obtenidos desde Airbnb, por lo que nos enfocaremos en analizar el impacto de esta plataforma en la ciudad y sus diversas variables. "
        "Entre lo que podras encontrar en este dashboard se encuentran: "
    )

    cols = st.columns(3)

    with cols[0]:
        st.info("📊 **Gráficas interactivas**")
    with cols[1]:
        st.info("📋 **Tablas de datos**")
    with cols[2]:
        st.info("🌡️ **Heatmap**")
    cols = st.columns(3)

    with cols[0]:
        st.info("🖼️ **Imágenes de Bozeman**")
    with cols[1]:
        st.info("🎞️ **Videos de Bozeman**")

    with cols[2]:
        st.info("🧠 **Análisis avanzado**")
    st.write("Pero antes, conozcamos un poco más sobre Bozeman, Montana.")


    def Home():
        st.markdown("<h1 style='text-align: center; color: #4ea4c9;background-color:#D6EBFF'>Bozeman</h1>", unsafe_allow_html=True)

        st.write(
            "Bozeman es una ciudad ubicada en el suroeste del estado de Montana, en los Estados Unidos. "
            "Rodeada por majestuosas montañas y paisajes naturales, es conocida por ser una puerta de entrada al Parque Nacional Yellowstone "
            "y por ofrecer una combinación única de naturaleza, ciencia, educación y cultura."
        )
        st.write(
            "La ciudad alberga la Universidad Estatal de Montana, lo que le da un ambiente vibrante y juvenil. "
            "Además, Bozeman es famosa por sus actividades al aire libre, como el senderismo, el esquí y la pesca, por lo que la mayoría de sus turistas "
            "van a la ciudad para vivir estas experiencias."
        )

        from PIL import Image
        img1 = Image.open("imagenes/bozeman.jpg").resize((250, 150))
        img2 = Image.open("imagenes/bozeman6.jpg").resize((250, 150))
        img3 = Image.open("imagenes/bozeman7.jpg").resize((250, 150))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img1)
        with col2:
            st.image(img2)
        with col3:
            st.image(img3)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4, gap="large")
        with c1:
            st.info("**Habitantes**")
            st.metric(label="En 2024", value="56,123")
        with c2:
            st.info("**Turistas en 2024**")
            st.metric(label="Visitantes", value="300,000")
        with c3:
            st.info("**Turistas en 2025**")
            st.metric(label="Visitantes", value="150,000")
        with c4:
            st.info("**Año fundación**")
            st.metric(label="", value="1864")

        column1, column2 = st.columns(2, gap="large")
        with column1:
            st.info("**Plataformas de alojamiento**")
            data_plataformas = {
                "Plataforma": ["Airbnb", "Booking.com", "Vrbo", "Expedia", "Otros"],
                "Turistas": [72000, 40000, 15000, 10000, 5000]
            }
            df_plataformas = pd.DataFrame(data_plataformas)
            fig = px.pie(
                df_plataformas,
                names="Plataforma",
                values="Turistas",
                title="Participación de plataformas de alojamiento (2024)",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        with column2:
            st.info("**¿Por qué visitan Bozeman?**")
            razones_data = {
                "Razón": [
                    "Naturaleza y senderismo",
                    "Parque Yellowstone",
                    "Esquí y deportes de invierno",
                    "Universidad Estatal de Montana",
                    "Eventos culturales y arte",
                    "Gastronomía y cervecerías"
                ],
                "Porcentaje": [30, 25, 20, 10, 10, 5]
            }
            df_razones = pd.DataFrame(razones_data)
            fig_razones_bar = px.bar(
                df_razones,
                x="Porcentaje",
                y="Razón",
                orientation='h',
                title="Principales razones del turismo en Bozeman (2024)",
                color="Razón",
                text="Porcentaje"
            )
            fig_razones_bar.update_layout(showlegend=False)
            fig_razones_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_razones_bar, use_container_width=True)

        st.markdown("<h1 style='text-align: center; color: #4ea4c9;background-color:#D6EBFF'>Conociendo la ciudad</h1>", unsafe_allow_html=True)
        st.markdown(" ")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            imagenes = [
                {"titulo": "Centro de Bozeman", "ruta": "imagenes/bozeman4.jpg"},
                {"titulo": "Paisaje de Bozeman", "ruta": "imagenes/bozeman5.jpg"}
            ]

            if "imagen_index" not in st.session_state:
                st.session_state.imagen_index = 0

            def anterior_imagen():
                st.session_state.imagen_index = (st.session_state.imagen_index - 1) % len(imagenes)

            def siguiente_imagen():
                st.session_state.imagen_index = (st.session_state.imagen_index + 1) % len(imagenes)

            imagen_actual = imagenes[st.session_state.imagen_index]
            st.markdown(f"### {imagen_actual['titulo']}")
            st.image(imagen_actual["ruta"], width=500)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.button("Anterior ", on_click=anterior_imagen)
            with col3:
                st.button("Siguiente ", on_click=siguiente_imagen)

        with c2:
            videos = [
                {"titulo": "Bozeman, Montana: Things to Know Before Moving to Bozeman", "url": "https://www.youtube.com/watch?v=PamjTC9I0rM10I?si=K4HtoxkZqz3LaIgg"},
                {"titulo": "Top 10 Best Things to Do in Bozeman, Montana", "url": "https://www.youtube.com/watch?v=6q2nFlw2fVQ"},
                {"titulo": "13 Things to Do in Bozeman, Montana", "url": "https://www.youtube.com/watch?v=M5132JPBgjM&t=1s"}
            ]

            if "video_index" not in st.session_state:
                st.session_state.video_index = 0

            def anterior():
                st.session_state.video_index = (st.session_state.video_index - 1) % len(videos)

            def siguiente():
                st.session_state.video_index = (st.session_state.video_index + 1) % len(videos)

            video_actual = videos[st.session_state.video_index]
            st.markdown(f"### {video_actual['titulo']}")
            st.video(video_actual["url"])

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.button("Anterior", on_click=anterior)
            with col3:
                st.button("Siguiente", on_click=siguiente)

        st.markdown(
            """
            <h4>📍 Ubicación de Bozeman, MT</h4>
            <iframe 
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d4068.9812246980496!2d-111.05407027989197!3d45.67942965714756!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x534545a37bfc0f27%3A0x77fc0bde1d54c20e!2sBozeman%2C%20MT%2C%20USA!5e0!3m2!1ses!2smx!4v1714439621423!5m2!1ses!2smx"
                width="100%" 
                height="400" 
                style="border:0; border-radius: 12px;" 
                allowfullscreen="" 
                loading="lazy" 
                referrerpolicy="no-referrer-when-downgrade">
            </iframe>
            """,
            unsafe_allow_html=True
        )
    Home()
st.markdown("""---""")
if view == 'Analisis Univariado':
    # Cargar datos
    df, numeric_cols, text_cols = load_data()
    df_descripciones, _, _ = load_data_variable()  # evitar sobrescribir variables previas

    st.markdown("<h1 style='text-align: center; color: #000000;'>Análisis Univariado</h1>", unsafe_allow_html=True)
    st.write('En este apartado podrás conocer más a fondo los datos recopilados para el análisis ' \
    'asi como una breve descripción de cada variable.')

    # Métricas resumen
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")
    with col1:
        st.info("**Variables**")
        st.metric(label="Totales", value=str(len(df.columns)))
    with col2:
        st.info("**Variables numéricas**")
        st.metric(label="Totales", value=str(len(numeric_cols)))
    with col3:
        st.info("**Variables categóricas**")
        st.metric(label="Totales", value=str(len(text_cols)))
    with col4:
        st.info("**Año de obtención**")
        st.metric(label="Datos actualizados", value="2024")
    with col5:
        st.info("**Lugar de obtención**")
        st.metric(label="", value="Airbnb")
    # Tabla de columnas + descripciones
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧾 Variables existentes")
        st.write(df.columns)

    with c2:
        st.subheader("📘 Significado de cada variable")
        with st.expander("🔍 Ver todas las variables y sus descripciones"):
            st.dataframe(df_descripciones, use_container_width=True)

        selected_var = st.selectbox("Selecciona una variable para ver su significado:", df_descripciones["Variable"])
        descripcion = df_descripciones[df_descripciones["Variable"] == selected_var]["Descripción"].values[0]
        st.info(f"**{selected_var}**: {descripcion}")

    #COLUMNAS que existen
    def graph():
        numeric_cols = df.select_dtypes(['float', 'int']).columns.tolist()
        graf_types = ['Pastel', 'Barras', 'Dispersión', 'Histograma', 'Boxplot']


        st.markdown("<h1 style='text-align: center; color: #000000;'>Análisis gráfico de una variable numérica</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox('Selecciona una variable numérica', numeric_cols)
        with col2:
            selected_graf = st.selectbox('Selecciona un tipo de gráfico', graf_types)
        c1, c2 = st.columns(2, gap="large")

        with c1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("📊 Tabla de frecuencia")
                freq_table = df[selected_col].value_counts().reset_index()
                freq_table.columns = [selected_col, "Frecuencia"]
                st.dataframe(freq_table, use_container_width=True)
            with col2:
                st.write("📊 Tabla de estadísticos")
                stats_table = df[selected_col].describe().reset_index()
                stats_table.columns = ["Estadístico", selected_col]
                st.dataframe(stats_table, use_container_width=True)
        with c2:
        # Gráfico según tipo seleccionado
            if selected_graf == 'Pastel':
                st.write('🥧 Gráfico de pastel')
                fig = px.pie(df, names=selected_col)
                #st.plotly_chart(fig, use_container_width=True)
            elif selected_graf == 'Barras':
                st.write('📊 Gráfico de barras')
                fig = px.bar(df, x=selected_col)
                #st.plotly_chart(fig, use_container_width=True)
            elif selected_graf == 'Dispersión':
                st.write('📈 Gráfico de dispersión')
                fig = px.scatter(df, x=selected_col, y=selected_col)
                #st.plotly_chart(fig, use_container_width=True)
            elif selected_graf == 'Histograma':
                st.write('📉 Histograma')
                fig = px.histogram(df, x=selected_col)
                #st.plotly_chart(fig, use_container_width=True)
            elif selected_graf == 'Boxplot':
                st.write('📦 Boxplot')
                fig = px.box(df, y=selected_col, title="Boxplot de la variable")
                #st.plotly_chart(fig, use_container_width=True)
            else:
                fig = None
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    graph()


if view == 'regresion lineal simple':
    df, numeric_cols, text_cols = load_data()
    st.title('📈 Regresión lineal simple')

    col1, col2 = st.columns(2, gap="large")
    with col1:
        selected_col = st.selectbox('Selecciona una variable dependiente (Y)', numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    with col2:
        selected_col2 = st.selectbox('Selecciona una variable independiente (X)', numeric_cols, index=0)

    x = df[selected_col2].to_numpy()
    y = df[selected_col].to_numpy()

    # Cálculo de coeficientes de regresión
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b


    st.markdown("### 🧮 Fórmula de la pendiente:")
    st.latex(r"m = \frac{n \sum(xy) - \sum x \sum y}{n \sum(x^2) - (\sum x)^2}")

    st.markdown("### 📌 Coeficientes de la recta:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pendiente (m)", f"{m:.4f}")
    with col2:
        st.metric("Intercepto (b)", f"{b:.4f}")

    st.latex(f"y = {m:.4f}x + {b:.4f}")

    # Correlación de Pearson y R²
    r, _ = pearsonr(x, y)
    r2 = r ** 2

    st.markdown("### 🔗 Correlaciones")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coef. de correlación (r)", f"{r:.4f}")
    with col2:
        st.metric("Coef. de determinación (R²)", f"{r2:.4f}")

    # Gráfico con línea de regresión
    fig = px.scatter(df, x=selected_col2, y=selected_col, title='Regresión lineal simple')
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Línea de regresión', line=dict(color='red')))
    st.plotly_chart(fig, use_container_width=True)

# Asegurar que numeric_cols sea lista
    if isinstance(numeric_cols, pd.Index):
        numeric_cols = list(numeric_cols)

    if not numeric_cols:
        st.warning("⚠️ No hay columnas numéricas disponibles para calcular correlaciones.")
    else:
        # Calcular matriz de correlación
        corr_matrix = df[numeric_cols].corr()

        # Mostrar como tabla
        st.subheader("📋 Tabla de Correlaciones")
        st.dataframe(corr_matrix.style.background_gradient(cmap='Blues').format("{:.2f}"))


if view == 'regresion lineal multiple':
    df, numeric_cols, text_cols = load_data()
    st.title('Regresion lineal multiple')
    col1, col2 = st.columns(2, gap="large")
    with col1:
        selected_col = st.selectbox('Selecciona una variable dependiente', numeric_cols)
    with col2:
        selected_col2 = st.multiselect( 'Selecciona variables independientes',numeric_cols)
    fig = px.scatter_matrix(df, dimensions=[selected_col]+selected_col2)
    st.plotly_chart(fig)


if view == 'regresion logistica':
    df_log, numeric_cols_log, text_cols_log = load_data_logistic()
    st.title('Regresión Logística')
    col1, col2 = st.columns(2, gap="large")
    with col1:
        dependent = st.selectbox('Selecciona la variable dependiente (binaria)', df_log.columns.tolist())
    with col2:
        independent = st.selectbox('Selecciona la variable independiente (numérica)', numeric_cols_log)

    if df_log[dependent].nunique() != 2:
        st.error("La variable dependiente seleccionada no es binaria. Por favor, selecciona otra variable.")
    else:
        X = df_log[[independent]]
        y = df_log[dependent]

        model_log = LogisticRegression()
        model_log.fit(X, y)
        st.write("Modelo entrenado correctamente.")
        st.write("Coeficiente:", round(model_log.coef_[0][0], 2))
        st.write("Intersección:", round(model_log.intercept_[0], 2))
        fig_log = px.scatter(df_log, x=independent, y=dependent, title="Scatter de Regresión Logística")
        st.plotly_chart(fig_log)







