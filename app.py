import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_resource

def load_data():
    df = pd.read_csv('titanic.csv',index_col='Name')
    numeric_df=df.select_dtypes(['float','int'])
    numeric_cols=numeric_df.columns

    text_df=df.select_dtypes(['object'])
    text_cols=text_df.columns

    categorical_column_sex=df['Sex']
    unique_categories_sex=categorical_column_sex.unique()
    return df,numeric_cols,text_cols,unique_categories_sex,numeric_df

df,numeric_cols,text_cols,unique_categories_sex,numeric_df=load_data()


#primer widget
view=st.selectbox(label='view',options=['view 1','view 2','view 3','view 4'])

if view=='view 1':
    st.title('Titanic')
    st.header('Panel principal')
    st.subheader('Line plot')

#
    st.sidebar.title("DASHBOARD")
    st.sidebar.header("sidebar")
    st.sidebar.subheader("Panel de selección")

    #widget 2:Checkbox
    check_box=st.sidebar.checkbox(label="mostrar dataset")

    if check_box:
        #mostramos el dataset
        st.write(df)
        st.write(df.columns)
        st.write(df.describe())

    numerics_vars_selected=st.sidebar.multiselect(label="variables graficadas",options=numeric_cols)
    category_selected=st.sidebar.selectbox(label="categoría",options=unique_categories_sex)
    Button=st.sidebar.button(label="Mostrar variables string")

    if Button:
        #mostramos el dataset
        st.write(text_cols)
    
    #grafica 1
    data=df[df['Sex']==category_selected]
    data_features=data[numerics_vars_selected]
    figure1=px.line(data_frame=data_features,x=data_features.index,
                    y=numerics_vars_selected,title=str('Features of Passengers'),
                    width=1600,height=600)
    Button2=st.sidebar.button(label="Mostrar gráfica tipo lineplot")
    if Button2:#intentar sacar el ploty del if para que sea dinamico 
        #mostramos el dataset
        st.plotly_chart(figure1)
elif view=="view 2":
    st.title('Titanic')
    st.header('Panel principal')
    st.subheader('Scatter plot')

    #scatterplor
    x_selected=st.sidebar.selectbox(label="x",options=numeric_cols)
    y_selected=st.sidebar.selectbox(label="y",options=numeric_cols)
    figure2=px.scatter(data_frame=numeric_df,x=x_selected,y=y_selected,
                        title='Dispersiones')
    st.plotly_chart(figure2)
elif view=="view 3":
    st.title('Titanic')
    st.header('Panel principal')
    st.subheader('Pie Plot')

    Variable_cat=st.sidebar.selectbox(label="Variable categórica",options=text_cols)
    Variable_num=st.sidebar.selectbox(label="Variable numérica",options=numeric_cols)
    #pieplot
    figure3=px.pie(data_frame=df,names=df[Variable_cat],values=df[Variable_num],title=str('Features of')+' ' + 'Passengers',width=1600,height=600)
    st.plotly_chart(figure3)

elif view=="view 4":
    st.title('Titanic')
    st.header('Panel principal')
    st.subheader('Bar Plot')

    #barplot
    Variable_cat=st.sidebar.selectbox(label="Variable categórica",options=text_cols)
    Variable_num=st.sidebar.selectbox(label="Variable numérica",options=numeric_cols)
    figure4=px.bar(data_frame=df,x=df[Variable_cat],y=df[Variable_num],title=str('Features of')+' ' + 'Passengers')
    figure4.update_xaxes(automargin=True)
    figure4.update_yaxes(automargin=True)
    st.plotly_chart(figure4)
