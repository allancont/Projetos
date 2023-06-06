
# Imports
import pandas            as pd
import streamlit         as st
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope
from io                     import BytesIO
from pycaret.classification import load_model, predict_model


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Função para realizar o pré-processamento

def preprocessamento(df):
    # Substituição de valores nulos pela média
    imputer = SimpleImputer(strategy='mean')
    df[['tempo_emprego']] = imputer.fit_transform(df[['tempo_emprego']])

    # Remoção de outliers utilizando Elliptic Envelope
    outliers_detector = EllipticEnvelope(contamination=0.01)
    outliers_detector.fit(df[['idade', 'tempo_emprego']])
    df = df[outliers_detector.predict(df[['idade', 'tempo_emprego']]) == 1]

    # Transformação em dummies
    df = pd.get_dummies(df)

    return df

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'PyCaret', \
        layout="wide",
        initial_sidebar_state='expanded'
        
    )
    
    # st.markdown("<div style='position: fixed; bottom: 10px; right: 10px; color: #999999; font-style: italic;'>developed by Allan</div>", unsafe_allow_html=True)
    st.write("<div style='position: fixed; bottom: 10px; right: 10px; color: #999999; font-style: italic;font-size: 10px;'>developed by Allan</div>", unsafe_allow_html=True)

    # Título principal da aplicação
    st.write("""## Escorando o modelo gerado no pycaret """)
    st.markdown("---")
    
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Base de crédito", type = ['csv','ftr'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)
        df_preprocessed = preprocessamento(df_credit)

        model_saved = load_model('Final LithtGBM 20220606')
        predict = predict_model(model_saved, data=df_preprocessed)

        df_xlsx = to_excel(predict)
        st.download_button(label='📥 Download',
                            data=df_xlsx ,
                            file_name= 'predict.xlsx')

    
    

if __name__ == '__main__':
    main()
    











