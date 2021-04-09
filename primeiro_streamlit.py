import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
import yfinance as yf
import pandas_datareader as data
from pandas_datareader import data as wb
import matplotlib.dates as mdates
from scipy.stats import norm
import plotly.express as px
import datetime as dt
import mplfinance as mpf
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from bs4 import BeautifulSoup
import requests

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h2 style='margin-top: -40px; font-family: Helvetica; font-weight:bold; margin-left: 140px'>Análise descritiva</h2>", unsafe_allow_html=True)


#---------------sidebar---------------------------------------------------------------------------------------------------------------------------------------

contador = 0
st.sidebar.header("Dados de entrada")
acao = st.sidebar.selectbox('Selecione a ação',
                              ( 'PETR3.SA', 'PETR4.SA','CSAN3.SA','BRKM5.SA','UGPA3.SA','KLBN11.SA','CYRE3.SA',"EQTL3.SA",
                                'ELET3.SA','USIM3.SA',"BBAS3.SA",'SANB11.SA','BBSE3.SA','BBDC3.SA','BBDC4.SA','ITSA4.SA','CIEL3.SA',
                                'BPAC11.SA','IGTA3.SA','PRIO3.SA','BRML3.SA',
                                'ABEV3.SA','VIVT3.SA','OIBR4.SA','BTOW3.SA','MGLU3.SA','LAME4.SA','LREN3.SA',
                                'RENT3.SA','RADL3.SA','GGBR4.SA','CCRO3.SA','EMBR3.SA','SBSP3.SA',
                                'MRVE3.SA','JBSS3.SA','MRFG3.SA','QUAL3.SA','IBOVESPA','BRENT','DÓLAR',"EURO"))


if(acao == 'IBOVESPA'):
    acao = '^BVSP'
if(acao == 'BRENT'):
    acao = 'BZ=F'
if(acao == 'DÓLAR'):
    acao = 'BRL=X'
if(acao == 'EURO'):
    acao = 'EURBRL=X'
    
modelo = st.sidebar.selectbox('Selecione o modelo preditivo',
                              ('Monte Carlo','Redes Neurais'))

x = np.arange(3,31)
tempo = st.sidebar.select_slider('Selecione o tempo de predição em dias',options=list(x))
col1, col2, col3 = st.sidebar.beta_columns(3)


submit = col2.button('Calcular')

st.sidebar.header("Indicadores")


#--------------------------------------ANALISE ESTATISTICA BASICA-----------------------------------------------------------------------------------------------

df = data.DataReader(name = acao, data_source='yahoo', start='2018-06-06', end=dt.datetime.now())
st.header("Estatística descritiva da ação "+acao)

st.write(df.loc[:, df.columns != 'Volume'].describe())
st.markdown("<p style='margin-bottom: 40px'>", unsafe_allow_html=True)

plt.figure(figsize=(16,8))
plt.title('Histórico de fechamento da '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(df['Adj Close'],color='r')
st.pyplot()

plt.figure(figsize=(16,8))
plt.title('Histórico de volumes negociados da '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(df['Volume'],color='r')
st.pyplot()


col10, col20, col30 = st.sidebar.beta_columns(3)

kk = f"**{round(df['Adj Close'].tail(1)[0],2)}**"
kk1 = f"**{round(100*(df['Adj Close'].iloc[-1] - df['Adj Close'].iloc[-2])/(df['Adj Close'].iloc[-1]),2)}**"
#st.sidebar.markdown('<p class="big-font">f"**{round(df["Adj Close"].tail(1)[0],2)}**"</p>', unsafe_allow_html=True)
st.sidebar.write("Valor de fechamento autalizado da "+acao+": "+kk)
st.sidebar.write("Variação diária de: "+kk1+"%")
#st.sidebar.write("{.:2f}".format(df['Adj Close'].tail(1)[0]))


#-------------------------------------ANALISE DE TENDENCIAS BASICA-----------------------------------------------------------------------------------------------

#---bandas de bollinger
st.markdown("<hr style='margin-bottom: 40px'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-top: -40px; font-family: Helvetica; font-weight:bold; margin-left: 140px'>Análise de tendências</h2>", unsafe_allow_html=True)

dff = df['2020-12-12':]
st.header("Tendência de evolução da "+acao+" pelas bandas de Bollinger - Período de 21 dias")
data = dff['Close'].values.tolist()
period = 21
bb_up = bb_up(data,period)
bb_mid = bb_mid(data,period)
bb_low = bb_low(data,period)
dff['bb_up'] = bb_up
dff['bb_mid'] = bb_mid
dff['bb_low'] = bb_low
apd = mpf.make_addplot(dff[['bb_up', 'bb_mid', 'bb_low']])
bollinger = mpf.plot(dff, type='candle', addplot=apd, volume=False)
st.pyplot(bollinger)


#-------ifr
stock = pd.DataFrame(df['Adj Close'].copy())
#
stock['Variation'] = stock['Adj Close'].diff()
stock = stock[1:] # remove first row once it does not have a variation
stock['Gain'] = np.where(stock['Variation'] > 0, stock['Variation'], 0) 
stock['Loss'] = np.where(stock['Variation'] < 0, stock['Variation'], 0)
n = 14 # define window interval
simple_avg_gain = stock['Gain'].rolling(n).mean()
simple_avg_loss = stock['Loss'].abs().rolling(n).mean()
# start off of simple average series
classic_avg_gain = simple_avg_gain.copy()
classic_avg_loss = simple_avg_loss.copy()

# iterate over the new series but only change values after the nth element
for i in range(n, len(classic_avg_gain)):
    classic_avg_gain[i] = (classic_avg_gain[i - 1] * (n - 1) + stock['Gain'].iloc[i]) / n
    classic_avg_loss[i] = (classic_avg_loss[i - 1] * (n - 1) + stock['Loss'].abs().iloc[i]) / n
stock['Simple RS'] = simple_avg_gain / simple_avg_loss
stock['Classic RS'] = classic_avg_gain / classic_avg_loss
stock['Simple RSI'] = 100 - (100 / (1 + stock['Simple RS']))
stock['Classic RSI'] = 100 - (100 / (1 + stock['Classic RS']))
#stock[['Simple RS', 'Classic RS']].head(20)
#st.write(stock.tail())
#st.write(stock.iloc[:].diff())
#stock = stock[1:] # remove first row once it does not have a variation
#stock.head()
#plt.title("IFR PETR4")
plt.figure(figsize=(16,8))
plt.title('Índice de Força Relativa da ação '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(stock['Classic RSI'],color='r')
st.pyplot()



#-------------------------TENDÊNCIA PELO ARIMA-----------------------------------------------------------------
st.header("Avaliação de tendência pelo algoritmo ARIMA da ação "+acao)
z = int(len(df['Adj Close'])/2)
decomposicao = seasonal_decompose(df['Adj Close'],freq=20)
tendencia = decomposicao.trend
plt.figure(figsize=(16,8))
plt.title('Tendência de valoração da ação '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(tendencia.iloc[:],color='r')
st.pyplot()



#--------------------------------------MEDIDAS DE RETORNO----------------------------------------------------------------------
st.markdown("<hr style='margin-bottom: 40px'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-top: -40px; font-family: Helvetica; font-weight:bold; margin-left: 140px'>Análise de retorno</h2>", unsafe_allow_html=True)
st.header("Retorno da ação "+acao)
st.write(100*df.loc[:, df.columns != 'Volume'].pct_change().describe())
plt.figure(figsize=(16,8))
plt.title('Retorno percentual diário da ação '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(100*df['Adj Close'].pct_change(),color='r')
st.pyplot()
retorno_acum = 1
ret = []
for i in range(1,len(df['Adj Close'])):
    retorno_acum *= (1 + df['Adj Close'].pct_change()[i])
    ret.append(100*retorno_acum - 100)

xf = pd.DataFrame(ret)
xf.columns = ['Retorno percentual']
xf.index = df.index[1:]
plt.figure(figsize=(16,8))
plt.title('Retorno acumulado da ação '+acao, fontsize=26)
plt.ylabel("Retorno (%)", fontsize=22)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(xf,color='r')
st.pyplot()

plt.figure(figsize=(16,8))
plt.title('Distribuição do retorno diário da ação '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
sns.distplot(100*df['Adj Close'].pct_change(), bins=100)
st.pyplot()


#--------------------------------------MEDIDAS DE VOLATILIDADE----------------------------------------------------------------------
st.markdown("<hr style='margin-bottom: 40px'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-top: -40px; font-family: Helvetica; font-weight:bold; margin-left: 140px'>Análise de volatilidade</h2>", unsafe_allow_html=True)

st.header("Volatilidade por drawdown da ação "+acao)
data1 = pd.DataFrame(df['Adj Close'].copy())
data1["Max"] = data1['Adj Close'].cummax()
data1["Delta"] = data1["Max"] - data1["Adj Close"]
data1["Drawdown"] = 100 * (data1["Delta"] / data1["Max"])
plt.figure(figsize=(16,8))
plt.title('Drawdown da ação '+acao, fontsize=26)
plt.xlabel('Data', fontsize=22)
plt.xticks(fontsize=16)
plt.plot(data1['Drawdown'],color='r')
plt.legend(['Drawdown'], loc='lower right')
st.pyplot()
#figura12 = px.line(title=" ")
#figura12.add_scatter(x=data1.index,y=data1['Drawdown'],name=acao)
#st.plotly_chart(figura12,use_container_width=True)

#------------------------covariancia------------------------------------------------------------------------------------------------------------------------------
act = ('PETR3.SA', 'PETR4.SA','CSAN3.SA','BRKM5.SA','UGPA3.SA',"EQTL3.SA","BBAS3.SA",
       'BBSE3.SA','BBDC3.SA','BBDC4.SA','ITSA4.SA','ABEV3.SA','VIVT3.SA','OIBR4.SA',
       'MGLU3.SA','LAME4.SA','LREN3.SA','RENT3.SA','GGBR4.SA','CCRO3.SA','EMBR3.SA',
       'SBSP3.SA','MRVE3.SA','KLBN11.SA','CYRE3.SA','SANB11.SA','QUAL3.SA','JBSS3.SA',
       'MRFG3.SA','CIEL3.SA','RADL3.SA','BTOW3.SA','BPAC11.SA','ELET3.SA','USIM3.SA',
       'IGTA3.SA','PRIO3.SA','BRML3.SA')

if(acao in act):
    #calculando indices por web scraping ------------------------------------------------
    #h = soup.find_all('span', class_="Trsdu(0.3s)")


    jp = "https://br.financas.yahoo.com/quote/"+acao+"/key-statistics"
    page1 = requests.get(jp)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    #h1 = soup1.find_all('td')
    pl = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="35">')[1].split('</td')[0]#str(h1).split(' class="Ta(end) Fw(600) Lh(14px)" data-reactid="107" data-test="DIVIDEND_AND_YIELD-value">')[1].split('</td')[0]
    market = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="28">')[1].split('</td')[0]
    capt = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="21">')[1].split('</td')[0]
    valor_empresa_receita = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="72">')[1].split('</td')[0]
    ebidta = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="79">')[1].split('</td')[0]
    div_yield = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="263">')[1].split('</td')[0]
    f_c_op = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="507">')[1].split('</td')[0]
    f_c_a = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="514">')[1].split('</td')[0]
    lpa = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="437">')[1].split('</td')[0]
    LL = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="430">')[1].split('</td')[0]
    dbt_pat = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="479">')[1].split('</td')[0]
    #payout = str(soup1.find_all('td', {'class': 'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})).split(' data-reactid="285">')[1].split('</td')[0]
    st.sidebar.write("Valor da empresa: "+market)
    st.sidebar.write("Capitalização de mercado: "+capt)
    st.sidebar.write("Lucro líquido: "+LL)
    st.sidebar.write("P/L passado: "+pl)
    st.sidebar.write("Lucro por ação: "+lpa)
    st.sidebar.write("Valor da empresa/receita: "+valor_empresa_receita)
    st.sidebar.write("Valor da empresa/EBITDA: "+ebidta)
    st.sidebar.write("Dividend yield: "+div_yield)
    st.sidebar.write("Fluxo de caixa operacional: "+f_c_op)
    st.sidebar.write("Fluxo de caixa livre alavancado: "+f_c_a)
    st.sidebar.write("Débito total/Patrimônio líquido: "+dbt_pat)
    datak = pd.DataFrame()
    tickers = [acao,'^BVSP']
## Coletando os dados do Yahoo Finance no período estipulado (dados de fechamento)
    for t in tickers:
        datak[t] = wb.DataReader(t, data_source='yahoo', start='2018-06-06', end=dt.datetime.now())['Adj Close']

    retorno_log = np.log( datak / datak.shift(1) )
    cov = retorno_log.cov() * 250
    ## Covariância AMZN em relação ao mercado (linha 0 e coluna 1)
    Covariancia_Mercado = cov.iloc[0,1]
    ## Variância do mercado (desconsiderando os finais de semana)
    Variancia_Mercado = retorno_log['^BVSP'].var() * 250
    ## Calculando o Beta da AMZN
    ACAO_beta = Covariancia_Mercado/ Variancia_Mercado
    bet = f"**{round(ACAO_beta,2)}**"
    if(ACAO_beta>1):
        st.write("A ação "+acao+" é agressiva, com valor de beta: "+bet)
    elif(ACAO_beta<=0):
        st.write("A ação "+acao+" não possui nenhuma relação com o mercado")
    else:
        st.write("A ação "+acao+" é defensiva, com valor de beta: "+bet)

#-------------------taxa de retorno-----------------------------------------------------------
    TL_BRASIL = 0.068
    RISCO = 0.1
    ACAO_CAPM = f"**{round(100*(TL_BRASIL + RISCO*ACAO_beta),2)}**"
    capm = TL_BRASIL + RISCO*ACAO_beta
    st.write("A ação "+acao+" possuirá retorno (perda ou ganho) esperado de: "+ACAO_CAPM+'%')
    sharpe = (capm - 0.068) / (retorno_log[acao].std() * 250 ** 0.5)
    sharp = f"**{round(sharpe,3)}**"
    st.write("O índice Sharpe da ação "+acao+" é: "+sharp)

st.markdown("<hr style='margin-bottom: 40px'>", unsafe_allow_html=True)
st.markdown("<h2 style='margin-top: -40px; font-family: Helvetica; font-weight:bold; margin-left: 140px'>Análise preditiva</h2>", unsafe_allow_html=True)
#-------------------------------------------PREVISAO---------------------------------------------------------------------------------------------------------------
def redes_neurais(df,contador):
    if(contador<1):
        dff = df['2019-06-06':]
        data = dff.filter(['Adj Close'])
        #Convert the dataframe to a numpy array
        dataset = data.values
        #Get the number of rows to train the model on
        training_data_len = math.ceil( len(dataset) * 0.8 )
        #Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        #Create the training data set 
        #Create the scaled training data set
        train_data = scaled_data[0:training_data_len, :]
        resto = math.ceil( len(dataset) * 0.2 )
        #Split the data into x_train and y_train data sets
        x_train = []
        y_train = []
        #We create a loop
        for i in range(resto, len(train_data)):
            x_train.append(train_data[i-resto:i, 0]) #Will conaint 60 values (0-59)
            y_train.append(train_data[i, 0]) #Will contain the 61th value (60)
        #Convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        #Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        #Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(30))
        model.add(Dense(1))

        model.add(Dense(units=1,activation='linear'))

        #Compile the model
        model.compile(optimizer='rmsprop', loss='mean_squared_error',metrics=['mean_absolute_error'])
       

        #Train the model
        model.fit(x_train, y_train, batch_size=3, epochs=5)
        test_data = scaled_data[training_data_len - 60:]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        #Convert the data to a numpy array
        x_test = np.array(x_test)

        #Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        #Get the model's predicted price values for the x_test data set
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        
        #Evaluate model (get the root mean quared error (RMSE))
        rmse = np.sqrt( np.mean( predictions - y_test )**2 )

        #Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        #Create the testing data set
        #Create a new array containing scaled values from index 1738 to 2247
        test_data = scaled_data[training_data_len - 60:]
        X_FUTURE = tempo
        predictions = np.array([])
        last = x_test[-1]
        
    from datetime import timedelta
    for i in range(X_FUTURE):
        curr_prediction = model.predict(np.array([last]))
        last = np.concatenate([last[1:], curr_prediction])
        predictions = np.concatenate([predictions, curr_prediction[0]])
    predictions = scaler.inverse_transform([predictions])[0]
    dicts = []
    curr_date = data.index[-1]
    for i in range(X_FUTURE):
        curr_date = curr_date + timedelta(days=1)
        dicts.append({'Predictions':predictions[i], "Data": curr_date})

    new_data = pd.DataFrame(dicts).set_index("Data")
    train = data
    previsao4 = new_data.copy()
    previsao4.rename(columns={0:'Valor da ação'}, inplace=True)
    previsao4['Dat'] = previsao4.index
    def corte(x):
        y = str(x).split('T')[0]
        yy = y.split(" ")
        return yy[0]
    previsao4['Data'] = previsao4['Dat'].apply(lambda x: corte(x))
    previsao5 = previsao4[['Data','Predictions']]
    previsao5.set_index('Data',inplace=True)
    st.write(previsao5)
    contador += 1
    #Visualize the data
    
    plt.figure(figsize=(16,8))
    plt.title('Modelo de Redes Neurais',fontsize=26)
    plt.xlabel('Data', fontsize=22)
    plt.xticks(fontsize=16)
    plt.plot(train['Adj Close'])
    plt.plot(new_data['Predictions'])
    plt.legend(['Dado real', 'Previsão'], loc='lower right')
    st.pyplot()


def Monte_Carlo(df):
    dff1 = df['Adj Close']
    LOG_RET = np.log(1 + dff1.pct_change()) #ajuste
    u = LOG_RET.mean() #media
    vari = LOG_RET.var() #variância
    drift = u - vari/2 #distribuição normal
    stdi = LOG_RET.std() #desvio padrão
    t_inter = int(tempo)+1 #dias previstos após dia mais recente 
    iteration = 10 #número de cenários
    d_r = np.exp(drift - stdi*norm.ppf(np.random.rand(t_inter,iteration))) #Modelo Browniano
    #st.write(d_r)
    price_list = np.zeros_like(d_r)
    s0 = dff1.iloc[-1] #preço da ação do dia mais recente
    price_list[0] = s0 #preço inicial para ação
    k = []
    std_min = []
    std_max = []
    for t in range(1,t_inter):
        price_list[t] = price_list[t-1]*d_r[t]
        k.append(np.mean(price_list[t]))
        std_min.append(np.mean(price_list[t]) - 2*np.std(price_list[t]))
        std_max.append(np.mean(price_list[t]) + 2*np.std(price_list[t]))
        
    temp = pd.date_range(dff1.tail(1).index[0], periods=int(tempo))
    previsao3 = pd.DataFrame(np.array([k,std_min,std_max]).T,index=temp)
    previsao3.rename(columns={0:'Valor médio da ação',1:'Valor mínimo da ação',2:'Valor máximo da ação'}, inplace=True)
    previsao3['Dat'] = previsao3.index
    def corte(x):
        y = str(x).split('T')[0]
        yy = y.split(" ")
        return yy[0]
    previsao3['Data'] = previsao3['Dat'].apply(lambda x: corte(x))
    previsao4 = previsao3[['Data','Valor médio da ação','Valor mínimo da ação','Valor máximo da ação']]
    previsao4.set_index('Data',inplace=True)
    st.header("Resultado do modelo preditivo por Monte Carlo")
    st.dataframe(previsao4.tail(int(tempo)))
    st.header("Previsão de fechamento da "+acao+" nos próximos " + str(int(tempo)) + " dias - Modelo de Monte Carlo")

    zp = ['Valor médio da ação','Valor mínimo da ação','Valor máximo da ação']
    plt.figure(figsize=(16,8))
    plt.title('Modelo de Monte Carlo',fontsize=26)
    plt.xlabel('Data', fontsize=22)
    plt.xticks(fontsize=12)
    plt.plot(previsao4[zp[0]])
    plt.plot(previsao4[zp[1]])
    plt.plot(previsao4[zp[2]])
    plt.legend([zp[0],zp[1],zp[2]], loc='upper left')
    st.pyplot()

if submit:
    if modelo == "Monte Carlo":
        Monte_Carlo(df)
    if modelo == 'Redes Neurais':
        redes_neurais(df,contador)

