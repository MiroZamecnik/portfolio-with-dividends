import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
import datetime
import sklearn.metrics
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from dateutil.parser import parse
 
def convert_to_datetime(input_str, parserinfo=None):
    return parse(input_str, parserinfo=parserinfo)

# setup of colors for graphs
bg_color = "#f8f9f8"
secondaryBackgroundColor="#a5f0fd"

# initial setup of portfolio if it is not in session_state yet, 
# if it already is - the current/changed values will not be overwritten
if 'stocks' not in st.session_state:
    st.session_state.stocks = OrderedDict()
    st.session_state.stocks['BMW.DE'] = 100
    st.session_state.stocks['DHL.DE'] = 250
    st.session_state.stocks['EXS1.DE'] = 100
    st.session_state.stocks['AAPL'] = 50
    st.session_state.stocks['KBC.BR'] = 150
    st.session_state.stocks['MBG.DE'] = 150
    st.session_state.stocks['MKS.L'] = 50

# ordered dictionary stocks - is used for storage of stock tickers and amount of stocks in the portfolio
stocks = st.session_state.stocks
    
# function to add ticker into stocks with 0 amount for portfolio    
def add_ticker(ticker):
    if ticker != '':
        stocks[ticker] = 0
        st.session_state.stocks[ticker] = 0
    #with open('stocks_files/tickers.txt', 'w') as f:
    #    ret =''
    #    for i in stocks:
    #        ret = ret + i +"::" + str(stocks[i]) + '\n'
    #    print(ret, file=f)
    return

# function to remove ticker from stocks with 0 amount for portfolio    
def remove_ticker(ticker):
    if ticker in stocks:
        del stocks[ticker]
        del st.session_state.stocks[ticker]
    #    with open('stocks_files/tickers.txt', 'w') as f:
    #        ret =''
    #        for i in stocks:
    #            ret = ret + i +"::" + str(stocks[i]) + '\n'
    #        print(ret, file=f)
    return

#read initial portfolio from data file
#file = open('stocks_files/tickers.txt', 'r')
#for row in file:
#    r = row.strip()
#    if r: 
#        stocks[r[:r.find('::')]]=int(r[r.find('::')+2:])
#file.close()

#setup of initial starting date
if 'set_sd' not in st.session_state:
    set_sd = '2015-03-16'

st.sidebar.header('Define your portfolio :)')

new = st.sidebar.text_input(label='Add/remove YAHOO symbol to/from portfolio. \n Use https://finance.yahoo.com to look up the symbol', value='e.g. AAPL')

col1, col2 = st.sidebar.columns(2)

button1 = col1.button('*Add*')
button2 = col2.button('*Remove*')

#on pressing left "Add" button, the text from the text_input will be included into the STOCKS list 
if button1:
    if new.strip().upper() not in stocks and new[:4]!='e.g.': 
        add_ticker(new.upper())
        
#on pressing right "Remove" button, the text from the text_input will be excluded from the STOCKS list       
if button2:
    if new.strip().upper() in stocks: 
        remove_ticker(new.upper())    
        
st.sidebar.subheader('Enter amounts of chosen stocks, ETFs, ... ')

temp_list = list()
for i, h in enumerate(stocks): 
    temp_list.append(h)

# tickers with text input appers only in case the ticker is in the portfolio
if len(stocks) > 0:
    amount_stocks_0 = st.sidebar.text_input(f"*enter amount of **{temp_list[0]}** stocks*", str(stocks[temp_list[0]]))
    stocks[temp_list[0]] = amount_stocks_0
if len(stocks) > 1:
    amount_stocks_1 = st.sidebar.text_input(f"*enter amount of **{temp_list[1]}** stocks*", str(stocks[temp_list[1]]))
    stocks[temp_list[1]] = amount_stocks_1
if len(stocks) > 2:
    amount_stocks_2 = st.sidebar.text_input(f"*enter amount of **{temp_list[2]}** stocks*", str(stocks[temp_list[2]]))
    stocks[temp_list[2]] = amount_stocks_2
if len(stocks) > 3:
    amount_stocks_3 = st.sidebar.text_input(f"*enter amount of **{temp_list[3]}** stocks*", str(stocks[temp_list[3]]))
    stocks[temp_list[3]] = amount_stocks_3
if len(stocks) > 4:
    amount_stocks_4 = st.sidebar.text_input(f"*enter amount of **{temp_list[4]}** stocks*", str(stocks[temp_list[4]]))
    stocks[temp_list[4]] = amount_stocks_4
if len(stocks) > 5:
    amount_stocks_5 = st.sidebar.text_input(f"*enter amount of **{temp_list[5]}** stocks*", str(stocks[temp_list[5]]))
    stocks[temp_list[5]] = amount_stocks_5
if len(stocks) > 6:
    amount_stocks_6 = st.sidebar.text_input(f"*enter amount of **{temp_list[6]}** stocks*", str(stocks[temp_list[6]]))
    stocks[temp_list[6]] = amount_stocks_6
if len(stocks) > 7:
    amount_stocks_7 = st.sidebar.text_input(f"*enter amount of **{temp_list[7]}** stocks*", str(stocks[temp_list[7]]))
    stocks[temp_list[7]] = amount_stocks_7
if len(stocks) > 8:
    amount_stocks_8 = st.sidebar.text_input(f"*enter amount of **{temp_list[8]}** stocks*", str(stocks[temp_list[8]]))
    stocks[temp_list[8]] = amount_stocks_8
if len(stocks) > 9:
    amount_stocks_9 = st.sidebar.text_input(f"*enter amount of **{temp_list[9]}** stocks*", str(stocks[temp_list[9]]))
    stocks[temp_list[9]] = amount_stocks_9
if len(stocks) > 10:
    amount_stocks_10 = st.sidebar.text_input(f"*enter amount of **{temp_list[10]}** stocks*", str(stocks[temp_list[10]]))
    stocks[temp_list[10]] = amount_stocks_10
if len(stocks) > 11:
    amount_stocks_11 = st.sidebar.text_input(f"*enter amount of **{temp_list[11]}** stocks*", str(stocks[temp_list[11]]))
    stocks[temp_list[11]] = amount_stocks_11
if len(stocks) > 12:
    amount_stocks_12 = st.sidebar.text_input(f"*enter amount of **{temp_list[12]}** stocks*", str(stocks[temp_list[12]]))
    stocks[temp_list[12]] = amount_stocks_12
if len(stocks) > 13:
    amount_stocks_13 = st.sidebar.text_input(f"*enter amount of **{temp_list[13]}** stocks*", str(stocks[temp_list[13]]))
    stocks[temp_list[13]] = amount_stocks_13
if len(stocks) > 14:
    amount_stocks_14 = st.sidebar.text_input(f"*enter amount of **{temp_list[14]}** stocks*", str(stocks[temp_list[14]]))
    stocks[temp_list[14]] = amount_stocks_14


#****************************************************************************************
#input all possible reference currencies and currencies of stocks
currency_list = ['EUR', 'CZK', 'USD', 'GBP', 'HUF', 'AUD', 'JPY', 'CAD']

# creating of list_all_tickers list of respective FX tickers for all ccies from currency_list and adding them to stocks tickers 
list_all_tickers = list(stocks) 
for c in currency_list:
    if c!='EUR':
        list_all_tickers.append('EUR'+c+'=X')

# cache memory trick to avoid multiple downloads of the same data
@st.cache_data

# function to download all tickers close rates based on input list of tickers and frequency of data, returns dataframe
def download_tickers_closing_rates(list_all_tickers, freq):
    data = yf.download(tickers=list_all_tickers, period='max', interval = freq, group_by='ticker', auto_adjust=True, prepost=True)
    df = pd.DataFrame(data)
    df= pd.DataFrame(data[list_all_tickers[0]].Close)

    for ticker in list_all_tickers: 
        df[ticker]= pd.DataFrame(data[ticker].Close)
    df = df.dropna()
    return df

# download of dividend pay outs in history for all tickers from the list starting from defined date - start_date
# returns enriched dataframe which was also input of the function
def download_dividends(a, list_all_tickers, start_date):
    a['was'] = 1
    for ticker in list_all_tickers:
        if ticker.find('=') == -1:
            #st.write('stahujem dividendy ', ticker)
            akcia = yf.Ticker(ticker)
            dividendy = akcia.dividends - akcia.dividends
            suma = 0
            for i in akcia.dividends.index: 
                if str(i)>str(start_date): 
                    suma += akcia.dividends[i]
                    dividendy[i] = suma
    
            b = dividendy.to_frame().reset_index() 
            b = b[b['Dividends']>0]
            b.rename(columns={"Dividends" : ticker+"_div"}, inplace = True) 
            a = a.reset_index()
            a = pd.concat([a, b])
            a = a.reset_index()
            a['Date'] = a['Date'].apply(lambda x:x.tz_localize(None))
            a = a.sort_values(by='Date')
            a[ticker+"_div"] = a[ticker+"_div"].fillna(method='ffill')
            a = a.fillna(value = 0)
            a = a[a['was']==1]
            a[ticker+'_total'] = a[ticker] + a[ticker+'_div']
            a = a.drop(['level_0', 'index', 'close'], axis=1, errors='ignore')
    return a

# retrieves currency of the ticker via yfinance
def get_currency_of_tickers(stocks):
    ccy = dict()
    for ticker in stocks:
        akcia = yf.Ticker(ticker)
        ccy[ticker] = akcia.info['currency'].upper()
    return ccy

# for each ticker - creating columns with original ccy and EUR ending
def make_orig_ccy_and_EUR(a, stocks):
    for ticker in stocks:
        if ccy[ticker]=='EUR':
            a[ticker+'_total_EUR'] = a[ticker+'_total']
            a[ticker+'_EUR'] = a[ticker]
        else: 
            a[ticker+'_total_EUR'] = a[ticker+'_total'] / a['EUR'+ccy[ticker]+'=X']
            a[ticker+'_EUR'] = a[ticker] / a['EUR'+ccy[ticker]+'=X']
            a[ticker+'_total_'+ccy[ticker]] = a[ticker+'_total']
            a[ticker+'_'+ccy[ticker]] = a[ticker]
    return a

def add_portfolio_all_ccies(a, stocks, currency_list):
 # adding portfolio columns in EUR
    a['portfolio_total_EUR'] = 0
    a['portfolio_EUR'] = 0
    for s in stocks:
        a['portfolio_total_EUR'] += float(stocks[s]) * a[s+'_total_EUR']
        a['portfolio_EUR'] += float(stocks[s]) * a[s+'_EUR']
# adding portfolio columns in all other possible reference currencies
    for c in currency_list:
        if c!='EUR':
            a['portfolio_total_'+c] = a['portfolio_total_EUR'] * a['EUR'+c+'=X']
            a['portfolio_'+c] = a['portfolio_EUR'] * a['EUR'+c+'=X']
    return a

#****************************************************************************************

st.title('Miro\'s stocks & ETFs portfolio app including dividends')

st.markdown("""
* This is my first Streamlit app (made in January 2024)
* **Data source:** market data downloaded using ***yfinance*** library
------------------------------------------------------------------------
""")
# function that will return key of the respective input val(ue) from indices
def get_ind_label(val):
    for k, v in indices.items():
        if v == val:
            return k

# function to draw price evolution of given ticker/symbol 
def price_plot(symbol, currency, fg_color=bg_color, bg_color=bg_color, place=st):
    deskrip = a[symbol+"_total_"+currency].describe(percentiles=[.8, .9, .95])
    fig, ax = plt.subplots(facecolor = bg_color)
    multip = 1
    suff = ""
    if deskrip['max']>5000:
        multip = 1000
        suff = 'k'
    if deskrip['max']>1000000:
        multip = 1000000
        suff = 'mln'
    
    if currency=='USD':
        pref = '$ %2.0f'
    elif currency=='EUR':
        pref = '€ %2.0f'
    elif currency=='GBP':
        pref = '£ %2.0f'
    else:
        pref = currency+'  %.0f'
    plt.grid(which='major', axis='y' ,linestyle = '-', linewidth = 1, alpha = 0.6)
    ax.set_facecolor(bg_color)
    #fig.axes.tick_params(color=fg_color, labelcolor=fg_color)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    fig.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter(pref+suff))
    plt.plot(a['Date'], a[symbol+"_"+currency]/multip, color="dodgerblue", linestyle='dashed', alpha=0.9, label=symbol+' price/value', linewidth=1)
    plt.plot(a['Date'], a[symbol+"_total_"+currency]/multip, color='dodgerblue', alpha=0.9, label='value with dividends since '+convert_to_datetime(str(start_date)).strftime("%b %d, %Y"), linewidth=1)
    plt.fill_between(a['Date'], a[symbol+"_"+currency]/multip, a[symbol+"_total_"+currency]/multip, color=secondaryBackgroundColor, alpha=0.8, label='dividend yields')

   # plt.axhline(y=deskrip['95%']/multip, color='lightgreen', linestyle='dashed', label='95 percentil of total value')
  #  plt.axhline(y=deskrip['90%']/multip, color='yellow', linestyle='dashed', label='90 percentil of total value')
   # plt.axhline(y=deskrip['80%']/multip, color='orange', linestyle='dashed', label='80 percentil of total value')
    plt.xticks(rotation=45)
    plt.title(symbol, fontweight='bold')
    #plt.xlabel('Date', fontweight='bold')   plt.ylabel('Value', fontweight='bold')
    plt.ylim(bottom = 0.3 * deskrip['min']/multip)
    plt.legend(labelcolor='linecolor', facecolor = bg_color, loc='lower left')      
    return place.pyplot(fig)

# function to return string with sign of the input number
def sign_m(number):
    z = ' + '
    if number<0:
        z = ' - '
    return z

# function to plot deltas - changes in symbol vs. changes in given index, with linear regression of the observations
def beta_plot(symbol, index, freq, year_min, year_max, fg_color=bg_color, bg_color=bg_color, place=st):
    if symbol == 'portfolio_total_EUR':
        symbol_label='portfolio'
    else: symbol_label = symbol
    
    p1, p2 = plt.subplots(facecolor = bg_color)
    p2.set_facecolor(bg_color)

    plt.grid(which='major', axis='both' ,linestyle = 'dashed', linewidth = 1)
    p1.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    p1.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    plt.xlabel(freq+' % change of '+ get_ind_label(index), fontweight='bold')
    plt.ylabel(freq+' % change of '+ symbol_label, fontweight='bold')

    X = lm[[index+'_delta', symbol+'_delta', 'year']]
    X = X[X['year'] >= year_min] 
    X = X[X['year'] <= year_max] 

    plt.scatter(X[index+'_delta'], X[symbol+'_delta'], marker = '.', color='dodgerblue')
    # adding of linear regression
    Y = X[symbol+'_delta'].values.reshape(-1, 1)[1:]  
    X = X[index+'_delta'].values.reshape(-1, 1)[1:] 
    lin_regressor = LinearRegression()
    lin_regressor = lin_regressor.fit(X, Y)
    rr = sklearn.metrics.r2_score(Y, lin_regressor.predict(X))
    equation = freq+' % of '+ symbol_label +' = '+ str(lin_regressor.intercept_[0])[:6]+sign_m(lin_regressor.coef_[0][0]) + str(lin_regressor.coef_[0][0])[:5]+ ' * (' + freq+' % of '+ get_ind_label(index)+')'+'\n'   
    plt.plot(X, lin_regressor.predict(X), color = 'red', linestyle='-', label = equation +'R squared = '+ str(rr)[:4])
    
    plt.title(str(year_min)+' - '+str(year_max), fontweight='bold', color = 'dodgerblue')
    plt.legend(labelcolor='linecolor', facecolor = bg_color)
    return place.pyplot(p1,p2)

# function to return table of statistics on dividends, price changes ands its percentages to starting value
def get_stat(tic, where=st):
    c = ccy[tic] + ' '
    if ccy[tic]=='EUR':
        c = '€ '
    if ccy[tic]=='USD':
        c = '$ '
    if ccy[tic]=='GBP':
        c = '£ '
    m = lm[['Date', tic, tic+'_div', tic+'_total', 'year']]
    first_lines = m.sort_values('Date').groupby('year').first()
    first_lines['type'] = 'first'
    last_lines = m.sort_values('Date').groupby('year').last()
    last_lines['type'] = 'last'
    all_lines = pd.concat([first_lines,last_lines])
    all_lines = all_lines.reset_index()
    starting_value = all_lines[all_lines['Date']==all_lines.describe()['Date']['min']][tic].iloc[0]
    final_value = all_lines[all_lines['Date']==all_lines.describe()['Date']['max']][tic].iloc[0]
    starting_date = str(all_lines.describe()['Date']['min'])[:10]
    final_date = str(all_lines.describe()['Date']['max'])[:10]
    where.write('Starting value of', tic, 'at '+ convert_to_datetime(starting_date).strftime("%b %d, %Y"),'was ', str(round(starting_value, 2)), c+'.')
    where.write('Final value of', tic, 'at '+ convert_to_datetime(final_date).strftime("%b %d, %Y"),'is ', str(round(final_value, 2)), c+'.')
    where.write('')
    where.write('Recap of', tic, 'performance up to now:')
    all_lines['total dividend amount per share'] = all_lines[tic+'_div']
    all_lines['      as % of '+starting_date+' value'] = all_lines[tic+'_div']/starting_value
    all_lines['      as % of '+starting_date+' value']=all_lines['      as % of '+starting_date+' value'].apply(lambda x: "{0:.1f}%".format(x*100))
    all_lines['share price change (+/-) since '+starting_date] = all_lines[tic]-starting_value
    all_lines['       as % of '+starting_date+' value'] = all_lines['share price change (+/-) since '+starting_date]/starting_value
    all_lines['       as % of '+starting_date+' value'] = all_lines['       as % of '+starting_date+' value'].apply(lambda x: "{0:.1f}%".format(x*100))
    all_lines['total yield per share'] = all_lines['total dividend amount per share']+all_lines['share price change (+/-) since '+starting_date]
    all_lines['        as % of '+starting_date+' value'] = all_lines['total yield per share']/starting_value
    all_lines['        as % of '+starting_date+' value'] = all_lines['        as % of '+starting_date+' value'].apply(lambda x: "{0:.1f}%".format(x*100)) 
    all_lines['total dividend amount per share'] = all_lines['total dividend amount per share'].apply(lambda x:c+"{0:,.1f}".format(x) if abs(x) < 5000 else (c+'{:.2f} k'.format(x/1000) if abs(x) < 1000000 else c+'{:.2f} mln'.format(x/1000000)))
    all_lines['share price change (+/-) since '+starting_date] = all_lines['share price change (+/-) since '+starting_date].apply(lambda x:c+"{0:,.1f}".format(x) if abs(x) < 5000 else (c+'{:.2f} k'.format(x/1000) if abs(x) < 1000000 else c+'{:.2f} mln'.format(x/1000000)))
    all_lines['total yield per share'] = all_lines['total yield per share'].apply(lambda x:c+"{0:,.1f}".format(x) if abs(x) < 5000 else (c+"{:.2f} k".format(x/1000) if abs(x) < 1000000 else c+'{:.2f} mln'.format(x/1000000)))
    #print(all_lines[all_lines.columns[:4]])
    final = all_lines.sort_values('Date').groupby('year').last()
    return final.transpose()[5:]

def get_portfolio_stat(currency, where=st):
    c = currency + ' '
    if currency=='EUR':
        c = '€ '
    if currency=='USD':
        c = '$ '
    if currency=='GBP':
        c = '£ '
    
    m = lm[['Date', 'portfolio_'+currency, 'portfolio_total_'+currency, 'year']]
    first_lines = m.sort_values('Date').groupby('year').first()
    first_lines['type'] = 'first'
    last_lines = m.sort_values('Date').groupby('year').last()
    last_lines['type'] = 'last'
    all_lines = pd.concat([first_lines,last_lines])
    all_lines = all_lines.reset_index()
    starting_value = round(all_lines[all_lines['Date']==all_lines.describe()['Date']['min']]['portfolio_total_'+currency].iloc[0], 0)
    final_value = round(all_lines[all_lines['Date']==all_lines.describe()['Date']['max']]['portfolio_total_'+currency].iloc[0], 0)
    starting_date = str(all_lines.describe()['Date']['min'])[:10] 
    final_date = str(all_lines.describe()['Date']['max'])[:10] 
    where.write('Starting value of portfolio at '+ convert_to_datetime(starting_date).strftime("%b %d, %Y") ,'was ', str(round(starting_value//1000))+' '+str(round(starting_value%1000)), c+'.')
    where.write('Final value of portfolio at '+ convert_to_datetime(final_date).strftime("%b %d, %Y"),'is ',str(round(final_value//1000))+' '+str(round(final_value%1000)), c+'.')
    where.write('')
    where.write('''**Overview of portfolio performance up to now:**''')
    starting_date_f = str(convert_to_datetime(starting_date).strftime("%b %d, %Y"))
    
    all_lines['total dividends since '+starting_date_f] = all_lines['portfolio_total_'+currency] - all_lines['portfolio_'+currency]

    all_lines['       as % of '+starting_date_f+' value'] = all_lines['total dividends since '+starting_date_f]/starting_value
    all_lines['       as % of '+starting_date_f+' value'] = all_lines['       as % of '+starting_date_f+' value'].apply(lambda x: "{0:.1f}%".format(x*100)) 
    
    all_lines['portfolio price change (+/-) since '+starting_date_f] = all_lines['portfolio_'+currency]-starting_value
    
    all_lines['      as % of '+starting_date_f+' value'] = all_lines['portfolio price change (+/-) since '+starting_date_f]/starting_value
    all_lines['      as % of '+starting_date_f+' value'] = all_lines['      as % of '+starting_date_f+' value'].apply(lambda x: "{0:.1f}%".format(x*100))
    
    all_lines['total value yield of portfolio'] = all_lines['total dividends since '+starting_date_f] + all_lines['portfolio price change (+/-) since '+starting_date_f]
    all_lines['        as % of '+starting_date_f+' value'] = all_lines['total value yield of portfolio']/starting_value
    all_lines['        as % of '+starting_date_f+' value'] = all_lines['        as % of '+starting_date_f+' value'].apply(lambda x: "{0:.1f}%".format(x*100))
    all_lines['total value yield of portfolio'] = all_lines['total value yield of portfolio'].apply(lambda x:c+"{0:,.0f}".format(x) if abs(x) < 5000 else (c+'{:.2f} k'.format(x/1000) if abs(x) < 1000000 else c+'{:.2f} mln'.format(x/1000000)))
    all_lines['portfolio price change (+/-) since '+starting_date_f] = all_lines['portfolio price change (+/-) since '+starting_date_f].apply(lambda x:c+"{0:,.0f}".format(x) if abs(x) < 5000 else (c+'{:.2f} k'.format(x/1000) if abs(x) < 1000000 else c+'{:.2f} mln'.format(x/1000000)))
    all_lines['total dividends since '+starting_date_f] = all_lines['total dividends since '+starting_date_f].apply(lambda x:c+"{0:,.0f}".format(x) if abs(x) < 5000 else (c+'{:.2f} k'.format(x/1000) if abs(x) < 1000000 else c+'{:.2f} mln'.format(x/1000000)))
    final = all_lines.sort_values('Date').groupby('year').last()
    return final.transpose()[4:]

# ********************************************************************************
# start of setup section in main section


colu1, colu2, colu3 = st.columns(3)
start_date = colu1.date_input("**What is your portfolio START date?**", datetime.date(int(set_sd[0:4]), int(set_sd[5:7]),int(set_sd[8:])))
set_sd = start_date
#with open('stocks_files/start_date.txt', 'w') as f:
#    print(str(start_date)+'\n', file=f)

ref_ccy = colu2.selectbox('  **Reference currency?**', currency_list)
freq = colu3.selectbox('  **Market data frequency?**', ('daily', 'monthly'), 1)

freq_dict = dict({'daily':'1d', 'monthly':'1mo' })


# function to return beta of the total - dividend included value of the ticker against given stock index
def beta(ticker, index, year, all_years=False):
    ticker=ticker + "_total"
    cova = lm[lm['year']==year][[index+'_delta', ticker+'_delta']].cov()
    if all_years:
        cova = lm[[index+'_delta', ticker+'_delta']].cov()
    return cova[ticker+'_delta'][index+'_delta']/cova[index+'_delta'][index+'_delta']

def betas_df(ticker, year_from, year_to):
    years = list()
    for y in range(year_from, year_to+1):
        years.append(y)
    years.append(str(year_from)+'-'+str(year_to))
    betas=np.zeros((len(indices), len(years)))
    betas_df = pd.DataFrame(betas, columns=years, index = np.array(indices.keys()))

    for y in range(year_from, year_to+1):
        for i in list(indices.keys()):   
            betas_df.at[i, y] = round(beta(ticker, indices[i], y), 2)
        betas_df[y] = betas_df[y].apply(lambda x: "{0:.2f}".format(x))
    for i in list(indices.keys()):   
        betas_df.at[i, str(year_from)+'-'+str(year_to)] = round(beta(ticker, indices[i], y, all_years=True), 2)
    betas_df[str(year_from)+'-'+str(year_to)] = betas_df[str(year_from)+'-'+str(year_to)].apply(lambda x: "{0:.2f}".format(x))
    return betas_df

a = download_tickers_closing_rates(list_all_tickers, freq_dict[freq])
a = download_dividends(a, list_all_tickers, start_date)
ccy = get_currency_of_tickers(stocks)
a = make_orig_ccy_and_EUR(a, stocks)
a = add_portfolio_all_ccies(a, stocks, currency_list)
a = a[a['Date']>=str(start_date)]

# definition of all indices and its labels to be used
indices = {'S&P 500':'^GSPC', 'DAX':'^GDAXI', 'Dow Jones':'^DJI', 'NASDAQ Composite':'^IXIC', 'FTSE 100':'^FTSE', 'CAC 40':'^FCHI', 'BEL 20':'^BFX', 'Euronext 100':'^N100'}
indices = dict(sorted(indices.items()))


ind = download_tickers_closing_rates(list(indices.values()), freq_dict[freq])
ind = ind.fillna(method='ffill')
lm = pd.merge(a, ind, how="left", on=["Date"])      
lm = lm.fillna(method='ffill')


for i in list(stocks)+list(indices.values())+["portfolio_EUR", "portfolio_total_EUR"]:
    if i in list(stocks):
        lm[i+'_total_delta'] = lm[i+'_total'].diff().fillna(0, downcast='infer')/(lm[i].shift(1)) 
    else:
        lm[i+'_delta'] = lm[i].diff().fillna(0, downcast='infer')/(lm[i].shift(1)) 
lm['year'] = lm['Date'].dt.year

# writing outpu of portfolio
st.subheader('Price and dividend earned on the defined portfolio since '+convert_to_datetime(str(start_date)).strftime("%b %d, %Y")+':')
price_plot('portfolio', ref_ccy, st)
st.write(get_portfolio_stat(ref_ccy))   ############################################
st.write()
st.write()

st.subheader('Correlation of the changes: defined portfolio vs. major stock indices')
#st.dataframe(betas_df('portfolio', int(str(start_date)[:4]), int(datetime.datetime.today().strftime('%Y'))-1).style.highlight_max(axis=0))

out01, out02 = st.columns([1, 3])
out01.write("  ")
out01.write("  ")
out01.write("  ")
year_slider0 = out01.slider('**years to draw?**', int(str(start_date)[:4])+1, 2024, (2022, 2023))
out01.write("  ")
out01.write("  ")
out01.write("  ")
out01.write("  ")
sel_index0 = out01.selectbox('**Which Index to compare '+ freq +' changes of your portfolio with?**', list(indices.keys()), 2)
sel_index0 = indices[sel_index0]


#out2.write('correlation of GDAXI and '+s+' during '+str(year_slider))

beta_plot('portfolio_total_EUR', sel_index0, freq, year_slider0[0], year_slider0[1], fg_color=bg_color, bg_color=bg_color, place=out02)
    
# writing output for chosen portfolio component/ticker
st.write('------------------------------------------------------------------')
st.subheader("Lets' have a look at particular portfolio component...")
s = st.selectbox('**Choose portfolio component:**', list(stocks))
st.write(f"Portfolio contains {stocks[s]} stocks of {s} in original currency {ccy[s]}.")        
price_plot(s, ccy[s])

st.write(get_stat(s))
st.write('')
st.write('')
st.write(f'Table of βetas ({s} vs. major world stock indices)',)
st.dataframe(betas_df(s, int(str(start_date)[:4]), int(datetime.datetime.today().strftime('%Y'))-1).style.highlight_max(axis=0))

    
    
out1, out2 = st.columns([1, 3])
out1.write("  ")
out1.write("  ")
out1.write("  ")
year_slider = out1.slider('**Period to draw?**  ', int(str(start_date)[:4])+1, 2024, (2022, 2023))
out1.write("  ")
out1.write("  ")
out1.write("  ")
out1.write("  ")
sel_index = out1.selectbox('**Index to compare '+ freq +' changes with?**', list(indices.keys()), 2)
sel_index = indices[sel_index]

beta_plot(s+"_total", sel_index, freq, year_slider[0], year_slider[1], fg_color=bg_color, bg_color=bg_color, place=out2)
