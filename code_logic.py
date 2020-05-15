# Libraries Importing
import fxcmpy
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys, os
import plotly.offline
import gunicorn
import _thread
import itertools
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

var_bars = ""
var_currency = 47
var_period = "m1"
var_market_value = 2
var_listing = []

token = '58f6b3b0d2fd0411de1d85d418345291d375e5a5'


def set_data(bars, currency, period, market_value):
    global var_bars
    global var_currency
    global var_period
    global var_market_value

    var_bars = bars
    var_currency = int(currency) #+1
    var_period = period
    var_market_value = int(market_value)


def get_currencies():
    global var_listing
    global token
    listing = []
    try:
        con1 = fxcmpy.fxcmpy(access_token=token, log_level='error', log_file=None)
        instruments = con1.get_instruments_for_candles()
    except:
        print('Exception in fxcmpy connect get_currencies()')
    # Currency Colection
    cnt = 0
    for i in range(int(len(instruments) / 4)):
        data = instruments[i * 4:(i + 1) * 4]
        for d in data:
            listing.append([cnt, d])
            cnt=cnt+1

    con1.close()
    var_listing = listing
    return listing


def Data_Collect():
    global var_bars
    global var_currency
    global var_period
    global var_market_value
    global token
    global var_listing
    global data
    print('Function [Data Collect] param:', 'Currency:', var_currency, 'Period:',var_period)
    # Creating list for Data collection
    con = fxcmpy.fxcmpy(access_token=token, log_level='error', log_file=None)
    """Listing = ['2']
    instruments = con.get_instruments_for_candles()
    # Currency Colection
    for i in range(int(len(instruments) / 4)):
        data = instruments[i * 4:(i + 1) * 4]
        for d in data:
            Listing.append(d)

    # input area
    i = 0
    for x in Listing:
        if i != 0:
            print(f'{i}. {x}')
        i += 1
    #choose = int(input("Select the market Currency by number: "))"""
    choose = var_currency

    #Period = input("Enter the Period: ")
    Period = var_period

    value = ['askopen', 'askclose', 'asklow', 'askhigh', 'bidopen', 'bidclose', 'bidhigh', 'bidlow']

    """i = 0
    for x in value:
        print(f"{i}. {x}")
        i += 1"""

    #option = int(input("Choose the Market value: "))
    option = var_market_value

    # Getting market prices by period
    print('Chosen Currency:', var_listing[choose][1])
    data = con.get_candles(var_listing[choose][1], period=Period, number=1000)  # daily data
    #data = con.get_candles("USD/CAD", period=Period, number=1000)  # daily data
    con.close()
    print(data)
    #data.to_csv('Data.csv', encoding='utf-8', index=True)
    # Fibonic_Series_Graph(data, value[option])
    CandleStick_Graph()


def show_in_window(fig):
    # Webview Candlestick Graph
    file_path = os.getcwd() + "/templates/name.html"
    plotly.offline.plot(fig, filename=file_path, auto_open=False)
    #app = QApplication(sys.argv)
    #web = QWebEngineView()
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    #web.load(QUrl.fromLocalFile(file_path))
    #web.show()
    #sys.exit(app.exec_())
    print('Return / Task completed from code_logic.py')
    return


def CandleStick_Graph():
    global var_bars
    global loc2
    global data
    #df = pd.read_csv('Data.csv')
    df = pd.DataFrame(data)
    #df2 = pd.read_csv('Data.csv')
    df = df[::-1]
    df.columns = ['o', 'h', 'l', 'c', 'o2', 'h2', 'l2', 'c2', 'v']
    df.drop(['v'], axis=1, inplace=True)
    df['mean'] = df.mean(axis=1)
    column = int(var_bars)

    sig = np.array(df.iloc[0:column, [8]])
    sig = sig[::-1]
    sig = sig.reshape(1, column)


    sig = sig - (np.mean(sig))
    N = (sig.shape[1])
    ft = np.fft.fft(sig)
    yy = int(np.round((N / 2) - 3))
    new_ft = np.delete(ft, np.s_[yy::1])
    mags = abs(new_ft)
    phases = np.angle(new_ft)
    dc_mag = mags[0]
    mags = np.delete(mags, np.s_[0:1:1])
    phases = np.delete(phases, np.s_[0:1:1])
    sorted_mags = np.sort(mags)
    sorted_mags = sorted_mags.reshape(1, len(sorted_mags))
    sorted_indices = np.argsort(mags)
    sorted_indices = sorted_indices.reshape(1, len(sorted_indices))
    sorted_mags = np.fliplr(sorted_mags)
    sorted_indices = np.fliplr(sorted_indices)
    sorted_phases = phases[sorted_indices]
    synth_op = np.zeros((N, 1), dtype=np.int)
    sig = sig - dc_mag
    n = np.arange(0, N, 1)
    extra_pos = sorted_mags[0, 0] / N * 2
    sig_pos = np.insert(sig, 0, extra_pos)
    sig_neg = np.insert(sig, 0, -extra_pos)
    ylim_max = max(sig_pos) * 1.05
    ylim_min = min(sig_neg) * 1.05
    ylims = np.array([ylim_min, ylim_max])
    # print(ylims)
    # plt.subplot(3,1,3)
    sig1 = sig.reshape(N, 1)
    # ax = plt.gca()
    # plt.plot(sig1)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("amplitude")
    # ax.set_title("Example Signal")
    # ax.set_ylim(ylim_min, ylim_max)
    # plt.show()
    # pause()
    n_s_i = sorted_indices + 1
    mm = max(max(sorted_mags / 100))
    significant_freqs = np.where(sorted_mags > mm)
    significant_freqs_len = (len(significant_freqs[1]))
    ran = n_s_i[0, :significant_freqs_len]
    freq_vals_to_display = (max(ran)) + 1
    NN = len(mags)
    n_s_i = sorted_indices + 1
    colors = itertools.cycle(["r", "g", "k", "b", "m"])
    #    plt.figure()
    
    for i in range(1):
        
        omega = 2 * np.pi * (n_s_i[0, i]) / N
        omega2 = 2 * np.pi * (n_s_i[0, 2]) / N
        omega3 = 2 * np.pi * (n_s_i[0, 3]) / N
        omega4 = 2 * np.pi * (n_s_i[0, 4]) / N
        omega5 = 2 * np.pi * (n_s_i[0, 5]) / N
        omega6 = 2 * np.pi * (n_s_i[0, 6]) / N
        omega7 = 2 * np.pi * (n_s_i[0, 7]) / N
        omega8 = 2 * np.pi * (n_s_i[0, 8]) / N

        sinusoid = np.cos(n * omega + sorted_phases[0, i]) * sorted_mags[0, i] / N * 2
        sinusoid = sinusoid.reshape(len(sinusoid), 1)

        sinusoid2 = np.cos(n * omega2 + sorted_phases[0, 2]) * sorted_mags[0, 2] / N * 2
        sinusoid2 = sinusoid2.reshape(len(sinusoid2), 1)

        sinusoid3 = np.cos(n * omega3 + sorted_phases[0, 3]) * sorted_mags[0, 3] / N * 2
        sinusoid3 = sinusoid3.reshape(len(sinusoid3), 1)

        sinusoid4 = np.cos(n * omega4 + sorted_phases[0, 4]) * sorted_mags[0, 4] / N * 2
        sinusoid4 = sinusoid4.reshape(len(sinusoid4), 1)

        sinusoid5 = np.cos(n * omega5 + sorted_phases[0, 5]) * sorted_mags[0, 5] / N * 2
        sinusoid5 = sinusoid5.reshape(len(sinusoid5), 1)

        sinusoid6 = np.cos(n * omega6 + sorted_phases[0, 6]) * sorted_mags[0, 6] / N * 2
        sinusoid6 = sinusoid6.reshape(len(sinusoid6), 1)

        sinusoid7 = np.cos(n * omega7 + sorted_phases[0, 7]) * sorted_mags[0, 7] / N * 2
        sinusoid7 = sinusoid7.reshape(len(sinusoid7), 1)

        sinusoid8 = np.cos(n * omega8 + sorted_phases[0, 8]) * sorted_mags[0, 8] / N * 2
        sinusoid8 = sinusoid8.reshape(len(sinusoid8), 1)

        if sorted_mags[0, i] < pow(10, -6):
            break

    synth_op = sinusoid + sinusoid2 + sinusoid3 + sinusoid4 + sinusoid5 + sinusoid6 + sinusoid7 + sinusoid8
    fig = plt.figure(figsize=(10, 9))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    plt.plot(sig1)
    plt.plot(synth_op)
    ax.set_xlabel("Time")
    ax.set_ylabel("amplitude")
    if i == 0:
        plt.title('signal and sinusoid shown in lower plot')
    else:
        plt.title("{} {} {}".format(loc2, ' signal and ', '8', ' sinusoids shown in middle plot added together.'))
    plt.subplot(3, 1, 2)

    plt.plot(sinusoid, color=next(colors))
    plt.plot(sinusoid2, color=next(colors))
    plt.plot(sinusoid3, color=next(colors))
    plt.plot(sinusoid4, color=next(colors))
    plt.plot(sinusoid5, color=next(colors))
    #plt.plot(sinusoid6, color=next(colors))
    #plt.plot(sinusoid7, color=next(colors))
    #plt.plot(sinusoid8, color=next(colors))
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("amplitude")
    plt.subplot(3, 1, 3)
    plt.plot(sinusoid2, color=next(colors))
    plt.plot(sinusoid3, color=next(colors))
    plt.plot(sinusoid4, color=next(colors))
    plt.plot(sinusoid5, color=next(colors))
    #plt.plot(sinusoid6, color=next(colors))
    #plt.plot(sinusoid7, color=next(colors))
    #plt.plot(sinusoid8, color=next(colors))
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("amplitude")    
    plotly_fig = tls.mpl_to_plotly(fig)
    show_in_window(plotly_fig)


def Fibonic_Series_Graph(data, value):
    price = []
    for x in data[value]:
        price.append(x)

    # Getting min and max values of the dataframe
    price_max = max(price)
    price_min = min(price)

    print(f"Maximum Price: {price_max}")
    print(f"Minimum Price: {price_min}")

    # Graph ploting
    fig, ax = plt.subplots()

    ax.plot(data, color='black')

    # Fibonacci Levels considering original trend as upward move
    diff = price_max - price_min

    level1 = price_max - 0.236 * diff
    level2 = price_max - 0.382 * diff
    level3 = price_max - 0.618 * diff

    print("Level", "Price")
    print("0 ", price_max)
    print("0.236", level1)
    print("0.382", level2)
    print("0.618", level3)
    print("1 ", price_min)

    # Coloring the points
    ax.axhspan(level1, price_min, alpha=0.4, color='lightsalmon')
    ax.axhspan(level2, level1, alpha=0.5, color='palegoldenrod')
    ax.axhspan(level3, level2, alpha=0.5, color='palegreen')
    ax.axhspan(price_max, level3, alpha=0.5, color='powderblue')

    # ploting on x and y axis
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend(loc=2)
    plt.show()


# Main Area
if __name__ == '__main__':
    print("Main")
    # You can add new Token here
    #token = '58f6b3b0d2fd0411de1d85d418345291d375e5a5'
    # Login in the Site with Socket
    #con = fxcmpy.fxcmpy(access_token=token, log_level='error', log_file=None)
    #Data_Collect()















