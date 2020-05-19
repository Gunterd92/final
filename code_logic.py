# Libraries Importing
import fxcmpy
import pandas as pd
import numpy as np
import datetime
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
Sun = 0
Moon = 0
Mercury = 0
Venus = 0
Mars = 0
Ceres = 0
Jupiter = 0
Saturn = 0
Uranus = 0 
Neptune = 0
Pluto = 0
Eris = 0
marketname = ""

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

def planetarycalculation(value, multiplier, high, low):
	planetArray = []
	NewValue = value * multiplier
	Increment = multiplier * 360
	NewValue=0
	while NewValue < high + Increment/3:
		if NewValue > low - Increment/3:
			planetArray.append(NewValue)
		NewValue += Increment
	return planetArray


def SetPlanetDegrees():
    global Sun
    global Moon 
    global Mercury
    global Venus
    global Mars
    global Ceres
    global Jupiter
    global Saturn
    global Uranus
    global Neptune
    global Pluto
    global Eris

    date_str = str(datetime.datetime.now().month) + "/" + str(datetime.datetime.now().day) + "/" + str(datetime.datetime.now().year)


    date_split = date_str.split('/')
    df = pd.read_excel('planet.xlsx', sheet_name='Sheet1')
    date_obj = datetime.datetime(int(date_split[2]), int(date_split[0]), int(date_split[1]), 0, 0)
    try:
        data_date = list(df[date_obj])
        print(data_date)
    except KeyError:
        print('No data found.')
        data_date = [0]*12

    Sun = data_date[0]
    Moon = data_date[1]
    Mercury = data_date[2]
    Venus = data_date[3]
    Mars = data_date[4]
    Ceres = data_date[5]
    Jupiter = data_date[6]
    Saturn = data_date[7]
    Uranus = data_date[8]
    Neptune = data_date[9]
    Pluto = data_date[10]
    Eris = data_date[10]


def Data_Collect():
    global var_bars
    global var_currency
    global var_period
    global var_market_value
    global token
    global var_listing
    global data
    global marketname
    print('Function [Data Collect] param:', 'Currency:', var_currency, 'Period:',var_period)
    # Creating list for Data collection
    con = fxcmpy.fxcmpy(access_token=token, log_level='error', log_file=None)

    #choose = int(input("Select the market Currency by number: "))"""
    choose = var_currency

    #Period = input("Enter the Period: ")
    Period = var_period

    value = ['askopen', 'askclose', 'asklow', 'askhigh', 'bidopen', 'bidclose', 'bidhigh', 'bidlow']
    #option = int(input("Choose the Market value: "))
    option = var_market_value
    # Getting market prices by period
    print('Chosen Currency:', var_listing[choose][1])
    data = con.get_candles(var_listing[choose][1], period=Period, number=1000)  # daily data
    #data = con.get_candles("USD/CAD", period=Period, number=1000)  # daily data
    con.close()
    print(data)
    # Fibonic_Series_Graph(data, value[option])

    marketname = var_listing[choose][1]
    CandleStick_Graph()


def show_in_window(fig):
    # Webview Candlestick Graph
    file_path = os.getcwd() + "/templates/name.html"
    plotly.offline.plot(fig, filename=file_path, auto_open=False)
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    print('Return / Task completed from code_logic.py')
    return


def CandleStick_Graph():
    global var_bars
    global var_listing
    global data
    global Sun
    global Moon 
    global Mercury
    global Venus
    global Mars
    global Ceres
    global Jupiter
    global Saturn
    global Uranus
    global Neptune
    global Pluto
    global Eris
    global marketname

    sun = []
    moon = []
    mercury = []
    venus = [] 
    mars = []
    ceres = []
    jupiter = []
    saturn = []
    uranus = []
    neptune = []
    pluto = []
    eris =[]
    mult = 0.0
    highs = 0
    lows = 0


    df = pd.DataFrame(data)
    df = df[::-1]
    df.columns = ['o', 'h', 'l', 'c', 'o2', 'h2', 'l2', 'c2', 'v']
    df.drop(['v'], axis=1, inplace=True)
    df['mean'] = df.mean(axis=1)
    column = int(var_bars)

    mean = np.array(df.iloc[0:column, [8]])
    mean = mean[::-1]
    sig = np.array(df.iloc[0:column, [8]])
    sig = sig[::-1]
    sig = sig.reshape(1, column)

    #Main calculation
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
    sig1 = sig.reshape(N, 1)
    n_s_i = sorted_indices + 1
    mm = max(max(sorted_mags / 100))
    significant_freqs = np.where(sorted_mags > mm)
    significant_freqs_len = (len(significant_freqs[1]))
    ran = n_s_i[0, :significant_freqs_len]
    freq_vals_to_display = (max(ran)) + 1
    NN = len(mags)
    n_s_i = sorted_indices + 1
    colors = itertools.cycle(["r", "g", "k", "b", "m"])

    #set planetary values
    SetPlanetDegrees()

    #set multiplier
    if mean.max() < 3:
        mult = .0001
    else: mult = .01
     
    #set highs and lows
    highs = mean.max()
    lows = mean.min()

    sun = planetarycalculation(Sun, mult, highs, lows)
    moon = planetarycalculation(Moon, mult, highs, lows)
    mercury = planetarycalculation(Mercury, mult, highs, lows)
    venus = planetarycalculation(Venus, mult, highs, lows)
    mars = planetarycalculation(Mars, mult, highs, lows)
    ceres = planetarycalculation(Ceres, mult, highs, lows)
    jupiter = planetarycalculation(Jupiter, mult, highs, lows)
    saturn = planetarycalculation(Saturn, mult, highs, lows)
    uranus = planetarycalculation(Uranus, mult, highs, lows)
    neptune = planetarycalculation(Neptune, mult, highs, lows)
    pluto = planetarycalculation(Pluto, mult, highs, lows)
    eris = planetarycalculation(Eris, mult, highs, lows)
    print("highs ", highs)
    print("lows ", lows)
    print(sun)
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

    #charting
    fig = plt.figure(figsize=(10, 11))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    plt.plot(mean)
    xaxis = []
    newsun = []
    newmoon = []
    newmercury = []
    newvenus = [] 
    newmars = []
    newceres = []
    newjupiter = []
    newsaturn = []
    newuranus = []
    newneptune = []
    newpluto = []
    neweris =[]

    #x axis
    for i in range(0,len(mean)):
        xaxis.append(i)
        newsun.append(0)
        newmoon.append(0)
        newmercury.append(0)
        newvenus.append(0)
        newmars.append(0)
        newceres.append(0)
        newjupiter.append(0)
        newsaturn.append(0)
        newuranus.append(0)
        newneptune.append(0)
        newpluto.append(0)
        neweris.append(0)






    #print(sun,moon,mercury,venus,mars,cere)
    if len(sun)>0:
        for p in range (0, len(sun)):
            for f in range(0, len(xaxis)):
                newsun[f]=sun[p]
            ax.plot(xaxis, newsun)

    if len(moon)>0:
        for p in range (0, len(moon)):
            for f in range(0, len(xaxis)):
                newmoon[f]=moon[p]
            ax.plot(xaxis, newmoon)

    if len(mercury)>0:
        for p in range (0, len(mercury)):
            for f in range(0, len(xaxis)):
                newmercury[f]=mercury[p]
            ax.plot(xaxis, newmercury)

    if len(venus)>0:
        for p in range (0, len(venus)):
            for f in range(0, len(xaxis)):
                newvenus[f]=venus[p]
            ax.plot(xaxis, newvenus)

    if len(mars)>0:
        for p in range (0, len(mars)):
            for f in range(0, len(xaxis)):
                newmars[f]=mars[p]
            ax.plot(xaxis, newmars)

    if len(ceres)>0:
        for p in range (0, len(ceres)):
            for f in range(0, len(xaxis)):
                newceres[f]=ceres[p]
            ax.plot(xaxis, newceres)

    if len(jupiter)>0:
        for p in range (0, len(jupiter)):
            for f in range(0, len(xaxis)):
                newjupiter[f]=jupiter[p]
            ax.plot(xaxis, newjupiter)

    if len(saturn)>0:
        for p in range (0, len(saturn)):
            for f in range(0, len(xaxis)):
                newsaturn[f]=saturn[p]
            ax.plot(xaxis, newsaturn)

    if len(uranus)>0:
        for p in range (0, len(uranus)):
            for f in range(0, len(xaxis)):
                newuranus[f]=uranus[p]
            ax.plot(xaxis, newuranus)

    if len(neptune)>0:
        for p in range (0, len(neptune)):
            for f in range(0, len(xaxis)):
                newneptune[f]=neptune[p]
            ax.plot(xaxis, newneptune)

    if len(pluto)>0:
        for p in range (0, len(pluto)):
            for f in range(0, len(xaxis)):
                newpluto[f]=pluto[p]
            ax.plot(xaxis, newpluto)

    if len(eris)>0:
        for p in range (0, len(eris)):
            for f in range(0, len(xaxis)):
                neweris[f]=eris[p]
            ax.plot(xaxis, neweris)



    #plt.plot(synth_op)
    ax.set_xlabel("Time")
    ax.set_ylabel("amplitude")
    if i == 0:
        plt.title('signal and sinusoid shown in lower plot')
    else:
        plt.title('time cycle analysis')
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

# Main Area
if __name__ == '__main__':
    print("Main")
    # You can add new Token here
    #token = '58f6b3b0d2fd0411de1d85d418345291d375e5a5'
    # Login in the Site with Socket
    #con = fxcmpy.fxcmpy(access_token=token, log_level='error', log_file=None)
    #Data_Collect()