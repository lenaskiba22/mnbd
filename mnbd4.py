import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
#zadanie 2
df = pd.read_csv('C:/Users/Magda/Desktop/road_traffic.csv')
df.head()
'''sprawdzanie brakujących wartości'''
brakujace = df.isnull().sum()
print('Brakujące wartości', brakujace)
'''Dane nie zawierają brakujących wartości'''

#zadanie 3

'''Wykorzystanie funkcji z pdfa do generownaia danych'''


def generate_weather_data(num_stations, num_days):
    """
    Funkcja generuje przykładowe dane meteorologiczne dla wielu stacji pomiarowych i dni
    i zapisuje je do pliku CSV.
    """
    temperatures1 = np.array([-2, 0, 5, 12, 18, 23, 26, 25, 21, 15, 8, 2])
    np.random.seed(0)
    dates = pd.date_range(start='2023-01-01', periods=num_days)
    station_ids = ['Station_' + str(i) for i in range(1, num_stations + 1)]
    data = {station: [] for station in station_ids}

    for day in range(num_days):
        month = dates[day].month - 1  # Indeksowanie od zera
        temperature1 = temperatures1[month]

        for station in station_ids:
            temperature = temperature1 + np.random.uniform(low=-2,
                                                           high=2) if station == 'Station_1' else temperature1 + np.random.uniform(
                low=-4, high=4)
            if day > 0 and np.random.rand() < 0.05:
                temperature += np.random.uniform(low=-10, high=10)
            if np.random.rand() < 0.1:  # wprowadzenie brakujących wartości
                temperature = np.nan
            data[station].append(temperature)

    df = pd.DataFrame(data)
    df['Date'] = dates
    df = df[['Date'] + station_ids]
    df.to_csv('weather_data.csv', index=False)


generate_weather_data(5, 15)
df_2 = pd.read_csv('weather_data.csv')
print("Pierwsze kilka wierszy", df_2.head())
print('Struktury danych')
df_2.info()


def load_and_clean_data(file_path):
    """Wczytuje dane z pliku CSV i obsługuje brakujące wartości poprzez interpolację."""
    df_2 = pd.read_csv(file_path, parse_dates=['Date'])
    df_2.set_index('Date', inplace=True)
    # interpolacja liniowa dla brakujących wartości
    df_2.interpolate(method='linear', inplace=True)
    return df_2


df_2 = load_and_clean_data('weather_data.csv')


def prognozowanie(df, station):
    """Prognozowanie temperatury za pomocą interpolacji splajnów B-sklejanych w prostszy sposób."""
    x = np.arange(len(df))
    y = df[station].interpolate().values
    cs = CubicSpline(x, y)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[station], 'o', label='Oryginalne dane')
    plt.plot(df.index, cs(x), '-', label='Interpolacja splajnami')
    plt.xlabel('Data')
    plt.ylabel('Temperatura (°C)')
    plt.title(f'Prognozowanie temperatury dla {station}')
    plt.legend()
    plt.show()


prognozowanie(df_2, 'Station_1')

#zadanie 4
print("Zadanie 4\n")
energy_df = pd.read_csv('energy.csv')
print(energy_df .head())
energy_df["timestamp"] = pd.to_datetime(energy_df["timestamp"])

#przekształcenie czasu na wartości liczbowe
energy_df["time_numeric"] = (energy_df["timestamp"] - energy_df["timestamp"].min()).dt.total_seconds() / 3600


subset = energy_df.iloc[:1000]

x_values = subset["time_numeric"].values
#zużycie energii
y_values = subset["load"].values  #zużycie energii

#interpolacja wielomianowa (stopień dopasowany do danych)
poly_fit = np.polyfit(x_values, y_values, deg=5)
poly_interp = np.poly1d(poly_fit)

#interpolacja splajnami naturalnymi
spline_interp = interp1d(x_values, y_values, kind='cubic')

#przedłużenie na przyszłe punkty-prognoza na 24 godziny
x_future = np.linspace(x_values.min(), x_values.max() + 48, 500)
y_poly = poly_interp(x_future)
y_spline = spline_interp(x_future[:-48])  # Splajn nie obsługuje przedłużania


plt.figure(figsize=(12, 6))
plt.scatter(x_values, y_values, color='black', label="Rzeczywiste dane", alpha=0.5)
plt.plot(x_future, y_poly, 'b-', label="Interpolacja wielomianowa")
plt.plot(x_future[:-48], y_spline, 'g-.', label="Interpolacja splajnami")

plt.xlabel("Godziny od początku pomiarów")
plt.ylabel("Zużycie energii (MW)")
plt.title("Porównanie metod interpolacji w prognozowaniu zużycia energii")
plt.legend()
plt.show()

#zadanie 5
print('Zadanie 5\n')
df = pd.read_csv('C:/Users/Magda/Desktop/stocks_data.csv')
print(df.head())

'''Wybieram dane dla jednej spółki jaką jest apple'''
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
x_values = np.arange(len(df))  # Indeksy dni jako wartości x
y_values = df["NVDA"].values

#szacowanie pochodnych jako różnicowych przybliżeń
y_derivatives = np.gradient(y_values, x_values)

spline = CubicHermiteSpline(x_values, y_values, y_derivatives)
x_interp = np.linspace(x_values.min(), x_values.max(), 500)
y_interp = spline(x_interp)

#lokalne ekstrema-scipy.signal import argrelextrema
local_maxima = argrelextrema(y_interp, np.greater)[0]
local_minima = argrelextrema(y_interp, np.less)[0]

plt.figure(figsize=(12, 6))
plt.plot(x_values, y_values, 'bo', label="Dane rzeczywiste NVDA")
plt.plot(x_interp, y_interp, 'r-', label="Interpolacja Hermite'a")

plt.scatter(x_interp[local_maxima], y_interp[local_maxima], color='g', label="Lokalne maksima", marker='^', s=100)
plt.scatter(x_interp[local_minima], y_interp[local_minima], color='orange', label="Lokalne minima", marker='v', s=100)

plt.xlabel("Dzień")
plt.ylabel("Cena akcji NVDA")
plt.title("Interpolacja kubiczna Hermite'a dla cen NVDA")
plt.legend()
plt.show()

#zadanie 6
print('Zadanie 6\n')
df_6 = pd.read_csv("C:/Users/Magda/Desktop/road_traffic.csv")
print(df_6.head())
#sprawdzenie brakujących wartości
missing_values = df_6.isnull().sum()

#konwersja kolumny daty i czasu na format datetime
df_6["Datetime"] = pd.to_datetime(df_6["Date"] + " " + df_6["Time"], format="mixed")


#wybór danych dla jednego czujnika i kierunku
sensor_data = df_6[(df_6["countlineName"] == "S16_PerneRoad_CAM003") &
                         (df_6["direction"] == "in")][["Datetime", "Car"]]

#sortowanie według czasu
sensor_data = sensor_data.sort_values("Datetime")


#przygotowanie wartości do interpolacji
x_values = np.arange(len(sensor_data))
y_values = sensor_data["Car"].values

#interpolacja kubiczna Hermite’a
y_derivatives = np.gradient(y_values, x_values)
hermite_spline = CubicHermiteSpline(x_values, y_values, y_derivatives)

#interpolacja wielomianowa
poly_fit = np.polyfit(x_values, y_values, deg=5)
poly_interp = np.poly1d(poly_fit)

#interpolacja splajnami naturalnymi
spline_interp = interp1d(x_values, y_values, kind='cubic')

#przedłużenie na przyszłe punkty (prognoza)
x_future = np.linspace(x_values.min(), x_values.max() + 24, 500)  # 24 dodatkowe godziny
y_hermite = hermite_spline(x_future)
y_poly = poly_interp(x_future)
y_spline = spline_interp(x_future[:-24])  # Splajn nie obsługuje przedłużania


plt.figure(figsize=(12, 6))
plt.scatter(x_values, y_values, color='black', label="Rzeczywiste dane", alpha=0.5)
plt.plot(x_future, y_hermite, 'r--', label="Interpolacja Hermite'a")
plt.plot(x_future, y_poly, 'b-', label="Interpolacja wielomianowa")
plt.plot(x_future[:-24], y_spline, 'g-.', label="Interpolacja splajnami")

plt.xlabel("Godziny od początku pomiarów")
plt.ylabel("Liczba samochodów")
plt.title("Porównanie metod interpolacji w prognozowaniu ruchu")
plt.legend()
plt.show()

'''Interpolacja kubiczna Hermitea dobrze odwzorowuje zmienność danych, ale może nie być stabilna na końcach przedziału'''
'''Interpolacja splajnami naturalnymi – dobrze dopasowuje się do istniejących danych, ale nie pozwala na rozszerzenie prognozy poza dostępne punkty'''
'''nterpolacja wielomianowa – może prowadzić do nadmiernych oscylacji przy dłuższych prognozach'''