# #zadanie 1
# print('Zadanie 1\n')
# from sklearn.linear_model import ElasticNet
# import numpy as np
# # Przygotowanie danych
# X = np.array([[1], [2], [3], [4]])
# y = np.array([2, 4, 5, 4])
# # Tworzenie modelu
# alpha = 0.1 # Parametr regularyzacji
# l1_ratio = 0.5 # Proporcja kary L1 w stosunku do kary L2
# model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# # Dopasowanie modelu do danych
# model.fit(X, y)
# # Przewidywanie wartości
# X_test = np.array([[5]])
# y_pred = model.predict(X_test)
# print("Przewidywana wartość dla X_test:", y_pred)
#
# from sklearn.svm import SVR
# # Przygotowanie danych
# X = np.array([[1], [2], [3], [4]])
# y = np.array([2, 4, 5, 4])
# # Tworzenie modelu
# model = SVR(kernel='linear')
# # Dopasowanie modelu do danych
# model.fit(X, y)
# # Przewidywanie wartości
# X_test = np.array([[5]])
# y_pred = model.predict(X_test)
# print("Przewidywana wartość dla X_test:", y_pred)

from sklearn import linear_model

#zadanie 2
print("zadanie 2\n")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# def generate_data(N):
#  #definiowanie listy, która będzie przechowywać dane
#     data = []
#  #generowanie N losowych wierszy danych
#     for _ in range(N):
#         area = random.randint(50, 120)
#         rooms = random.randint(1, 5)
#         floor = random.randint(1, 10)
#         year_of_construction = random.randint(1950, 2022)
#         price = random.randint(150000, 1000000)
#         data.append([area, rooms, floor, year_of_construction, price])
#  # Tworzenie obiektu DataFrame z listy danych
#     df = pd.DataFrame(data, columns=['area', 'rooms', 'floor',
# 'year_of_construction', 'price'])
#     # Zapisanie danych do pliku CSV
#     df.to_csv('appartments.csv', index=False)
#     print(f"Plik 'appartments.csv' został wygenerowany z {N} wierszami danych.")
# generate_data(100)
# from sklearn.metrics import mean_squared_error, r2_score
#po stworzeniu danych będę je odczytywać
# df = pd.read_csv('appartments.csv')
# print(df.head())
# #w regresji bedę łączyć cechy które składają się potem na cenę mieszkania
# x = df[['area', 'rooms', 'floor', 'year_of_construction']]
# y = df['price']
#
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
#
# model = linear_model.LinearRegression()
# model.fit(xTrain, yTrain)
#
# y_pred = model.predict(xTest)
#
# mse_1 = mean_squared_error(yTest, y_pred)
# r2_1 = r2_score(yTest, y_pred)
# print('Wartość MSE wynosi:', mse_1)
# print('Wartośc r2 wynosi:', r2_1)
#
# if r2_1 <0:
#     print('model działa gorzej niż naiwne przewidywanie średniej ceny')
# elif r2_1 == 1:
#     print('model jest bardzo dobry')
# else:
#     print('model nie przewiduje lepiej niż losowe zgadywanie')
# plt.scatter(yTest, yTest, alpha = 0.5, color = 'salmon')
# plt.plot(yTest, y_pred, color='blue', linestyle = '--', linewidth=2)
# plt.xlabel('Rzeczywista cena')
# plt.ylabel('Przewidywana cena')
# plt.title('Przewidywana oraz rzeczywista cena mieszkań')
# plt.show()

# #zadanie 4
# df = pd.read_csv('C:/Users/Magda/Desktop/time_n,temperature,energy_consumpti.csv')
# print(df.head())
#
# df['time_n'] = pd.to_datetime(df['time_n'])
# df['time_n'] = df['time_n'].map(pd.Timestamp.toordinal)
#
#
# x = df[['time_n', 'temperature']]
# y = df['energy_consumption']
#
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
# degree = 2
# model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
# model.fit(xTrain, yTrain)
# y_pred = model.predict(xTest)
# mse_4 = mean_squared_error(yTest, y_pred)
# r_4 = r2_score(yTest, y_pred)
# print("Mse wynosi", mse_4)
# print("R2 wynosi", r_4)
# #to poprawić wyżej
# if r_4 < 0:
#     print('Model działa gorzej niż losowe przewidywanie średniej wartości.')
# elif r_4 > 0.8:
#     print('Model dobrze przewiduje zużycie energii.')
# elif r_4 > 0.5:
#     print('Model ma umiarkowaną skuteczność.')
# else:
#     print('Model słabo przewiduje zużycie energii.')
#
# plt.scatter(yTest, yTest, alpha = 0.5, color = 'yellow')
# plt.plot(yTest, y_pred, color='blue', linestyle = '--', linewidth=2)
# plt.xlabel('Rzeczywiste zużycie energii')
# plt.ylabel('Przewidywane zużycie energii')
# plt.title('Przewidywana oraz rzeczywista cena mieszkań')
# plt.show()

#zadanie 5
print('Zadanie 5\n')
