#zadanie 1
print('Zadanie 1\n')
from sklearn.linear_model import ElasticNet
import numpy as np
# Przygotowanie danych
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 4])
alpha = 0.1 #parametr regularyzacji
l1_ratio = 0.5 #proporcja kary L1 w stosunku do kary L2
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
#dopasowanie modelu do danych
model.fit(X, y)
X_test = np.array([[5]])
y_pred = model.predict(X_test)
print("Przewidywana wartość dla X_test:", y_pred)

from sklearn.svm import SVR
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 4])


#tworzenie modelu
model = SVR(kernel='linear')
#dopasowanie modelu do danych
model.fit(X, y)
#przewidywanie wartości
X_test = np.array([[5]])
y_pred = model.predict(X_test)
print("Przewidywana wartość dla X_test:", y_pred)

from sklearn import linear_model
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
#zadanie 2
print("zadanie 2\n")
import pandas as pd
import random

def generate_data(N):
    data = []
    for _ in range(N):
        area = random.randint(50, 120)  #powierzchnia mieszkania
        rooms = random.randint(1, 5)  #l pokoi
        floor = random.randint(1, 10)  #piętro
        year_of_construction = random.randint(1950, 2022)  #rok budowy

        #żeby cena zależała od powierzchni
        base_price_per_m2 = random.randint(3000, 10000)  # Cena za m²
        price = (area * base_price_per_m2) + (rooms * 2000) + (floor * 5000) + (
                    (year_of_construction - 1950) * 1000) + random.randint(-20000, 20000)

        data.append([area, rooms, floor, year_of_construction, price])

    df = pd.DataFrame(data, columns=['area', 'rooms', 'floor', 'year_of_construction', 'price'])
    df.to_csv('appartments.csv', index=False)

    print(f"Plik 'appartments.csv' został wygenerowany z {N} wierszami danych.")

#generowanie 100 mieszkań
generate_data(100)

from sklearn.metrics import mean_squared_error, r2_score
#po stworzeniu danych będę je odczytywać
df = pd.read_csv('appartments.csv')
print(df.head())
#w regresji bedę łączyć cechy które składają się potem na cenę mieszkania
x = df[['area', 'rooms', 'floor', 'year_of_construction']]
y = df['price']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

y_pred = model.predict(xTest)

mse_1 = mean_squared_error(yTest, y_pred)
r2_1 = r2_score(yTest, y_pred)
print('Wartość MSE wynosi:', mse_1)
print('Wartośc r2 wynosi:', r2_1)

if r2_1 <0:
    print('model działa gorzej niż naiwne przewidywanie średniej ceny')
elif r2_1 == 1:
    print('model jest bardzo dobry')
else:
    print('model nie przewiduje lepiej niż losowe zgadywanie')
plt.scatter(yTest, yTest, alpha = 0.5, color = 'salmon')
plt.scatter(yTest, y_pred, color='blue', linewidth=2)
plt.xlabel('Rzeczywista cena')
plt.ylabel('Przewidywana cena')
plt.title('Przewidywana oraz rzeczywista cena mieszkań')
plt.show()

#zadanie 4
print("Zadanie 4\n")
df = pd.read_csv('C:/Users/Magda/Desktop/time_n,temperature,energy_consumpti.csv')
print(df.head())

df['time_n'] = pd.to_datetime(df['time_n'])
df['time_n'] = df['time_n'].map(pd.Timestamp.toordinal)

x = df[['time_n', 'temperature']]
y = df['energy_consumption']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(xTrain, yTrain)
y_pred = model.predict(xTest)
mse_4 = mean_squared_error(yTest, y_pred)
r_4 = r2_score(yTest, y_pred)
print("Mse wynosi", mse_4)
print("R2 wynosi", r_4)

if r_4 < 0:
    print('Model działa gorzej niż losowe przewidywanie średniej wartości.')
elif r_4 > 0.8:
    print('Model dobrze przewiduje zużycie energii.')
elif r_4 > 0.5:
    print('Model ma umiarkowaną skuteczność.')
else:
    print('Model słabo przewiduje zużycie energii.')

plt.scatter(yTest, yTest, alpha = 0.5, color = 'yellow')
plt.scatter(yTest, y_pred, color='blue', linestyle = '--', linewidth=2)
plt.xlabel('Rzeczywiste zużycie energii')
plt.ylabel('Przewidywane zużycie energii')
plt.title('Przewidywana oraz rzeczywista cena mieszkań')
plt.show()

#zadanie 5
print('Zadanie 5\n')
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Magda/Desktop/time_n,temperature,energy_consumpti.csv')

#konwersja czasu na liczby
df['time_n'] = pd.to_datetime(df['time_n'])
df['time_n'] = df['time_n'].map(pd.Timestamp.toordinal)

#zdefiniowanie zmiennych
x = df[['time_n', 'temperature']]
y = df['energy_consumption']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

#regresja grzbietowa
alpha = 0.1
model_ridge = Ridge(alpha=alpha)
model_ridge.fit(xTrain, yTrain)
y_pred_ridge = model_ridge.predict(xTest)

#regresja liniowa
model_linear = LinearRegression()
model_linear.fit(xTrain, yTrain)
y_pred_linear = model_linear.predict(xTest)

#metody ocena modeli
mse_ridge = mean_squared_error(yTest, y_pred_ridge)
r2_ridge = r2_score(yTest, y_pred_ridge)
print("Regresja grzbietowa - MSE:", mse_ridge, "R2:", r2_ridge)

mse_linear = mean_squared_error(yTest, y_pred_linear)
r2_linear = r2_score(yTest, y_pred_linear)
print("Regresja liniowa - MSE:", mse_linear, "R2:", r2_linear)

plt.scatter(yTest, y_pred_ridge, color='black', label='Regresja grzbietowa',alpha=0.3)
plt.scatter(yTest, y_pred_linear, color='pink', label='Regresja liniowa', alpha=0.3)
plt.plot(yTest, yTest, color='black', linestyle='--', label='Idealne dopasowanie')  # Linia y=x
plt.xlabel("Rzeczywiste wartości zużycia energii")
plt.ylabel("Przewidywane wartości zużycia energii")
plt.title("Porównanie rzeczywistych i przewidywanych wartości")
plt.legend()
plt.show()


#zadanie 6
print("Zadanie 6\n")
from sklearn.svm import SVR

df = pd.read_csv("wiek,BMI,cisnienie_krwi,poziom_gluk.csv")
print(df.head(5))
x = df[['wiek', 'BMI', 'cisnienie_krwi', 'poziom_glukozy', 'cholesterol', 'kretynina']]
y = df['czas_przezycia']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)
for i in [0.1,0.5, 1, 10]:
    model_SVR = SVR(kernel='linear', C=i)
    model_SVR.fit(xTrain, yTrain)
    y_pred = model_SVR.predict(xTest)

    model_SVR = SVR(kernel='linear')
    mse_svr = mean_squared_error(yTest, y_pred)
    r2_svr = r2_score(yTest, y_pred)
    print("Wyniki dla SVR:",'MSE:',  mse_svr, "R2:", r2_svr)

'''Widzimy, że model osiągnął najlepsze rezutaty dla c=10,jednak tak duża wartość tego parametru może doprowadzić do przeuczenia więc weźmiemy 1'''
model_SVR = SVR(kernel='linear', C=1)
model_SVR.fit(xTrain, yTrain)
y_pred = model_SVR.predict(xTest)

model_SVR = SVR(kernel='linear')
mse_svr = mean_squared_error(yTest, y_pred)
r2_svr = r2_score(yTest, y_pred)
print("Wyniki dla SVR:", 'MSE:', mse_svr, "R2:", r2_svr)
alpha = 0.1
model_grzbietowa = Ridge(alpha=alpha)
model_grzbietowa.fit(xTrain, yTrain)
y_pred_grzbiet = model_grzbietowa.predict(xTest)

mse = mean_squared_error(yTest, y_pred_grzbiet)
r2_grzebietowa = r2_score(yTest, y_pred_grzbiet)
print("Wyniki dla regresji grzbietowej: mse", mse, "r2", r2_grzebietowa)

model_regresja = linear_model.LinearRegression()
model_regresja.fit(xTrain, yTrain)

y_pred_regresja = model_regresja.predict(xTest)

mse_regresja = mean_squared_error(yTest, y_pred_regresja)
r2_regresja = r2_score(yTest, y_pred_regresja)

print("Wyniki dla klasycznej regresji: mse", mse_regresja,"r_2", r2_regresja)
'''Klasyczne metody regresji lepiej dopasowywują się do danych-niższe mse wychodzi'''
plt.scatter(yTest, y_pred_grzbiet, color='black', label='Regresja grzbietowa',alpha=0.3)
plt.scatter(yTest, y_pred_regresja, color='pink', label='Regresja liniowa', alpha=0.3)
plt.scatter(yTest, y_pred, color = 'yellow', label = 'Regresja SVR', alpha=0.3)
plt.plot(yTest, yTest, color='red', label='Idealne dopasowanie')
plt.xlabel("Rzeczywiste wartości czasu przeżycia")
plt.ylabel("Przewidywane wartości czasu przeżycia")
plt.title("Porównanie rzeczywistych i przewidywanych wartości dla różnych modeli")
plt.legend()
plt.show()
'''Wykres potwierdza powyższe obliczenie, jednak nie jest to bardzo widoczne'''
