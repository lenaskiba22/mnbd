import numpy as np
#zadanie 2
print("Zadanie 2")
x = np.array([1,2,3,4,5,6,7,8,9,10])
minimum = np.min(x)
maximum = np.max(x)
srednia = np.mean(x)
odchylenie = np.std(x)
print("Minumim to:", minimum, "Maximim to:", maximum)
print("Średnia to", srednia, "Odchylenei standardowe to", odchylenie)

#zadanie 3
print("Zadanie 3")
tablica_3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(tablica_3)
#print(tablica_3[0][1])
print(tablica_3[:2, 1:])

#zadanie 4
print("Zadnaie 4\n")
tablica_4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(tablica_4)
tablica4_4 = np.array_split(tablica_4, 2)
print(tablica4_4)#z array

#bez array ale nie wiem czy tak można
for i in tablica4_4:
    print(i.tolist())
transponowana = np.transpose(tablica4_4)
print(transponowana)

#zadanie 5
print("Zadanie 5")
tablica_5 = np.random.randint(5,size = (2,5))
tablica_5_2 = np.random.randint(5,size = (2,5))
print("Tablica 1\n",tablica_5)
print("Tablica 2\n",tablica_5_2)
print("Suma \n",tablica_5+tablica_5_2)
print("Pomnożona przez skalar tablica 1", 3*tablica_5)

#zadanie 6
print("Zadanie 6\n")
tablica6 =np.array([[1,2,3], [4,5,6], [7,8,9]])
print(tablica6)
wymiar_do_dodania = np.array([1,4,5])
nowe = np.column_stack((tablica6,wymiar_do_dodania))
print("Nowa tablica\n", nowe)
mnozenie = np.array([10,2,6,8])
print("Nowa tablica pomnożona \n", nowe*mnozenie)

#zadanie7
print("Zadanie 7\n")
tablice_7 = np.random.randint(1,100,(100,))
print(tablice_7)
suma_2 = np.sum(tablice_7)
srednia_2 = np.mean(tablice_7)
standarowe = np.std(tablice_7)
tablica6 =np.array([[1,2,3], [4,5,6], [7,8,9]])
print(tablica6)
wymiar_do_dodania = np.array([1,4,5])
nowe = np.column_stack((tablica6,wymiar_do_dodania))
print("Nowa tablica\n", nowe)
mnozenie = np.array([10,2,6,8])
print("Nowa tablica pomnożona \n", nowe*mnozenie)
#sumowanie i mnożenie skumulowane
sumowanie_skumulowane = np.cumsum(tablica6)
mnozenie_skumulowane = np.cumprod(tablica6)
print("Sumowanie", sumowanie_skumulowane)
print("Mnozenie", mnozenie_skumulowane)
#zadanie8
print("Zadanie 8\n")
tablice_7 = np.random.randint(1,100,(100,))
print(tablice_7)
posorotwana = np.sort(tablice_7)
print(posorotwana)

#wyszukiwanie biarne
szukana = 20
poszukiwana = np.searchsorted(tablice_7, szukana)
if tablice_7[poszukiwana] == szukana:
    print(f"Wartość {szukana} znaleziona na indeksie {szukana}.")
else:
    print("Wartość nie została znaleziona w tablicy.")

#zadanie 9

import pandas as pd
print("Zadanie 9")
df = pd.read_csv("C:\\Users\\Magda\\Desktop\\netflix_users.csv")
#wiersze i kolumny
# print("Liczba wierszy", df.shape[0])
# print("Liczba kolumn", df.shape[1])
# #pierwsze kolumny
# print(df.head())

#zadanie 10
print("Zadanie 10\n")
kolumny  = df[['Name' ,'Age', 'Favorite_Genre']]
print(kolumny.head())
#
wiek  = df[(df['Age'] == 30) & (df['Favorite_Genre'] == 'Documentary')]
print('Osoby w wieku 30 lat lubiące dokumenty to:',wiek)

imiona = wiek['Name']
print('Imiona tych osób',imiona)

#zadanie 11
#print("Zadanie 11\n")
#usuwanie brakujących wartości
df = df.dropna()
#liczba duplikatów
print("Liczba duplikatów:", df.duplicated().sum())
#gdyby były duplikaty to
#df = df.drop_duplicates()
#typy danych
print('Typy danych', df.dtypes)
#zmiana z liczby na słowa-wyłączone do reszty zadań
# from num2words import num2words
# df['Age'] = df['Age'].apply(lambda x: num2words(x,lang = 'pl'))
print(df['Age'])
#zadanie 12
print("Zadanie 12\n")
sredni_wiek = df['Age'].mean()
print("Średni wiek", sredni_wiek)
grupowane = df.groupby('Favorite_Genre')['Age'].mean()
print("Średni wiek dla gatunków\n", grupowane)
agregacja = df.groupby('Favorite_Genre').agg({'Age': ['mean', 'min', 'max'], 'User_ID': 'count'})
print(agregacja)
#zadanie 13
print('Zadanie 13\n')
#dodanie kolumny z minutami obejrzanymi
df['Watch_Minutes'] = df['Watch_Time_Hours'] * 60
print(df[['Watch_Minutes', 'Watch_Time_Hours']].head())
def wiek_ogladajacych(age):
    if age <18:
        return "Nieletni"
    else:
        return "Dorosły"
df['Age_Category'] = df['Age'].apply(wiek_ogladajacych)
print(df[['Age', 'Age_Category']])

df['Name'] = df['Name'].str.upper()
print(df['Name'])

df['Initials'] = df['Name'].str[0]
print(df[['Name', 'Initials']])
#zadanie 14
print('Zadanie 14\n')
import matplotlib.pyplot as plt
df['Favorite_Genre'].value_counts().plot(kind='bar', figsize=(10, 5), color='salmon', edgecolor='black')
plt.xlabel("Ulubiony gatunek filmowy")
plt.ylabel("Liczba użytkowników")
plt.title("Wybierane gatunki filmowe")
plt.legend(['Liczba użytkowników'])
plt.show()

df['Favorite_Genre'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title("Procentowy udział gatunków filmowych")
plt.ylabel("")
plt.show()

df.plot(x='Age', y='Watch_Time_Hours', kind='scatter', figsize=(8, 5), color='green', alpha=0.1, s = 10, edgecolors='black')
plt.xlabel("Wiek użytkownika")
plt.ylabel("Godziny oglądania")
plt.title("Zależność między wiekiem a liczbą godzin oglądania")
plt.show()

#zadanie 15
print("Zadanie 15\n")
df1 = pd.DataFrame({'User_ID': [1, 2, 3], 'Name': ['Anna', 'Kuba', 'Ola'], 'Age': [25, 30, 22]})
df2 = pd.DataFrame({'User_ID': [1, 2, 3], 'Favorite_Genre': ['Drama', 'Comedy', 'Action']})

merged_df = pd.merge(df1, df2, on='User_ID')
print(merged_df)

df3 = pd.DataFrame({'User_ID': [4, 5], 'Name': ['Tomek', 'Ewa'], 'Age': [28, 35]})
concatenated_df = pd.concat([df1, df3])
print(concatenated_df)

df_wide = pd.DataFrame({
    'User_ID': [1, 2],
    'Drama': [5, 3],
    'Comedy': [2, 8]
})

print(df_wide)
df_long = df_wide.melt(id_vars=['User_ID'], var_name='Favorite_Genre', value_name='Watch_Time_Hours')
print(df_long)

df_pivot = df_long.pivot(index = 'User_ID', columns = 'Favorite_Genre', values = 'Watch_Time_Hours')
print(df_pivot)

df_data = pd.DataFrame({
    'User_ID': [1, 2, 3],
    'Join_Date': ['2023-01-10', '2023-05-15', '2022-11-20']
})
print(df_data)

df_data['Join_Date'] = pd.to_datetime(df_data['Join_Date'])
print(df_data)

sprawdzanie = df_data[df_data['Join_Date'] > '2023-01-10']
print(sprawdzanie)
#dodawanie roku
df_data['Year'] = df_data['Join_Date'].dt.year
print(df_data)

# #zadanie 16
#print("Zadanie 16\n")
import matplotlib.pyplot as plt
x = np.random.randint(0,10, 5)
y = np.random.randint(0,4, 5)
print(x,y)
plt.plot(x,y, label = "Wykres")
plt.xlabel("oś x")
plt.ylabel("oś y")
plt.title("Wykres liniowy")
plt.legend()
plt.show()

#zadanie 17
print("Zadanie 17\n")
z = np.random.randint(0,10, 5)
k = np.random.randint(0,4, 5)
print(z,k)
plt.scatter(z,k, label = "Punkty", s=100, color = "black", alpha = 0.8, marker = "o", edgecolor = "blue")
plt.xlabel("Oś x")
plt.ylabel("Oś y")
plt.title("Wykres kropkowy", fontweight="bold", fontsize="large")
plt.legend()
plt.show()

#zadanie 18
print("Zadanie 18\n")
liczba_znajomych = [1,2,4,1,3,4,1]
urodzeni_w_miesiacu = ('Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec', 'Lipiec')
plt.bar(urodzeni_w_miesiacu,liczba_znajomych, label="Liczba znajomych", color = "blue")
plt.ylabel('liczba znajomych', fontsize = 16)
plt.xlabel('miesiące', fontsize = 16)
plt.title("Miesiąc urodzenia znajomych")
plt.legend()
plt.show()

#zadanie 19-histogram
print("Zadanie 19\n")
dane = np.random.randint(20,36, 100)
plt.hist(dane, bins = 20, color = 'salmon', edgecolor = 'black')
plt.xlabel("wiek osób")
plt.ylabel("Liczba osób")
plt.title("Mieszkania wśród osób pomiędzy wiekiem 20-35 lat")
plt.show()


#zadanie 20-wykres kołowy
print("Zadanie 20\n")
label = ['Apple', 'Orange', 'Banana', 'Mango', 'Peach']
ilosc  = np.random.randint(1,100,5)
plt.pie(ilosc, labels =label, autopct='%1.1f%%', colors = ['firebrick', 'orange', 'yellow', 'forestgreen', 'salmon'])
plt.title("Ilość owoców spożywanych w ciągu tygodnia")
plt.show()

#zadanie 21-wykres wątków bocznych
print("Zadanie 21\n")
x = np.linspace(0, 10, 100)
y1 = np.tan(x)
y2 = np.random.randint(1, 10, 100)
z = np.random.randn(100)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))


axes[0, 0].plot(x, y1, color='blue')
axes[0, 0].set_title('Tangens')
axes[0,0].set_xlabel("Wartość x")
axes[0,0].set_ylabel("Wartość  y")
axes[0, 0].legend(["tan(x)"])

axes[0, 1].plot(x, y2, color='red')
axes[0, 1].set_title('Rozkład danych')
axes[0,1].set_xlabel("Oś x")
axes[0,1].set_ylabel("Oś y")
axes[0, 1].legend(["Losowe wartości"])

axes[1, 0].scatter(x, z, color='green', alpha=0.6, label="Punkty losowe")
axes[1, 0].set_title("Wykres punktowy")
axes[1,0].set_xlabel("Oś x")
axes[1,0].set_ylabel("Oś y")
axes[1, 0].legend()

axes[1, 1].hist(y2, bins=10, color='salmon', edgecolor = 'black',alpha=0.6, label="Histogram")
axes[1, 1].set_title("Histogram")
axes[1,1].set_xlabel("Liczba ludzi")
axes[1,1].set_ylabel("Liczby butów")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

#zadanie 22
print("Zadanie 22\n")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Magda/Desktop/netflix_users.csv")

grouped = df.groupby(['Age', 'Favorite_Genre'])['Watch_Time_Hours'].sum().unstack()

grouped.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20', edgecolor='black')

plt.title("Skumulowany wykres godzin oglądania według wieku i gatunku", fontsize=16)
plt.xlabel("Wiek użytkownika")
plt.ylabel("Sumaryczne godziny oglądania")
plt.legend(title="Gatunek filmowy", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
