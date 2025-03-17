#zadanie 1
for i in [0, 1, 2, 3, 5]:
 print("Wartosc i wynosi: ", i)

for i in range(0,5):
 print("Wartosc i wynosi: ", i)

lista=[3, 2, 1, 'Hello', 1, 2, 3, 'and bye! ']
for x in lista:
 print(x)

for ch in "alfabet":
 print(ch)
i=0
while i < 5:
 print("Wartosc i wynosi: ", i)
 i=i+1
i=0
while True: #Petla bez kryterium wyjscia
 print ("Wartosc i wynosi: ", i);
 if i == 10:
    break
 i=i+1

lista5=[]
for x in range(0, 100, 10):
 lista5.append(x)
for i in lista5:
 print(i, end=' ')

krotka0=tuple()
print(krotka0)
krotka1 = ('bialy', 'czerwony', 'zielony', 'niebieski', 'czarny')

krotkamieszana = ('jeden', 2, [3, 4.56])
print(krotka1[0])
print(krotka1[-1])
print(krotka1[2:])

krotkasuma=krotka1+krotkamieszana
print(krotkasuma)
krotka2=('sto', 'lat!')*3+('niech', 'zyje', 'nam')
print (krotka2)
print(len(krotka2))
list1=list(krotka1)
list1.append('szary')
krotka3=tuple(list1)
print(krotka3)
from copy import deepcopy
krotka3copy = deepcopy(krotka3)


krotka4 = (-30,250,-70)
wynik=sum(krotka4)
print('Wynik sumowania wartości z krotki4 wynosi: ',wynik)

from array import *
tab1 = array('i', [21,13,8,5,3,2,1])
for i in tab1:
 print(i)
print("Dostęp do poszczególnych elementów tablicy.")
print(tab1[0])
print(tab1[1])
print(tab1[2])


tab4 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for i in range(len(tab4)):
    for j in range(len(tab4[i])):
        print( tab4[i][j], end=' ')
    print()

#zadanie2
print("\nZadanie 2\n")
for i in range(31):
    if (i % 2) != 0:
        continue
    print(i)

#zadanie3
print("\nZadanie 3\n")
x = input("Podaj liczbę:")
x = int(x)
i = 0
suma = 0
while i<=x:
    suma += i
    i +=1
print("Suma", suma)

#zadanie4
print("\nZadanie 4\n")
def suma_parzystych(x):
    suma_2 = 0
    for i in range(1, x + 1):
        if i % 2 != 0:
            continue
        suma_2 += i
    return suma_2

x = int(input("Podaj liczbę: "))
print("Suma po parzystych:", suma_parzystych(x))


#zadanie5
print("\nZadanie 5\n")
def liczby():
    tablica = []
    for j in range(100, -101, -1):
        if j%2 !=0:
            continue
        if j %3 ==0 or j%8 ==0:
            continue
        tablica.append(j)
    return tablica
print(liczby())

#zadanie 6
print("\nZadanie 6\n")
def jodelka(n):

    for i in range(1, n+1):
        for j in range(1, n+1):
                print(min(j, i), end = '')
        print()

jodelka(5)
#zadanie 7
print("\nZadanie 7\n")
n = 4
lista_7 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O','P']
def rozdzielanie(lista,n):
    lista_2 = []
    for i in range(n):
        lista_2.append([])
    for i in range(len(lista)):
        element = lista[i]
        #do której podlisty
        indx = i %n
        lista_2[indx].append(element)
    return lista_2
print(rozdzielanie(lista_7,4))

#zadanie 8
print("\nZadanie 8\n")
lista_8 = [100, 90, 80, 70, 60, 50]
lista_88 = [49, 39, 29, 19]
def zastepstwo(lista_1, lista_2):
    lista_1.pop()
    for i in lista_2:
        lista_1.append(i)
    return lista_1
print(zastepstwo(lista_8,lista_88))

#zadanie 9
print("\nZadanie 9\n")
lista_9 =  ['A', 'B', 'C', 'D']
slowo_1 = "Exit"
def laczenie(slowo,lista):
    koncowa = []
    for i in lista:
        koncowa.append(f"{slowo} {i}")
    return koncowa
print(laczenie(slowo_1,lista_9))

#zadanie10
print("\nZadanie 10\n")

list2 = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
def zeroostatnie(lista):
    list10 = []
    for i in lista:
        lista_3 = list(i)
        lista_3[-1] = 0
        list10.append(tuple(lista_3))
    return list10
print(zeroostatnie(list2))

#zadanie 11
print("\nZadanie 11\n")
lista_11 =[(), (), ('',), ('i1', 'i2'), ('i1', 'i2', 'i3'), ('i4')]
listka_11 = []
for i in lista_11:
    if i != ():
        listka_11.append(i)
print(listka_11)

#zadanie 12
print("\nZadanie 12\n")
k = 1
dict12 = { 'f1': 4.8, 'f2': 2.4, 'f3': 1.2, 'f4': 0.6}
for v in dict12.values():
    k = k*v
print(k)
#zadanie 13
print('\nZadanie 13\n')
n = int(input('Podaj liczbę n: '))
dict13 = {}
for i in range(1,n + 1):
    dict13[i] = i*i*i*i
print('Podany słownik to', dict13)

#zadanie 14
print('\nZadanie 14\n')
dict14 = {'a': 'A201', 'b': 'B202', 'c': 'B202', 'd': 'H018', 'e': 'H018', 'f': 'A007', 'g': 'G230'}
lista = []
for k in dict14.values():
        lista.append(k)
lista = set(lista)
print(lista)
#bardziej optymalne
wartosci = set(dict14.values())
print(wartosci)
