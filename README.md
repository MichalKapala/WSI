# WSI

## Autorzy

* Anna Ryfa
* Piotr Makosiej
* Wojciech Rojek
* Michał Kapała

## Regresja cen pojazdu na podstawie jego danych 

### Cel projektu
Stworzenie modelu sztucznej inteligencji, do wyceny pojazdu na podstawie jego danych

### Wskaźniki sukcesu
1. Model z małym błędem przewiduje ceny pojazdu.

### Wyznaczenie zakresu projektu
1. Zebranie danych za pomocą wybranego API
2. Przegląd i przygotowanie danych
3. Analiza danych
4. Trenowanie wybranych modeli uczenia maszynowego do predykcji cen samochodu
5. Dostrajanie modeli
6. Ocena na zbiorze testowym
7. Wprowadzenie modelu.

### Określenie korzyści z realizacji
1. Szybsza wycena aut z rynku wtórnego
2. Możliwośc automatyzacji serwisów giełd samochodowych 

## Technologie
* Python
* Pandas
* Numpy
* Plotly
* API do zbierania danych 

## Zestaw danych
Zebrany ręcznie

## Dokumentacja
### Biblioteki
* Sklearn
* Numpy
* Pandas
* Matplotlib
* Seaborn 

### Określenie potrzeb sprzętowych
Minimalne wymagania sprzętowe to: 
RAM: 2+ GB 
CPU: 2+ rdzenie 
Dysk: 20 GB
Są one wymagane aby uruchomić serwer jupyter notebook 

### Zarzadzanie kodem i wersjami 

Zarządzanie kodem odbywa się za pomocą systemu kontroli wersji git oraz za pomocą strony guthub.com.
Aktualnie dostępna jest jedna wersja projektu. 

### Struktura projektu

W głównym folderze znajdują się takie pliki jak cardatapreparation.ipynb oraz ModelSelection.ipynb 

Cardatapreparation.ipynb odpowiedzialny jest za obróbkę danych znajdujących się w “data/used_cars.csv”. 
Znajduje się tam między innymi: 
- wyciągane cechy do wielu kolumn z kolumn zawierających wiele informacji, 
- uzupełniane brakujące dane 
- analiza danych 
- usuwane odstające wartości 
- wybór cech 
- zapisywanie danych do pliku “cleaned_car_data.csv” w folderze “data” 

W ModelSelection.ipynb testowana jest efektywność modeli z różnymi parametrami. Te modele to:
- Regresja Liniowa
- Drzewo decyzyjne
- Las losowy 
- Regresja Ridge 
- Regresja Lasso
- SVM
- Gradient Boosting Regressor
- AdaBoost
- Bagging
- Stacking 

Oceniane są one na podstawie średniego błędu bezwględnego (MAE), średniego błędu kwadratowego (MSE), mediany błędu bezwzględnego (MedAE) oraz czasu wykonania. 