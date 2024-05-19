import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.exceptions import ConvergenceWarning

# Ignorowanie konkretnego ostrzeżenia
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Załaduj dane
data = pd.read_csv('../data/cleaned_car_data.csv')

# Podgląd danych
print(data)

# wybieranie cech i wartości wyznaczanej
features = data[['model_year', 'milage', 'accident', 'horsepower', 'brand_alfa', 'brand_aston', 'brand_audi', 'brand_bentley', 'brand_bmw', 'brand_bugatti', 'brand_buick', 'brand_cadillac', 'brand_chevrolet', 'brand_chrysler', 'brand_dodge', 'brand_ferrari', 'brand_fiat', 'brand_ford', 'brand_genesis', 'brand_gmc', 'brand_honda', 'brand_hummer', 'brand_hyundai', 'brand_infiniti', 'brand_jaguar', 'brand_jeep', 'brand_karma', 'brand_kia', 'brand_lamborghini', 'brand_land', 'brand_lexus', 'brand_lincoln', 'brand_lotus', 'brand_lucid', 'brand_maserati', 'brand_maybach', 'brand_mazda', 'brand_mclaren', 'brand_mercedes-benz', 'brand_mercury', 'brand_mini', 'brand_mitsubishi', 'brand_nissan', 'brand_plymouth', 'brand_polestar', 'brand_pontiac', 'brand_porsche', 'brand_ram', 'brand_rivian', 'brand_rolls-royce', 'brand_saab', 'brand_saturn', 'brand_scion', 'brand_smart', 'brand_subaru', 'brand_suzuki', 'brand_tesla', 'brand_toyota', 'brand_volkswagen', 'brand_volvo', 'fuel_type_e85 flex fuel', 'fuel_type_electric', 'fuel_type_gasoline', 'fuel_type_hybrid', 'fuel_type_hydrogen', 'fuel_type_plug-in hybrid', 'transmission_category_manual']]
target = data['price']

# Konwersja danych na numpy arrays
X = features.values
y = target.values

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'MedAE': 'neg_median_absolute_error'
}

modelNames = ['Regresja Liniowa', 'Drzewo decyzyjne', 'Drzewo decyzyjne(3,5)', 'Drzewa decyzyjne(5,10)',
              'Las losowy', 'Las losowy(100,10)', 'Las losowy(30,none)', 'Regresja Ridge', 'Regresja Lasso',
              'SVM', 'SVM(linear, 0.1)', 'SVM(rbf, 2.0, scale)', 'Gradient Boosting Regressor',
              'Gradient Boosting Regressor(100,0.01)', 'Gradient Boosting Regressor(200,0.1)',
              'AdaBoost', 'AdaBoost(50, 1.0)', 'AdaBoost(100, 0.5)', 'Bagging(50)', 'Bagging(10)', 'Bagging(150)',
              'Stacking(lr+svr+rf)', 'Stacking(rf+svr)', 'Stacking(gb+dt)', 'Stacking(rf+br)']

models = {
    modelNames[0]: LinearRegression(),
    modelNames[1]: DecisionTreeRegressor(random_state=42),
    modelNames[2]: DecisionTreeRegressor(max_depth=3, min_samples_leaf=5),
    modelNames[3]: DecisionTreeRegressor(max_depth=5, min_samples_split=10),
    modelNames[4]: RandomForestRegressor(random_state=42),
    modelNames[5]: RandomForestRegressor(n_estimators=100, max_depth=10),
    modelNames[6]: RandomForestRegressor(n_estimators=50, max_depth=None),
    modelNames[7]: Ridge(),
    modelNames[8]: Lasso(),
    modelNames[9]: LinearSVR(random_state=42, dual='auto', max_iter=10000),
    modelNames[10]: LinearSVR(random_state=42, dual='auto', max_iter=10000, C=0.1),
    modelNames[11]: SVR(kernel='rbf', C=2.0, gamma='scale'),
    modelNames[12]: GradientBoostingRegressor(random_state=42),
    modelNames[13]: GradientBoostingRegressor(n_estimators=100, learning_rate=0.01),
    modelNames[14]: GradientBoostingRegressor(n_estimators=200, learning_rate=0.1),
    modelNames[15]: AdaBoostRegressor(random_state=42, n_estimators=50),
    modelNames[16]: AdaBoostRegressor(n_estimators=50, learning_rate=1.0),
    modelNames[17]: AdaBoostRegressor(n_estimators=100, learning_rate=0.5),
    modelNames[18]: BaggingRegressor(random_state=42, n_estimators=50),
    modelNames[19]: BaggingRegressor(n_estimators=10),
    modelNames[20]: BaggingRegressor(n_estimators=150),
    modelNames[21]: StackingRegressor(estimators=[
                                                     ('lr', LinearRegression()),
                                                     ('svr', LinearSVR(random_state=42, dual='auto')),
                                                     ('rf', RandomForestRegressor(random_state=42))
                                                 ], final_estimator=Ridge()),
    modelNames[22]: StackingRegressor(estimators=[
                                                     ('rf', RandomForestRegressor(n_estimators=10)),
                                                     ('svr', LinearSVR(random_state=42, dual='auto'))
                                                 ], final_estimator=Ridge()),
    modelNames[23]: StackingRegressor(estimators=[
                                                     ('gb', GradientBoostingRegressor(n_estimators=50)),
                                                     ('dt', DecisionTreeRegressor(max_depth=5))
                                                 ], final_estimator=Ridge()),
    modelNames[24]: StackingRegressor(estimators=[
                                                     ('rf', RandomForestRegressor(random_state=42)),
                                                     ('br', BaggingRegressor(n_estimators=150))
                                                 ], final_estimator=Ridge())
}


results = {}
normResultsMAE=[]
normResultsMSE=[]
normResultsMedAE=[]
timeResults=[]

for index, element in enumerate(modelNames):
    start_time = time.time()
    results[element] = cross_validate(models[element], X, y, cv=kf, scoring=scoring, return_train_score=False)
    end_time = time.time()
    timeResults.append(end_time - start_time)
    normResultsMAE.append(-np.mean(results[element]['test_MAE']))
    normResultsMSE.append(-np.mean(results[element]['test_MSE']))
    normResultsMedAE.append(-np.mean(results[element]['test_MedAE']))
    print(f"{element}: {normResultsMAE[-1]:.2f}, {normResultsMSE[-1]:.2f}, {normResultsMedAE[-1]:.2f}, {timeResults[-1]:.2f}")

# Wyświetl wyniki
resultsMAE = pd.DataFrame({'Model': modelNames,'MAE': normResultsMAE})
resultsMSE = pd.DataFrame({'Model': modelNames, 'MSE': normResultsMSE})
resultsMedAE = pd.DataFrame({'Model': modelNames,'MedAE': normResultsMedAE})
resultsTime = pd.DataFrame({'Model': modelNames,'Time': timeResults})

print(resultsMAE.sort_values(by='MAE'),end="\n\n")
print(resultsMSE.sort_values(by='MSE'),end="\n\n")
print(resultsMedAE.sort_values(by='MedAE'),end="\n\n")
print(resultsTime.sort_values(by='Time'),end="\n\n")

# Wykres błędów
plt.figure(figsize=(12, 6))
plt.bar(resultsMAE['Model'], resultsMAE['MAE'], color='purple')
plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('Porównanie średniego błędu bezwzględnego')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(resultsMSE['Model'], resultsMSE['MSE'], color='blue')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('Porównanie średniego błedu kwadratowego')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(resultsMedAE['Model'], resultsMedAE['MedAE'], color='green')
plt.xlabel('Model')
plt.ylabel('MedAE')
plt.title('Porównanie mediany błędu bezwzględnego')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(resultsTime['Model'], resultsTime['Time'], color='red')
plt.xlabel('Model')
plt.ylabel('Time')
plt.title('Porównanie czasu wykonania')
plt.show()
