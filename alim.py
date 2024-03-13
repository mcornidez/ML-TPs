import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

xls = pd.ExcelFile(r"Datos Alimenticios.xls")

sheet = xls.parse(0)

sheet = np.array(sheet)
numsheet, genders = np.split(sheet, [-1], axis=1)
numsheet = numsheet.astype(float)
index = ["Grasas_sat", "Alcohol", "Calorías"]

# Replace not found values
numsheet[np.isin(numsheet, [999.99])] = np.nan
mean = np.nanmean(numsheet, axis=0)
std = np.nanstd(numsheet, axis=0)

numsheet[:, 0][np.isnan(numsheet[:, 0])] = mean[0]
numsheet[:, 1][np.isnan(numsheet[:, 1])] = mean[1]
numsheet[:, 2][np.isnan(numsheet[:, 2])] = mean[2]


kurtosis = kurtosis(numsheet, axis=0)
skew = skew(numsheet, axis=0)

# General normalization
normsheet = (numsheet - mean) / std

print("General means:")
print(mean)
print("General stds:")
print(std)
print("General kurtosis:")
print(kurtosis)
print("General skew:")
print(skew)


df = pd.DataFrame(normsheet, columns=index)
df.boxplot()

plt.title("Boxplot General")
plt.ylabel("Valores Normalizados")
# plt.show()

# Sex dependent normalization
sheet = np.concatenate((numsheet, genders), axis=1)

female = sheet[sheet[:, 3] == "F", :3].astype(float)
male = sheet[sheet[:, 3] == "M", :3].astype(float)

f_mean = np.nanmean(female, axis=0)
f_std = np.nanstd(female, axis=0)
m_mean = np.nanmean(male, axis=0)
m_std = np.nanstd(male, axis=0)

female_norm = (female - f_mean) / np.nanstd(female, axis=0)
male_norm = (male - m_mean) / np.nanstd(male, axis=0)


print("Female means:")
print(f_mean)
print("Female stds:")
print(f_std)

df = pd.DataFrame(female_norm, columns=index)
df.boxplot()

plt.title("Boxplot Mujeres")
plt.ylabel("Valores Normalizados")
# plt.show()

print("Male means:")
print(m_mean)
print("Male stds:")
print(m_std)

df = pd.DataFrame(male_norm, columns=index)
df.boxplot()

plt.title("Boxplot Hombres")
plt.ylabel("Valores Normalizados")
# plt.show()

print("Comparacion Medias Hombres:")
print((mean - m_mean) / std)
print("Comparacion Medias Mujeres:")
print((mean - f_mean) / std)

# df = pd.DataFrame({'Media Hombre': (mean-m_mean)/mean, 'Media Mujer': (mean-f_mean)/mean}, index=index)
# df.plot.bar(rot=0)
# plt.title('Media Hombre-Mujer')
# plt.ylabel('Valores Normalizados')
# plt.show()

# Alcohol vs calories

sheet = numsheet[:, 1:]

calories = sheet[:, 1]

low = sheet[calories < 1100, :1]
mid = sheet[(calories > 1100) & (calories < 1700), :1]
high = sheet[calories > 1700, :1]

# sheet = np.array([low, mid, high])

# df = pd.DataFrame({"Data1": low, "Data2": mid, "Data3": high})
# df = pd.DataFrame(sheet, columns=['CATE 1', 'CATE 2', 'CATE 3'])
# df.boxplot()

# plt.title("Alcohol sobre Calorias")
# plt.ylabel("Alcohol")
# plt.show()


Grasas_sat = numsheet[:, 0]
Alcohol = numsheet[:, 1]
Calorias = numsheet[:, 2]


# Scatter plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.scatter(Grasas_sat, Alcohol)
plt.xlabel("Grasas_sat")
plt.ylabel("Alcohol")
plt.title("Grasas_sat vs Alcohol")

plt.subplot(1, 3, 2)
plt.scatter(Grasas_sat, Calorias)
plt.xlabel("Grasas_sat")
plt.ylabel("Calorías")
plt.title("Grasas_sat vs Calorías")

plt.subplot(1, 3, 3)
plt.scatter(Alcohol, Calorias)
plt.xlabel("Alcohol")
plt.ylabel("Calorías")
plt.title("Alcohol vs Calorías")

plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.hist(Grasas_sat, bins=20)
plt.xlabel("Grasas_sat")
plt.ylabel("Frequency")
plt.title("Distribution of Grasas_sat")

plt.subplot(1, 3, 2)
plt.hist(Alcohol, bins=20)
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.title("Distribution of Alcohol")

plt.subplot(1, 3, 3)
plt.hist(Calorias, bins=20)
plt.xlabel("Calorías")
plt.ylabel("Frequency")
plt.title("Distribution of Calorías")

plt.tight_layout()
plt.show()


# Box plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.boxplot(Grasas_sat)
plt.ylabel("Grasas_sat")
plt.title("Box plot of Grasas_sat")

plt.subplot(1, 3, 2)
plt.boxplot(Alcohol)
plt.ylabel("Alcohol")
plt.title("Box plot of Alcohol")

plt.subplot(1, 3, 3)
plt.boxplot(Calorias)
plt.ylabel("Calorías")
plt.title("Box plot of Calorías")

plt.tight_layout()
plt.show()


# Extracting columns for female and male
Grasas_sat_female = female[:, 0]
Alcohol_female = female[:, 1]
Calorias_female = female[:, 2]

Grasas_sat_male = male[:, 0]
Alcohol_male = male[:, 1]
Calorias_male = male[:, 2]

# Scatter plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(Grasas_sat_female, Alcohol_female, color="magenta", label="Femenino")
plt.scatter(Grasas_sat_male, Alcohol_male, color="blue", label="Masculino")
plt.xlabel("Grasas_sat")
plt.ylabel("Alcohol")
plt.title("Grasas_sat vs Alcohol")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(Grasas_sat_female, Calorias_female, color="magenta", label="Femenino")
plt.scatter(Grasas_sat_male, Calorias_male, color="blue", label="Masculino")
plt.xlabel("Grasas_sat")
plt.ylabel("Calorías")
plt.title("Grasas_sat vs Calorías")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(Alcohol_female, Calorias_female, color="magenta", label="Femenino")
plt.scatter(Alcohol_male, Calorias_male, color="blue", label="Masculino")
plt.xlabel("Alcohol")
plt.ylabel("Calorías")
plt.title("Alcohol vs Calorías")
plt.legend()

plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(Grasas_sat_female, bins=20, color="magenta", alpha=0.5, label="Femenino")
plt.hist(Grasas_sat_male, bins=20, color="blue", alpha=0.5, label="Masculino")
plt.xlabel("Grasas_sat")
plt.ylabel("Frecuencia")
plt.title("Distribucion de Grasas_sat")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(Alcohol_female, bins=20, color="magenta", alpha=0.5, label="Femenino")
plt.hist(Alcohol_male, bins=20, color="blue", alpha=0.5, label="Masculino")
plt.xlabel("Alcohol")
plt.ylabel("Frecuencia")
plt.title("Distribucion de Alcohol")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(Calorias_female, bins=20, color="magenta", alpha=0.5, label="Femenino")
plt.hist(Calorias_male, bins=20, color="blue", alpha=0.5, label="Masculino")
plt.xlabel("Calorías")
plt.ylabel("Frecuencia")
plt.title("Distribucion de Calorías")
plt.legend()

plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.boxplot([Grasas_sat_female, Grasas_sat_male], labels=["Femenino", "Masculino"])
plt.ylabel("Grasas_sat")
plt.title("Box plot de Grasas_sat")

plt.subplot(1, 3, 2)
plt.boxplot([Alcohol_female, Alcohol_male], labels=["Femenino", "Masculino"])
plt.ylabel("Alcohol")
plt.title("Box plot de Alcohol")

plt.subplot(1, 3, 3)
plt.boxplot([Calorias_female, Calorias_male], labels=["Femenino", "Masculino"])
plt.ylabel("Calorías")
plt.title("Box plot de Calorías")

plt.tight_layout()
plt.show()

## Categorias

low = sheet[calories < 1100, :1].flatten()
mid = sheet[(calories > 1100) & (calories < 1700), :1].flatten()
high = sheet[calories > 1700, :1].flatten()


max_length = max(len(low), len(mid), len(high))
# Pad the shorter arrays with NaNs to have the same length
if len(low) < max_length:
    padding = np.empty(max_length - len(low)) * np.nan
    low = np.append(low, padding)
if len(mid) < max_length:
    padding = np.empty(max_length - len(mid)) * np.nan
    mid = np.append(mid, padding)
if len(high) < max_length:
    padding = np.empty(max_length - len(high)) * np.nan
    high = np.append(high, padding)


df = pd.DataFrame({"CATE 1": low, "CATE 2": mid, "CATE 3": high})
df.boxplot()

plt.title("Alcohol sobre Calorias")
plt.ylabel("Alcohol")
plt.show()

# Normalized plot

low = (low - np.nanmean(low)) / np.nanstd(low)
mid = (mid - np.nanmean(mid)) / np.nanstd(mid)
high = (high - np.nanmean(high)) / np.nanstd(high)

df = pd.DataFrame({"CATE 1": low, "CATE 2": mid, "CATE 3": high})
df.boxplot()

plt.title("Alcohol sobre Calorias Normalizado")
plt.ylabel("Alcohol")
plt.show()


"""

# Definir categorías de calorías
sheet_categories = np.copy(sheet)
sheet_categories[:, 2] = np.where(sheet[:, 2] <= 1100, 'CATE 1', 
                        np.where(sheet[:, 2] <= 1700, 'CATE 2', 'CATE 3'))

# Filtrar datos por categorías de calorías
cate_1 = sheet_categories[sheet_categories[:, 2] == 'CATE 1']
cate_2 = sheet_categories[sheet_categories[:, 2] == 'CATE 2']
cate_3 = sheet_categories[sheet_categories[:, 2] == 'CATE 3']

# Boxplot para Alcohol por categorías de calorías
plt.figure(figsize=(10, 5))
plt.boxplot([cate_1[:, 1], cate_2[:, 1], cate_3[:, 1]], labels=['CATE 1', 'CATE 2', 'CATE 3'])
plt.ylabel('Alcohol')
plt.title('Boxplot de Alcohol por categorías de Calorías')
plt.show()

# Calcular la media de alcohol para cada categoría de calorías
mean_alcohol_cate_1 = np.mean(cate_1[:, 1])
mean_alcohol_cate_2 = np.mean(cate_2[:, 1])
mean_alcohol_cate_3 = np.mean(cate_3[:, 1])

# Crear un gráfico de barras
categories = ['CATE 1', 'CATE 2', 'CATE 3']
mean_alcohol = [mean_alcohol_cate_1, mean_alcohol_cate_2, mean_alcohol_cate_3]

plt.figure(figsize=(10, 6))
plt.bar(categories, mean_alcohol, color='skyblue')
plt.xlabel('Categorías de Calorías')
plt.ylabel('Consumo Medio de Alcohol')
plt.title('Consumo Medio de Alcohol por Categorías de Calorías')
plt.show()

"""
