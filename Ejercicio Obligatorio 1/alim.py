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
plt.show()

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
plt.show()

print("Male means:")
print(m_mean)
print("Male stds:")
print(m_std)

df = pd.DataFrame(male_norm, columns=index)
df.boxplot()

plt.title("Boxplot Hombres")
plt.ylabel("Valores Normalizados")
plt.show()

print("Comparacion Medias Hombres:")
print((mean - m_mean) / std)
print("Comparacion Medias Mujeres:")
print((mean - f_mean) / std)

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

# Alcohol vs calories

sheet = numsheet[:, 1:]

calories = sheet[:, 1]

low = sheet[calories < 1100, :1].flatten()
mid = sheet[(calories > 1100) & (calories < 1700), :1].flatten()
high = sheet[calories > 1700, :1].flatten()

plt.boxplot([low, mid, high], labels=["CATE 1", "CATE 2", "CATE 3"])
plt.title("Alcohol sobre Calorias")
plt.ylabel("Alcohol")
plt.show()

# Normalized plot

low = (low - low.mean()) / low.std()
mid = (mid - mid.mean()) / mid.std()
high = (high - high.mean()) / high.std()

plt.boxplot([low, mid, high], labels=["CATE 1", "CATE 2", "CATE 3"])
plt.title("Alcohol sobre Calorias Normalizado")
plt.ylabel("Alcohol")
plt.show()
