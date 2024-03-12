import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

xls = pd.ExcelFile(r"Datos Alimenticios.xls")

sheet = xls.parse(0)

sheet = np.array(sheet)
numsheet, genders = np.split(sheet, [-1], axis=1)
numsheet = numsheet.astype(float)

#Replace not found values
numsheet[np.isin(numsheet, [999.99])] = np.nan
mean = np.nanmean(numsheet, axis=0)

numsheet[:,0][np.isnan(numsheet[:,0])] = mean[0]
numsheet[:,1][np.isnan(numsheet[:,1])] = mean[1]
numsheet[:,2][np.isnan(numsheet[:,2])] = mean[2]

# df = pd.DataFrame(sheet, columns=['Grasas_sat', 'Alcohol', 'Calorías', 'Sexo'])
# df = df.replace(999.99, float('nan'))
# df['Grasas_sat'].fillna(df['Grasas_sat'].mean(), inplace=True)
# df['Alcohol'].fillna(df['Alcohol'].median(), inplace=True)
# df['Calorías'].fillna(df['Calorías'].mode().iloc[0], inplace=True)

#General normalization
normsheet = (numsheet - mean) / np.nanstd(numsheet, axis=0)

df = pd.DataFrame(normsheet, columns=['Grasas_sat', 'Alcohol', 'Calorías'])
df.boxplot()

plt.title('Boxplot General')
plt.ylabel('Valores Normalizados')
plt.show()

#Bar graphs (normalized)

df = pd.DataFrame(normsheet, columns=['Grasas_sat', 'Alcohol', 'Calorías'])
df.plot.hist()

plt.ylabel('Values')
plt.title('Bar Graph')
plt.show()

#Sex dependent normalization 

sheet = np.concatenate((numsheet, genders), axis=1)

Ms = sheet[sheet[:, 3] == 'M', :3].astype(float)
Fs = sheet[sheet[:, 3] == 'F', :3].astype(float)
Ms = (Ms- np.nanmean(Ms, axis=0)) / np.nanstd(Ms, axis=0)
Fs = (Fs- np.nanmean(Fs, axis=0)) / np.nanstd(Fs, axis=0)


df = pd.DataFrame(Ms, columns=['Grasas_sat', 'Alcohol', 'Calorías'])
df.boxplot()

plt.title('Boxplot Mujeres')
plt.ylabel('Valores Normalizados')
plt.show()

df = pd.DataFrame(Fs, columns=['Grasas_sat', 'Alcohol', 'Calorías'])
df.boxplot()

plt.title('Boxplot Hombres')
plt.ylabel('Valores Normalizados')
plt.show()

# Alcohol vs calories

sheet = numsheet[:, 1:]

calories = sheet[:, 1]

low = sheet[calories < 1100, :1]
mid = sheet[(calories > 1100) & (calories < 1700), :1]
high = sheet[calories > 1700, :1]

sheet = np.array([low, mid, high])

df = pd.DataFrame({'Data1': low, 'Data2': mid, 'Data3': high})
#df = pd.DataFrame(sheet, columns=['CATE 1', 'CATE 2', 'CATE 3'])
#df.boxplot()

plt.title('Alcohol sobre Calorias')
plt.ylabel('Alcohol')
plt.show()
