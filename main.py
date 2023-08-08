import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing  import StandardScaler
from scipy.stats import norm

df = pd.read_excel(r"C:\Users\Akhil\PycharmProjects\regression\Heartattack_prediction_data.xlsx")

scale = StandardScaler()

x = df[['Age', 'Smoker', 'Hypertension','Sex']]
y = df['RiskOfCHD']

x[['Age', 'Smoker', 'Hypertension','Sex']] = scale.fit_transform(x[['Age', 'Smoker', 'Hypertension','Sex']].to_numpy())

print (x)
est = sm.OLS(y, x).fit()

print(est.summary())

y = df['RiskOfCHD']
y.groupby(df['Heart Rate']).mean()
Z= [y.groupby(df['Heart Rate']).mean(),y.groupby(df['Glucose']).mean(),y.groupby(df['Smoker']).mean(),y.groupby(df['Hypertension']).mean()]
print(Z)

stddev = y.groupby(df['Heart Rate']).std()
mean = y.groupby(df['Heart Rate']).mean()
print(stddev)
print(mean)

