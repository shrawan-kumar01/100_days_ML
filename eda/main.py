import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Titanic-Dataset.csv')
# print(df.head)
# # print(df.sample)
# sns.countplot(x = df['Survived'])
# plt.pie(df['Survived'].value_counts())
# plt.hist(df['Age'],bins=5)
# sns.distplot(df['Age'])
sns.boxplot(df['Age'])

plt.show()
