import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('delaney_solubility_with_descriptors.csv')
print()
#  data seperation as X and Y

y = df['logS']
# print(y)

x = df.drop('logS',axis=1)
# print(x)

# split the dataset in training set and est set 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)
# print(x_train)
# print(y_train)
# print(x_test)    //[229 rows x 4 columns]


# model building 

lr = LR()
lr.fit(x_train,y_train)
y_lr_tain_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
# print(y_lr_tain_pred)
# print(y_lr_test_pred)

#  EVALUATE MODEL; PERFORMANCE 

lr_train_mse = mean_squared_error(y_train ,y_lr_tain_pred)
lr_train_r2scoreb = r2_score(y_train,y_lr_tain_pred)

lr_train_mse = mean_squared_error(y_test ,y_lr_tain_pred)
lr_train_r2scoreb = r2_score(y_test,y_lr_tain_pred)

print('LR MSE (TRAIN) : ' , lr_train_mse)
print("r2_score (train)" , lr_train_r2scoreb)
