import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("loan_data.csv")

df = df.dropna(axis=0)

df['Gender']=df['Gender'].replace({"Male":1.0,'Female':0.0})
df['Married']=df['Married'].replace({'Yes':1.0,'No':0.0})
df['Dependents']=df['Dependents'].replace({'3+':3.0}).astype(int)
df['Education']=df['Education'].replace({'Graduate':1.0,'Not Graduate':0.0})
df['Self_Employed']=df['Self_Employed'].replace({'Yes':1.0,'No':0.0})
df['Property_Area']=df['Property_Area'].replace({'Rural':0.0,'Semiurban':1.0,'Urban':2.0})
df['Loan_Status']=df['Loan_Status'].replace({'Y':1.0,'N':0.0})

x = df[list(df.columns)[1:len(list(df.columns))-1]]
y = df.Loan_Status
scaler=StandardScaler().fit(x)
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns,index=x.index)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
model = keras.Sequential([
    layers.Dense(2048, activation="relu"),
    layers.Dense(1024, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# This is the best config I could get with a validation accuracy of about 87%


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=30, validation_data =(x_test,y_test))

values = model.predict(x_test)

val_array = []

for value in values:
    if value > 0.5:
        val_array.append("Yes")
    else:
        val_array.append("No")
print(val_array)


