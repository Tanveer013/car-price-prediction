import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
car_dataset = pd.read_csv('car data.csv')
import datetime
date_time=datetime.datetime.now()
car_dataset['Age'] = date_time.year - car_dataset['Year']
car_dataset.drop('Year',axis=1,inplace=True)
#MODIFY EXISTING DATA FRAME
car_dataset= car_dataset[~(car_dataset['Selling_Price']>=33.0) & (car_dataset['Selling_Price']<=35.0)]
# Fuel_Type
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# Seller_Type
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# Transmission
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)#INDEPENDENT VARIABLES
Y = car_dataset['Selling_Price']#TARGET VARIABLE
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']
#Traing data in X_train
#Testin Data in X_train
#Price value of X_train in Y_train
#Price value of X_test in Y_test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
#0.1% Data = 10% Data is Testing Data
#90% Data is Training Data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor 
lr = LinearRegression()#lr is instance of linearRegression    
lr.fit(X_train,Y_train)

rf = RandomForestRegressor()
rf.fit(X_train,Y_train)

xg = XGBRegressor()
xg.fit(X_train,Y_train)

xgb = GradientBoostingRegressor()
xgb.fit(X_train,Y_train)
Y_pred1 = lr.predict(X_test)
Y_pred2 = rf.predict(X_test)
Y_pred3 = xgb.predict(X_test)
Y_pred4 = xg.predict(X_test)
#THE MORE THE VALUE OF R SQARED(NEAR TO 1.0)THE BEST THE MODEL IS
from sklearn import metrics
score1 = metrics.r2_score(Y_test,Y_pred1)#ACTUAL VALUE,PREDICTED VALUES
score2 = metrics.r2_score(Y_test,Y_pred2)#ACTUAL VALUE,PREDICTED VALUES
score3 = metrics.r2_score(Y_test,Y_pred3)#ACTUAL VALUE,PREDICTED VALUES
score4 = metrics.r2_score(Y_test,Y_pred4)#ACTUAL VALUE,PREDICTED VALUES
#CREATING PANDAS DATAFRAME AND USING PYTHON DICTIONARY
final_data = pd.DataFrame({'Models':['LR','RF','GBR','XG'],
              'R2_SCORE':[score1,score2,score3,score4]})
xg = XGBRegressor()
xg_final = xg.fit(X,Y)#train the best model on entire data set
import joblib
joblib.dump(xg_final,'Car_Bazaar')
model = joblib.load('Car_Bazaar')
import pandas as pd
data_new = pd.DataFrame({
    'Present_Price':5.59,
    'Kms_Driven':27000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':0,
    'Owner':0,
    'Age':0
},index=[0])
from tkinter import *
root=Tk()

root.geometry("644x344")

def new():
    new=Toplevel(root)
    new.geometry("710x500")
    new.resizable(0,0)
    new.title("CAR BAZAAR")
    new.config(bg="#FFE4C4")
    Label(new,text="CAR BAZAAR",font="italic 20 bold",pady=30,padx=50,bg="#FFE4C4").grid(row=0,column=0)
    

    def show_entry_feilds():
        p1=float(e1.get())
        p2=float(e2.get())
        p3=float(e3.get())
        p4=float(e4.get())
        p5=float(e5.get())
        p6=float(e6.get())
        p7=float(e7.get())

        model = joblib.load('Car_Bazaar')
        data_new = pd.DataFrame({
            'Present_Price':p1,
            'Kms_Driven':p2,
            'Fuel_Type':p3,
            'Seller_Type':p4,
            'Transmission':p5,
            'Owner':p6,
            'Age':p7
        },index=[0])

        result = model.predict(data_new)
        Label(new,text="Car Purchase Amount").grid(row=8)
        Label(new,text=result).grid(row=10)
        print("Car Purchase Amount : ",result[0])

    # master =Tk()
    # master.title("Car Price Prediction Using Machine Learning")
    # label = Label(master,text="Car Price Prediction Using Machine Learning",bg="black",fg="white").grid(row=0,columnspan=2)

    Label(new, text="Present_Price",bg="#FFE4C4").grid(row=1,column=0)
    Label(new, text="Kms_Driven",bg="#FFE4C4").grid(row=2,column=0)
    Label(new, text="Fuel_Type",bg="#FFE4C4").grid(row=3,column=0)
    Label(new, text="Seller_Type",bg="#FFE4C4").grid(row=4,column=0)
    Label(new, text="Transmission",bg="#FFE4C4").grid(row=5,column=0)
    Label(new, text="Owner",bg="#FFE4C4").grid(row=6,column=0)
    Label(new, text="Age",bg="#FFE4C4").grid(row=7,column=0)

    e1 = Entry(new)
    e2 = Entry(new)
    e3 = Entry(new)
    e4 = Entry(new)
    e5 = Entry(new)
    e6 = Entry(new)
    e7 = Entry(new)

    e1.grid(row=1,column=1)
    e2.grid(row=2,column=1)
    e3.grid(row=3,column=1)
    e4.grid(row=4,column=1)
    e5.grid(row=5,column=1)
    e6.grid(row=6,column=1)
    e7.grid(row=7,column=1)

    Button(new,text='PREDICT',command=show_entry_feilds).grid()


# photo = PhotoImage(file="p.png")
# label = Label(root,image=photo)
# label.pack()


# image1 = Image("C:\Users\Tanveer Singh\Desktop\images\red car.png")
# img = ImageTk.PhotoImage(Image.open("red_car.jpg"))
# Label(root, image = img).grid(row=0,column=7)

# photo = PhotoImage(file = "C:\Users\Tanveer Singh\Desktop\images\red car.png")
# photo = photo.subsample(2)
# lbl = Label(root,image = photo)
# lbl.image = photo
# lbl.grid(column=7, row=0)

root.geometry("710x500")
root.title("CAR BAZAAR")
root.config(bg='#00FFFF')
root.resizable(0,0)
Label(root,text="A1 CAR BAZAAR",font="CooperBlack 24 bold ",pady=30,padx=225,fg="blue",bg="skyblue",relief=RIDGE).grid(row=0,column=5)
Label(root,text="Want To Find Out The BEST Selling Price Of Your Used Car ?",font="BerlinSansFBDemi 15 bold",padx=5,pady=40,bg="#00FFFF").grid(row=1,column=5)
Label(root,text=" ",padx=5,pady=25,bg='#00FFFF').grid(row=2,column=5)
Button(text="LETS GO!!!",font="ArialBlack 15 bold",padx=10,pady=20,relief=RIDGE,bg="#F0FFFF",fg="red",command=new).grid(row=5,column=5)

# root.title("CAR BAZAAR")
# root.config(bg='#00FFFF')
# # root.resizable(0,0)
# Label(root,text="A1 CAR BAZAAR",font="Arial 24 bold ",pady=30,padx=225,fg="blue",bg="skyblue",relief=RIDGE).grid(row=0,column=5)
# Label(root,text="Want To Find Out The BEST Selling Price Of Your Used Car ?",font="BerlinSansFBDemi 13 bold",padx=5,pady=20,bg="#00FFFF").grid(row=1,column=5)
# Label(root,text=" ",padx=5,pady=25,bg='#00FFFF').grid(row=2,column=5)
# Button(text="LEST GO!!!",font="ArialBlack 15 bold",padx=10,pady=20,relief=RIDGE,bg="#F0FFFF",fg="red",command=new).grid(row=5,column=5)


root.mainloop()