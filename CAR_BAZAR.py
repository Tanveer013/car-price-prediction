from tkinter import *
import joblib
import pandas as pd
import seaborn as sns
from sklearn import metrics

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
    Label(master,text="Car Purchase Amount").grid(row=8)
    Label(master,text=result).grid(row=10)
    print("Car Purchase Amount",result[0])

master =Tk()
master.title("Car Price Prediction Using Machine Learning")
label = Label(master,text="Car Price Prediction Using Machine Learning",bg="black",fg="white").grid(row=0,columnspan=2)

Label(master, text="Present_Price").grid(row=1)
Label(master, text="Kms_Driven").grid(row=2)
Label(master, text="Fuel_Type").grid(row=3)
Label(master, text="Seller_Type").grid(row=4)
Label(master, text="Transmission").grid(row=5)
Label(master, text="Owner").grid(row=6)
Label(master, text="Age").grid(row=7)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)

Button(master,text='PREDICT',command=show_entry_feilds).grid()

mainloop()
