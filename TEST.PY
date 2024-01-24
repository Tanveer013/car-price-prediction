
from tkinter import *
root=Tk()

root.geometry("710x500")
root.title("CAR BAZAAR")
root.config(bg='#00FFFF')
root.resizable(0,0)
Label(root,text="A1 CAR BAZAAR",font="CooperBlack 24 bold ",pady=30,padx=225,fg="blue",bg="skyblue",relief=RIDGE).pack()
Label(root,text="Want To Find Out The BEST Selling Price Of Your Used Car ?",font="BerlinSansFBDemi 15 bold",padx=5,pady=40,bg="#00FFFF").pack()
# Label(root,text=" ",padx=5,pady=25,bg='#00FFFF').grid(row=2,column=5)
Button(text="LETS GO!!!",font="ArialBlack 15 bold",padx=10,pady=20,relief=RIDGE,bg="#F0FFFF",fg="red",command=exit).pack()

photo = PhotoImage(file="p.png")
label = Label(root,image=photo)
label.pack(anchor=E)

root.mainloop()