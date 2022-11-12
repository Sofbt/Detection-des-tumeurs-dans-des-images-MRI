import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import numpy
from keras.models import load_model
model = load_model('C:/Users/lenovo/Brain_Tumor_Classification/tumortest9_10epoch.h5')
classes = { 
    0:'Normal',
    1:'Tumeur'
}

top=tk.Tk()
top.geometry('800x600')
top.title('Classification des Tumeurs Cérébrales')
top.configure(background='#e7ebe0')
label=Label(top,background='#e7ebe0', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path).convert('RGB')
    image = image.resize((224,224))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image/255
    pred = model.predict(image)
    pred = np.argmax(pred, axis= 1)
    print(pred)
    if pred > 0.9:
        pred = 1
    else: 
        pred = 0
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#00203f', text=sign)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classifier L'Image",
    command=lambda: classify(file_path),
    padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)


def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Importer une Image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Classification des Tumeurs Cérébrales",pady=20, font=('arial',20,'bold'))
heading.configure(background='#e7ebe0',foreground='#364156')
heading.pack()
top.mainloop()