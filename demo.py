#import required libraries
from cgitb import text
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageTk
import mlflow.pytorch
import models
from torchvision import transforms
import write_2_obj

#load WasteNet model
print("loading model...")
logged_model = 'runs:/db4d5b48bfc54862a18d07f40e116f34/model'
model = mlflow.pytorch.load_model(logged_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

#get predictions from model
def predict(model, filename):
    image = Image.open(filename).convert('L')
    image = transform_s(image).unsqueeze(0)
    model.setup_test(image)
    model.forward()
    return model.y_fake.detach()


size = 128
transform_s = transforms.Compose([
    transforms.Resize(size=(size, size), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

invTrans = transforms.Compose([
    transforms.Normalize((0.),(1/0.5)),
    transforms.Normalize((-0.5),(1.))
])

#first column in window for selecting file
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [sg.Image(key="-SEG-")]
]

#column for showing selected image and prediction plot
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text("Predictions Plot")],
    [sg.Image(key="-TEXTURE-")],
    [sg.Image(key="-HEIGHT-")]
]

#set whole window layout
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

#create window
window = sg.Window("WasteNet Demo", layout, finalize=True, 
    element_justification="center", font="Helvetica 14",)

#main logic loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    #make list of file names from selected folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        #set list of files ending with .png or .jpg
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg"))
        ]
        #update window
        window["-FILE LIST-"].update(fnames)
        
    #when a file is selected from list
    elif event == "-FILE LIST-":
        #get filename
        filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
        
        #get prediction from image and display plot in canvas
        terrain = predict(model, filename)
        terrain = invTrans(terrain)
        terrain = terrain.squeeze().detach().cpu().numpy()*255
        texture = Image.fromarray(terrain[:][:][0:3].transpose(1, 2, 0).astype(np.uint8)).resize((400, 400))
        height = Image.fromarray(terrain[:][:][3].astype(np.uint8)).resize((400, 400))
        seg = Image.open(filename).resize((256, 256))
        
        window["-SEG-"].update(data=ImageTk.PhotoImage(seg))
        window["-TEXTURE-"].update(data=ImageTk.PhotoImage(texture))
        window["-HEIGHT-"].update(data=ImageTk.PhotoImage(height))
        
        write_2_obj.write_obj("3D", np.asarray(texture), np.asarray(height))

window.close()