import pygame
import PySimpleGUI as sg



# #colours
# BLUE = (0,0,255)


# isPressed = False

# screen = pygame.display.set_mode((512,512))

# def drawCircle(screen,x,y):
#   pygame.draw.circle(screen,BLUE,(x,y),5)


# while True:
#   for event in pygame.event.get():
#     if event.type == pygame.MOUSEBUTTONDOWN:
#       isPressed = True
#     elif event.type == pygame.MOUSEBUTTONUP:
#       isPressed = False
#     if event.type == pygame.MOUSEMOTION and isPressed == True:
#       (x,y) = pygame.mouse.get_pos()
#       drawCircle(screen,x,y)
#   pygame.display.flip()



image_drawer_column = [
    [sg.Text('Image Drawer')],
    [sg.Text('', size=(10, 1), key='-OUTPUT-')],
    [sg.Button('Clear', size=(10, 1)), sg.Button('Exit', size=(10, 1))],
]

image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text("Predictions Plot")],
    [sg.Canvas(key="-CANVAS-")],
]


layout = [
    [
        sg.Column(image_drawer_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("WasteNet Demo", layout, finalize=True, 
    element_justification="center", font="Helvetica 14",)

#main logic loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
      
window.close()