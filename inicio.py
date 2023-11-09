import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math

st.header("TEST IMAGES")
st.write("¡Bienvenido a nuestra interfaz de prueba de imágenes! Selecciona una sección dentro del sidebar para empezar.")

dropped_images_folder = "DroppedImages"

os.makedirs(dropped_images_folder, exist_ok=True)

# File uploader
uploaded_files = st.file_uploader("Drop your images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Process and save uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(dropped_images_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    st.success(f"Saved all files sucessfull.")

def calculate_nitidez(image_path):
    #calcular nitidez

    return True

# Calcular la intensidad promedio en los canales R, G y B
def getImageIntensity(imagePath):
    # define limits
    min_limit = 170
    max_limit = 250
    img = cv2.imread(imagePath)
    
    roi = img[330:360, 290:340] # región cerca del centro (inferior)
    intensidad_rojo = np.mean(roi[:, :, 2])  # Canal Rojo
    intensidad_verde = np.mean(roi[:, :, 1])  # Canal Verde
    intensidad_azul = np.mean(roi[:, :, 0])   # Canal Azul

    roi2 = img[100:160, 290:340] # región cerca del centro (superior)
    intensidad_rojo2 = np.mean(roi2[:, :, 2])  # Canal Rojo
    intensidad_verde2 = np.mean(roi2[:, :, 1])  # Canal Verde
    intensidad_azul2 = np.mean(roi2[:, :, 0])   # Canal Azul

    roi3 = img[190:290, 190:240] # región cerca del centro (izquierda)
    intensidad_rojo3 = np.mean(roi3[:, :, 2])  # Canal Rojo
    intensidad_verde3 = np.mean(roi3[:, :, 1])  # Canal Verde
    intensidad_azul3 = np.mean(roi3[:, :, 0])   # Canal Azul

    roi4 = img[190:290, 420:460] # región cerca del centro (derecha)
    intensidad_rojo4 = np.mean(roi4[:, :, 2])  # Canal Rojo
    intensidad_verde4 = np.mean(roi4[:, :, 1])  # Canal Verde
    intensidad_azul4 = np.mean(roi4[:, :, 0])   # Canal Azul

    avg_redIntensity = (intensidad_rojo + intensidad_rojo2 + intensidad_rojo3 + intensidad_rojo4) / 4
    avg_greenIntensity = (intensidad_verde + intensidad_verde2 + intensidad_verde3 + intensidad_verde4) / 4
    avg_blueIntensity = (intensidad_azul + intensidad_azul2 + intensidad_azul3 + intensidad_azul4) / 4

    testRed = False
    testGreen = False
    testBlue = False

    if avg_redIntensity >= min_limit and avg_redIntensity <= max_limit : 
        testRed = True;
    
    if avg_greenIntensity >= min_limit and avg_greenIntensity <= max_limit : 
        testGreen = True;
    
    if avg_blueIntensity >= min_limit and avg_blueIntensity <= max_limit : 
        testBlue = True;
    
    return testRed, testGreen, testBlue

def calculate_red_intensity(image_path):
    r,g,a = getImageIntensity(image_path)
    return r

def calculate_green_intensity(image_path):
    r,g,a = getImageIntensity(image_path)
    return g

def calculate_blue_intensity(image_path):
    r,g,a = getImageIntensity(image_path)
    return a

def calculate_centrado(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    height, width, _ = img.shape
    new_img = np.ones((height, width), dtype=np.uint8) * 255
    new_img[1:height - 1 , 1:width - 1] = thresh[1:height - 1 , 1:width - 1]
    thresh = new_img
    center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
    lower = np.array([60,60,60])
    higher = np.array([250,250,250])
    mask = cv2.inRange(img, lower, higher)
    mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] < 0:
            # This is an external contour
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        else:
            # This is an internal contour
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    # Calculate the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the center of the bounding box
    centerS_x = x + (w / 2)
    centerS_y = y + (h / 2)

    # Centrado de la imagen con respecto al centro del cuadro
    # print("Distancia del centro real al centro de la imagen (",center_x - centerS_x,",",center_y - centerS_y,")")
    if ((center_x - centerS_x)>10 or (center_y - centerS_y)>10 or (center_x - centerS_x)<-10 or (center_y - centerS_y)<-10):
        result = False
    else:
        result = True
    return result

def quadrant_of_contour(contour, quadrants):
    x,y,w,h = cv2.boundingRect(contour)
    for idx, quadrant in enumerate(quadrants):
        if quadrant[0] <= x < quadrant[0] + quadrant[2] and \
            quadrant[0] <= x+w < quadrant[0] + quadrant[2] and \
            quadrant[1] <= y < quadrant[1] + quadrant[3] and \
            quadrant[1] <= y+h < quadrant[1] + quadrant[3]:
            return idx
    return -1

def calculate_orientacion(img, crop_margin=(3, 20)):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    height, width = thresh_img.shape
    mx, my = crop_margin

    right_x = width - mx
    while thresh_img[height - my][right_x] == 0 and right_x >= 0:
        right_x -= 1

    left_x = mx
    while thresh_img[my][left_x] == 0 and left_x >= 0:
        left_x += 1
    
    cropped = thresh_img.copy()
    cropped.fill(1)
    cropped[my:height - my, left_x:right_x] = thresh_img[my:height - my, left_x:right_x]

    contours, _ = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    quadrants = [
        (0, 0, width//2, height//2),
        (width//2, 0, width//2, height//2),
        (0, height//2, width//2, height//2),
        (width//2, height//2, width//2, height//2),
    ]

    for countour_idx, contour in enumerate(contours):
        quadrant = quadrant_of_contour(contour, quadrants)
        area = cv2.contourArea(contour)
        if quadrant == -1:
            continue
        if area < 2000:
            continue
        if quadrant == 1:
            return (True)
        
    return (False)

# Display results table
st.header("Resultados")
if uploaded_files:
    # Get a list of image names from the "DroppedImages" folder
    dropped_images_folder = "DroppedImages"
    image_names = [file for file in os.listdir(dropped_images_folder) if os.path.isfile(os.path.join(dropped_images_folder, file))]

    # Create a table with the image names and parameters
    st.write("### Image Results Table")

    # Table columns
    columns = ["Name", "Nitidez", "Red Intensity", "Green Intensity", "Blue Intensity", "Centrado", "Orientación"]

    # Initialize the data for the table
    table_data = {"Name": image_names}

    # List of parameter calculation functions
    parameter_functions = [
        calculate_nitidez,
        calculate_red_intensity,
        calculate_green_intensity,
        calculate_blue_intensity,
        calculate_centrado,
        calculate_orientacion
    ]

    # Calculate parameters and update the table data
    for column, parameter_function in zip(columns[1:], parameter_functions):
        parameter_values = [parameter_function(os.path.join(dropped_images_folder, name)) for name in image_names]
        table_data[column] = ["Pass" if value else "Fail" for value in parameter_values]

    # Display the table
    st.table(table_data)
else:
    st.write("No hay imagenes para desplegar. Por favor, ingresar imágenes para mostrar resultados.")
