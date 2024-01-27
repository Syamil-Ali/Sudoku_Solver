import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core import img_preprocessing as sp
from core import backtrack_sudoku_solver as bss
from core import small_ocr_model as som
#import time
#import timeout_decorator

#from streamlit_webrtc import webrtc_streamer

#webrtc_streamer(key="sample")
# define max process time
max_execution_time = 180

# debug table cmd
def dimension_array(sudoku_holder_model):
    sudoku_board_vir = [[]]

    row = 0 #start at first row
    for counter, num in enumerate(sudoku_holder_model): #loop by column
        if counter % 9 == 0 and counter != 0:
            sudoku_board_vir.append([])
            row += 1
            
        sudoku_board_vir[row].append(num)

    return sudoku_board_vir

def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")

# Working Funtion
#@timeout_decorator.timeout(max_execution_time, timeout_exception=TimeoutError)
def working(img, hline_cond):

    # To read image file buffer with OpenCV:
    bytes_data = img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    #st.write(cv2_img.shape)
    # do the img preprocessing
    result_img, result_img_bin = sp.img_preprocess(cv2_img, hline_cond)

    # do the second part
    result_img, coor = sp.preprocess_p2(result_img, result_img_bin)
    #print(result_img)

    st.write(' ')
    st.write(' ')
    st.write('Preprocessing')
    st.write(result_img.shape)
    st.image(result_img)
    st.image(result_img_bin)




    # using the ocr model to do identify sudoku number

    sudoku_holder_model = [] # to hold the pred sudoku board num
    #image_copy = sudoku_img_crop_binary.copy()



    for coordinate in coor:
        
        
        cell = result_img_bin[coordinate[0][1]:coordinate[1][1], coordinate[0][0]:coordinate[1][0]]

        #----------- 3.1 ---------# 
        #to get mid coordinate to use for cropping the number only 
        mid_x = cell.shape[0] // 2 
        mid_y = cell.shape[1] // 2 
        
        #----------- 3.2 ---------# 
        #predefine the radius  
        radius = int((cell.shape[0] * (3/4)) // 2) 
        
        #----------- 3.3 ---------# 
        # crop the image num
        num = cell[mid_x-radius:mid_x+radius, mid_y-radius:mid_y+radius]
        
        num = sp.preprocessed_mini_num(num, mid_x,mid_y)
        img = num.astype('float32',casting='same_kind')
        
        cell_predict = som.predict_img(img)
        sudoku_holder_model.append(cell_predict)


        #st.image(num)
        #st.write(cell_predict)
    st.image(result_img_bin)
    #st.write(sudoku_holder_model)
    #print(sudoku_holder_model)

    #st.write(sudoku_holder_model)

    # check if the clue less than 20 then exist
    filtered_list = [value for value in sudoku_holder_model if value != 0]

    if len(filtered_list) < 20:
        return result_img, sudoku_holder_model

    # do the backtracking
    sudoku_solved = bss.main_backtrack_solve(sudoku_holder_model)


    # output the result
    output_sudoku = sp.draw_sudoku(result_img, sudoku_holder_model, sudoku_solved, coor)
    st.write(' ')
    st.write(' ')
    st.write('Answer:..')
    
    debug_sudoku = dimension_array(sudoku_holder_model)
    print('')
    print_board(debug_sudoku)

    return output_sudoku, sudoku_solved




st.title(' 🤓 Sudoku Solver 🤓', anchor=None)
st.divider()

# create a two button 
# 1. use camera, 2. upload file

genre = st.radio(
    "**Mode**",
    ["Camera 📸", "Upload Image 🖼️", "Sample Image ☹️"],
    captions = ["Take a picture of sudoku using camera", "Upload sudoku image", "Lazy? try out sample image that works"])




if genre == "Camera 📸":

    st.divider()



    img_file_buffer = st.camera_input("Take a picture")


    if img_file_buffer is not None:

        output_sudoku, sudoku_solved = working(img_file_buffer, True)

        # try:
        #     start_time = time.time()
        #     output_sudoku, sudoku_solved = working(img_file_buffer, True)
        # except TimeoutError:
        #     print("Execution time exceeded 3 minutes.")
        #     output_sudoku, sudoku_holder_model = working(img_file_buffer, False)

        if 0 in sudoku_solved:

            output_sudoku, sudoku_holder_model = working(img_file_buffer, False)


        st.image(output_sudoku)

        

else:

    st.divider()
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        output_sudoku, sudoku_solved = working(uploaded_file, True)

        # try:
        #     start_time = time.time()
        #     output_sudoku, sudoku_solved = working(uploaded_file, True)
        # except TimeoutError:
        #     print("Execution time exceeded 3 minutes.")
        #     output_sudoku, sudoku_holder_model = working(uploaded_file, False)

        if 0 in sudoku_solved:

            output_sudoku, sudoku_holder_model = working(uploaded_file, False)


        st.image(output_sudoku)
        
