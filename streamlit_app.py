import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core import img_preprocessing as sp
from core import backtrack_sudoku_solver as bss
from core import small_ocr_model as som

#from streamlit_webrtc import webrtc_streamer

#webrtc_streamer(key="sample")

img_file_buffer = st.camera_input("Take a picture")


def visualize(img, gray = False, title = ""):
    
    if gray == False:
        #plt.figure(figsize=[5,5]) 
        #plt.axis('off')
        #plt.title(title)
        #plt.imshow(img[:,:,::-1])
        #plt.show()
        return img[:,:,::-1]

    else:
        plt.figure(figsize=[5,5]) 
        plt.axis('off')
        plt.title(title)
        plt.imshow(img, cmap = 'gray')


if img_file_buffer is not None:

    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    #st.write(cv2_img.shape)
    # do the img preprocessing
    result_img, result_img_bin = sp.img_preprocess(cv2_img)

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
    # st.image(result_img_bin)
    # st.write(sudoku_holder_model)

    #st.write(sudoku_holder_model)

    # do the backtracking
    sudoku_solved = bss.main_backtrack_solve(sudoku_holder_model)


    # output the result
    output_sudoku = sp.draw_sudoku(result_img, sudoku_holder_model, sudoku_solved, coor)
    st.write(' ')
    st.write(' ')
    st.write('Answer:..')
    st.image(output_sudoku)
    
