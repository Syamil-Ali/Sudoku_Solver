import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

#setting up matplotlip img viewer

def visualize(img, gray = False, title = ""):
    
    if gray == False:
        plt.figure(figsize=[5,5]) 
        plt.axis('off')
        plt.title(title)
        plt.imshow(img[:,:,::-1])
        #plt.show()
    else:
        plt.figure(figsize=[5,5]) 
        plt.axis('off')
        plt.title(title)
        plt.imshow(img, cmap = 'gray')
    

# function to be apply to detect the distance of contour with the middle of pic
def detect_close_contour(img, contour_list):

    # sort the contour from largest to smallest
    sort_contourse = sorted(contour_list, key=cv2.contourArea, reverse=True)

    # get the mid coor of the image
    scx = int(img.shape[0] / 2)
    scy = int(img.shape[1] / 2)

    # create a threshold len (close with the middle)
    if img.shape[0] > img.shape[1]:
        threshold_len = int(img.shape[0] * 0.2)
    else:
        threshold_len = int(img.shape[1] * 0.2)


    # place to store the contour
    new_contour_hold = []

    # start the process
    for c in sort_contourse[:10]:
    
        #  get the mid of contour
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cv2.circle(img, (cx, cy), 7, (0, 255, 0), -1)
            
            
            # check the distance 
            distance = math.dist((scx, scy), (cx,cy))
            
            if distance < threshold_len:
                new_contour_hold.append(c)


    return new_contour_hold



# function to clean the number inside the image (use for when predicting)
def preprocessed_mini_num(num, mid_x, mid_y):
    
    #kernel = np.ones((1, 1),np.uint8)
    #num = cv2.dilate(num, kernel, iterations = 1)
    #num = cv2.erode(num, kernel, iterations=1) 
    
   # num = num.astype(np.uint8)

    mini_contours, _ = cv2.findContours(num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    new_mini_contour_hold = []

    # give threshold len
    if num.shape[0] > num.shape[1]:
        threshold_len = int(num.shape[0] * 0.45)
    else:
        threshold_len = int(num.shape[1] * 0.45)
        
        
    #print(len(mini_contours))
    #print(len(new_mini_contour_hold))

    

    for c in mini_contours:
        #print('contour detected')
        
        area = cv2.contourArea(c)
        #print(area)
        
        #print(c)

        #  get the mid of contour
        M = cv2.moments(c)
        if M['m00'] != 0:
            #print('here')
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cv2.circle(image_copy_2, (cx, cy), 7, (0, 255, 0), -1)

            # check the distance 
            distance = math.dist((mid_x, mid_y), (cx,cy))
            #print(distance)
            #print(threshold_len)

            if distance >= threshold_len or area < 20:
                #print('here huhhu')
                #new_mini_contour_hold.append(c)
                #num = cv2.drawContours(num, c, contourIdx=-1, color=(255,255,255), thickness=cv2.FILLED)
                num = cv2.drawContours(num, [c], 0, (0, 0, 0), thickness=cv2.FILLED)
                #cv2.drawContours(binary_image, [contour_coordinates], 0, (0, 0, 0), thickness=cv2.FILLED)


        else:
            #print('get 76 part')
            
            num = cv2.drawContours(num, c, 0, 0, thickness=cv2.FILLED)
    
    
    return num


## PLACEHOLDER
#sudoku_img = cv2.imread('result.png')


def img_preprocess(sudoku_img):


    # test to resize image
    scale_percent = 150 # percent of original size
    width = int(sudoku_img.shape[1] * scale_percent / 100)
    height = int(sudoku_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    sudoku_img = cv2.resize(sudoku_img, dim, interpolation = cv2.INTER_AREA)


    # denoising the img first
    sudoku_img = cv2.fastNlMeansDenoisingColored(sudoku_img,None,7,7,5,15)

    # resize the image - to focus on the center part -> crop 10% of t,b,l,r
    x_shape = sudoku_img.shape[0]
    y_shape = sudoku_img.shape[1]

    # get 3% from height and width
    x_remove = int(x_shape * 0.03)
    y_remove = int(y_shape * 0.03)

    sudoku_img = sudoku_img[0 + x_remove:x_shape - x_remove, 0 + y_remove:y_shape-y_remove]



    # a few steps to do the preprocessing

    # 1. convert the image to grayscale color
    sudoku_gray = cv2.cvtColor(sudoku_img,cv2.COLOR_BGR2GRAY)
    # 2 --> Invert the image in which the black -> white and vice versa (to fit the condition for perfect contour)
    sudoku_gray_inverted = cv2.bitwise_not(sudoku_gray)
    # 3 --> apply appropriate threshold to the image
    #the value inverted is found based on try and error*
    _, sudoku_binary = cv2.threshold(sudoku_gray_inverted, 100, 225, cv2.THRESH_BINARY) #if pixel color > 50 then it will be 255 = white
    # 4 --> apply contour detection 
    contours, hierarchy = cv2.findContours(sudoku_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 4.update -> get the contour close with the mid of image
    new_contour = detect_close_contour(sudoku_binary, contours)
    
    # 5 --> find the largest area of the contour image and get the corresponding coordinate
    biggest_contour = max(new_contour, key = cv2.contourArea)
    
    # 6 --> crop the image
    #get the x,y, width and height from the biggest contour found
    x, y, w, h = cv2.boundingRect(biggest_contour)
        
    #applied the crop
    sudoku_img_crop = sudoku_img[y:y+h,x:x+w]
    sudoku_img_crop_binary = sudoku_binary[y:y+h,x:x+w]


    # 6.1 beta do erode the pic -> to get the num more better
    kernel = np.ones((1, 1),np.uint8)
    sudoku_img_crop_binary = cv2.erode(sudoku_img_crop_binary, kernel, iterations=1) 


    # 7. eed to remove excess line using hough lines detection
    lines = cv2.HoughLines(sudoku_img_crop_binary, 1, np.pi / 180, threshold=180)

    # Convert the detected lines to black color
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv2.line(sudoku_img_crop_binary, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return sudoku_img_crop, sudoku_img_crop_binary

# ## ----------------------------------------------- ##
# ### PART 2 for

def preprocess_p2(sudoku_img_crop, sudoku_img_crop_binary): # to format properly into sudoku table

    # 1 --> divide the image into 9 section (horizontal and vertical)

    #will use the crop sudoku image for divide the section
    x_range = sudoku_img_crop.shape[0] // 9
    y_range = sudoku_img_crop.shape[1] // 9
    image_copy = sudoku_img_crop.copy()


    # 2 --> get intersection coordinate by using the divided value

    x_start = 0
    y_start = 0

    coor = []

    for row in range(9):
        x_bottom = 0
        x_start = 0
        for column in range(9):
            x_bottom = x_start + x_range
            y_bottom = y_start + y_range
            cv2.rectangle(image_copy,(x_start,y_start),(x_bottom, y_bottom),(0,255,0),3)
            coor.append([[x_start,y_start],[x_bottom,y_bottom]])
            x_start += x_range
            
        y_start += y_range


    return image_copy, coor



# Draw output result
def draw_sudoku(img, sudoku_holder_model, sudoku_solved, coor):

    for counter, coordinate in enumerate(coor):


        scale_x = ((coordinate[1][0] - coordinate[0][0]) // 2) * (1/3)
        scale_y = ((coordinate[0][1] - coordinate[1][1]) // 2) * (7/3)

        mid_x = ((coordinate[1][0] - coordinate[0][0]) // 2) + coordinate[0][0] - int(scale_x)
        mid_y = ((coordinate[0][1] - coordinate[1][1]) // 2) + coordinate[0][1] - int(scale_y)

        if sudoku_holder_model[counter] == 0:
            #print('true')
            new = cv2.putText(img, str(sudoku_solved[counter]), (mid_x,mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (255,0,0), 2)


        
    return img



    
