import cv2
import numpy as np

# Import your picture
img_name = 'oak_floor.jpg'
input_picture = cv2.imread('Test_Images/'+img_name)
change_mask = input_picture[:,:,2] == 255
input_picture[change_mask]=[0, 0, 0]

# Color it in gray
gray = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)

# Create our mask by selecting the non-zero values of the picture
ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

cont, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in cont]
max_index = np.argmax(areas)

cnt=cont[max_index]


contour = cont[max_index].reshape(len(cont[max_index]),2)
rect = []
for i in range(len(contour)):
    x1, y1 = contour[i]
    for j in range(len(contour)):
        x2, y2 = contour[j]
        area = abs(y2-y1)*abs(x2-x1)
        rect.append(((x1,y1), (x2,y2), area))

# the first rect of all_rect has the biggest area, so it's the best solution if he fits in the picture
all_rect = sorted(rect, key = lambda x : x[2], reverse = True)


# we take the largest rectangle we've got, based on the value of the rectangle area
# only if the border of the rectangle is not in the black part

# if the list is not empty
if all_rect:
    
    best_rect_found = False
    index_rect = 0
    nb_rect = len(all_rect)
    
    # we check if the rectangle is  a good solution
    while not best_rect_found and index_rect < nb_rect:
        
        rect = all_rect[index_rect]
        (x1, y1) = rect[0]
        (x2, y2) = rect[1]
        
        valid_rect = True
        
        # we search a black area in the perimeter of the rectangle (vertical borders)
        x = min(x1, x2)
        while x <max(x1,x2)+1 and valid_rect:
            if mask[y1,x] == 0 or mask[y2,x] == 0:
                # if we find a black pixel, that means a part of the rectangle is black
                # so we don't keep this rectangle
                valid_rect = False
            x+=1
        
        y = min(y1, y2)
        while y <max(y1,y2)+1 and valid_rect:
            if mask[y,x1] == 0 or mask[y,x2] == 0:
                valid_rect = False
            y+=1
            
        if valid_rect:
            best_rect_found = True
        
        index_rect+=1
        
    if best_rect_found:
        # Finally, we crop the picture and store it
        result = input_picture[min(y1, y2):max(y1, y2), min(x1,x2):max(x1,x2)]
        print(result.shape)
        if result.shape[0] > 224 and result.shape[1] > 224:  
            result = result[0:224, 0:224]
            
            cv2.imwrite('Cropped_Images/'+str(img_name), result)
        elif result.shape[0] < 224 and result.shape[1] > 224:
            result = result[::, 0:224]
            
            cv2.imwrite('Cropped_Images/'+str(img_name), result)

        elif result.shape[0] > 224 and result.shape[1] < 224:
            result = result[0:224, ::]
          
            cv2.imwrite('Cropped_Images/'+str(img_name), result)    

        else:
            
            cv2.imwrite('Cropped_Images/'+str(img_name), result)  
    else:
        print("No rectangle fitting into the area")
    
else:
    print("No rectangle found")

print('done')    