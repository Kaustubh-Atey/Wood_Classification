# Wood_Classification

Steps for Implementing the Wood Classification Model

Download the weights folder from the drive

link - https://drive.google.com/drive/folders/1XO8w1BFL0SaFq14GbOTwoqFaSSAYBdhY?usp=sharing

1. Store the output of the segmentation model in Test_Images folder (the floor.jpg image)

2. In the cropped.py file (line no:-5 - change the img_name to the output of the segmentation model(floor img) i.e stored in the Test_Images folder)

Example - img_name = 'oak_floor.jpg' 
          input_picture = cv2.imread('Test_Images/'+img_name)

3. Run the cropped.py file

Example - python cropped.py

Output - same image name but will be stored in the Cropped_Images folder.

Example - oak_floor.jpg image stored in the Cropped_Images folder.


4. In the inference.py(line no:-50 - change the image name to the output of the cropped.py stored in the Cropped_Images folder)

Example - image_path = 'Cropped_Images/oak_floor.jpg'

5. Run the inference.py file

Example - python inference.py

Output - Prediction and confidence


# New wood model for 5 new classes

Weight link - https://drive.google.com/drive/folders/1a7oDX51mzy0QWBraSr-D9knCWjCZqqfz?usp=sharing
