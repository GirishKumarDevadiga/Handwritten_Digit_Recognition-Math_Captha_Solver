from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
import base64
import re
from io import BytesIO

app = Flask(__name__, template_folder='template', static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/', methods=['GET', 'POST'])
def home():

    # sorting the contours
    def sort_contours(cnts, method="left-to-right"):
    	# initialize the reverse flag and sort index
    	reverse = False
    	i = 0
    	# handle if we need to sort in reverse
    	if method == "right-to-left" or method == "bottom-to-top":
    		reverse = True
    	# handle if we are sorting against the y-coordinate rather than
    	# the x-coordinate of the bounding box
    	if method == "top-to-bottom" or method == "bottom-to-top":
    		i = 1
    	# construct the list of bounding boxes and sort them from top to
    	# bottom
    	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    		key=lambda b:b[1][i], reverse=reverse))
    	# return the list of sorted contours and bounding boxes
    	return (cnts)

    image_path = 'image/predicted_output.png'
    new_model = tf.keras.models.load_model('saved_model/my_model/')
    output = ''

    if request.method == 'GET':
            return render_template('index.html', outPut = output)
    else:   
            if request.form['btn'] != 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAncAAAE8CAYAAACivZdQAAAQOElEQVR4Xu3WQREAAAgCQelf2h43awMWH+wcAQIECBAgQIBARmCZJIIQIECAAAECBAiccecJCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQMO78AAECBAgQIEAgJGDchcoUhQABAgQIECBg3PkBAgQIECBAgEBIwLgLlSkKAQIECBAgQMC48wMECBAgQIAAgZCAcRcqUxQCBAgQIECAgHHnBwgQIECAAAECIQHjLlSmKAQIECBAgAAB484PECBAgAABAgRCAsZdqExRCBAgQIAAAQLGnR8gQIAAAQIECIQEjLtQmaIQIECAAAECBIw7P0CAAAECBAgQCAkYd6EyRSFAgAABAgQIGHd+gAABAgQIECAQEjDuQmWKQoAAAQIECBAw7vwAAQIECBAgQCAkYNyFyhSFAAECBAgQIGDc+QECBAgQIECAQEjAuAuVKQoBAgQIECBAwLjzAwQIECBAgACBkIBxFypTFAIECBAgQICAcecHCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQMO78AAECBAgQIEAgJGDchcoUhQABAgQIECBg3PkBAgQIECBAgEBIwLgLlSkKAQIECBAgQMC48wMECBAgQIAAgZCAcRcqUxQCBAgQIECAgHHnBwgQIECAAAECIQHjLlSmKAQIECBAgAAB484PECBAgAABAgRCAsZdqExRCBAgQIAAAQLGnR8gQIAAAQIECIQEjLtQmaIQIECAAAECBIw7P0CAAAECBAgQCAkYd6EyRSFAgAABAgQIGHd+gAABAgQIECAQEjDuQmWKQoAAAQIECBAw7vwAAQIECBAgQCAkYNyFyhSFAAECBAgQIGDc+QECBAgQIECAQEjAuAuVKQoBAgQIECBAwLjzAwQIECBAgACBkIBxFypTFAIECBAgQICAcecHCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQMO78AAECBAgQIEAgJGDchcoUhQABAgQIECBg3PkBAgQIECBAgEBIwLgLlSkKAQIECBAgQMC48wMECBAgQIAAgZCAcRcqUxQCBAgQIECAgHHnBwgQIECAAAECIQHjLlSmKAQIECBAgAAB484PECBAgAABAgRCAsZdqExRCBAgQIAAAQLGnR8gQIAAAQIECIQEjLtQmaIQIECAAAECBIw7P0CAAAECBAgQCAkYd6EyRSFAgAABAgQIGHd+gAABAgQIECAQEjDuQmWKQoAAAQIECBAw7vwAAQIECBAgQCAkYNyFyhSFAAECBAgQIGDc+QECBAgQIECAQEjAuAuVKQoBAgQIECBAwLjzAwQIECBAgACBkIBxFypTFAIECBAgQICAcecHCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQMO78AAECBAgQIEAgJGDchcoUhQABAgQIECBg3PkBAgQIECBAgEBIwLgLlSkKAQIECBAgQMC48wMECBAgQIAAgZCAcRcqUxQCBAgQIECAgHHnBwgQIECAAAECIQHjLlSmKAQIECBAgAAB484PECBAgAABAgRCAsZdqExRCBAgQIAAAQLGnR8gQIAAAQIECIQEjLtQmaIQIECAAAECBIw7P0CAAAECBAgQCAkYd6EyRSFAgAABAgQIGHd+gAABAgQIECAQEjDuQmWKQoAAAQIECBAw7vwAAQIECBAgQCAkYNyFyhSFAAECBAgQIGDc+QECBAgQIECAQEjAuAuVKQoBAgQIECBAwLjzAwQIECBAgACBkIBxFypTFAIECBAgQICAcecHCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQMO78AAECBAgQIEAgJGDchcoUhQABAgQIECBg3PkBAgQIECBAgEBIwLgLlSkKAQIECBAgQMC48wMECBAgQIAAgZCAcRcqUxQCBAgQIECAgHHnBwgQIECAAAECIQHjLlSmKAQIECBAgAAB484PECBAgAABAgRCAsZdqExRCBAgQIAAAQLGnR8gQIAAAQIECIQEjLtQmaIQIECAAAECBIw7P0CAAAECBAgQCAkYd6EyRSFAgAABAgQIGHd+gAABAgQIECAQEjDuQmWKQoAAAQIECBAw7vwAAQIECBAgQCAkYNyFyhSFAAECBAgQIGDc+QECBAgQIECAQEjAuAuVKQoBAgQIECBAwLjzAwQIECBAgACBkIBxFypTFAIECBAgQICAcecHCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQMO78AAECBAgQIEAgJGDchcoUhQABAgQIECBg3PkBAgQIECBAgEBIwLgLlSkKAQIECBAgQMC48wMECBAgQIAAgZCAcRcqUxQCBAgQIECAgHHnBwgQIECAAAECIQHjLlSmKAQIECBAgAAB484PECBAgAABAgRCAsZdqExRCBAgQIAAAQLGnR8gQIAAAQIECIQEjLtQmaIQIECAAAECBIw7P0CAAAECBAgQCAkYd6EyRSFAgAABAgQIGHd+gAABAgQIECAQEjDuQmWKQoAAAQIECBAw7vwAAQIECBAgQCAkYNyFyhSFAAECBAgQIGDc+QECBAgQIECAQEjAuAuVKQoBAgQIECBAwLjzAwQIECBAgACBkIBxFypTFAIECBAgQICAcecHCBAgQIAAAQIhAeMuVKYoBAgQIECAAAHjzg8QIECAAAECBEICxl2oTFEIECBAgAABAsadHyBAgAABAgQIhASMu1CZohAgQIAAAQIEjDs/QIAAAQIECBAICRh3oTJFIUCAAAECBAgYd36AAAECBAgQIBASMO5CZYpCgAABAgQIEDDu/AABAgQIECBAICRg3IXKFIUAAQIECBAgYNz5AQIECBAgQIBASMC4C5UpCgECBAgQIEDAuPMDBAgQIECAAIGQgHEXKlMUAgQIECBAgIBx5wcIECBAgAABAiEB4y5UpigECBAgQIAAAePODxAgQIAAAQIEQgLGXahMUQgQIECAAAECxp0fIECAAAECBAiEBIy7UJmiECBAgAABAgSMOz9AgAABAgQIEAgJGHehMkUhQIAAAQIECBh3foAAAQIECBAgEBIw7kJlikKAAAECBAgQeFrkAT0V9yMVAAAAAElFTkSuQmCC':
                # fetch and save image
                image_b64 = request.form['btn']
                image_data = re.sub('^data:image/.+;base64,', '', image_b64)
                image_data = base64.b64decode(image_data)
                image_PIL = Image.open(BytesIO(image_data))
                image_np = np.array(image_PIL)
                
                im = Image.fromarray(image_np)
                im.save("image/download.png")

                # loading the image
                image = cv2.imread('image/download.png', 1)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # apply binary thresholding
                ret,thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
                # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
                contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)   # cv2.RETR_TREE                 
                # draw contours on the original image
                image_copy = image.copy()
                cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                contours = sort_contours(contours)
     
                for i, cnt in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        top = int(0.05 * thresh.shape[0])
                        bottom = top
                        left = int(0.05 * thresh.shape[1])
                        right = left
                        roi = thresh[y-top:y+h+bottom, x-left:x+w+right]
                        try:
                                img = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
                        except:
                                return render_template('error.html')  

                        img = img.reshape(1, 28, 28, 1)
                        img = img / 255.0

                        pred = new_model.predict([img])[0]

                        final_pred = np.argmax(pred)
 
                        if final_pred == 10:
                            data = '-' + ' ' + str(int(max(pred)*100))+'%'
                        elif final_pred == 11:
                            data = '+' + ' ' + str(int(max(pred)*100))+'%'
                        elif final_pred == 12:
                            data = '*' + ' ' + str(int(max(pred)*100))+'%'
                        elif final_pred == 13:
                            data = '/' + ' ' + str(int(max(pred)*100))+'%'
                        else:
                            data = str(final_pred) + ' ' + str(int(max(pred)*100))+'%'


                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        color = (255, 255, 255)
                        thickness = 1
                        cv2.putText(image_copy, data, (x, y-5), font, fontScale, color, thickness)
                        output = output + data.split(' ')[0]
                try:
                        output = output + ' = ' + str(eval(output))
                except:
                        output = output + ' (Error parsing, default string)'   

                cv2.imwrite('static/image/predicted_output.png', image_copy)

                return render_template('index.html', imagePath = image_path, outPut = output)
            else:
                # loading the image
                image = cv2.imread('image/download.png', 1)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # apply binary thresholding
                ret,thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
                # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
                contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)                    
                # draw contours on the original image
                image_copy = image.copy()
                cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                contours = sort_contours(contours)
    
                for i, cnt in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        top = int(0.05 * thresh.shape[0])
                        bottom = top
                        left = int(0.05 * thresh.shape[1])
                        right = left
                        roi = thresh[y-top:y+h+bottom, x-left:x+w+right]
                        try:
                                img = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
                        except: 
                                return render_template('error.html')                                
                        img = img.reshape(1, 28, 28, 1)
                        img = img / 255.0

                        pred = new_model.predict([img])[0]

                        final_pred = np.argmax(pred)

                        if final_pred == 10:
                            data = '-' + ' ' + str(int(max(pred)*100))+'%'
                        elif final_pred == 11:
                            data = '+' + ' ' + str(int(max(pred)*100))+'%'
                        elif final_pred == 12:
                            data = '*' + ' ' + str(int(max(pred)*100))+'%'
                        elif final_pred == 13:
                            data = '/' + ' ' + str(int(max(pred)*100))+'%'
                        else:
                            data = str(final_pred) + ' ' + str(int(max(pred)*100))+'%'

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        color = (255, 255, 255)
                        thickness = 1
                        cv2.putText(image_copy, data, (x, y-5), font, fontScale, color, thickness)
                        output = output + data.split(' ')[0]
                try:
                        output = output + ' = ' + str(eval(output))
                except:
                        output = output + ' (Error parsing, default string)'   


                cv2.imwrite('static/image/predicted_output.png', image_copy)

                return render_template('index.html', imagePath = image_path, outPut = output)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
   app.run()
