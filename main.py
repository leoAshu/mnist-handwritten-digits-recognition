import cv2 as cv
import numpy as np
import keras.models as models

def main():
    # initialize model
    model = models.load_model('model/mnistModel.h5')

    # initialize video writer object
    out = cv.VideoWriter('data/output.mp4', -1, 20.0, (512,512))


    # input user choice
    # 0 - Use Webcam
    # 1 - Use Saved Video
    option = input('Enter:\n0 - Capture Webcam\n1 - Capture Saved Video\n')

    # loop until correct choice entered
    while option != '0' and option != '1':
        option = input('Incorrect Choice.... Re-enter:\n0 - Capture Webcam\n1 - Capture Saved Video\n')

    video = None
    if option == '0':
        # webcam videocapture object
        video = cv.VideoCapture(0)
    else:
        # saved video videocapture object
        video = cv.VideoCapture('data/video.mov')
    
    flag = True
    while(flag):
        # read frame of the video/webcam
        ret, frame = video.read()

        # exit when video finished
        if not ret:
            print('video end!')
            break

        # resize the frame: 512 x 512
        frame = cv.resize(frame, (512, 512))

        # preprocess frame
        # convert to grayscale
        # invert the image
        # binarize the image
        frame_preprocessed = preprocess_image(frame)

        # extract contours
        contours = cv.findContours(frame_preprocessed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # list to store individual digits
        digits = []

        for contour in contours:
            # extract co-ordinates and dimensions of boxes
            x, y, w, h = cv.boundingRect(contour)
            x = max(x-10, 0)
            y = max(y-10, 0)
            w = min(w+20, 512)
            h = min(h+20, 512)

            # draw bounding box on frame
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            
            # extract the digits for prediction
            roi = frame_preprocessed[y:y+h, x:x+w]

            # resize rois
            # convert from rectange to square
            # no change to aspect ratio
            roi = resize(roi, max(w, h))

            # add a digit to the list
            digits.append(roi)

            # make roi model-input compatible
            # resize
            roi = cv.resize(roi, (28, 28))
            # reshape
            roi = roi.reshape((1, 28, 28, 1))
            # normalize
            roi = roi.astype('float32') / 255

            # predict the probabilities for the roi
            prediction_prob = list(model.predict(roi)[0])

            # extract the label
            # the index of the highest probability
            num = prediction_prob.index(max(prediction_prob))

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(
                frame, 
                str(num), 
                (x, y), 
                font,
                1, 
                (0, 255, 0), 
                2, 
                cv.LINE_AA
            )

        # display digits
        cv.imshow('roi-1', digits[0])
        cv.imshow('roi-2', digits[1])
        cv.imshow('roi-3', digits[2])
        cv.imshow('roi-4', digits[3])

        # display image with bounding boxes
        cv.imshow('output', frame)

        # write the frame to the output video
        out.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv.destroyAllWindows()


def preprocess_image(image):
    # convert color image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # invert the grayscaled image
    # ensures proper contour extraction
    image_gray_inv = cv.bitwise_not(image_gray)

    # binarize the image
    (_, image_binary) = cv.threshold(image_gray_inv, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # thresh = 128
    # image_binary = cv.threshold(image_gray_inv, thresh, 255, cv.THRESH_BINARY)[1]

    return image_binary



def resize(image, size):
    if 0 in image.shape:
        return image
    if size == 0:
        return image
    # get dimensions of the image
    h, w = len(image), len(image[0])

    # horizontal padding
    if h == size:
        pad = size - w
        if pad == 1:
            image = np.insert(image, 0, 0, axis=1)
        else:
            while pad > 0:
                idx = len(image[0])
                if pad == 1:
                    image = np.insert(image, 0, 0, axis=1)
                else:
                    image = np.insert(image, idx-1, 0, axis=1)
                    image = np.insert(image, 0, 0, axis=1)
                pad -= 2
    # vertical padding
    else:
        pad = size - h
        if pad == 1:
            image = np.insert(image, 0, 0, axis=0)
        else:
            while pad > 0:
                idx = len(image)
                if pad == 1:
                    image = np.insert(image, 0, 0, axis=0)
                else:
                    image = np.insert(image, idx-1, 0, axis=0)
                    image = np.insert(image, 0, 0, axis=0)
                pad -= 2
    
    # return the square image
    return image



if __name__ == '__main__':
    main()