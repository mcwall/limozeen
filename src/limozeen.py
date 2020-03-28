import numpy as np
import cv2 as cv
import time


def load_video_file():
    cap = cv.VideoCapture('../content/metal.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cv.imshow('frame', gray)
        cv.imshow('frame', frame)
        time.sleep(0.012)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def get_fret_coords(frame):
    (height, width, _) = frame.shape

    bottomScale = 0.27
    topScale = 0.425
    heightScale = 0.46

    bottomL = width*bottomScale
    bottomR = width - bottomL
    bottomY = height - 2

    topL = width*topScale
    topR = width - topL
    topY = height * heightScale

    # top-left clockwise
    return np.array([[topL,topY], [topR,topY], [bottomR,bottomY], [bottomL,bottomY]], np.int32)
    

def get_transformed_img(frame, coords):
    fret_img = np.copy(frame)
    topLeft = coords[0]
    bottomLeft = coords[3]
    bottomRight = coords[2]

    print(bottomLeft[0], bottomRight[0])

    return fret_img[topLeft[1]:bottomLeft[1], bottomLeft[0]:bottomRight[0]]

def output(frame, fret):
    (height, frameWidth, _) = frame.shape
    (fretHeight, fretWidth, _) = fret.shape
    width = int((frameWidth + fretWidth) * 1.1)

    output = np.zeros((height, width, 3), np.uint8)
    output[0:height, 0:frameWidth] = frame
    output[0:fretHeight, width-fretWidth:width] = fret

    return output


def process_frame(frame):
    (height, width, _) = frame.shape

    cv.putText(frame,'Width: ' + str(width), (0,height-60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA, False)
    cv.putText(frame,'Height: ' + str(height), (0,height-30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA, False)
    
    pts = get_fret_coords(frame)
    poly_pts = pts.reshape((-1,1,2))
    cv.polylines(frame,[pts],True,(0,255,255))

    fret_img = get_transformed_img(frame, pts)

    return output(frame, fret_img)


def transform_video_file():
    # init stream and window
    cap = cv.VideoCapture('../content/metal.mp4')
    cv.namedWindow('vidstream', cv.WINDOW_NORMAL)
    cv.resizeWindow('vidstream', 1280, 720)

    paused = False
    while cap.isOpened():
        ret, frame = cap.read()

        # check for input
        key_press = cv.waitKey(1)
        if key_press & 0xFF == ord('q'):
            break
        if key_press & 0xFF == ord('p'):
            paused = not paused
            #cv.imwrite('../content/screenshot.png',frame)

        if paused:
            continue

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        process_frame(frame)
        cv.imshow('vidstream', frame)

        #time.sleep(0.0012)

    cap.release()
    cv.destroyAllWindows()

def transform_img():
    frame = cv.imread('../content/screenshot.png',cv.IMREAD_UNCHANGED)
    cv.namedWindow('vidstream', cv.WINDOW_NORMAL)
    cv.resizeWindow('vidstream', 1920, 720)
    
    img = process_frame(frame)
    cv.imshow('vidstream', img)    

    cv.waitKey()
    cv.destroyAllWindows()


transform_img()

