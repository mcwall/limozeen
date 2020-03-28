import numpy as np
import cv2 as cv
import time, math

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
    frameHeight, frameWidth, ch = frame.shape
    topLeft = coords[0]
    topRight = coords[1]
    bottomLeft = coords[3]
    bottomRight = coords[2]

    width = bottomRight[0] - bottomLeft[0] + topRight[0] - topLeft[0]
    width = int(width / 2)
    height = int(math.sqrt(math.pow(bottomLeft[1] - topLeft[1], 2) + math.pow((bottomRight[0] - topRight[0]), 2))) + heightExp
    #height = frameHeight + heightExp

    in_coords = np.float32(coords)
    out_coords = np.float32([[0,0], [width,0], [width,height], [0,height]])
    M = cv.getPerspectiveTransform(in_coords, out_coords)
    # M, status = cv.findHomography(in_coords, out_coords)
    warped_img = cv.warpPerspective(frame, M, (width, frameHeight))

    return warped_img


def output(frame, fret):
    height, frameWidth, ch = frame.shape
    fretHeight, fretWidth, ch = fret.shape
    width = int((frameWidth + fretWidth) * 1.1)

    output = np.zeros((int(height), width, 3), np.uint8)
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
    cap = cv.VideoCapture('../content/metal1080.mp4')
    cv.namedWindow('vidstream', cv.WINDOW_NORMAL)
    cv.resizeWindow('vidstream', 1920, 720)

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
        
        cv.imshow('vidstream', process_frame(frame))

    cap.release()
    cv.destroyAllWindows()


def transform_img():
    global heightExp
    frame = cv.imread('../content/screenshot.png',cv.IMREAD_UNCHANGED)
    cv.namedWindow('vidstream', cv.WINDOW_NORMAL)
    cv.resizeWindow('vidstream', 1920, 720)

    while True:
        # check for input
        key_press = cv.waitKey(1)
        if key_press & 0xFF == ord('q'):
            break
        if key_press & 0xFF == ord('o'):
            heightExp += 1
        if key_press & 0xFF == ord('l'):
            heightExp -= 1

        cv.imshow('vidstream', process_frame(frame))

    cv.destroyAllWindows()

heightExp = 0

# Notes are distorted vertically; as the distance from top increases, note height decreases linearly
# Top note height has been measured to be 12% of total fret board size at the top, with a scaling factor of 1/12
# In other words note height shrinks by 1 pixel for every 12 pixels a note travels down the screen
# Rendered note height can thus be calculated as: renderedNoteHeigh = baseNoteHeight - (y * noteDistortion)
baseNoteHeight = 12
noteDistortion = 1 / 12

transform_img()
# transform_video_file()
