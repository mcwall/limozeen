import numpy as np
import cv2 as cv
import time, math, copy, uuid, sys


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
    height = int(math.sqrt(math.pow(bottomLeft[1] - topLeft[1], 2) + math.pow((bottomRight[0] - topRight[0]), 2)))

    in_coords = np.float32(coords)
    out_coords = np.float32([[0,0], [width,0], [width,height], [0,height]])
    M = cv.getPerspectiveTransform(in_coords, out_coords)
    # M, status = cv.findHomography(in_coords, out_coords)
    warped_img = cv.warpPerspective(frame, M, (width, height))

    return warped_img


def filter_func(img):
    # output = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    global filter_grayscale, filter_type, filter_threshold

    if filter_grayscale:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, filter_threshold, 255, filter_type)
        output = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    else:
        ret, output = cv.threshold(img, filter_threshold, 255, filter_type)

    return output


def detection_filter(fret, zone):
    height, width, ch = fret.shape

    left = int(zone.left*width)
    right = int(zone.right*width)
    top = int(zone.top*height)
    bottom = int(zone.bottom*height)

    img = fret[top:bottom, left:right]
    filtered_image = filter_func(img)
    fret[top:bottom, left:right] = filtered_image

    zone.match(filtered_image)


def output(frame, fret, detection_rows):
    global font, colors, confidence_threshold

    # Initialize canvas
    height, frameWidth, ch = frame.shape
    fretHeight, fretWidth, ch = fret.shape
    width = int((frameWidth + fretWidth) * 1.4)
    output = np.zeros((int(height), width, 3), np.uint8)

    # Draw video frame
    output[0:height, 0:frameWidth] = frame

    # Draw detection zones on fret projections
    detection_bottom = height - 5
    detection_top = fretHeight
    for i, detection_row in enumerate(detection_rows):
        x = width - (len(detection_rows) - i) * fretWidth
        fret_copy = np.copy(fret)
        for idxZone, zone in enumerate(detection_row.zones):
            zleft = int(zone.left*fretWidth)
            zright = int(zone.right*fretWidth)
            ztop = int(zone.top*fretHeight)
            zbottom = int(zone.bottom*fretHeight)

            detection_filter(fret_copy, zone)
            cv.rectangle(fret_copy, (zleft, ztop), (zright, zbottom), (0,0,255), 3)

            x1 = int(x + (idxZone / 5 * fretWidth))
            x2 = int(x1 + fretWidth / 5)

            bar_top = detection_top + 35
            bar_bottom = detection_bottom - 35
            bar_height = bar_bottom - bar_top
            y = int(bar_bottom - zone.match_confidence * bar_height)

            fill = cv.FILLED if zone.match_confidence > confidence_threshold else 2
            cv.rectangle(output, (x1+2, y), (x2-2, bar_bottom), colors[zone.i], fill)
            cv.putText(output, str(int(100*zone.match_confidence)), (x1, detection_bottom), font, 1, colors[zone.i], 2, cv.LINE_AA)

        # Draw fret projection
        output[0:fretHeight, x: x+fretWidth] = fret_copy

    return output



class DetectionZone:
    def __init__(self, i, top, bottom):
        self.i = i
        self.top = top
        self.bottom = bottom
        self.left = i / 5
        self.right = self.left + (1 / 5)
        self.match_confidence = 0.0
        self.prev_match = False # was the previous frame a match?

    # Find a better implementation for this
    def match(self, img):
        global confidence_threshold

        height, width, ch = img.shape        
        # consider logarithmic scale instead of linear to avoid false negatives
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.match_confidence = cv.countNonZero(gray) / (width * height)
        
        if self.match_confidence > confidence_threshold:
            if not self.prev_match:
                # We found a new match, add to queue
                do_nothing = True

            self.prev_match = True
        else:
            self.prev_match = False



# Detection zones are represented as a percentage of overall fret height
class DetectionRow:
    def __init__(self, center_ratio):
        global base_note_height, note_distortion

        self.center = center_ratio
        self.height = (base_note_height - (center_ratio * note_distortion))
        self.top = self.center - (self.height / 2)
        self.bottom = self.center + (self.height / 2)

        self.zones = [DetectionZone(i, self.top, self.bottom) for i in range(5)]


def detect(fret):
    global detection_rows

    return detection_rows


def process_frame(frame):
    height, width, ch = frame.shape

    cv.putText(frame,'Width: ' + str(width), (0,height-60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA, False)
    cv.putText(frame,'Height: ' + str(height), (0,height-30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA, False)
    
    # Get fret region
    pts = get_fret_coords(frame)
    poly_pts = pts.reshape((-1,1,2))
    cv.polylines(frame,[pts],True,(0,255,255), 3)

    fret = get_transformed_img(frame, pts)

    # Detection zones
    detection_rows = detect(fret)

    return output(frame, fret, detection_rows)


def run(fname):
    # init stream and window
    
    cv.namedWindow('vidstream', cv.WINDOW_NORMAL)
    cv.resizeWindow('vidstream', 1920, 720)

    is_video = ".mp4" in fname
    if is_video:
        frame = None
        cap = cv.VideoCapture(fname)
    else:
        frame = cv.imread(fname, cv.IMREAD_UNCHANGED)
        cap = None

    paused = False
    ret = None
    while True:
        global detection_height, filter_threshold, filter_grayscale, detection_rows

        if is_video and not paused:
            ret, frame = cap.read()

        # Flow
        key_press = cv.waitKey(1)
        if key_press & 0xFF == ord('q'):
            break
        if key_press & 0xFF == ord('p'):
            paused = not paused
        if key_press & 0xFF == ord('f') and is_video:
            ret, frame = cap.read()
            paused = True
        
        # Detection heights
        if key_press & 0xFF == ord('i'):
            detection_height[0] -= 0.005
            detection_rows[0] = DetectionRow(detection_height[0])
        if key_press & 0xFF == ord('k'):
            detection_height[0] += 0.005
            detection_rows[0] = DetectionRow(detection_height[0])
        if key_press & 0xFF == ord('o'):
            detection_height[1] -= 0.005
            detection_rows[1] = DetectionRow(detection_height[1])
        if key_press & 0xFF == ord('l'):
            detection_height[1] += 0.005
            detection_rows[1] = DetectionRow(detection_height[1])

        # Detection settings
        if key_press & 0xFF == ord('t'):
            filter_threshold += 2
            print(filter_threshold)
        if key_press & 0xFF == ord('g'):
            filter_threshold -= 2
            print(filter_threshold)
        if key_press & 0xFF == ord('z'):
            filter_grayscale = not filter_grayscale
        
        if key_press & 0xFF == ord('s') and is_video:
            cv.imwrite('../content/screenshot_' + str(uuid.uuid1()) + '.png', frame)
        
        if is_video and (not ret or not cap.isOpened()):
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv.imshow('vidstream', process_frame(frame))

    if cap:
        cap.release()

    cv.destroyAllWindows()

filter_grayscale = False
filter_type = cv.THRESH_TOZERO
filter_threshold = 60

font = cv.FONT_HERSHEY_SIMPLEX

# Notes are distorted vertically; as the distance from top increases, note height decreases linearly
# Top note height has been measured to be 12% of total fret board size at the top, with a scaling factor of 1/12
# In other words note height shrinks by 1 pixel for every 12 pixels a note travels down the screen
# Rendered note height can thus be calculated as: renderedNoteHeigh = base_note_height - (y * note_distortion)
base_note_height = 0.12
note_distortion = 1 / 12

confidence_threshold = 0.55

detection_height = [0.25, 0.5]
detection_rows = [DetectionRow(detection_height[0]), DetectionRow(detection_height[1])]


GREEN = (50, 255, 50)
RED = (50, 50, 255)
YELLOW = (70, 255, 255)
BLUE = (255, 180, 0)
ORANGE = (50, 130, 200)


colors = [GREEN, RED, YELLOW, BLUE, ORANGE]


fname = sys.argv[1] if len(sys.argv) > 1 else '../content/flames720.mp4'
run(fname)
