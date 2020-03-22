import cv2
import dlib
import argparse
import datetime
import time
import math
from numpy import linalg
from imutils import face_utils

ap = argparse.ArgumentParser()
threshold_msg = "Threshold to determine close eyes." \
                " If it does not detect well," \
                " adjust the threshold according to the size of the eyes." \
                " (0.2 to 0.3 is recommended)"
ap.add_argument("-t", "--threshold", type=float, default=0.23, help=threshold_msg)
ap.add_argument("-v", "--verbose", const=True, nargs='?', default=False, help="Show frame on your face")

args = vars(ap.parse_args())
EYE_CONSEC_FRAMES = 3
EYE_SIZE_AVG_THRESHOLD = args['threshold']
BLINK_CNT_PER_MS = 20

RGB_RED = (0, 0, 255)
RGB_BLUD = (255, 0, 0)
RGB_GREEN = (0, 150, 0)
RGB_WHITE = (255, 255, 255)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
IS_VERBOSE = args["verbose"]

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(LEYE_START, LEYE_END) = face_utils.FACIAL_LANDMARKS_68_IDXS.get("left_eye")
(REYE_START, REYE_END) = face_utils.FACIAL_LANDMARKS_68_IDXS.get("right_eye")


def now():
    """
    get crreunt time data

    return: string format time (%Y-%m-%d %H:%M:%S), floating point number time
    """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), time.time()


def eye_ratio(eye):
    """
    The denominator is given a weight of 2.0
    because there are 2 sets of vertical points
    and 1 set of horizontal points

    Equation: (p2 - p6) + (p3 - p5) / 2(p1 - p4)

    :param eye: shape
    :return: float
    """
    A = linalg.norm(eye[1] - eye[5])  # p2 ~ p6
    B = linalg.norm(eye[2] - eye[4])  # p3 ~ p5
    C = linalg.norm(eye[0] - eye[3])  # p1 ~ p4

    return (A + B) / (2.0 * C)


def eye_size_average(shape):
    """
    Calculate mean value of eye aspect ratio size

    :param shape:
    :return: float
    """
    leye_ratio = eye_ratio(shape[LEYE_START:LEYE_END])
    reye_ratio = eye_ratio(shape[REYE_START:REYE_END])

    return (leye_ratio + reye_ratio) / 2.0


def recommend_blink_cnt_per_min(m):
    """
    Calculate recommend blink count per minute

    :param m: elapsed time (minute)
    :return: int
    """
    return math.ceil(m) * BLINK_CNT_PER_MS


def draw_face_frame(frame, landmarks):
    """
    draw a face frame

    :param frame:
    :param landmarks:
    :return: None
    """
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 3, RGB_BLUD, -1)


def run():
    """
    main function

    :return: None
    """
    count = 0
    blink_count = 0
    start_time, fpn_start_time = now()

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            for face in faces:
                landmarks = predictor(gray, face)
                shape = face_utils.shape_to_np(landmarks)
                average = eye_size_average(shape)

                if average > EYE_SIZE_AVG_THRESHOLD:
                    count += 1
                else:
                    if count >= EYE_CONSEC_FRAMES:
                        blink_count += 1
                    count = 0

                # left eye ~ right eye
                if IS_VERBOSE:
                    draw_face_frame(frame, landmarks)

                now_time, fpn_now_time = now()
                elasped_time = (fpn_now_time - fpn_start_time) / 60
                cv2.putText(frame, "START: {}".format(start_time), (10, 30), DEFAULT_FONT, 0.7, RGB_GREEN, 2)
                cv2.putText(frame, "TIME: {}".format(now_time), (10, 50), DEFAULT_FONT, 0.7, RGB_GREEN, 2)
                cv2.putText(frame, "ELAPSED TIME: {} minute ".format(round(elasped_time, 2)), (10, 70), DEFAULT_FONT, 0.7, RGB_GREEN, 2)
                cv2.putText(frame, "BLINK: {}".format(blink_count), (10, 130), DEFAULT_FONT, 0.7, RGB_RED, 2)
                cv2.putText(frame, "Recommended number of blinks: {}".format(recommend_blink_cnt_per_min(elasped_time)), (10, 150), DEFAULT_FONT, 0.7, RGB_RED, 2)
                cv2.putText(frame, "EXIT: press 'q'", (10, 190), DEFAULT_FONT, 0.7, RGB_WHITE, 2)
        else:
            cv2.putText(frame, "Not found your face", (10, 30), DEFAULT_FONT, 0.7, RGB_RED, 2)
            cv2.putText(frame, "EXIT: press 'q'", (10, 70), DEFAULT_FONT, 0.7, RGB_WHITE, 2)

        cv2.imshow("Blink Detector", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == '__main__':
    run()
    cv2.destroyAllWindows()
