import numpy as np
import cv2


def remove_extremes(contours, percentage=0.2):
    py_list_contours = list(contours)
    py_list_areas = list([cv2.contourArea(c) for c in contours])
    py_list_both = list(zip(py_list_areas, py_list_contours))

    py_list_both.sort(key=lambda c: c[0])
    sorted = np.array(py_list_both)
    cut_size = int(len(contours) * percentage)
    # remove <percentage>% in each side
    if cut_size != 0:
        sorted = sorted[cut_size:-cut_size]

    return sorted


def get_actual_value(contours):
    # a tuple in the format (contour_area, contour)
    no_extremes = remove_extremes(contours, 0.2)
    average = np.median([c[0] for c in no_extremes])
    with_extremes = remove_extremes(contours, 0.0)

    MIN_AREA = int(average * 0.6)
    MAX_AREA = int(average * 1.4)

    total = 0
    for c in with_extremes:
        area = c[0]

        if area < MIN_AREA:
            continue
        elif area > MAX_AREA:
            rices_in_blob = area // average
            total += rices_in_blob
        else:
            total += 1

    return total
