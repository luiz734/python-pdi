import sys
import numpy as np
import cv2
import rice_counter

CHANNELS = 3
images = ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']
INPUT_IMAGE = '60.bmp'   # luz
INPUT_IMAGE = '82.bmp'   # normal
INPUT_IMAGE = '114.bmp'  # normal
INPUT_IMAGE = '150.bmp'  # luz, FUNDO, aglomerado
INPUT_IMAGE = '205.bmp'  # luz, fundo, aglomerado


def create_output_img(img, background, name_postfix=""):
    contours, h = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background = cv2.drawContours(background, contours, -1, (0, 0, 255), 2)

    count = rice_counter.get_actual_value(contours)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    outline = (255, 255, 255)
    color = (0, 0, 255)
    thickness = 2
    background = cv2.putText(background, f'total: {count}', org, font, fontScale,
                             outline, thickness * thickness, cv2.LINE_AA)
    background = cv2.putText(background, f'total: {count}', org, font, fontScale,
                             color, thickness, cv2.LINE_AA)

    output_name = "out.bmp" if name_postfix == "" else f"out_{name_postfix}.bmp"
    cv2.imwrite(output_name, background)


def main(input_image):
    img = cv2.imread(f"{input_image}", cv2.IMREAD_COLOR)
    # img = cv2.imread(f"input/{INPUT_IMAGE}", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    # img = img.astype(np.float32) / 255

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(f"0-gray.bmp", gray)

    gaussian = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imwrite(f"1-gaussian.bmp", gaussian)

    median = cv2.medianBlur(gaussian, 5)
    cv2.imwrite(f"2-median-0.bmp", median)

    median = cv2.medianBlur(gaussian, 7)
    cv2.imwrite(f"3-median-1.bmp", median)

    median = cv2.medianBlur(gaussian, 13)
    cv2.imwrite(f"4-median-2.bmp", median)

    adaptative_threshold = cv2.adaptiveThreshold(
        median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    cv2.imwrite(f"5-adpative_threshold.bmp", adaptative_threshold)

    median = cv2.medianBlur(adaptative_threshold, 11)
    cv2.imwrite(f"6-median-3.bmp", median)

    kernel = np.ones((5, 5))

    opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"7-opening.bmp", opening)

    # closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"7-colsing.bmp", closing)

    # opening = cv2.morph(median, kernel)
    # cv2.imwrite(f"8-dilate.bmp", dilate)
    # erode = cv2.erode(dilate, kernel)
    # erode = cv2.erode(erode, kernel)

    # out = closing
    out = opening

    postfix = input_image.split("/")[-1]
    postfix = postfix.split(".")[0]
    return create_output_img(out, img, name_postfix=postfix)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for img_path in images:
        main(f"input/{img_path}")
    # main("rice.png")
