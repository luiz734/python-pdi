import numpy as np
import cv2
import argparse


# ---------------------------------- params ---------------------------------- #
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

DEBUG_WIN_SCALE = 1
DEBUG_VECTOR_SCALE = 1
DEBUG_MODE = False
DEBUG_SHOW_FLOW = DEBUG_MODE and True
DEBUG_SHOW_FLOW_STEP = DEBUG_MODE and False
EMPTY_BG = False
DEBUG_ARROW_TIP = 0.25
FLOW_WINDOW_SIZE = 101
FLOW_WINDOW_HALF = FLOW_WINDOW_SIZE // 2
RED = (0, 0, 255)

# ----------------------------------- debug ---------------------------------- #


def show_flow(old_pts, pts, img):
  for i, (new, old) in enumerate(zip(pts, old_pts)):
    a, b = new.ravel()
    c, d = old.ravel()
    # u e v devem ser diferente de zero
    u = a - c + 0.001
    v = b - d + 0.001
    uv = [u, v]
    norm = np.linalg.norm(uv)
    # não mostra vetores claramente errados
    # if norm > 5:
    # continue
    start_point = (int(c), int(d))
    end_point = (int(c + uv[0] * DEBUG_VECTOR_SCALE),
                 int(d + uv[1] * DEBUG_VECTOR_SCALE))
    img = cv2.arrowedLine(img, start_point, end_point,
                          RED, 1, tipLength=DEBUG_ARROW_TIP)

  img = cv2.resize(img, (0, 0), fx=DEBUG_WIN_SCALE, fy=DEBUG_WIN_SCALE)
  cv2.imshow('frame', img)

  while (cv2.waitKey(0) != 32):
    pass


# ----------------------------------- input/output ---------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='path to video input')
args = parser.parse_args()
input_video = cv2.VideoCapture(args.video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = args.video.split(".")[0]
filename += "_nobg" if EMPTY_BG else ""
filename += "_out.mp4"
output_video = cv2.VideoWriter(
    filename=filename,
    fourcc=fourcc,
    fps=input_video.get(cv2.CAP_PROP_FPS) * 2,
    frameSize=(
        int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))


# ---------------------------------------------------------------------------- #
def double_fps():
  old_ret, old_frame = input_video.read()
  old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
  output_video.write(old_frame)
  RES_X = old_frame.shape[1]
  RES_Y = old_frame.shape[0]

  # buffer to draw the current flow
  buffer = np.empty_like(old_frame)
  while input_video.isOpened():
    # próximo frame
    new_ret, new_frame = input_video.read()
    if not new_ret:
      break

    old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    avg_frame = cv2.addWeighted(old_frame, 0.5, new_frame, 0.5, 0)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_points, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray, new_gray, old_points, None, **lk_params)

    if new_points is not None:
      old_good = old_points[status == 1]
      new_good = new_points[status == 1]

    buffer = avg_frame.copy() if not EMPTY_BG else np.zeros_like(new_frame)

    for i, (p_old, p_new) in enumerate(zip(old_good, new_good)):
      old_left = int(p_old[0] - FLOW_WINDOW_HALF)
      old_top = int(p_old[1] - FLOW_WINDOW_HALF)
      old_right = int(p_old[0] + (FLOW_WINDOW_HALF + 1))
      old_bottom = int(p_old[1] + (FLOW_WINDOW_HALF + 1))

      new_left = int(p_new[0] - FLOW_WINDOW_HALF)
      new_top = int(p_new[1] - FLOW_WINDOW_HALF)
      new_right = int(p_new[0] + (FLOW_WINDOW_HALF + 1))
      new_bottom = int(p_new[1] + (FLOW_WINDOW_HALF + 1))

      # ignora janelas fora da imagem
      try:
        old_win = old_frame[old_top:old_bottom, old_left:old_right]
        new_win = new_frame[new_top:new_bottom, new_left:new_right]

        # média das janlelas do ponto de interesse
        avg_win = cv2.addWeighted(old_win, 0.5, new_win, 0.5, 0)
        win_start_x = (new_left + old_left) // 2
        win_start_y = (new_top + old_top) // 2
        win_end_x = (new_right + old_right) // 2
        win_end_y = (new_bottom + old_bottom) // 2

        # copia a avg_window para a posição adequada
        buffer[win_start_y:win_end_y,
               win_start_x:win_end_x] = avg_win

        if DEBUG_SHOW_FLOW_STEP:
          show_flow(old_points, new_points, buffer)
      except:
        pass

    if DEBUG_SHOW_FLOW:
      show_flow(old_points, new_points, buffer)

    # cópia para a próxima iteração
    old_gray = new_gray.copy()
    old_points = new_points.copy()
    old_frame = new_frame.copy()
    output_video.write(buffer)
    output_video.write(new_frame)

  # salva os vídeos
  cv2.destroyAllWindows()
  input_video.release()
  output_video.release()


if __name__ == "__main__":
  double_fps()
