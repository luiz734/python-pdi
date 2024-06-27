import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='path to video input')
args = parser.parse_args()
input_video = cv2.VideoCapture(args.video)
NEW_FRAME_RATE = 8


def change_frame_rate(video_path, new_frame_rate, output_path):
  # abre o vídeo
  vid = cv2.VideoCapture(video_path)
  # frame rate atual
  orig_frame_rate = vid.get(cv2.CAP_PROP_FPS)
  # utlizado para determinar quais frames pegar do vídeo
  scaling_factor = orig_frame_rate / new_frame_rate

  # vídeo de saída
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(output_path, fourcc, new_frame_rate, (int(
      vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

  while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
      break

    # subamostragem
    if scaling_factor > 1:
      for i in range(int(scaling_factor)):
        ret, frame = vid.read()
        if not ret:
          break
    elif scaling_factor < 1:
      frame = cv2.resize(frame, None, fx=scaling_factor,
                         fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # salva o frame na saída
    out.write(frame)

  vid.release()
  out.release()


output_path = f'{args.video.split(".")[0]}_{NEW_FRAME_RATE}.mp4'
change_frame_rate(args.video, NEW_FRAME_RATE, output_path)
