def detect(link):
  import mediapipe as mp
  from typing import List, Optional, Tuple
  import cv2
  import dataclasses
  import matplotlib.pyplot as plt
  from mediapipe.framework.formats import landmark_pb2
  import numpy as np
  import json

  mp_pose = mp.solutions.pose
  mp_drawing = mp.solutions.drawing_utils 
  mp_drawing_styles = mp.solutions.drawing_styles

  PRESENCE_THRESHOLD = 0.5
  RGB_CHANNELS = 3
  BLACK_COLOR = (0, 0, 0)
  RED_COLOR = (0, 0, 255)
  GREEN_COLOR = (0, 128, 0)
  BLUE_COLOR = (255, 0, 0)
  VISIBILITY_THRESHOLD = 0.5

  @dataclasses.dataclass
  class DrawingSpec:
    # Color for drawing the annotation. Default to the green color.
    color: Tuple[int, int, int] = (0, 255, 0)
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2

  def _normalize_color(color):
      return tuple(v / 255. for v in color)

  def plot_landmarks(
          landmark_list: landmark_pb2.NormalizedLandmarkList,

          connections: Optional[List[Tuple[int, int]]] = None,

          landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR, thickness=5),

          connection_drawing_spec: DrawingSpec = DrawingSpec(color=BLACK_COLOR, thickness=5),

          elevation: int = 10,

          azimuth: int = 10
      ):
    global first, cap, fps, video, frame, total_frames

    if not landmark_list:
      return
    
    plt.switch_backend('AGG')
    fig = plt.figure(num=1, figsize=(10, 10), constrained_layout=True)
    plt.rcParams['figure.constrained_layout.h_pad'] = 0
    plt.rcParams['figure.constrained_layout.w_pad'] = 0
    plt.rcParams['figure.constrained_layout.wspace'] = 0
    plt.rcParams['figure.constrained_layout.hspace'] = 0
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}

    for idx, landmark in enumerate(landmark_list.landmark):
      if ((landmark.HasField('visibility') and
          landmark.visibility < VISIBILITY_THRESHOLD) or
          (landmark.HasField('presence') and
          landmark.presence < PRESENCE_THRESHOLD)):
        continue

      ax.scatter3D(
          xs=[-landmark.z],
          ys=[landmark.x],
          zs=[-landmark.y],
          color=_normalize_color(landmark_drawing_spec.color[::-1]),
          linewidth=landmark_drawing_spec.thickness)
      plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

    if connections:
      num_landmarks = len(landmark_list.landmark)
      # Draws the connections if the start and end landmarks are both visible.

      for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
          raise ValueError(f'Landmark index is out of range. Invalid connection '
                          f'from landmark #{start_idx} to landmark #{end_idx}.')

        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
          landmark_pair = [ plotted_landmarks[start_idx], plotted_landmarks[end_idx] ]

          ax.plot3D(
              xs=[landmark_pair[0][0], landmark_pair[1][0]],
              ys=[landmark_pair[0][1], landmark_pair[1][1]],
              zs=[landmark_pair[0][2], landmark_pair[1][2]],
              color=_normalize_color(connection_drawing_spec.color[::-1]),
              linewidth=connection_drawing_spec.thickness)

    # plt.savefig("frames/image_" + str(frame) + ".png", transparent=True, bbox_inches='tight')
    # plt.show(block=False)
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if first:
      dimensions = img.shape
      fourcc = cv2.VideoWriter_fourcc(*'avc1')
      video = cv2.VideoWriter('Pose_Detect_App/output/output.mp4', fourcc, fps, (dimensions[1], dimensions[0]))
      first = False
    
    video.write(img)
    fig.canvas.flush_events()
    fig.clear()
    frame += 1

  global first, cap, fps, video, frame, total_frames, pose
  cap = cv2.VideoCapture(link)
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  frame = 0
  first = True
  video = ''
  # with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
  
  pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
  lands = ''
  while cap.isOpened():
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      if frame >= total_frames:
        f = open("Pose_Detect_App/output/landmarks(in meters).txt", 'w')
        f.write(lands)
        video.release()
        cap.release()
        break

      time = total_frames/fps
      current_time = frame/fps
      
      lands_first = True

      success, img = cap.read()
      results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

      # Print the real-world 3D coordinates of nose in meters with the origin at
      # the center between hips.
      # print('Nose world landmark:')
      # print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
      #if len(results.pose_world_landmarks) >0:
      # Plot pose world landmarks.
      plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
      for landmark in results.pose_world_landmarks.landmark:
        if lands_first:
          lands += f'{landmark.x}, {landmark.y}, {landmark.z}'
          lands_first = False
        else:
          lands += f', {landmark.x}, {landmark.y}, {landmark.z}'
      if frame < total_frames:
        lands += '\n'
      # print("drew")
      #else:
          #print("No landmarks")
      # print(frame)
      yield json.dumps({"time": current_time, "total_time": time}) + "|"
      # if cv2.waitKey(1) & 0xFF == 27:
      #     break
  return

