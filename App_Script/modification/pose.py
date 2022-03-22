from attr import evolve


def detect(link, type=""):
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

  indices = {
    "nose": [0, "nose"],
    "left_eye_inner": [1, "left eye inner"],
    "left_eye": [2, "left eye"],
    "left_eye_outer": [3, "left eye outer"],
    "right_eye_inner": [4, "right eye inner"],
    "right_eye": [5, "right eye"],
    "right_eye_outer": [6, "right eye outer"],
    "left_ear": [7, "left ear"],
    "right_ear": [8, "right ear"],
    "mouth_left": [9, "mouth left"],
    "mouth_right": [10, "mouth right"],
    "left_shoulder": [11, "left shoulder"],
    "right_shoulder": [12, "right shoulder"],
    "left_elbow": [13, "left elbow"],
    "right_elbow": [14, "right elbow"],
    "left_wrist": [15, "left wrist"],
    "right_wrist": [16, "right wrist"],
    "left_pinky": [17, "left pinky"],
    "right_pinky": [18, "right pinky"],
    "left_index": [19, "left index"],
    "right_index": [20, "right index"],
    "left_thumb": [21, "left_thumb"],
    "right_thumb": [22, "right thumb"],
    "left_hip": [23, "left hip"],
    "right_hip": [24, "right hip"],
    "left_knee": [25, "left knee"],
    "right_knee": [26, "right knee"],
    "left_ankle": [27, "left ankle"],
    "right_ankle": [28, "right ankle"],
    "left_heel": [29, "left heel"],
    "right_heel": [30, "right heel"],
    "left_foot_index": [31, "left foot index"],
    "right_foot_index": [32, "right foot index"]
  }

  deviation = 2/100
  
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
    
##    plt.switch_backend('AGG')
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
    plt.show(block=False)
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if first:
      dimensions = img.shape
      fourcc = cv2.VideoWriter_fourcc(*'avc1')
      video = cv2.VideoWriter('Pose_Detect_App/output/output.mp4', fourcc, fps, (dimensions[0], dimensions[1]))
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

  wrist_pos = []
  wrist_count = 0
  b_reps = 0

  hip_pos = []
  hip_count = 0
  q_reps = 0

  elb_pos = []
  elb_count = 0
  s_reps = 0

  while cap.isOpened():
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      if frame >= total_frames:
##        f = open("Pose_Detect_App/output/landmarks(in meters).txt", 'w')
##        f.write(lands)
        video.release()
        cap.release()
        break

      time = total_frames/fps
      current_time = frame/fps
      
      lands_first = True

      success, img = cap.read()
      if cap.isOpened():
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

      # Print the real-world 3D coordinates of nose in meters with the origin at
      # the center between hips.
      # print('Nose world landmark:')
      points = results.pose_landmarks.landmark

      
      
      if points[indices["left_shoulder"][0]] and points[indices["right_shoulder"][0]] and len(type) > 0:
        
        
        # BICEP CURLS LOGIC ---------------------------------------------------------------------------------------------------------------------------------------
        
        if type == "biceps":

          shoulders = False
          elbows = False
          legs = False

          if abs(points[indices["left_shoulder"][0]].y - points[indices["right_shoulder"][0]].y) < deviation:
            if not shoulders:
              shoulders = True
            print("Shoulders Straight")
          else:
            if shoulders:
              shoulders = False
            print("Shoulders Not Straight")

          if abs(points[indices["left_elbow"][0]].x - points[indices["left_shoulder"][0]].x) < (deviation + 1/100) and abs(points[indices["right_elbow"][0]].x - points[indices["right_shoulder"][0]].x) < (deviation + 1/100):
            if not elbows:
              elbows = True
            print("Elbows Aligned")
          else:
            if elbows:
              elbows = False
            print("Elbows Not Aligned")

          wrist_pos = [(points[indices["left_wrist"][0]].y - points[indices["left_elbow"][0]].y)/abs(points[indices["left_wrist"][0]].y - points[indices["left_elbow"][0]].y), (points[indices["right_wrist"][0]].y - points[indices["right_elbow"][0]].y)/abs(points[indices["right_wrist"][0]].y - points[indices["right_elbow"][0]].y)]

          if wrist_pos[0] == 1 and wrist_pos[1] == 1:
            if wrist_count != 0:
              if shoulders and elbows:
                b_reps += 1
              wrist_count = 0
            print("Hands Down")
          elif wrist_pos[0] == -1 and wrist_pos[1] == -1:
            if wrist_count == 0:
              wrist_count = 1
            print("Hands Up")
          else:
            wrist_count = wrist_count
          #  wrist_count = 0
          #  reps = 0
            print("Hands Not in Sync")
          # print("\n")
          print("Bicep Curls: ", b_reps)

          print("\n")


        # SQUATS LOGIC ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        if type == "squats":

          shoulders = False
          legs = False

          if abs(points[indices["left_shoulder"][0]].y - points[indices["right_shoulder"][0]].y) < deviation:
            if not shoulders:
              shoulders = True
            print("Shoulders Straight")
          else:
            if shoulders:
              shoulders = False
            print("Shoulders Not Straight")
          
          if (0 < (points[indices["left_foot_index"][0]].x - points[indices["left_hip"][0]].x) and 0 < (points[indices["left_knee"][0]].x - points[indices["left_hip"][0]].x) and (0 > (points[indices["right_foot_index"][0]].x - points[indices["right_hip"][0]].x) and 0 > (points[indices["right_knee"][0]].x - points[indices["right_hip"][0]].x))) :
            if not legs:
              legs = True
            print("Legs Straight") 
          else:
            if legs:
              legs = False
            print("Legs Not Straight")

          hip_pos = [0 < (points[indices["left_knee"][0]].y - points[indices["left_hip"][0]].y) < (deviation + 10/100), 0 < (points[indices["right_knee"][0]].y - points[indices["right_hip"][0]].y) < (deviation + 10/100)]

          if hip_pos[0] and hip_pos[1]:
            if hip_count != 0:
              if shoulders and legs:
                q_reps += 1
              hip_count = 0
            print("Hips Down")
          elif not hip_pos[0] and not hip_pos[1]:
            if hip_count == 0:
              hip_count = 1
            print("Hips Up")
          else:
            hip_count = hip_count
            print("Hips Not in Sync")

          print("Squats: ", q_reps)
          
          print("\n")


        # STRAP LOGIC --------------------------------------------------------------------------------------------------------------------------------------------------------

        if type == "strap":

          bent = False
          knees = False

          if abs(points[indices["left_shoulder"][0]].x - points[indices["left_hip"][0]].x) > (deviation) and abs(points[indices["right_shoulder"][0]].x - points[indices["right_hip"][0]].x) > (deviation):
            if not bent:
              bent = True
            print("Bent")
          else:
            if bent:
              bent = False
            print("Not Bent")
          
          if abs(points[indices["left_knee"][0]].x - points[indices["left_hip"][0]].x) < (deviation + 2/100) and abs(points[indices["left_knee"][0]].x - points[indices["left_hip"][0]].x) < (deviation + 2/100):
            if not knees:
              knees = True
            print("Knees Straight")
          else:
            if knees:
              knees = False
            print("Knees Not Straight")

          elb_pos = [points[indices["left_elbow"][0]].y - points[indices["left_shoulder"][0]].y < (deviation + 3/100), points[indices["right_elbow"][0]].y - points[indices["right_shoulder"][0]].y < (deviation + 3/100)]

          if elb_pos[0] and elb_pos[1]:
            if elb_count != 0:
              if bent and knees:
                s_reps += 1
              elb_count = 0
            print("Elbows Up")
          elif not elb_pos[0] and not elb_pos[1]:
            if elb_count != 1:
              elb_count = 1
            print("Elbows Down")
          else:
            elb_count = elb_count
            print("Elbows Not in Sync")

          print("Strap Pulls: ", s_reps)

          print("\n")
        
##      print(results.pose_world_landmarks.landmark[indices["left_knee"][0]] if results.pose_world_landmarks.landmark[indices["left_knee"][0]] else "None", indices["left_knee"][1])
      #if len(results.pose_world_landmarks) >0:
      # Plot pose world landmarks.
      mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
      cv2.imshow("Test", img)
##      plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
##      for landmark in results.pose_world_landmarks.landmark:
##        if lands_first:
##          lands += f'{landmark.x}, {landmark.y}, {landmark.z}'
##          lands_first = False
##        else:
##          lands += f', {landmark.x}, {landmark.y}, {landmark.z}'
##      if frame < total_frames:
##        lands += '\n'
      # print("drew")
      #else:
          #print("No landmarks")
      # print(frame)
      #yield json.dumps({"time": current_time, "total_time": time}) + "|"
      # if cv2.waitKey(1) & 0xFF == 27:
      #     break
  return

detect('bicep curls correct.mp4', "biceps")
