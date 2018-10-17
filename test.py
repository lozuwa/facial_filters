"""
TODO:
- Objects are not being drawn ok on top of the image.
"""
import os
import cv2
import numpy as np

from biometrics.face_operations import face_detectors, face_aligners

def load_frame():
  path:str = 'trump.jpg'
  frame:np.ndarray = cv2.imread(path)
  assert isinstance(frame, np.ndarray), 'image is not valid'
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return frame

def merge_background_and_object(frame_roi, object_to_merge):
  # Create mask with black foreground meaning the object to be placed.
  # Background image.
  #bck = cv2.imread(frame_roi)
  bck = frame_roi.copy()
  #bck = cv2.cvtColor(bck, cv2.COLOR_BGR2RGB)
  #bck = cv2.resize(bck, (600, 536))

  # Create mask that allows pixels of the object pass.
  #image_4channel = cv2.imread('mexican_puro.png', cv2.IMREAD_UNCHANGED)
  image_4channel = object_to_merge.copy()
  alpha = image_4channel[:, :, 3].copy()
  alphaf = alpha.flatten()
  # Makes black the background.
  # alphaf[np.where(alphaf==0)[0]] = 0
  # alphaf[np.where(alphaf>=1)[0]] = 1
  # Makes black the region where the puro should be.
  alphaf[np.where(alphaf==0)[0]] = 1
  alphaf[np.where(alphaf!=1)[0]] = 0
  height, width = alpha.shape[:2]
  a = alphaf.reshape(height, width)
  empty = np.zeros([height, width, 3], np.uint8)
  empty[:, :, 0] = a
  empty[:, :, 1] = a
  empty[:, :, 2] = a

  bck = empty*bck
  # plot_image(bck)

  # Create object with black background.
  # image_4channel = cv2.imread('mexican_puro.png', cv2.IMREAD_UNCHANGED)
  alpha_channel = image_4channel[:,:,3]
  rgb_channels = image_4channel[:,:,:3]

  # White Background Image
  white_background_image = np.zeros_like(rgb_channels, dtype=np.uint8)

  # Alpha factor
  alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) #/ 255.0
  alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

  # Transparent Image Rendered on White Background
  base = rgb_channels.astype(np.float32) * alpha_factor
  white = white_background_image.astype(np.float32) * (1 - alpha_factor)
  final_image = base + white
  final_image = final_image.astype(np.uint8)
  # plot_image(final_image)

  # In order to merge the object and the background.
  new = cv2.add(final_image, bck)
  #plot_image(new)
  return new

def load_mtcnn():
  mtcnn = face_detectors.MTCNNV2()
  return mtcnn

def load_mexican_sombrero():
  path:str = 'mexican_sombrero.png'
  frame:np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return frame

def load_mexican_bigote():
  path:str = 'mexican_bigote.png'
  frame:np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return frame

def load_mexican_puro():
  path:str = 'mexican_puro.png'
  frame:np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return frame

def dlib_landmarks():
  # Load frame.
  frame = load_frame()
  # Load face detector.
  mtcnn:any = load_mtcnn()
  # Find faces.
  face_locations = mtcnn.findFaces(img=frame, keypoints=False)
  # Find landmarks.
  landmarks = face_aligners.find_landmarks_dlib(image=frame, face_locations=face_locations)[0]
  # Draw landmarks.
  print(landmarks)
  for key in landmarks.keys():
    points = landmarks[key]
    for point in points:
      cv2.circle(frame, point, 2, (0, 155, 255), 5, -1)
  cv2.imshow('__frame__', frame)
  cv2.waitKey(5000)
  exit()

def mtcnn_landmarks():
  # Load frame.
  frame = load_frame()
  height, width = frame.shape[:2]
  # Load face detector.
  mtcnn:any = load_mtcnn()
  # Find faces.
  predictions = mtcnn.findFaces(img=frame, keypoints=True)
  # Draw the landmarks on the image.
  # for index, _ in enumerate(predictions):
  #   bounding_box = predictions[index]['box']
  #   keypoints = predictions[index]['keypoints']
  #   top, right, bottom, left = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
  #   cv2.rectangle(frame, (left, top), (right, bottom), (0, 155, 255), 3)
  #   cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 5, -1)
  #   cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 5, -1)
  #   cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 5, -1)
  #   cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 5, -1)
  #   cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 5, -1)

  # Face locations.  
  bounding_box = predictions[0]['box']
  keypoints = predictions[0]['keypoints']
  top, right, bottom, left = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

  # Draw the mexican sombrero.
  # Load sombrero.
  frame_sombrero:np.ndarray = load_mexican_sombrero()
  sombrero_height, sombrero_width = frame_sombrero.shape[:2]
  # Calculate space to put the sombrero.
  new_sombrero_height = top
  new_sombrero_width = (right-left)
  compensate_sombrero_width = int(width//4)
  new_sombrero_width += 2*compensate_sombrero_width
  frame_sombrero = cv2.resize(frame_sombrero, (new_sombrero_width, new_sombrero_height))
  yi = 0
  yo = top
  xi = left - compensate_sombrero_width
  xo = right + compensate_sombrero_width
  merged = merge_background_and_object(frame[yi:yo, xi:xo], frame_sombrero)
  frame[yi:yo, xi:xo] = merged.copy()

  # Draw the mexican bigote.
  # Load bigote.
  frame_bigote:np.ndarray = load_mexican_bigote()
  bigote_height, bigote_width = frame_bigote.shape[:2]
  # Calculate space to put the bigote.
  new_bigote_height = keypoints['mouth_right'][1] - keypoints['nose'][1]
  new_bigote_width = keypoints['mouth_right'][0] - keypoints['mouth_left'][0]
  compensate_bigote_width = int(new_bigote_width//4)
  new_bigote_width += 2*compensate_bigote_width
  frame_bigote = cv2.resize(frame_bigote, (new_bigote_width, new_bigote_height))
  yi = keypoints['nose'][1]
  yo = keypoints['mouth_right'][1]
  xi = keypoints['mouth_left'][0] - compensate_bigote_width
  xo = keypoints['mouth_right'][0] + compensate_bigote_width
  merged = merge_background_and_object(frame[yi:yo, xi:xo], frame_bigote)
  frame[yi:yo, xi:xo] = merged.copy()

  # Draw the mexican puro.
  # Load puro.
  frame_puro:np.ndarray = load_mexican_puro()
  puro_height, puro_width = frame_puro.shape[:2]
  # Calculate space to put the puro.
  puro_center_y = ((keypoints['mouth_left'][1] + keypoints['mouth_right'][1])//2) + 1
  puro_center_x = ((keypoints['mouth_left'][0] + keypoints['mouth_right'][0])//2) + 1
  new_puro_height = bottom - puro_center_y
  new_puro_width = keypoints['mouth_right'][0] - puro_center_x
  frame_puro = cv2.resize(frame_puro, (new_puro_width, new_puro_height))
  yi = puro_center_y
  yo = bottom
  xi = puro_center_x
  xo = keypoints['mouth_right'][0]
  merged = merge_background_and_object(frame[yi:yo, xi:xo], frame_puro)
  frame[yi:yo, xi:xo] = merged.copy()

  # Display result.
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  cv2.imwrite('mexican_transformation.png', frame)

if __name__ == '__main__':
  mtcnn_landmarks()
