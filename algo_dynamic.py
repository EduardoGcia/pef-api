#!/usr/bin/env python
#:*- coding: utf-8:*-
import csv
import copy
import argparse
import ast
import itertools
import re
from collections import Counter
from collections import deque
from numpy import loadtxt
from algo_static import static_model

import cv2 as cv
import numpy as np
import mediapipe as mp
import base64


def dynamic_model(frames, gesture, steps, THUMB_TRESHOLD = 0.15, INDEX_TRESHOLD =0.15, MIDDLE_TRESHOLD=0.15, RING_TRESHOLD=0.15, PINKY_TRESHOLD=0.15):
    #frame viene como arreglo de los frames
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    pose_dictionary = {
        0: "nose",
        1: "left eye (inner)",
        2: "left eye",
        3: "left eye (outer)",
        4: "right eye (inner)",
        5: "right eye",
        6: "right eye (outer)",
        7: "left ear",
        8: "right ear",
        9: "mouth (left)",
        10: "mouth (right)",
        11: "left shoulder",
        12: "right shoulder",
        13: "left elbow",
        14: "right elbow",
        15: "left wrist",
        16: "right wrist",
        17: "left pinky",
        18: "right pinky",
        19: "left index",
        20: "right index",
        21: "left thumb",
        22: "right thumb",
        23: "left hip",
        24: "right hip",
        25: "left knee",
        26: "right knee",
        27: "left ankle",
        28: "right ankle",
        29: "left heel",
        30: "right heel",
        31: "left foot index",
        32: "right foot index",
    }

    # Se regresa un arreglo en tipo [[tempo 1], [tempo 2], [tempo 3], ...]
    pose_messages = []
    hand_messages = []
    gesture_data = load_gesture_data(gesture)


    matching_frames, hand_messages = find_best_matching_frames(frames, gesture_data, gesture, THUMB_TRESHOLD, INDEX_TRESHOLD, MIDDLE_TRESHOLD, RING_TRESHOLD, PINKY_TRESHOLD)
    for match in matching_frames:
        keypoints_to_move = get_keypoints_to_move(match)
        movement_direction = determine_movement_direction(keypoints_to_move)
        if len(movement_direction) == 0:
                pose_messages.append("Correcto")
        else:
            pose_messages.append(movement_direction)

    return pose_messages, hand_messages







    #for index, framePerSec in enumerate(frames):
    for index in range(0, steps):
        for frame in frames:
        # Se podría quitar el fingers_done?
            if frame.startswith('data:'):
                frame = re.sub('^data:image/.+;base64,', '', frame)
            hand_message, fingers_done = static_model(frame, gesture,THUMB_TRESHOLD, INDEX_TRESHOLD, MIDDLE_TRESHOLD, RING_TRESHOLD, PINKY_TRESHOLD, index=index, dynamic=True)
            if hand_message == "No hay mano detectada":
                continue

            image = np.frombuffer(base64.b64decode(frame), np.uint8)
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            
            landmark_list = image_to_landmarks(image, results)
            pre_processed_landmark_list = pre_process_landmark(
                                landmark_list)
            # Se regresa un arreglo en tipo [[dif tempo 1], [dif tempo 2], [dif tempo 3], ...]
            difference = calculate_difference(gesture_data[index], pre_processed_landmark_list)
            keypoints_to_move = get_keypoints_to_move(difference)
            movement_direction = determine_movement_direction(keypoints_to_move)
            # Cambiar
            if len(movement_direction) == 0:
                    pose_messages.append("Correcto")
            else:
                pose_messages.append(movement_direction)
            hand_messages.append(hand_message)
            break
        (pose_messages)
        (hand_messages)
    return pose_messages, hand_messages


def image_to_landmarks(image, results):
    landmark_list = []
    if results.pose_landmarks:
        for idx,point in enumerate (results.pose_landmarks.landmark):
            x = min(int(point.x * image.shape[1]), image.shape[1] - 1)
            y = min(int(point.y * image.shape[0]), image.shape[0] - 1)
            landmark_list.append([x, y])
    return landmark_list


# Normalize the landmarks to save them in csv
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def load_gesture_data(gesture):
    gesture_data = []
    
    csv_path = 'model/keypoint_classifier/keypoint_image_dynamic.csv'
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row) < 2:
                continue
            if row[0].lower() == gesture.replace(" ", "").lower():
                # The first column is the gesture number, so we skip that column
                gesture_data.append([float(cell) for cell in row[2:]])
    return gesture_data


# Function to calculate the difference between real-time coordinates and reference coordinates
def calculate_difference(gesture_data, landmarks_in_real_time):
    try:
        if not gesture_data:
            return []
        if len(landmarks_in_real_time) != len(gesture_data):
            raise ValueError("Las listas de coordenadas no tienen la misma longitud")

        difference = []
        num_keypoints = len(gesture_data)
        for i in range(0, num_keypoints, 2):
            x1 = gesture_data[i]
            y1 = gesture_data[i+1]
            x2 = landmarks_in_real_time[i]
            y2 = landmarks_in_real_time[i+1]
            diff_x = x2 - x1
            diff_y = y2 - y1
            difference.append((diff_x, diff_y))
        return difference
    except ValueError as e:
        # Manejar la excepción, por ejemplo, imprimir un mensaje de error y devolver un valor predeterminado
        print(f"Error: {e}")
        return []  # O devuelve otro valor predeterminado si es más adecuado


# Function to determine which keypoints should be moved based on differences
def get_keypoints_to_move(difference):
    keypoints_to_move = []
    treshold = 0.20
    #print(difference)
    for i, (diff_x, diff_y) in enumerate(difference):
        # Calculate the magnitude of the Euclidean difference
        diff_magnitude = (diff_x**2 + diff_y**2)**0.5
        # PARA HOLA COMPARAR 20, 18, 16, 22??
        #if gesture.lower() == 'hola':
        #if i == 20 or i == 18 or i == 16 or i == 22:
        if i == 19 or i == 17 or i == 15 or i == 21:
            if diff_magnitude > treshold:
                keypoints_to_move.append([i, diff_x, diff_y]) 
        
    return keypoints_to_move


def get_keypoints_to_move_mean(difference):
    total = 0
    for i, (diff_x, diff_y) in enumerate(difference):
        # Calculate the magnitude of the Euclidean difference
        diff_magnitude = (diff_x**2 + diff_y**2)**0.5
        # PARA HOLA COMPARAR 20, 18, 16, 22??
        #if gesture.lower() == 'hola':
        #if i == 20 or i == 18 or i == 16 or i == 22:
        if i == 19 or i == 17 or i == 15 or i == 21:
            total += diff_magnitude
    total /= 4
    return total

# Determine the direction of movement based on the difference in x and y coordinates
def determine_movement_direction(keypoints_to_move):
    movement_direction = []

    for (i, diff_x, diff_y) in keypoints_to_move:
        if diff_x > 0 and abs(diff_x) > abs(diff_y):
            movement_direction.append([i, "Izquierda"])
        elif diff_x < 0 and abs(diff_x) > abs(diff_y):
            movement_direction.append([i, "Derecha"])
        elif diff_y > 0 and abs(diff_y) > abs(diff_x):
            movement_direction.append([i, "Arriba"])
        elif diff_y < 0 and abs(diff_y) > abs(diff_x):
            movement_direction.append([i, "Abajo"])
        else:
            movement_direction.append([i, "Sin movimiento"])

    return movement_direction


def find_best_matching_frames(frames, target_frames, gesture,THUMB_TRESHOLD, INDEX_TRESHOLD, MIDDLE_TRESHOLD, RING_TRESHOLD, PINKY_TRESHOLD):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    indexes = [0] * len(target_frames)
    hand_messages = [""] * len(target_frames)
    best_matches = [[]] * len(target_frames)
    best_matches_differences = [[]] * len(target_frames)

    frame_counter = 0
    for frame in frames:
        if frame.startswith('data:'):
            frame = re.sub('^data:image/.+;base64,', '', frame)

        image = np.frombuffer(base64.b64decode(frame), np.uint8)
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        
        landmark_list = image_to_landmarks(image, results)
        pre_processed_landmark_list = pre_process_landmark(
                                landmark_list)
        
        if frame_counter == 0:
            index = 0
            for target_frame in target_frames:
                best_matches[index] = pre_processed_landmark_list
                index += 1
        else:

            # Evaluar si el frame actual cumple con las condiciones de algún momento "correcto"
            index = 0
            for target_frame in target_frames:
                difference_actual_frame = calculate_difference(target_frame, pre_processed_landmark_list)
                difference_actual_frame2 = get_keypoints_to_move_mean(difference_actual_frame)
                difference_best_match = calculate_difference(target_frame, best_matches[index])
                difference_best_match2 = get_keypoints_to_move_mean(difference_best_match)
                # Implementa tu lógica de comparación aquí, por ejemplo, utilizando una función de similitud
                if difference_actual_frame2  < difference_best_match2:
                    print("entré")
                    hand_message, fingers_done = static_model(frame, gesture,THUMB_TRESHOLD, INDEX_TRESHOLD, MIDDLE_TRESHOLD, RING_TRESHOLD, PINKY_TRESHOLD, index=index, dynamic=True)
                    if hand_message == "No hay mano detectada":
                        continue
                    best_matches[index] = pre_processed_landmark_list
                    best_matches_differences[index] = difference_actual_frame
                    indexes[index] = frame_counter
                    hand_messages[index] = hand_message
                    index += 1
                else:
                    print("no entré")
        frame_counter += 1
    return best_matches_differences, hand_messages

