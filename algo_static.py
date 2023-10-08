#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import ast
import itertools
from collections import Counter
from collections import deque
from numpy import loadtxt

import cv2 as cv
import numpy as np
import mediapipe as mp
import base64

from treshold import THUMB_TRESHOLD, INDEX_TRESHOLD, MIDDLE_TRESHOLD, RING_TRESHOLD, PINKY_TRESHOLD


def static_model(frame, palabra):
    #print(palabra)
    with open("datos_recibidos.txt", "r") as archivo:
        contenido = archivo.read()
    fingers_done = ast.literal_eval(contenido)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    # Defines the name for each keypoint in the hand
    hand_dictionary = {
        0: "Muñeca (Wrist)",
        1: "Pulgar - Articulación carpometacarpiana (Thumb_cmc)",
        2: "Pulgar - Articulación metacarpofalángica (Thumb_mcp)",
        3: "Pulgar - Articulación interfalángica proximal (Thumb_ip)",
        4: "Pulgar - Punta del finger (Thumb_tip)",
        5: "finger índice - Articulación metacarpofalángica (Index_finger_mcp)",
        6: "finger índice - Articulación interfalángica proximal (Index_finger_pip)",
        7: "finger índice - Articulación interfalángica distal (Index_finger_dip)",
        8: "finger índice - Punta del finger (Index_finger_tip)",
        9: "finger medio - Articulación metacarpofalángica (Middle_finger_mcp)",
        10: "finger medio - Articulación interfalángica proximal (Middle_finger_pip)",
        11: "finger medio - Articulación interfalángica distal (Middle_finger_dip)",
        12: "finger medio - Punta del finger (Middle_finger_tip)",
        13: "finger anular - Articulación metacarpofalángica (Ring_finger_mcp)",
        14: "finger anular - Articulación interfalángica proximal (Ring_finger_pip)",
        15: "finger anular - Articulación interfalángica distal (Ring_finger_dip)",
        16: "finger anular - Punta del finger (Ring_finger_tip)",
        17: "finger meñique - Articulación metacarpofalángica (Pinky_mcp)",
        18: "finger meñique - Articulación interfalángica proximal (Pinky_pip)",
        19: "finger meñique - Articulación interfalángica distal (Pinky_dip)",
        20: "finger meñique - Punta del finger (Pinky_tip)"
    }
    #fingers_done = [False, False, False, False, False]
    gesture_number = -1

    image = np.frombuffer(base64.b64decode(frame), np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    #image = cv.flip(image, 1) 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    messages = []
    fingers_done_return = [False, False, False, False, False]
    # #print(messages)
    # #print("------------------")
    landmarks_list = []
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for idx, point in enumerate(landmarks.landmark):
                x = min(int(point.x * image.shape[1]), image.shape[1] - 1)
                y = min(int(point.y * image.shape[0]), image.shape[0] - 1)
                landmarks_list.append([x, y])
            if(len(landmarks_list)) > 21:
                continue
            base_landmark = landmarks_list[0]
            pre_processed_landmark_list = pre_process_landmark(
                    landmarks_list)
            gesture_data = load_gesture_data(palabra)
            difference = calculate_difference(gesture_data, pre_processed_landmark_list)
            keypoints_to_move, fingers_done_return = get_keypoints_to_move(difference, fingers_done, palabra)
            movement_direction = determine_movement_direction(keypoints_to_move)
            if len(movement_direction) == 0:
                messages.append("Correcto")
            elif movement_direction is not None:
                fingers = {
                    "pulgar": {},
                    "dedo índice": {},
                    "dedo medio": {},
                    "dedo anular": {},
                    "dedo meñique": {}
                }

                hand_movement = {} 

                for (keypoint_hand, correction) in movement_direction:
                    if keypoint_hand in hand_dictionary:
                                
                        if 1 <= keypoint_hand <= 4:
                            finger = "pulgar"
                        elif 5 <= keypoint_hand <= 8:
                            finger = "dedo índice"
                        elif 9 <= keypoint_hand <= 12:
                            finger = "dedo medio"
                        elif 13 <= keypoint_hand <= 16:
                            finger = "dedo anular"
                        elif 17 <= keypoint_hand <= 20:
                            finger = "dedo meñique"
                                
                        if correction in fingers[finger]:
                            fingers[finger][correction] += 1
                        else:
                            fingers[finger][correction] = 1

                        # Check if a hand turn message should be generated
                        if keypoint_hand in {1, 5, 9, 13, 17}:
                            if correction in hand_movement:
                                hand_movement[correction] += 1
                            else:
                                hand_movement[correction] = 1

                # Messages for the most frequent correction by hand movement
                if hand_movement:
                    most_frequent_hand_movement = max(hand_movement, key=hand_movement.get)
                    hand_message = f"Gira tu mano hacia {most_frequent_hand_movement}"
                    if most_frequent_hand_movement == "Arriba" or most_frequent_hand_movement == "Abajo":
                        hand_message = f"Gira tu mano hacia {most_frequent_hand_movement}"
                    else:
                        hand_message = f"Gira tu mano hacia la {most_frequent_hand_movement}"
                    messages.append(hand_message)
                    # cv.putText(image, hand_message, (10, y_position),
                    #         cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    #         cv.LINE_AA)
                    # y_position += 20

                # Messages for the most frequent correction by finger
                for finger, movements in fingers.items():
                    if movements:
                        most_frequent_correction = max(movements, key=movements.get)
                        if most_frequent_correction == "Arriba" or most_frequent_correction == "Abajo":
                            message = f"Mueve el {finger} para {most_frequent_correction}"
                        else:
                            message = f"Mueve el {finger} para la {most_frequent_correction}"
                        messages.append(message)
                        # cv.putText(image, message, (10, y_position),
                        #     cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        #     cv.LINE_AA)
                        # y_position += 20
    else:
        messages.append("No hay mano detectada")
    ##print(messages)
    ##print(fingers_done_return)
    # #print("------------------")
    return [messages, fingers_done_return]


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


# Function to load gesture data from CSV file
def load_gesture_data(gesture_number):
    gesture_data = []
    
    csv_path = 'model/keypoint_classifier/keypoint_image.csv'
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row) < 2:
                continue
            if row[0] == gesture_number.lower():
                # The first column is the gesture number, so we skip that column
                gesture_data.append([float(cell) for cell in row[1:]])
    return gesture_data


# Function to calculate the difference between real-time coordinates and reference coordinates
def calculate_difference(gesture_data, landmarks_in_real_time):
    if not gesture_data:
        return []
    if len(landmarks_in_real_time) != len(gesture_data[0]):
        raise ValueError("Las listas de coordenadas no tienen la misma longitud")

    difference = []
    num_keypoints = len(gesture_data[0])
    for i in range(0, num_keypoints, 2):
        x1 = gesture_data[0][i]
        y1 = gesture_data[0][i+1]
        x2 = landmarks_in_real_time[i]
        y2 = landmarks_in_real_time[i+1]
        diff_x = x2 - x1
        diff_y = y2 - y1
        difference.append((diff_x, diff_y))
    
    return difference

def treshold_calculator(gesture_number, i):
    treshold = 0.15
    if 1 <= i <= 4:
        print(THUMB_TRESHOLD[gesture_number])
        return THUMB_TRESHOLD[gesture_number]
    elif 5 <= i <= 8:
        return INDEX_TRESHOLD[gesture_number]
    elif 9 <= i <= 12:
        return MIDDLE_TRESHOLD[gesture_number]
    elif 13 <= i <= 16:
        return RING_TRESHOLD[gesture_number]
    elif 17 <= i <= 20:
        return PINKY_TRESHOLD[gesture_number]
    
    return treshold


# Function to determine which keypoints should be moved based on differences
def get_keypoints_to_move(difference, fingers_done, gesture_number):
    keypoints_to_move = []
    fingers_done_count = [True, True, True, True, True]
    treshold_done=0.3
    for i, (diff_x, diff_y) in enumerate(difference):
        # Calculate the magnitude of the Euclidean difference
        diff_magnitude = (diff_x**2 + diff_y**2)**0.5
        treshold = treshold_calculator(gesture_number, i)
        
        if 1 <= i <= 4:
            if fingers_done[0]:
                #print(i, diff_magnitude, "treshold")
                if diff_magnitude > treshold_done:
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[0] = False
            else:
                #print(i, diff_magnitude, "tresh")
                if diff_magnitude > treshold:
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[0] = False
        elif 5 <= i <= 8:
            if fingers_done[1]:
                #print(i, diff_magnitude, "treshold")
                if diff_magnitude > treshold_done:
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[1] = False
            else:
                if diff_magnitude > treshold:
                    #print(i, diff_magnitude, "tresh")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[1] = False
        elif 9 <= i <= 12:
            if fingers_done[2]:
                if diff_magnitude > treshold_done:
                    #print(i, diff_magnitude, "treshold")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[2] = False
            else:
                if diff_magnitude > treshold:
                    #print(i, diff_magnitude, "tresh")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[2] = False
        elif 13 <= i <= 16:
            if fingers_done[3]:
                if diff_magnitude > treshold_done:
                    #print(i, diff_magnitude, "treshold")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[3] = False
            else:
                if diff_magnitude > treshold:
                    #print(i, diff_magnitude, "tresh")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[3] = False
        elif 17 <= i <= 20:
            if fingers_done[4]:
                if diff_magnitude > treshold_done:
                    #print(i, diff_magnitude, "treshold")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[4] = False
            else:
                if diff_magnitude > treshold:
                    #print(i, diff_magnitude, "tresh")
                    keypoints_to_move.append([i, diff_x, diff_y])
                    fingers_done_count[4] = False
            
    for i in range(0, len(fingers_done_count)):
        if fingers_done_count[i]:
            fingers_done[i] = True
        
    return keypoints_to_move, fingers_done


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

""" if __name__ == '__main__':
    () """