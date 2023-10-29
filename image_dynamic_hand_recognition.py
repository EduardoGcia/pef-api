import cv2
import mediapipe as mp
import csv
import copy
import itertools
import os

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    try:
        csv_path = 'model/keypoint_classifier/keypoint_image_hand_dynamic.csv'
        os.remove(csv_path)
    except:
        pass
    directorio = 'model/images_dynamic_hand_change/'
    try:
        for dir in os.listdir(directorio):
            new_dir = os.path.join(directorio, dir)
            if os.path.isdir(new_dir):
                files = [element for element in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, element))]
                num_files = len(files)
                for file in files:
                    try:
                        # Image load
                        image = cv2.imread(os.path.join(new_dir, file))
                        image = cv2.flip(image, 1) 
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results_hands = hands.process(image)

                        image.flags.writeable = True
                        name, extension = os.path.splitext(file)
                        name, name_ord = name.split('_')
                        image_to_landmarks(image, results_hands, name, name_ord)
                    except:
                        pass
    except:
        print("e 46 image recognition")


# Get the landmarks from an image
def image_to_landmarks(image, results, name, name_ord):
    landmarks_list = []
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for idx, point in enumerate(landmarks.landmark):
                x = min(int(point.x * image.shape[1]), image.shape[1] - 1)
                y = min(int(point.y * image.shape[0]), image.shape[0] - 1)
                landmarks_list.append([x, y])
            if(len(landmarks_list)) > 21:
                continue
            pre_processed_landmark_list = pre_process_landmark(
                        landmarks_list)
            logging_csv(pre_processed_landmark_list, name, name_ord)
    else:
        #TODO VER POR QUÉ NO FUNCIONA EL SI #1
        print(name)
        print(name_ord)
        pass
    return


# Normalize the data
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


# Saves the data to a csv
def logging_csv(landmarks_list, name, name_ord):
    csv_path = 'model/keypoint_classifier/keypoint_image_hand_dynamic.csv'
    with open(csv_path, 'a', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([name.lower(), name_ord, *landmarks_list])
        
    return


if __name__ == '__main__':
    main()