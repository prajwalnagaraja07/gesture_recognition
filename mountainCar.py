import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
import gym 
import random
from train_sign_detection.sign_detection import Detection

   
class controlCar:
    
    def __init__(self,mp_holistic,mp_drawing):
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
    
    def prob_viz(self,res, actions, input_frame, colors):
        
        '''Displays the predicted action'''
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return output_frame
    
    def enableGame(self,episodes):
        
        '''Enable the game for live control of mountain car'''
        
        for episode in range(1, episodes+1):
            state = env.reset()
            score = 0
            sequence = []
            sentence = []
            threshold = 0.8
        

            cap = cv2.VideoCapture(0)
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():

                    # Read feed
                    ret, frame = cap.read()
                    frame = cv2.flip(frame,1)
                    
                    # Make detections
                    image, results = detection.mediapipe_detection(frame, holistic)
                    print(results)
            
                    # Draw landmarks
                    detection.draw_styled_landmarks(image, results)
            
                    # 2. Prediction logic
                    keypoints = detection.extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
            
                    if len(sequence) == 30:
                    
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        do = np.argmax(res)
                        print(do)
                    
                        #while not done:
                        env.render()
                        n_state, reward, done, info = env.step(do)
                        score+=reward
                        print('Episode:{} Score:{}'.format(episode, score))
                    
                        if done==True:
                            break

                        # Viz probabilities
                        image = self.prob_viz(res, actions, image, colors)
                    
                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    
                    cv2.imshow('OpenCV Feed', image)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
        
        
    
if __name__ == "__main__":
    
    
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    
    control_car = controlCar(mp_holistic,mp_drawing)
    detection = Detection(mp_holistic,mp_drawing)
    
    colors = [(245,117,16), (117,245,16), (16,117,245)] # Stack overflow
    
    # Create the environment
    env = gym.make('MountainCar-v0')
    
    states = env.observation_space.shape[0]
    
    # actions to perform based on states
    actions = np.array(['left', 'stop', 'right'])
    
    # Load the model for prediction
    model = load_model('C:/Git_Code/co2nn/~/sign_detection.h5')
    
    
    # Enable the game
    control_car.enableGame(episodes=10)
    

    