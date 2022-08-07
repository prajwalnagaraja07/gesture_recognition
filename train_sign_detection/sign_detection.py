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
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

'''


'''

class Detection:
    
    def __init__(self,mp_holistic,mp_drawing):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
      
    def mediapipe_detection(self,image, model):
        
        ''' Takes an image and model for prediction'''
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_landmarks(self,image, results):
        
        ''' Takes an image and draws connections'''
        
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
        
    
    def draw_styled_landmarks(self,image, results):
        
        ''' Takes an image and returns landmarks'''
        
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, 
                             self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                             self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                             self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
        # Draw right hand connections  
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                             self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    def extract_keypoints(self,results):
        
        ''' Extracts Kepoints''' 
        
        # Stack Overflow
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    def collect_traindata(self):
        
        ''' Creates Training Data'''
        
        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
            # NEW LOOP
            # Loop through actions
            for action in actions:
                # Loop through sequences aka videos
                for sequence in range(no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(sequence_length):

                        # Read feed
                        ret, frame = cap.read()
                        frame = cv2.flip(frame,1)

                        # Make detections
                        image, results = self.mediapipe_detection(frame, holistic)
                        #print(results)

                        # Draw landmarks
                        self.draw_styled_landmarks(image, results)
                
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        # NEW Export keypoints
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    
            cap.release()
            cv2.destroyAllWindows()
            
    
    def build_model(self):
        
        ''' Build Model for Training'''
        
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=False, activation='relu'))
        #model.add(LSTM(64, return_sequences=False, activation='relu'))
        #model.add(Dense(64, activation='relu'))
        #model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def plot_metrics(self,history):
        
        ''' Plots accuracy and loss curves'''
        
        # PLotting acuracy curves
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train","val"],loc ="upper left")
        plt.savefig("Accuracy_plot.png")
        plt.show()


        # PLotting loss curves

        # PLotting acuracy curves
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train","val"],loc ="upper left")
        plt.savefig("Loss_plot.png")
        plt.show()
        
    def plot_confusion_matrix(self,y_pred,y_true):
    
        # Plotting Confusion Matrix for Test Data

        cm = confusion_matrix(y_pred,y_true)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels,
            cmap="Blues")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show(block=False)
        
        

if __name__ == "__main__":
    
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    

    sequences, labels = [], []
    
    detection = Detection(mp_holistic,mp_drawing)
    
    # Path for creating dataset
    DATA_PATH = os.path.join('MP_Data')
    
    # Actions that we try to detect
    actions = np.array(['left', 'stop', 'right'])
    
    display_labels =["left","stop","right"]
    
    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30
    
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    
    # Collect Training Data
    #train = detection.collect_traindata() # Comment if no need of new dataset
    
    # Create Label Map
    label_map = {label:num for num, label in enumerate(actions)}
    print(label_map)
    
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    
    # Creating Train data and Labels
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # Log dir for storing training logs
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    
    # Build and fit the model
    model = detection.build_model()
    
    print(model.summary())
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_test,y_test),
                        epochs=2000, callbacks=[tb_callback])
    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1).tolist()
    y_true = np.argmax(y_test, axis=1).tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    print("The accuracy is "+ str(accuracy))
    
    recall = recall_score(y_true, y_pred,average='weighted')
    print("The recall score is "+ str(recall))
    
    # Plotting Loss curves
    plot = detection.plot_metrics(history)
    
    # Plotting Confusion Matrix
    confusion_plot = detection.plot_confusion_matrix(y_pred,y_true)
    
    # Save the weights of the model
    saved_model = model.save('sign_detection.h5')
    
    
    
    
    
    
    
    