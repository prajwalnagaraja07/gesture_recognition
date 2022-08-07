# GestureRecognition

The script **sign_detection.py** is dedicated for new dataset collection and training model:

The following are methods implemented in Detection class
```
def mediapipe_detection(self,image, model):
   ''' Takes an image and model for prediction'''
   return image,results
```
```
def extract_keypoints(self,results):
  ''' Extracts Kepoints and returns a numpy array'''
  
  return np.concatenate([pose, face, lh, rh])
```
```
def collect_traindata(self):
   '''Creates Training Data and saves it in directory'''
   pass
```
```
def plot_metrics(self,history):      
   ''' Plots accuracy and loss curves'''
   pass
```
```
def plot_confusion_matrix(self,y_pred,y_true):
   '''Plotting Confusion Matrix'''
   pass
```
Comment out the following line in sign_detection.py, if new dataset is not needed.
```
train = detection.collect_traindata() # Comment if no need of new dataset
```

Run the script **mountainCar.py** for controlling the mountain car in OpenAI gym env with gestures.

It has following methods, also few methods are inherited from class Detection:

```
def enableGame(self,episodes):
   '''Enable the game for live control of mountain car for specified episodes'''
   pass
```

```
def prob_viz(self,res, actions, input_frame, colors):
   '''Displays the predicted action'''   
   return frame
```

Following are the gestures allowed:

Left: Lift your left hand to move the car to left.

Right: Lift your right hand to move the car to right.

Stop: Lift your both the hands to stop the car.

<img width="795" alt="image" src="https://user-images.githubusercontent.com/110788191/183310805-e05c2728-6259-44a4-a60c-b61de9fda411.png">

The script **model_visul.py** is to see and debug the model (VGG16) learnings using tf-keras-vis library.

