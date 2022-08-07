# GestureRecognition

The script sign_detection.py is dedicated for new dataset collection and training model:

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
  '''''' Creates Training Data and saves it in directory'''
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

Run the script mountainCar.py for controlling the mountain car in OpenAI gym env with gestures.

Following are the gestures allowed

<img width="795" alt="image" src="https://user-images.githubusercontent.com/110788191/183310805-e05c2728-6259-44a4-a60c-b61de9fda411.png">