# run in bash directly. DO NOT RUN THIS FILE

python -c "
import numpy
import pandas
import scipy
import sklearn
import torch
import mediapipe
import cv2
import ultralytics
import yaml
import joblib
import tqdm
import matplotlib
import seaborn

print('======================================')
print('DEPENDENCY CHECK')
print('======================================')
print('NumPy       :', numpy.__version__)
print('Pandas      :', pandas.__version__)
print('SciPy       :', scipy.__version__)
print('Scikit-learn:', sklearn.__version__)
print('PyTorch     :', torch.__version__)
print('CUDA        :', torch.version.cuda)
print('GPU enabled :', torch.cuda.is_available())
print('GPU         :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('MediaPipe   :', mediapipe.__version__)
print('OpenCV      :', cv2.__version__)
print('Ultralytics :', ultralytics.__version__)
print('PyYAML      : OK')
print('Joblib      :', joblib.__version__)
print('tqdm        :', tqdm.__version__)
print('Matplotlib  :', matplotlib.__version__)
print('Seaborn     :', seaborn.__version__)
print('======================================')
"