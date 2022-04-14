# Import essential libraries

import face_recognition
import urllib.request 
from deepface import DeepFace
import csv
import os
import glob
import tqdm

class FACIAL_RACE_AGE_ETHNICS_EMOTIONS_FWHR:
    def __init__(self):
#        takes path where images located
        self.img_path = input('Enter input images path:(eg: .../.../SEC_Filings/):')
        self.strt = int(input('Enter start year:'))
        self.end = int(input('Enter end year:'))
        self.create_excel()
        
    def get_image_link(self, path, year):
        
        pth_by_year = path + '/' + str(year)+'/'
        all_qtr = os.listdir(pth_by_year)
        
        for qtr in all_qtr:
            pth_by_qtr =pth_by_year + qtr + '/' + 'Images/' + '/**/*'
            imgPth = glob.glob(pth_by_qtr)
            for imgpath in imgPth:
                yield  imgpath.replace("\\","/")
                
    def get_face_points(self, points, method='average', top='eyebrow'):
        width_left, width_right = points[0], points[16]
        
        if top == 'eyebrow':
            top_left = points[18]
            top_right = points[25]
            
        elif top == 'eyelid':
            top_left = points[37]
            top_right = points[43] 
            
        else:
            raise ValueError('Invalid top point, use either "eyebrow" or "eyelid"')
            
        bottom_left, bottom_right = points[50], points[52]
        
        if method == 'left':
            coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])
            
        elif method == 'right':
            coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])
            
        else:
            top_average = int((top_left[1] + top_right[1]) / 2)
            bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
            coords = (width_left[0], width_right[0], top_average, bottom_average)
            
        ## Move the line just a little above the top of the eye to the eyelid    
        if top == 'eyelid':
            coords = (coords[0], coords[1], coords[2] - 4, coords[3])
            
        return {'top_left' : (coords[0], coords[2]),
                'bottom_left' : (coords[0], coords[3]),
                'top_right' : (coords[1], coords[2]),
                'bottom_right' : (coords[1], coords[3])
               }
    
    def good_picture_check(self, p, debug=False):
    ## To scale for picture size
        width_im = (p[16][0] - p[0][0]) / 100
        
        ## Difference in height between eyes
        eye_y_l = (p[37][1] + p[41][1]) / 2.0
        eye_y_r = (p[44][1] + p[46][1]) / 2.0
        eye_dif = (eye_y_r - eye_y_l) / width_im
        
        ## Difference top / bottom point nose 
        nose_dif = (p[30][0] - p[27][0]) / width_im
        
        ## Space between face-edge to eye, left vs. right
        left_space = p[36][0] - p[0][0]
        right_space = p[16][0] - p[45][0]
        space_ratio = left_space / right_space
        
        if debug:
            print(eye_dif, nose_dif, space_ratio)
        
        ## These rules are not perfect, determined by trying a bunch of "bad" pictures
        if eye_dif > 5 or nose_dif > 3.5 or space_ratio > 3:
            return False
        else:
            return True
        
    def FWHR_calc(self, corners):
        width = corners['top_right'][0] - corners['top_left'][0]
        height = corners['bottom_left'][1] - corners['top_left'][1]
        return float(width) / float(height)
    
    def get_fwhr(self, path, url=False, show=True, method='average', top='eyelid'):
        image = face_recognition.load_image_file(path)
        landmarks = face_recognition.api._raw_face_landmarks(image)
        landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]
        
        if self.good_picture_check(landmarks_as_tuples): 
            corners = self.get_face_points(landmarks_as_tuples, method=method, top = top)
            fwh_ratio = self.FWHR_calc(corners)
        else:
            fwh_ratio = 0.0
        return fwh_ratio
    
    def facial_features(self, path):
        obj = DeepFace.analyze(path, actions = ['age', 'gender', 'race', 'emotion'])
        x = [obj["age"] ,obj["dominant_race"], obj["dominant_emotion"], obj["gender"]]
        emotions = obj['emotion']
        emotion_scores = [emotions['angry'], emotions['disgust'], emotions['fear'], emotions['happy'], 
                          emotions['sad'], emotions['surprise'], emotions['neutral']]
        races = obj['race']
        race_scrores = [races['asian'], races['indian'], races['black'], races['white'],
        races['middle eastern'], races['latino hispanic']]
        
        return [x, race_scrores, emotion_scores]

    def Image_Features_calculation(self, path):
        y = self.facial_features(path)
        print(y[0])
        file_path = path
        # ...........................................
        r_scores = y[1]
        Asian_score	= r_scores[0]
        White_score	= r_scores[3]
        Middle_Eastern_score = r_scores[4]
        Indian_score = r_scores[1]
        Latino_score = r_scores[5]
        Black_score = r_scores[2]
        # .........................................
        e_scores = y[2]
        Angry_score	= e_scores[0]					
        Fear_score = e_scores[2]
        Neutral_score = e_scores[6]
        Sad_score = e_scores[4]
        Disgust_score = e_scores[1]
        Happy_score = e_scores[3]
        Surprise_score = e_scores[5]
        # ...........................................
        a_r_e_g = y[0]
        age = a_r_e_g[0]
        male = female = 0 
        if a_r_e_g[3] == 'Man':
            male = 1
            female = 0
        elif a_r_e_g[3] == 'Woman':
            male = 0
            female = 1
        x = self.get_fwhr(path)
        angry = fear = neutral = sad = disgust = happy = surprise = 0
        # ............................................
        if a_r_e_g[2] == 'angry':
            angry = 1
            fear = neutral = sad = disgust = happy = surprise = 0
        elif a_r_e_g[2] == 'fear':
            fear = 1
            angry = neutral = sad = disgust = happy = surprise = 0
        elif a_r_e_g[2] == 'natural':
            neutral = 1
            angry = fear = sad = disgust = happy = surprise = 0
        elif a_r_e_g[2] == 'sad':
            sad = 1
            angry = fear = neutral = disgust = happy = surprise = 0
        elif a_r_e_g[2] == 'disgust':
            disgust = 1
            angry = fear = neutral = sad = happy = surprise = 0
        elif a_r_e_g[2] == 'happy':
            happy = 1
            angry = fear = neutral = sad = disgust = surprise = 0
        elif a_r_e_g[2] == 'surprise':
            angry = fear = neutral = sad = disgust = happy = 0
            surprise = 1
        # ..........................................     
        asian = white = middle_eastern = indian = latino = black = 0
        
        if a_r_e_g[1] == 'asian':
            asian = 1
            white = middle_eastern = indian = latino = black = 0
        elif a_r_e_g[1] == 'white':
            white = 1
            asian = middle_eastern = indian = latino = black = 0
        elif a_r_e_g[1] == 'middle_eastern':
            middle_eastern = 1
            asian = white = indian = latino = black = 0
        elif a_r_e_g[1] == 'indian':
            indian = 1
            asian = white = middle_eastern = latino = black = 0
        elif a_r_e_g[1] == 'latino':
            latino = 1
            asian = white = middle_eastern = indian = black = 0
        elif a_r_e_g[1] == 'black':
            asian = white = middle_eastern = indian = latino = 0
            black = 1
            
        lst = [file_path, age, male, female, x, angry, fear, neutral, sad, disgust, 
               happy, surprise, asian, white, middle_eastern, indian, latino, black,
               Angry_score, Fear_score,	Neutral_score, Sad_score, Disgust_score, 
               Happy_score, Surprise_score, Asian_score, White_score, 
               Middle_Eastern_score, Indian_score, Latino_score, Black_score] 
        return lst
    
    def create_excel(self):
      keys = ['file_path', 'Age', 'Male', 'Female', 'FWHR', 'Angry', 'Fear', 'Neutral', 'Sad', 'Disgust', 
               'Happy', 'Surprise', 'Asian', 'White', 'Middle_Eastern', 'Indian', 
               'Latino', 'Black', 'Angry_score', 'Fear_score',	'Neutral_score', 'Sad_score', 
               'Disgust_score', 'Happy_score', 'Surprise_score', 'Asian_score', 'White_score', 
               'Middle_Eastern_score', 'Indian_score', 'Latino_score', 'Black_score'] 
      with open('output.csv', 'w', newline = '') as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerow(keys)
        for year in range(self.strt, self.end+1):
            img_path_gen = self.get_image_link(self.img_path, year)
            for img_link in img_path_gen:
              if not img_link.endswith('.gif'):
                  try:
                      lst = self.Image_Features_calculation(img_link)
                      if lst[4] != 0:
                          csv_writer.writerow(lst)
                  except:
                      pass
            
      return
