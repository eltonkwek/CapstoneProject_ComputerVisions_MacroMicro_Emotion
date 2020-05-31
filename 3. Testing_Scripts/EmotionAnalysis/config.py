
HAAR_CASCADE = "../Haarcascades/haarcascade_frontalface_default.xml"

class_labels={0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Surprise = 2
# Happy =1 
# Neutral =0
# Disgust=-1
# Sad=-2
# Fear=-3
# Angry=-4
pos_neg_mapping={'Angry':-4, 'Disgust': -1, 'Fear':-3, 'Happy':1, 'Neutral':0, 'Sad':-2, 'Surprise':2}

#BGR format (for still image using cv)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
PURPLE = (148, 0, 211)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
PINK = (197, 135, 249)
BLACK=(0, 0, 0)
ORANGE = (0, 165, 255)

#RGB formt (for video)
RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 0, 255)
RGB_PURPLE = (211, 0, 148)
RGB_PINK = (249, 135, 197)
RGB_ORANGE = (255, 165, 0)

# map each emotion to the color used for the bounding box
#for still image
EMOTION_COLOR_MAP = {'Angry':RED, 'Disgust':BLUE, 'Fear': GREEN, 'Happy':PINK, 
                     'Neutral':PURPLE, 'Sad':GRAY, 'Surprise':ORANGE}

#for video mapping
RGB_EMOTION_COLOR_MAP = {'Angry':RGB_RED, 'Disgust':RGB_BLUE, 'Fear': GREEN, 'Happy':RGB_PINK, 
                     'Neutral':RGB_PURPLE, 'Sad':GRAY, 'Surprise':RGB_ORANGE}


#Font size for the emotion label
fontsize=0.7