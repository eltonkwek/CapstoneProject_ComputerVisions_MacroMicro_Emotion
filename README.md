# CapstoneProject_ComputerVisions_MacroMicro_Emotion
This repository is for the storage of code related to my capstone project relating to computer vision for micro/macro emotion for XXXX XX Pte Ltd

Disclaimers: 
 - This is not the typical format of what a README.md should contain
 - Some of the images were not uploaded - such as the Confusion Matrix Images, Precision/Recall
 - Neural Network layers were not included as a snapshot
 - Images used to train the model are not publicly available - some are from private centres of education. Please reach out to the individual owners for access. I will not be providing them. 
 - Company name has been blanked out and referred to as XXXX XX Pte Ltd 

## Background
People voluntarily and involuntarily express their emotions through facial expression in communication, and such emotion could reflect better than the content of the speech. In a video of a speech, people’s continuous movement can be dissected to static pieces of images, and the emotions could be identified and analysed from independent images and connected image sequences. <br />

XXXX XX Pte Ltd offers an end to end platform that facilitates the entire hiring process from shortlisting to offer letter. Hiring processes often involve interviews and XXXX has a video interview feature in their platform. The intent is to screen through the offline video interview once it's completed and to provide a facial analytics summary to the recruiter. <br />

The ultimate goal of the capstone project is to have a prototype model which can function to detect emotions and give reasonable output. Each project participant is expected to gather relevant information to train a model for emotion classification and innovate ideas to generate insights on the outputs. The results can be validated against out-of-sample videos to understand the model performance. <br />
 
## 2. Methodology and Design
The project can be divided into 3 sections:
 - Phase 1: Data Collection/Pre-Processing
 - Phase 2: Model Training/Model Selection
 - Phase 3: Testing/Results Generation and Review 

### Phase 1: Data Collection/Pre-processing
In order to build a suitable dataset for training the model, the members sought out all publicly available dataset of images which have been pre-annotated with the emotions of Anger, Disgust, Fear, Happy, Neutral, Sad and Surprise.<br />

The following image datasets were identified: 
1.	CAS-PEAL Face Database
2.	DISFA+
3.	MMI Facial Expression Database
4.	AffectNet 
5.	JAFFE
6.	KDEF/AKDEF
7.	FER2013<br />

In addition, videos from the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) was used to test the effective of the emotion detection on strips of video which have been emotionally annotated. This provides a strong basis for further investigation on the effectiveness of the trained model. <br />

Images were manually reviewed and a sample size of over 60,000 training images was finalized which consisted of samples from FER2013, KDEF/AKDEF, JAFFE, CASPEAL and AffectNet. <br /> 

A further subset of 6,000 images (out of training sample) were used for model validation purposes. 
One of the obstacles encountered was the smaller sample size for images which were annotated with the emotion “disgust”. (<3,600 images) As there were insufficient samples to train the model, this would be something which we need to take note when reviewing the results. <br />

### Phase 2: Model Training/Model Selection
Phase 2 consists of the following
1)	Facial Localization
2)	Deep Learning Network for Emotion Recognition
3)	Image Resolution  

1) Facial Localization 
Facial localization involves using Haar Cascades.
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of  features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.
(Source: http://www.willberger.org/cascade-haar-explained/)
Only the front-view detection of the Haar Cascade algorithm is used.

2) Deep Learning Network for Emotion Recognition
To facilitate learning, we approached the project from 2 angles to determine which will produce better results. 
Approach 1: Training a new Convolutional Neural Network (CNN) from scratch 
Approach 2: Apply transfer learning using pre-trained model -> VGG-16

Approach 1
We created a neural network with 9 layers using a combination of Dense Layers and Drop-out Layers. (See Appendix A for details). The model is able to achieve 60% accuracy.

Approach 2
Use transfer learning by freezing the pre-trained VGG-16 model

Approach 2 was dropped due to accuracy issues. Approach 1 was used as the final model.
(See Appendix B for accuracy comparison)
The CNN model is approach 1 was trained for 200 epochs and was used for the testing of sample videos. 
3) Image Resolution 
Typically VGG-16 models use 224x224 pixels for their training images. In our approach, the image were re-sized to 48x48 greyscale pictures in order to speed up the training of the model and the greyscale prohibits any bias towards a particular skin colour tone. 

Phase 3: Testing/Results Generation and Review 
For image/video testing of the model, the workflow is as follows:   

Input -> Facial Localisation -> Emotion Prediction -> Analysis

User selects a video input (e.g. mp4) and loads into the script

Facial Localisation
Machine detects face using Haar Cascade detection algorithm

Emotion Prediction
CNN predicts emotion based on the trained model and displays the emotion classification

Analysis
Machine provides a chart based on the prediction and a summary of the emotions displayed by subject in video 

### 3. Findings
In the real world, current studies on compound emotions are limited to use data sets with limited number of categories and unbalanced data distributions. The labels are obtained automatically by machine learning-based algorithms which could lead to inaccuracies. 
The task of emotion detection is challenging due to high similarities of compound facial emotions from different categories. Experiments indicate that pairs of compound emotion (e.g., surprisingly-happy vs happily-surprised) are more difficult to be recognized if compared with the seven basic emotions. However, we hope the proposed data set can help to pave the way for further research on compound facial emotion recognition1.
In relation to the difficulties of differentiating compound emotions, there are some potentially some solutions which can help to solve this. The “dlib” library has a pre-trained facial landmark detector which is used to estimate the location of close to 68 (x,y)-coordinates that map to facial structures on the face. This can help to improve the accuracy of the model if we leverage the dlib library to train facial landmark detectors or customized shape predictors of our own. 
To improve the accuracy of the training models, there are a couple of suggestions.
Firstly, using imbalanced training data will cause the model to be biased. This is particularly so in this model as there were insufficient publicly available data relating to emotions such as “disgust”. Secondly, researchers can also engage behavioural experts to augment their research by providing their feedback on annotated images in order to judge the accuracy of trained emotion classification models. 

### 4. Evaluation and Analysis
### Model Training
In the course of training model selection, we experimented with two way to build the emotion classification models. 
The first was to utilise transfer learning using the Visual Geometry Group’s 16 layers model (VGG16). We froze the VGG layers and trained the last 4 years close to the output. 
Model training was done using the pre-selected 60,000+ images which shows the 7 emotions. However, despite training the layers for over 100 epochs, there was minimal improvement. The accuracy was below 50%. 
In contrast, we built a 9-layer VGG CNN network. Using the same training images, we were able to achieve a training accuracy of over 60% done over 200 epochs. 
Based on the results, the VGG-9 model was selected and used in the eventual model. 

### Haar Cascade Algorithm 
The Haar Cascade algorithm is unable to detect slanted faces. This means that there will be no emotion prediction in the event of the subject tilts his/her head during an interview. 
Haar Cascade classifier employs a machine learning approach for visual object detection which is capable of processing images extremely rapidly and achieving high detection rates. This can be attributed to three main reasons:
•	Haar classifier employs 'Integral Image' concept which allows the features used by the detector to be computed very quickly.
•	The learning algorithm is based on AdaBoost. It selects a small number of important features from a large set and gives highly efficient classifiers.
•	More complex classifiers are combined to form a 'cascade' which discard any non-face regions in an image, thereby spending more computation on promising object-like regions.
The assumption here is that the video should ideally have the subject in full frontal view in order to maximise the effectiveness of the emotion detection. 

### Testing Data
In order to test for the effectiveness of the model in emotion detection, we utilised the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). This is a dynamic, multimodal set of facial and vocal expressions in North American English. To test various dataset, we created strips of videos which are less than 2 minutes on average and loaded into the test model and the output was observed via the graphical plots and automated analysis summary which was generated. There were 4 video samples which were created:
Sample 1: Calm-Angry-Calm (Male Subject) 
Sample 2: Happy-Calm-Happy (Female Subject) 
Sample 3: Calm-Angry-Calm (Female Subject) 
Sample 4: Happy-Angry-Happy (Male Subject)

As the emotions were played by actual actors, we noticed that the model had detected a larger range of emotions. This was further validated when we entered with phase 2 of the test data.

Phase 2 of the test data consists of 3 videos scrapped from YouTube which consists of job interviews of both Caucasian men and women. A higher degree of stability in the observed results and less fluctuation in the emotions was observed. 

Sample 1: Caucasian Male
Sample 2: Caucasian Female
Sample 3: Caucasian Male 

### 5. Results
The results from the testing of the model was divided into 2 separate stages.  
Phase 1 consists of testing the model’s effectiveness in emotion detection using professionally annotated strips of video.  Phase 2 is emotion detection on videos scrapped from YouTube. 

### Scenario 1: Calm-Angry-Calm
Description: Clips of video with the "Calm" state of the actor was introduced at the start. This simulates the candidate entering the interview in a calm and poised state. Thereafter the interviewee is posed a question which triggers the emotion of "Anger" inside him. After replying to the interviewer's question, the candidate returns to his "Calm" state. 
Results
Detector is predicting "Sad" as the emotion for the first 1/3 of the video. Detector is predicting "Angry" as the emotion in the middle of the video. Detector is predicting “Sad” as the emotion for the last 1/3 of the video. 
Conclusions
Despite the annotation that the actor is displaying the "Calm" emotion, the detector is consistently picking up "Sad" as the primary emotion displayed by the actor. This may be due to the imperfection of the trained model or it could be displaying a unseen emotion. 
This might be useful in detecting what is the underlying temperament of the candidate and this can be further investigated when trying to use character profiling tests. 

### Scenario 2: Happy-Calm-Happy
Description: A “Happy” video was inserted at the start of the video to simulate a candidate who is entering the interview room feeling happy. Thereafter, the candidate is presented with a question which might be particularly difficult or uncomfortable to the candidate. After reply to the interviewer with a satisfactory answer, the candidate proceeds on with the interview in her usual jovial self. 
Results
First third of the detection generally showed "Happy" with some elements of "Sad" (Model detection). Middle of the detection showed "Surprise" with some elements of "Neutral". Last third of the detection showed "Happy" consistently 
Conclusion
Detector is able to pick up the condition "Happy" competently. In the middle of the video, during the "Calm" phase, the detector detects the emotion as "Surprise" and not as "Calm". Detector may not be able to differentiate between "Surprise" and "Calm" but is able to identify a change in the emotional state of the interviewer. Thereafter, the detector is able to detect "Happy" again. "Surprise" and "Calm" are distinctly different emotions so it may be worthwhile to train the model further to separate "Surprise" and "Calm".
Scenario 3: Calm-Angry-Calm
Description: A video of the actress displaying the "calm" emotion was setup t omimic the candidate starting the interview as being subdued and relaxed. The "angry" video was placed in the middle to simulate a tough question being asked by the interviwer. The candidate under stress, shows signs of anger.The candidate regains her composure after answering the question and returns to the calm state. 
Results
Results are not as clear cut as scenario 1. The detector picks out a range of emotions with the emotion "Sad" being detected throughout the length of the video. There are fluctuations seen through the first 1/3 of the video - displaying "Happy" and "Surprised" which also repeats itself again the last 3rd of the video. There are scattering of "Neutral" which is the target emotion reflecting "Calm" through the video but they are fewer in frequency. 
Conclusion
In Scenario 1, a male candidate was introduced to this scenario to determine the first instance of the test. Thereafter, the female candidate was introduced to determine the effectiveness of the detector on female candidates. The desired emotion "calm" was not reflected in the majority of the frames in the first 1/3 and last 1/3 of the video. Emotions like "happy" and "surprised" were detected. As the actors were predictably more trained in a larger range of emotions, they might put their own "spin" on how such emotions which introduces more noise into the detection. On a positive note, the middle portion of the detection does show a large negative drop into "angry" which is the desired result. Adjustments may be made to the sampling rate to perhaps reduce the capture of images so as to reduce background noise. 

### Scenario 4: Happy-Angry-Happy
Description: A video of the candidate with behaviour happy was introduced at the start of the interview. Thereafter, a stressful situation was introduced and the candidate got upset. Thereafter, the candidate responded and returned to his usual "Happy" state.
Results
The dip observed in the middle of the frames. This concurs with the portion of the video where the "Anger" emotion was displayed. The first 1/3 and last 1/3 of the video. This showed that the candidate was generally happy with some frames which had elements of "Disgust. “Happy” was at a higher percentage compared with "Disgust". This might be due to some classification errors between "Happy" and "Disgust". 

### Conclusion
The desired detection of the emotion "angry" was observed in the middle of the video as shown by the graph. However, there was some issue with the detector distinguishing happy and disgust. Happy was the well-trained emotion due to a greater availability of the dataset on a public domain, hence the model is well-trained according to the confusion matrix. Perhaps, the other way is to create negative images of "Not Equal to Disgust" might be effective for training the model and improving the accuracy. Phase 2 consists of testing the model’s effectiveness in emotion detection using publicly available video interview data. 

### Phase 2 Data Testing 
Phase 2 data testing consists of 3 videos extracted from YouTube (public data). Overall, the following items were observed:
a)	Subjects were more composed and did not display a greater range of emotions
b)	Detector was able to detect consistent emotions
c)	Fewer range of emotions detected
d)	Emotions were more consistently representative of what we feel should be expressed during an interview


### Conclusion
Commercially available applications such as HireVue has a very comprehensive suite of applications which utilize elements of Computer Vision/Emotion Detection to improve the hiring process. Emotion detection in the interview process would help to remove any biases towards candidates especially when a deeper assessment towards the mental and emotional reaction of the candidates is required. 
The final model which is built for the project is a good first prototype model and allows for opportunities for the development team to further improve or enhance in the long term. 

### Recommendations
a)	Incorporate Natural Language Processing (NLP) capability to the Emotion Detection framework to isolate specific words/instances which result in certain emotions being displayed by the candidate

b)	Larger dataset for training the model. With improvements in the scope and size of the dataset, particularly in emotions such as “Disgust” and “Fear”, it may help to improve the accuracy of the model with a more balanced dataset. 



