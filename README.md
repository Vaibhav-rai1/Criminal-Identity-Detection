# Criminal-Identity-Detection

CRIMINAL IDENTITY DETECTION
Author -1 Vaibhav Rai,20BCE2651,School of Computer Science and Engineering Author -2 Sai Abhinav Kolli,20BCE2272, School of Computer Science and Engineering Author 3 Srujan Chinnam 19BCB0022, CSE with specialization in Bioinformatics Author 4 Dr. Swarnalatha. P, professor, ,School of Computer Science and Engineering


 

INDEX

S.NO	CONTENTS
1)	Abstract
2)	Introduction
3)	Literature Survey
4)	Project description and Goals
5)	Proposed System
6)	Methodology
7)	Experimental Setup
8)	Results and Discussion
9)	Findings
10)	Conclusion and Future Work
11)	References
12)	Appendices
Abstract:

There are more and more criminals, and the number of crimes is going up quickly. This makes security problems very scary. Crime prevention and figuring out who did it are the police's top goals, as security of property and life is the main goal of the security forces. However, there aren't enough police to fight crime as much as they'd like to. As security technology has improved, CCTV and other cameras have been put up in many public and private places to keep an eye on things. CCTV video could be used to figure out who was at the scene of the crime. In this project, a cascade- based part of Haar was used to come up with an idea for a criminal identity system. This system will be able to find faces, and it will be able to do so by hand and automatically, in real time. The exact position of the face is still hard, and there are other small difficulties.
Face scanners are shared by groups like OpenCV that are open to the public. Criminal Identification, face
 
recognition, Haar classifier, manual and real-time, OpenCV, fisher recognizer, training, and tensor are some of the keywords that describe these things

1)	Introduction:
Several security measures have been put in place over the years to help keep data safe and reduce the chance of security leaks. Face recognition is one of the few fingerprint methods that is both accurate and doesn't cause much trouble. is a computer program that uses a picture or video frame of a person's face to instantly identify and verify the person [1, 2, 3]. It matches certain facial features from an image to a facial recognition system or maybe a piece of hardware that checks a person's identity.
•	This technology is a fingerprints system that is often used to verify, authorize, validate, and identify people. Face recognition has been used on security cams, door locks, and much more by many businesses. Face recognition has been used on Facebook's website to make a digital record of each person who uses the site. In developed countries, police set up a face mask that can be used with their face recognition system to match any suspect with a website. On the other hand, in Malaysia, most cases are looked into by using thumbprints to find out who might be responsible. But because the Internet gives people access to a lot of information, many thieves know how to recognize thumbprints.
•	Because of this, they are very careful to leave a thumbs up by always wearing gloves. This paper suggests a face recognition system for a website about crimes, where suspects are identified by looking at their faces instead of giving a thumbs-up.
•	There are two goals for this study:
•	Getting a good match between a face and a database.
•	Using principal component analysis to find the similarities between many images by looking for traits that make them different.
 
1)	Objective:
the other that gives very few false negatives. We've talked
So, we show a smart way to reduce violence by detecting about some of the problems with the face recognition
 
and identifying criminal behavior in the field of view of a
certain CCTV camera and having a drone identify, track, and chase the criminal until he or she is caught using our face detection system.

2)	Motivation:

Better efficiency: Finding out who a criminal is takes a lot of time and work, especially when it comes to matching large amounts of data. Using deep learning methods like CNN, the process can be automated and made much more efficient.

Accuracy: Convolutional neural networks are very accurate and have been shown to do better than people at several computer vision tasks, including facial recognition. By using CNN to identify criminals, we can reduce the number of false hits and make sure we are right more often.

people Safety: To keep the people safe, it is important to find and catch criminals. With a better and more accurate method for identifying criminals, police can respond to crimes and bring the people responsible to justice faster.

Cost-effective: Identifying criminals by hand can be expensive because it takes a lot of time and work from people. By using CNN to automate the process, we can cut down on the costs of identifying criminals.
3)	Literature Survey:

1. Web front end Realtime Face Recognition Based on TFJS

In this paper, we suggested a web-based face recognition system that works in real time. We came up
 
process, like how the pictures of the faces are lit and what's in the background.
In the future, our system could be improved by making a face recognition algorithm that is less likely to be wrong or fail and works well no matter what color the skin is. A more complete set of features would also make it harder for someone to trick the system by changing their face.



3.	Face Detection and Recognition System using Digital Image Processing


Since PCA transformation limits the number of Eigen faces that can be used, the system's accuracy for both human and automatic face recognition was less than 90%. More work needs to be done in the area of a fully automated frontal view face recognition system that shows almost perfect accuracy when it is shown.
The way this method will work in the real world will be much more accurate. In order to get a high rate of accuracy, the system that was planned and built was not strong enough.
One of the main reasons for this flaw is that the sub-system of the face recognition system doesn't show small changes in how steady the scale or rotation of the segmented face picture is.
The performance of this system can only be compared to face detection done by hand if the eye detection system is added to the system that has been created.


4.	Facial emotion recognition using deep learning: review and insights


This paper talked about current research on FER, which
 
with a plan by looking at different ways of doing things, let us know what's new in this field. We've talked about
 
standard APIs, and frameworks. Then we put it into practice and tried it in different ways. The finding shows that front-end face recognition in real time is both possible and acceptable to users. In the future, the
 
the different architectures of CNN and CNN-LSTM that
have recently been proposed by different researchers. We've also shown some different databases with spontaneous images from the real world and images made
 
system will be connected to a site for live or on-demand in the lab (see Table.1), so that we can accurately detect
 
video viewing so that face searching and face swapping can be done.


2. Criminal Face Recognition System

A This system uses our version of a face recognition system, which looks at colors, shapes, and distances on a person's face to figure out who they are. With its two degrees of freedom, our system has two ways to work: one that gives very few fake positives and
 
how people feel. We also talk about the high rate that researchers found, which shows that machines will be able to understand feelings better in the future. This means that interactions between humans and machines will become more natural.
 
5.	"A Survey on Criminal Identification using Convolutional	the potential for biases in the algorithms. They also suggest
Neural Networks" by J. R. Huang and J. W. Huang, published several future research directions, such as the development of
 
in the International Journal of Intelligent Systems and Applications in 2019

This Paper provides an overview of the recent developments in criminal identification using CNN, including facial recognition, gait analysis, and body part recognition. The authors also discuss the challenges and future research directions in this field.

The authors also review the challenges faced in the field, such as the need for large datasets and the potential for biases in the algorithms. They also suggest several future research directions, including the development of more accurate and reliable CNN-based criminal identification systems.

Overall, Huang and Huang's survey provides a comprehensive overview of the recent advances and limitations in criminal identification using CNN, and it highlights the potential for future research in this field. Their survey serves as a valuable resource for researchers and practitioners interested in criminal identification using CNN.
 
more accurate and robust deep learning-based criminal identification systems and the exploration of multi-modal data fusion techniques.

Overall, Arora and Kulkarni's review provides a comprehensive analysis of the recent advances and limitations in criminal identification using deep learning, and it highlights the potential for future research in this field. Their review serves as a valuable resource for researchers and practitioners interested in criminal identification using deep learning techniques.

8. "A Survey on Convolutional Neural Networks for Criminal Recognition" by W. Hu, Y. Wu, and H. Wang, published in the Journal of Multimedia in 2020.

The authors provide an overview of recent developments in criminal recognition using convolutional neural networks (CNN). They discuss various applications of CNN in criminal recognition, including facial recognition, gait analysis, and object recognition.
 

 
6.	"Deep Learning Techniques for Criminal Identification: A Survey" by S. S. Kulkarni and S. S. Kulkarni

The authors provide an overview of recent advances in criminal identification using deep learning techniques, including CNN. They discuss various applications of deep learning in criminal identification, such as facial recognition, gait analysis, and behavioural biometrics.

The authors also highlight the challenges faced in this field, including the need for large datasets, privacy concerns, and the potential for biases in the algorithms. They also suggest several future research directions, such as the development of more robust and reliable deep learning-based criminal identification systems.

Overall, Kulkarni and Kulkarni's survey provides a comprehensive overview of the recent advances and limitations in criminal identification using deep learning techniques, and it highlights the potential for future research in this field. Their survey serves as a valuable resource for researchers and practitioners interested in criminal identification using deep learning techniques.

7.	"Criminal Identification Using Deep Learning: A Comprehensive Review" by S. P. Arora and S. S. Kulkarni

The authors provide a detailed overview of recent developments in criminal identification using deep learning, including CNN. They discuss various applications of deep learning in criminal identification, such as facial recognition, gait analysis, and behavioral biometrics, and also analyze the performance of these methods using various evaluation metrics.
 
The authors also review the challenges faced in this field, such as the need for large datasets and the potential for biases in the algorithms. They also suggest several future research directions, including the development of more accurate and robust CNN-based criminal recognition systems.

Overall, Hu, Wu, and Wang's survey provides a comprehensive overview of the recent advances and limitations in criminal recognition using CNN, and it highlights the potential for future research in this field. Their survey serves as a valuable resource for researchers and practitioners interested in criminal recognition using CNN.


9.	"A Comprehensive Survey of Deep Learning Techniques for Criminal Identification" by S. P. Arora and S. S. Kulkarni, published in the Journal of Intelligent and Fuzzy Systems in 2021.

The authors provide a comprehensive overview of recent developments in criminal identification using deep learning techniques, including CNN. They discuss various applications of deep learning in criminal identification, such as facial recognition, gait analysis, and body part recognition.

10.	"A Survey on Criminal Identification Techniques Using Deep Learning" by M. S. Khan, S. A. Raza, and F. Iqbal, published in the Journal of Computational Science in 2021.
 

 
 



The authors review the challenges faced in this field, such as the need for large datasets, privacy concerns, and the potential for biases in the algorithms. They also suggest
 
Methodology:

1.	Import the required modules:
For face detection to work, you need cv2, OS, image module, and NumPy. Cv2 is a part of OpenCV, and it has methods for
 
several future research directions, including the development face recognition and recognition. The OS will be used to
 
of more accurate and robust deep learning-based criminal The authors also provide an overview of recent developments

gait analysis, and behavioral biometrics.


Project Description and Goals
The number of crimes is going up quickly, and there are more and more crooks. Problems with security are very scary because of this. The police's main goals are to stop crime and find out
 
control the images and the names of the directories. First, we use this module to get the names of the pictures from the catalogue. Then, we use these names to turn each image into a number, which is used as the face label on the image. Since the pictures in the data set are in gif format, which OpenCV no longer supports, the image module from PIL is used to read images in grayscale format. Images are stored in a lot of similar ways

2.	Load the face detection Cascade:
The first step to getting a face on each picture is to upload a face recognition cascade. Once we find a field of interest that has people in it, we use it to train the dreamer. We will use the Haar Cascade tool that OpenCV gives us to find faces. The
 
who did it. The main goal of the security forces is to keep people OpenCV startup guide can be used to find the Haar cascades
 
and things safe. But there aren't enough cops for them to fight crime as much as they would like. CCTV and other cameras have been put up in many public and private places to keep an eye on things as security technology has gotten better.
 
that come with OpenCV. Face recognition is done with the help of Haar cascade frontal-face default.xml. The cv2 Cascade Classifier method takes the path to the cascade xml file and loads Cascade. If the XML file is on the active list, the related
 
CCTV footage could be used to find out who was there when the method is used.
crime happened. A cascade-based part of Haar was used in this
 
project to come up with an idea for an illegal identity system. This system will be able to find faces, both manually and automatically, and it will be able to do so in real time. It's still hard to put the face in the right place, and there are other small problems. Groups like OpenCV that are open to the public share face sensors. Some words that can be used to explain these things are criminal identification, face recognition, Haar classifier, manual and real-time, OpenCV, fisher recognizer, training, and tensor.

4)	Proposed System

The suggested face recognition system is better than the current face recognition system in some ways. It works by taking the most important features from a set of human faces in the database and doing math on the numbers that match to them. So, when a new image is fed into the system to be recognized, the key features are taken out and used to figure out how far apart the new image is from the images that have already been stored. So, a new face picture that needs to be recognized can have a little bit of a different look. When a person's new picture is different from the pictures of that person already in the database, the system will be able to recognize the new face and figure out who it belongs to.
The suggested system is better because it uses only parts of the face instead of the whole face. The good things about it are:

•	Better ability to recognize and tell things apart Computational cost because smaller images (main features) require less processing to teach the PCA. Because dominant features are used, the PCA can be used as an effective method of authentication.
 
3• Create the Face Recognizer Object:
The next step is to make a mask for your face. FaceRecognizer is one of the things that the face recognition tool can do.train means to teach the visioner and FaceRecognizer something new.guess the face to see it. OpenCV has Eigenface Recognizer, Fisherface Recognizer, and Local Binary Patterns Histograms (LBPH) Face Recognizer. We used LBPH vision because nothing is ever wrong in real life. We can't promise that your pictures or 10 different personal photos will have perfect lighting. LBPH is all about taking out local details from pictures. The idea is not to look at the whole picture as a high- magnitude vector, but to describe only the small parts of it.

4.	Testing:
To test Face Recognizer, we put a trained face in front of the camera and look for the predicted label. This tells us if the detection was right. The name of the sample picture folder is used with the OS module and character unit functions to get the label. The more confident a prediction is, the less points there are.

5.	OpenCV:
To get a face, we used an OpenCV-based cassette-based cascade divider. It is a method based on machine learning in which many positive and negative images are used to teach the cascade function. Then, it is used to find things in other pictures.
 


 
6.	LBPH Algorithm:
 
Haar classifier:
 

Also, we found the faces with the help of Local Binary Patterns Histograms (LBPH). Some good things about this method are: Choosing a trait that works, Scale and static location detector, instead of measuring the image itself, scaling features like a standard acquisition method are used That can be taught to find different kinds of things (e.g., cars, signboards, phone numbers etc.). The LBPH detector can find faces with great accuracy even when the light is changing. Also, LBPH can see very clearly if only one training picture is used for a person. Some bad things about our app are: The detector works best with images that are looking forward. You can't turn 45° in both the vertical and horizontal direction

Technical Specifications:
Criminal Identification:
Criminal identification is the process of figuring out who a criminal is or if someone is accused of being a criminal. In this process, the suspect's facial features, fingerprints, or DNA are compared to current records in criminal databases to see if there is a match. The goal of criminal identity is to find and catch criminals so that they can be brought to justice. It is a key part of law enforcement and is very important for keeping the people safe. Convolutional Neural Networks (CNNs) and other new technologies have made identifying criminals faster, more accurate, and less expensive.
Facial recognition:	Haar classifier is a machine learning-based approach used for object detection in computer vision. It is based on the Haar Wavelet technique and is commonly used for detecting faces in images and videos.
	Haar classifiers work by analysing image features at different scales and orientations to detect the presence of an object in an image. The technique involves the use of a set of rectangular Haar-like features that describe the intensity variations of an object's pixels. The classifier then uses these features to train a machine learning model, typically a cascade classifier, which can accurately detect the object of interest in an image.
	The Haar classifier approach has been widely used in various applications, including face detection, object detection, and pedestrian detection. Its effectiveness is due to its ability to detect objects in real-time, even under challenging conditions such as variations in lighting and background clutter. However, the technique also has its limitations, such as its sensitivity to image noise and occlusions.
	OpenCV:
	OpenCV stands for Open Source Computer Vision Library. It is a cross-platform, open-source library of programming functions primarily designed for real-time computer vision and machine learning applications. OpenCV was originally developed by Intel in 1999 and has since become one of the most popular libraries in the field of computer vision.
Facial recognition is a technology that uses artificial intelligence and machine learning algorithms to identify and verify a person's identity by analysing their facial features. The process involves capturing an image or video of a person's face and comparing it to a database of known faces to find a match. Facial recognition technology can be used for various purposes, including security, surveillance, access control, and marketing.	
OpenCV provides a range of functions and tools for performing various tasks in computer vision, including image and video processing, object detection and recognition, feature detection and matching, camera calibration, and machine learning. It supports multiple programming languages such as C++, Python, and Java, making it widely accessible to developers.

Fisher recognizer:
Facial recognition works by analysing unique features of a person's face, such as the distance between the eyes, the shape of the nose, and the contours of the jawline. These features are compared to a database of faces to identify a match. The technology has advanced significantly in recent years, and facial recognition systems are now capable of identifying faces with high accuracy, even in challenging conditions such as low light or with partial facial occlusion. However, the use of facial recognition technology has also raised concerns
about privacy and potential misuse, prompting calls for regulations to ensure its responsible use.	Fischer’s recognizer, also known as Fischer’s linear discriminant analysis, is a pattern recognition algorithm used for feature extraction and classification.

It works by projecting high dimensional feature data into a lower dimensional space while maximizing the class separability. Algorithm computes a set of linear discriminant
functions that maximize the ratio.
	
	
	
 
Application of Shneideramn’s Eight Golden Principles for this Project:

	1.		First, we had chosen an appropriate CNN architecture for the face detection task. Popular architectures for face detection include Faster
R-CNN, YOLO, and SSD.
		
	2. Then, we collected and prepared a high-quality face detection data set that is diverse and representative of the target population. Ensure that the data set is free from biases and errors.

		
	3. After the first 2 steps, we Augmented the training data set by using techniques such as image rotation, scaling, and flipping. This can help to improve the robustness and accuracy of
the CNN.
		
	4. And then used transfer learning to fine-tune pre-trained CNN models on the specific face detection task. This can help to reduce the amount of training data required and improve
the accuracy of the system.

		
	5. Then we Regularized the CNN model to prevent overfitting and improve the generalization performance. This can be done using techniques such as dropout, weight decay, and early
stopping.
		
	6. After this, we used appropriate loss functions for the face detection task. For example, use binary cross-entropy loss for binary classification tasks and categorical cross-
entropy loss for multi-class classification tasks.

		
	7. And then Optimized the hyperparameters of the CNN model using techniques such as grid search, random search, and Bayesian optimization. This can help to improve the
performance of the system.
		
	8. Then evaluated the performance of the face detection CNN model using appropriate metrics such as precision, recall, and F1-score. Use cross-validation and hold-out validation to
ensure that the results are statistically significant and robust.

Some of the Architectural Diagrams which helped us for implementing this project are shown in these figures:

Use Case Diagram
 
Activity Diagram ER Diagram Sequence Diagram
Deployment Diagram

 






 





Uniqueness of this Project and CNN:

In this Project, we used video surveillance for which it Has many uses. It is a kind of live criminal detection. Here, when a person stands in front of the camera, then Based on the stored dataset, the model can detect if he is
The criminal or not. (Input based on the stored dataset only, otherwise, it cannot detect. So storing the details in dataset is must). After detecting the criminal, it provides their details
In the next step automatically. Thus, their entire data can be known just by Video Surveillance module.

Convolutional Neural Networks (CNN) are a unique way to identify criminals because they can learn and recognize trends in large sets of visual and non-visual data, like images, videos, and biometric signals. CNNs are made so that they can easily find features and patterns in these data sets without having to be explicitly programmed. This makes CNNs good for a wide range of criminal identification tasks, such as face recognition, gait analysis, and behavioral biometrics.

CNNs are also unique because they can learn hierarchical models of the data they are given. The first layers of a CNN usually learn simple features like edges and corners, while the higher levels learn more complicated patterns and structures. This lets CNNs successfully capture the rich and complex information in criminal identification data sets, which can be hard to analyze and understand using traditional methods.

Lastly, the use of deep learning methods, such as CNNs, has made criminal identification systems much more accurate and reliable. Now, these systems can correctly identify people from big, noisy data sets, like those that are collected during real-world criminal investigations. This is important for law enforcement and the criminal justice system because it can help make criminal cases more efficient and accurate and make it easier to identify criminals.



Screenshots of implementation:


1.	User Interface 2.Criminal Registration 3.Image Recognition 4.Criminal Details 5.Failed Case
 


 

8)	Results and Discussion:
So, using the steps above, we were able to finish our "CNN-Based Criminal Identification" project using the TensorFlow JS Library and the P5 JS Library. We used ml5.js to train our model and got an accuracy of 76.19%.
We've used VS Code Editor, which is a source code editor that is small but strong. It has built-in support for JavaScript, which we used to teach CNN how to identify criminal faces.
Our system's main job is to find and pull out faces from a picture and figure out if the person is a criminal or not. For this, we've made a database with more than 500 pictures of criminals and people who aren't criminals. Each of these images is 6464 pixels. For training and testing, it is further split in an 8:2 ratio.
Our system successfully achieved the result by recognizing Criminal faces, checking from all aspects such as Blur images, face rotation, and addition of other features like glasses, caps, etc

9)	Findings:
In this project, we can use photos and videos to find and identify the names of criminals. The main reason we did this project was to show how important AI will be in the future and how it can help avoid crime in many different fields and domains:
Terrorist acts: Stores and pharmacies can use advanced AI tools to find customers who buy unusually large amounts of drugs that can be used to start terrorist acts.
Illegal shipping: Using AI, transport companies can figure out how likely it is that packages contain illegal items like drugs and report them to the right people.
Human trafficking: Shipping companies can save lives by using their data and AI to find packages that might be used for human trafficking.
 
1) Accuracy of the system:












Total number of Test images: 40

•	Total Number of Correct Prediction Positive (TP): 10

•	Total Number of Correct Prediction Negative (TN): 6

•	Total Number of Correct Prediction True Positive and True Negative: 16
•	Total Number of Wrong Prediction Positive (FP): 0

•	Total Number of Wrong Prediction Negative (FN): 5 •
Accuracy: 76.19047619047619

•	Recall: 0.6666666666666666

Conclusion and Future Work: Conclusion:

In this project, we can use pictures and videos to find out who the criminals are and what their names are. The main reason we did this project was to show how important AI will be in the future and how it can help prevent crime in many different fields and domains:
•	Terrorist acts: Stores and pharmacies can use advanced AI tools to find customers who buy strangely large amounts of drugs that could be used to start terrorist acts.
•	Illegal shipping: Using AI, shipping companies can figure out how likely it is that packages contain illegal things like drugs and report them to the right people.
•	Human slavery: Shipping companies can save lives by using their data and AI to find packages that could be used for human trafficking.
 

 
Future Scope:







1.	Improving the accuracy of CNN-based facial recognition systems by developing more advanced CNN architectures and algorithms, as well as improving the quality and diversity of training data.
2.	Addressing issues of bias and discrimination in CNN- based facial recognition systems by developing methods to ensure that the systems are trained on diverse and representative datasets and evaluating their performance across different demographic
groups.
3.	Developing systems that can recognize faces in real- time and under varying lighting conditions and camera angles, which can be particularly useful for law enforcement agencies in identifying suspects in
real-time.
4.	Integrating CNN-based facial recognition systems with other technologies such as surveillance cameras, drones, and social media platforms to enhance the accuracy and efficiency of criminal identification.
5.	Investigating the ethical implications of using CNN- based facial recognition systems for criminal identification, particularly with regard to privacy and
civil rights.
 


5.	https://www.sciencedirect.com/science/article/pii/ S22147853210 47672

6.	https://www.researchgate.net/profile/A-S- SyedNavaz/publication/235950165_FACE_RECOGNITI ON_USING_PRIN CIPAL_COMPONENT_ANALYSIS_AND_NEURAL_NETW ORKS/links/ 00b495226b7a6e5985000000/FACE- RECOGNITION-USINGPRINCIPAL-COMPONENT- ANALYSIS-AND-NEURAL-NETWORKS.pdf

7.
https://link.springer.com/chapter/10.1007/978- 3-030-75680- 2_41

8. https://link.springer.com/article/10.3758/s13423- 014-0641-2

12)	Appendices:


1.	A dataset description: This would include details about the dataset used to train and test the CNN model, such as the size of the dataset, the number of images per class, and any pre processing steps taken.

2.	Code snippets: This would include code snippets for any custom code used in the project, such as the CNN model architecture, data pre processing, and training and evaluation scripts.


3.	Results tables and graphs: This would include tables and graphs showing the performance of the CNN model on the test data, such as accuracy, precision, recall, and F1 score.
 

 
. 11)References:


•	1.https://heinonline.org/HOL/LandingPage?handle=hei n.journals/br anlaj49&div=30&id=&page 2.https://ieeexplore.ieee.org/abstract/document/9377 205 3.https://ieeexplore.ieee.org/abstract/document/9137 927
4.	https://link.springer.com/article/10.3758/s13423- 014-0641-2
 
4.	Ethical considerations: This would include a discussion of any ethical considerations related to the use of facial recognition technology for criminal identification, such as privacy, bias, and discrimination.


5.	Future work: This would include a discussion of potential future work related to the project, such as improving the CNN model architecture, expanding the dataset, or exploring new applications of facial recognition technology in criminal identification.
6.
Overall, the appendices for a criminal identification project using CNN would provide additional details and supporting materials that can help better understand.
 


















.
