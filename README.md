# Human-Identification-in-non-cooperative-scenarios

% To execute the code follow the steps below:

  1. Download the dataset from: http://csip.di.ubi.pt/
  2. The data downloaded from the CSIP platform contains 50 subjects images with a total of 2000 images, to separate the images by subjects      and to structurize the code      run the file given by name: "structurize_code.py" that calls the method struct_data. Provide the          required three arguments to method        (Destination path, Source path, Number of subjects).
  3. By executing the above code, the data is now split in three folders: training_set, test_set, and validation_set by the weightages of        70,20,10 % respectively with the subjects folder in it containing the images of same subjects in its folder.
  4. To train the model run the code provided in the file: "Model.py" that will read the data from dataset/ folder and train the model. By      executing this code, the file will be saved in its path by name: "PModel.pkl". This will contain the trained model on the provided          dataset (Note: This step may take some time to train the model).
  5. Once the model is trained in pkl file, you can use it to recognize any person from the provided/trained subjects. To check, put the        images in the folder named: "toRecognize_images" and execute the code in the file "Executer.py". This will read the images in              toRecognize_images folder and will output the recognized subjects. 
  
% Project Description:

By analyzing various techniques and researching some possible solutions to our problem, domain we came up with our own possible architecture to obtain the better results and to test it in real-time environment, General Architecture and modules of our project are presented below:

1) Dataset:
The dataset for our problem domain is gathered from CSIP. By analyzing the datasets of various open source platforms, we found CSIP as the most suitable and better dataset for our problem as it contains the half-cut face images from various angles, directions (left half or right half) and captured in different light sources. It contains 50 subjects to be recognized giving a total of 2000 images. The image samples are divided into three categories: Training set (Containing 70% of the data), Test set (Containing 20% of the data), and Validation set (Containing 10% of the data). These datasets are divided into the mentioned categories using the code provided by name: “structurize_code.py”.

2) Segmentation:
The image sample taken from the dataset is passed to the segmentation module that segments and extracts the face part or to be specific periocular part from the image. For this module, we tested some techniques mentioned in literature, however two of these techniques were found to perform better on our dataset:
a. Viola Jones to first detect and extract eye from the image and take some ratio part of the image around the detected eye to get the periocular part in the image.
b. GrabCut Algorithm that is based on graph cutting segmentation and extracts the face from the background without any training.
However, both of these performed well on our dataset but out of these Viola Jones was found to perform just better than the Grab Cut as it is based on perfect detection of eye and taking the area around it.

3) Feature Extraction:
Once the image is segmented, its features are extracted using pre-trained deep learning model. Various deep learning feature extraction models were used based on ImageNet Dataset, out of which VGG19 was found to work amazingly on our problem.

4) Classifier:
As the feature are extracted from above module, these are passed to the classifying module that contains the classifier taking the input of features and classifying the data on 50 subjects/classes provided. For this purpose, various Machine Learning based classifiers were tested and the best suitable was found to be linear kernel support vector machine (SVM) as the features extracted from above module on our dataset is linearly separable and performs well on linear kernel.

5) Evaluation:
Once the model is trained it is saved using pkl and is tested on various testing images from the dataset using metrics of: Accuracy and Confusion Metrics for subjects provided.
