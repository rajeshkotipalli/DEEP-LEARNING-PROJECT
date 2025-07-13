# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: K RAJESH

INTERN ID: CTO4DG2597

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTHOSH

In this task i have implemented an image classification deep learning model like convolutional neural networks [CNN'S] with fashion MNIST dataset. this project fpocused on developing a basic, yet powerful model capable of identifying cothing items from grayscale 
TOOLS AND TECHNOLOGIES USED:
Google Colab: A free cloud-based Jupyter notebook environment by Google. I used this due to TensorFlow installation issues on my LAPTOP, it was suggest by open ai 
TensorFlow & Keras are The core libraries used to build, train, and test the deep learning model.
Matplotlib: Used to visualize training accuracy, loss graphs, and sample predictions.
Python 3.10: The programming language used, which is compatible with TensorFlow 2.x.

Fashion MNIST Dataset: This is a dataset consisting of 70,000 grayscale images (60,000 for training and 10,000 for testing), representing 10 different categories of clothing such as T-shirts, trousers, shoes, coats, and bags.
The primary goal was to build a CNN that could:
Learn features from fashion images using convolution layers
Classify images into one of 10 clothing categories
Provide visual feedback on training performance (accuracy/loss)
Predict new, unseen test data and output the predicted label

Position	Label  	Meaning
1  	      9     	Ankle Boot
2	        0      	T-shirt/Top
3  	      0	       T-shirt/Top
4        	3	       Dress
5	        0	     T-shirt/Top
6	        2	     Pullover
7	        7	      Sneaker
8	        2      	Pullover
9	        5      	Sandal
Model Architecture:

The model consists of:
3 Convolutional layers with increasing filter sizes (64, 128, 256)
MaxPooling layers to reduce spatial dimensions
Flatten layer to convert feature maps into a 1D vector
Fully connected Dense layers (including a softmax output layer for multi-class classification)

The model was trained for 10 epochs with a batch size of 64, using the Adam optimizer and sparse categorical crossentropy loss function. A validation split of 20% was applied to monitor generalization.

output images visualizations:
During training, two key graphs were generated:
Accuracy vs Epochs – showed how well the model learned from training data.
Loss vs Epochs – helped identify if the model was overfitting.

Additional outputs included:
A 3x3 grid of sample training images
A prediction result showing the predicted clothing label for a random test image
All output images and model weights were saved in the DL_Task2_Outputs folder for easy access and sharing.
Challenges and Issues Faced:
Initially, I tried to build and run this project locally using Python 3.12 in vs code , but TensorFlow was not supported on that version. I then installed Python 3.10, which supports TensorFlow, but still faced environment path issues like i when i was searching for the locations of python3.10 i cant find it , virtual environment complications, and failed pip installations on Windows.

To overcome this, I shifted to Google Colab[whichh was suggested by ai when i asked it , which provided a pre-installed setup for all required libraries (TensorFlow, Keras, etc.). This significantly improved the speed of model development and helped avoid all local configuration issues. Additionally, I faced slight editor issues on Colab like runtime timeouts during long training sessions, which I handled by saving intermediate results and training with fewer epochs.
 Real-World Application:
This type of image classification project is highly relevant in real-world applications such as:
#E-commerce: Automated tagging of clothing products by type
#Retail Analytics: Smart mirrors that recognize clothes in changing rooms
#Security Systems: Detecting clothing styles in surveillance footage
#Fashion Recommendation Systems: Personalized outfit suggestions

Even though the model was trained on the basic Fashion MNIST dataset, the architecture is scalable to more complex datasets with color images and higher resolutions.as im a starter i just tried my best using platforms , with someones suggestions later im sure that ill defenitely upskill my self to develop and do projects ehich can give better output

OUTPUT:
The image shown below is a 3x3 grid of sample images from the Fashion MNIST training dataset. Each image is 28x28 pixels and represents a type of clothing item, such as a T-shirt, sandal, sneaker, or pullover. These images were used to train the CNN model to identify patterns and classify them into one of 10 categories. Labels like "Label: 0" refer to the integer class index in the dataset, which corresponds to specific clothing types. This visual confirms that the dataset was correctly loaded and understood by the model during training.

output:
<img width="800" height="800" alt="Image" src="https://github.com/user-attachments/assets/81d6b811-f7bd-4c56-8c22-46be40dfba04" />
<img width="1049" height="642" alt="Image" src="https://github.com/user-attachments/assets/5b825c3c-985b-4798-a24d-e1a4a0ad41c1" />







