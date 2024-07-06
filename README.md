# Anatomy of Deep Learning:


Deep learning: One of the machine learning technique that learns features directly from data.

Why deep learning: When the amounth of data is increased, machine learning techniques are insufficient in terms of performance and deep learning gives better performance like accuracy.

What is amounth of big: It is hard to answer but intuitively 1 million sample is enough to say "big amounth of data".

Usage fields of deep learning: Speech recognition, image classification, natural language procession (nlp) or recommendation systems.

What is difference of deep learning from machine learning:

Machine learning covers deep learning.

Features are given machine learning manually.

On the other hand, deep learning learns features directly from data.

![image](https://github.com/Tiwari666/Anatomy_of_Deep_Learning/assets/153152895/4a77e8b6-8422-4c04-a58d-c8338103de1f)




# Deep Learning vs Machine Learning:

# 1 Automatic Feature Extraction

--DL Advantage:  Deep learning has ability to automatically learn features due to its architecture in input layer, hidden layer and output layer.

--ML Limitation: Traditional ML often requires manual feature extraction and selection, which relies heavily on domain expertise and may not capture complex patterns hidden in raw data as effectively as DL.

# 2. Handling Large Amounts of Data: 

--DL  handles large datasets

--ML Limitation: traditional ML models might struggle due to computational and performance limitations , suffering from the overfiting  or underfiting when trained on large datasets without extensive preprocessing and feature engineering, potentially limiting their performance and generalization.

# 3.  Complex Non-linear Relationships:

--DL Advantage: DL capture model complex relationships (non-linear relation) due to its deep neural networks with multiple layers that is crucial for tasks like image classification, language translation, and sequence prediction.

--ML Limitation: Traditional ML models, such as linear regression or decision trees, are limited in their ability to model complex relationships unless feature engineering is meticulously performed. This can restrict their performance in tasks where data exhibits intricate patterns.

# 4. Hierarchical Representation Learning:

--DL Advantage: Deep learning architectures, like convolutional neural networks (CNNs) and recurrent neural networks (RNNs), are designed to learn hierarchical representations of data. This means they can capture features at multiple levels of abstraction, which is beneficial for tasks like image and speech recognition.

--ML Limitation: Traditional ML models often rely on shallow representations of data and may struggle to learn hierarchical features without manual intervention. This limitation can hinder their performance in tasks requiring understanding of complex relationships within data.

# 5. State-of-the-Art Performance in Specific Tasks:

DL Advantage: Deep learning has achieved state-of-the-art performance in various domains, including computer vision (e.g., object detection, image segmentation), natural language processing (e.g., machine translation, sentiment analysis), and speech recognition. These achievements are often driven by the ability of deep learning models to leverage large-scale data and powerful computational resources.

ML Limitation: Traditional ML approaches may not match the performance of deep learning models in tasks where data complexity and volume are significant factors. They may require more effort in feature engineering and tuning to achieve comparable results.

# 6 Interpretability
Last but not the least, we have interpretability as a factor for comparison of machine learning and deep learning. This factor is the main reason deep learning is still thought 10 times before its use in industry.

Let’s take an example. Suppose we use deep learning to give automated scoring to essays. The performance it gives in scoring is quite excellent and is near human performance. But there’s is an issue. It does not reveal why it has given that score. Indeed, mathematically we can find out which nodes of a deep neural network were activated, but we don’t know what there neurons were supposed to model and what these layers of neurons were doing collectively. 
---------------So we fail to interpret the results from the deep neural.----------------------

On the other hand, machine learning algorithms like decision trees give us crisp rules as to why it chose what it chose, so it is particularly easy to interpret the reasoning behind it. Therefore, algorithms like decision trees and linear/logistic regression are primarily used in industry for interpretability.

# Conclusion:

Deep learning's ability to automatically learn features, handle large datasets, model complex relationships, and achieve state-of-the-art performance in various domains distinguishes it from traditional machine learning methods. While both DL and ML are valuable tools depending on the task and available data, deep learning excels in scenarios where raw data can be leveraged effectively to solve complex problems without extensive manual feature engineering.

----------------------------------------------------------------------------------------------------------------------------------------------------

# Step-by-Step Explanation of How Deep Learning Works

# 1. Data Collection and Preparation
 
Data Collection: Gather a large dataset relevant to the problem you want to solve. For example, a dataset of handwritten digits (MNIST dataset) for digit recognition.

Data Preparation: Preprocess the data, which may include normalization, splitting into training and testing sets, and handling missing values.

# 2. Building the Neural Network

Architecture Design: Decide on the architecture of the neural network. This involves choosing the number of layers, types of layers (e.g., dense layers, convolutional layers), and activation functions.

Input Layer: The first layer of the network receives the input data, which could be images, text, or numerical data.

Hidden Layers: Intermediate layers between the input and output layers where the network learns representations. These layers perform operations using weights (parameters) that are learned during training.

The second type of layer is called the hidden layer. Hidden layers are either one or more in number for a neural network. Hidden layers are the ones that are actually responsible for the excellent performance and complexity of neural networks. They perform multiple functions at the same time such as data transformation, automatic feature creation, etc.

Output Layer: The final layer produces the output of the network, which could be class probabilities for classification tasks or numerical values for regression tasks.

The last type of layer is the output layer. The output layer holds the result or the output of the problem. Raw images/data get passed to the input layer and we receive output in the output layer.

# Activation function
The activation function calculates a weighted total and then adds bias to it to decide whether a neuron should be activated or not. The Activation Function’s goal is to introduce non-linearity into a neuron’s output. A Neural Network without an activation function is basically a linear regression model in Deep Learning, since these functions perform non-linear computations on the input of a Neural Network, enabling it to learn and do more complex tasks.

# Why do we need  the activation function?
Non-linear activation functions: Without an activation function, a Neural Network is just a linear regression model. The activation function transforms the input in a non-linear way, allowing the model to learn and as well as accomplish more complex tasks.

# 3. Training the Model

Forward Propagation: During training, data is fed forward through the network layer by layer. Each layer applies a transformation to the input based on its weights and activation function, producing an output.

Loss Calculation: Compare the model's output with the actual target (ground truth) using a loss function (e.g., categorical cross-entropy for classification, mean squared error for regression).

----------------------Probabilistic Loss Functions for classification problems------------------------------------
--Binary Cross-Entropy Loss: Binary cross-entropy is used to compute the cross-entropy between the true labels and predicted outputs. It’s used when two-class problems arise like curse and boon  classification [1 or 0].

--Categorical Crossentropy Loss: The Categorical crossentropy loss function is used to compute loss between true labels and predicted labels. It’s mainly used for multiclass classification problems.


----------------------Regression Losses-----------------------------------------------------

-- Means Squared Error (MSE):

--Mean Absolute Error:

--Cosine Similarity Loss:

--


Backward Propagation (Backpropagation): The error (difference between predicted and actual output) is propagated backward through the network. This involves calculating gradients of the loss function with respect to the weights of the network using techniques like gradient descent.

--Neurons in a Neural Network work following their weight, bias, and activation function. Changing the weights and biases of the neurons in a Neural Network based on the output error is called the Back-propagation. Because the gradients are supplied simultaneously with the error to update the weights and biases, activation functions, therefore, enable back-propagation.

Gradient Descent: Update the weights of the network to minimize the loss function, making use of optimization algorithms (e.g., stochastic gradient descent, Adam optimizer).

# 4. Validation and Fine-tuning
Validation: Evaluate the trained model on a validation dataset to ensure it generalizes well to unseen data.

Hyperparameter Tuning: Adjust hyperparameters such as learning rate, batch size, and number of epochs to improve model performance.

# 5. Testing and Deployment

Testing: Once satisfied with the model's performance, test it on a separate test dataset to assess its accuracy and generalization.

Deployment: Deploy the trained model to make predictions on new, unseen data in real-world applications.

--------------------------------------------------------------------------------------------------------------------------------------------
# Variants Of Activation Function: activation function in all hidden layer

A few rules for choosing the activation function for your output layer based on the type of prediction problem that you are solving:

A) Regression - Linear Activation Function

B) Binary Classification—Sigmoid/Logistic Activation Function
![image](https://github.com/Tiwari666/Anatomy_of_Deep_Learning/assets/153152895/be544d84-4b82-4010-83b1-7644b1d8608c)

C) Multiclass Classification—Softmax  
![image](https://github.com/Tiwari666/Anatomy_of_Deep_Learning/assets/153152895/47fb06b7-de5b-4e2c-90f6-354d748edc22)


--The Softmax function is described as a combination of multiple sigmoids. 

It calculates the relative probabilities. Similar to the sigmoid/logistic activation function, the SoftMax function returns the probability of each class. 

It is most commonly used as an activation function for the last layer of the neural network in the case of multi-class classification. 


D) Multilabel Classification—Sigmoid [0,1]

--The activation function used in hidden layers is typically chosen based on the type of neural network architecture.

E) Convolutional Neural Network (CNN): ReLU activation function
![image](https://github.com/Tiwari666/Anatomy_of_Deep_Learning/assets/153152895/1480c331-86f1-4de3-b5e2-db58ba945ef0)


![image](https://github.com/Tiwari666/Anatomy_of_Deep_Learning/assets/153152895/c58d9707-dc27-409b-9b4c-bc41462633c3)

F) Recurrent Neural Network: Tanh and/or Sigmoid activation function

# CONCLUSION:
As a rule of thumb, we can begin with using the ReLU activation function and then move over to other activation functions if ReLU doesn’t provide optimum results.

And here are a few other guidelines to help you out.

--ReLU activation function should only be used in the hidden layers.

--Sigmoid/Logistic and Tanh functions should not be used in hidden layers as they make the model more susceptible to problems during training (due to vanishing gradients).

--Swish function is used in neural networks having a depth greater than 40 layers.


----------------------------------------------------------------------------------------------------------------------------------------------
In deep learning, neural networks are composed of different types of layers, each serving specific purposes and functions in processing and transforming data. Some common types of layers in deep learning and when to use them are as follows:

1. Dense (Fully Connected) Layer:
   
This layer connects every neuron from the previous layer to every neuron in its layer, facilitating learning of complex relationships in data.

Purpose: Each neuron in a dense layer is connected to every neuron in the previous layer, making it capable of learning complex relationships between features.


When to Use: Dense layers are typically used in the early stages of a neural network for tasks like classification or regression where learning relationships between all input features and output is important.

2. Convolutional Layer: Convolutional layers apply filters to input data, making them particularly effective for processing grid-like data such as images.
 
Purpose: Convolutional layers apply convolution operations to the input data, which is particularly effective for processing grid-like data such as images.
When to Use: Convolutional layers are essential in convolutional neural networks (CNNs) used for image recognition, object detection, and other tasks involving spatial relationships in data.

3. Pooling (Pooling) Layer: Pooling layers reduce the spatial dimensions of the input data, helping to reduce computation and control overfitting in convolutional neural networks.
 
Purpose: Pooling layers reduce the spatial dimensions (width and height) of the input volume, effectively reducing computation and controlling overfitting.
When to Use: Pooling layers are often used after convolutional layers in CNNs to progressively reduce the spatial size of the representation and to generalize learned features.

4. Recurrent Layer (RNN, LSTM, GRU): Recurrent layers process sequential data by maintaining an internal state that captures information about previous inputs, essential for tasks like time series prediction or natural language processing.


Purpose: Recurrent layers process sequential data by maintaining an internal state (memory) that captures information about previous inputs.
When to Use: Recurrent layers like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are used for tasks such as speech recognition, natural language processing, and time series prediction where sequence information is crucial.

6. Batch Normalization Layer:

Purpose: Batch normalization normalizes the input layer by adjusting and scaling activations, which helps in faster training and better generalization.
When to Use: Batch normalization layers are often added after dense or convolutional layers to improve the stability and speed of training deep neural networks.

6. Dropout Layer
   
Purpose: Dropout layers randomly drop a fraction of neurons during training to prevent overfitting by reducing interdependencies among neurons.
When to Use: Dropout layers are typically applied after dense or convolutional layers in deep networks, especially when dealing with large datasets and complex architectures.

7. Activation Layer (e.g., ReLU, Sigmoid, Tanh)
    
Purpose: Activation layers introduce non-linearity into the network, allowing it to learn and approximate complex mappings between inputs and outputs.
When to Use: Activation layers are used after each layer (except the output layer) to introduce non-linearities and enable the network to learn more complex functions.

8. Output Layer
   
Purpose: The output layer produces the final output of the network based on the task (e.g., class probabilities for classification, regression values for regression).

When to Use: Output layers are designed based on the specific task and type of data being predicted or classified.

Usage Considerations:
Task Requirements: Choose layers based on the requirements of the task (e.g., spatial relationships in images, sequential dependencies in text).
Data Type: Different layers are suitable for different types of data (e.g., images, text, time series).
Model Architecture: The architecture of the neural network (e.g., CNN, RNN) determines which layers are most appropriate.



-------------------------------------------------------------------------------------------------------------------------------
# Two challenges while training deep neural networks:

# A) Vanishing Gradients
Like the sigmoid function, certain activation functions squish an ample input space into a small output space between 0 and 1. 

Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small. For shallow networks with only a few layers that use these activations, this isn’t a big problem. 

However, when more layers are used, it can cause the gradient to be too small for training to work effectively. 

# B) Exploding Gradients
Exploding gradients are problems where significant error gradients accumulate and result in very large updates to neural network model weights during training. 

An unstable network can result when there are exploding gradients, and the learning cannot be completed. 

The values of the weights can also become so large as to overflow and result in something called NaN values. 





-------------------------------------------------------------------
# References:

1. Link1:  https://www.kaggle.com/code/kanncaa1/deep-learning-tutorial-for-beginners

2. Link2: https://www.analyticsvidhya.com/blog/2021/03/basics-of-neural-network/

3. Link3: https://www.analyticsvidhya.com/blog/2021/05/guide-for-loss-function-in-tensorflow/?utm_source=reading_list&utm_medium=https://www.analyticsvidhya.com/blog/2017/07/debugging-neural-network-with-tensorboard/

4. Link4: https://www.v7labs.com/blog/neural-networks-activation-functions
   
5. Link5: https://www.analyticsvidhya.com/blog/2017/04/comparison-between-deep-learning-machine-learning/?utm_source=reading_list&utm_medium=https://www.analyticsvidhya.com/blog/2021/06/a-comprehensive-tutorial-on-deep-learning-part-2/

