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

Output Layer: The final layer produces the output of the network, which could be class probabilities for classification tasks or numerical values for regression tasks.

# 3. Training the Model

Forward Propagation: During training, data is fed forward through the network layer by layer. Each layer applies a transformation to the input based on its weights and activation function, producing an output.

Loss Calculation: Compare the model's output with the actual target (ground truth) using a loss function (e.g., categorical cross-entropy for classification, mean squared error for regression).

Backward Propagation (Backpropagation): The error (difference between predicted and actual output) is propagated backward through the network. This involves calculating gradients of the loss function with respect to the weights of the network using techniques like gradient descent.

Gradient Descent: Update the weights of the network to minimize the loss function, making use of optimization algorithms (e.g., stochastic gradient descent, Adam optimizer).

# 4. Validation and Fine-tuning
Validation: Evaluate the trained model on a validation dataset to ensure it generalizes well to unseen data.

Hyperparameter Tuning: Adjust hyperparameters such as learning rate, batch size, and number of epochs to improve model performance.

# 5. Testing and Deployment

Testing: Once satisfied with the model's performance, test it on a separate test dataset to assess its accuracy and generalization.

Deployment: Deploy the trained model to make predictions on new, unseen data in real-world applications.

--------------------------------------------------------------------------------------------------------------------------------------------

# References:

1. Link1:  https://www.kaggle.com/code/kanncaa1/deep-learning-tutorial-for-beginners
