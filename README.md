# 📚 Reading: Deep Neural Networks & Supervised Learning Tasks

<h2>Introduction</h2>
<p>Deep neural networks are widely used in supervised machine learning tasks. In supervised learning, the goal is to train a model to learn the mapping between input data and corresponding target labels. Deep neural networks excel in capturing complex patterns and relationships in the data, making them a powerful tool for such tasks. Deep neural networks are called&nbsp;<em>deep</em> since they have a number of layers, i.e.&nbsp;<em>depth</em>.</p>
<p>&nbsp;</p>
<h3>Objectives</h3>
<ul>
<li>Determine how deep neural networks are used in supervised machine learning tasks</li>
<li>Identify the applications of supervised learning by neural networks</li>
</ul>
<h2>Neural Networks</h2>
<h3>Overview of Supervised Learning Tasks</h3>
<p>Here's a general overview of how deep neural networks are used in supervised machine learning:</p>
<ol>
<li>
<p><strong>Data Preparation</strong>: The first step is to prepare the labeled training data. This involves splitting the data into input features (often represented as a numerical vector) and their corresponding target labels. The data is typically divided into training and validation sets, where the training set is used to train the model, and the validation set is used to assess its performance during training.</p>
</li>
<li>
<p><strong>Network Architecture</strong>: Next, the architecture of the deep neural network is defined. This includes specifying the number and type of layers, the number of neurons in each layer, and the activation functions used. Common types of layers in deep neural networks include input layers, hidden layers, and output layers.</p>
</li>
<li>
<p><strong>Forward Propagation</strong>: During the training process, the input data is fed into the neural network through the input layer. The data passes through the hidden layers, where each neuron performs a weighted sum of its inputs, applies an activation function, and passes the result to the next layer. This process continues until the output layer, which produces the predicted output of the network.</p>
</li>
<li>
<p><strong>Loss Calculation</strong>: After the forward propagation, the predicted output of the network is compared to the true target labels. A loss function is used to measure the dissimilarity between the predicted and actual values. Common loss functions include mean squared error, categorical cross-entropy, and binary cross-entropy.</p>
</li>
<li>
<p><strong>Backpropagation</strong>: The next step is to update the network's parameters (weights and biases) based on the calculated loss. Backpropagation is used to propagate the error backward through the network, calculating the gradient of the loss function with respect to each parameter. This gradient is then used to update the parameters using optimization algorithms like stochastic gradient descent (SGD) or Adam.</p>
</li>
<li>
<p><strong>Iteration and Optimization</strong>: Steps 3 to 5 are repeated iteratively for a defined number of epochs or until a convergence criterion is met. The model continues to refine its parameters, gradually improving its ability to predict the target labels accurately.</p>
</li>
<li>
<p><strong>Model Evaluation</strong>: Once the training is complete, the model is evaluated using the validation set to assess its generalization performance. Various metrics, such as accuracy, precision, recall, or F1 score, are used to evaluate the model's performance on the validation set.</p>
</li>
<li>
<p><strong>Prediction</strong>: Finally, the trained model can be used to make predictions on new, unseen data by feeding it through the network's input layer and obtaining the output from the output layer.</p>
</li>
</ol>
<h3>Summary</h3>
<p><span>Deep neural networks have demonstrated great success in various supervised learning tasks, including image classification, object detection, speech recognition, natural language processing, and more. The ability of deep neural networks to automatically learn hierarchical representations from the data allows them to capture intricate patterns and improve predictive performance.</span></p>