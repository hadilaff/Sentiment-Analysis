# Sentiment-Analysis of MDB Movie Reviews
The main goal of this project is to develop model that can accurately classify the sentiment of movie reviews and implement a transformer-based model from scratch
You can find the model in this link: https://drive.google.com/file/d/1-0TZpJwLLNJdElzlUw-ct7S3x6I3qDMa/view?usp=sharing

**Dataset**
the IMDb dataset consist of 50,000 movie reviews labeled as positive or negative.

**Data Preprocessing**
The following preprocessing steps are performed:
1. Remove the HTML tags.
2. Remove the stop words
3. Remove the special  characters
4. Remove the multiple spaces: Extra.

**Tokenization**
To prepare the textual data for model training, the sentences are split into individual words or tokens through a process using a tokenizer. This step is crucial for generating input sequences that can be fed into the models.
**Transformer Block**
Includes multi-head attention, feed-forward layers, and layer normalization.
**Sentiment Classifier**
A Transformer-based model combining embeddings, positional encoding, and multiple Transformer blocks for sentiment classification.

## Architecture and Hyperparameters
I developed various models for sentiment analysis of movie reviews, but this architecture has produced the best model and results.
1. Embedding Layer: Converts token IDs into dense vectors of dimension `d_model` (128). This dense representation captures semantic information about the tokens.

2. Positional Encoding: Essential for sequence tasks to provide order information to the model, which is critical in understanding context.

3. Transformer Blocks:
Number of Blocks: 4
Each Block Contains:
    - Multi-Head Attention: Allows the model to focus on different parts of the input sequence simultaneously, capturing various aspects of dependencies.
    - LayerNorm:Normalizes the inputs to each layer, stabilizing training and improving convergence.
    - Dropout:Regularizes the model by randomly dropping units during training, reducing overfitting.
    - Feed-Forward Network:Applies transformations to the attention outputs, allowing the model to learn complex representations.
    - Final Linear Layer: Maps the final representation to the number of output classes (2 in this case), producing the final logits for classification.

**Hyperparameters** 
The model is trained using the following hyperparameters:

- **`d_model = 128`:** A standard dimension that provides a good trade-off between representational power and computational efficiency.
- **`num_heads = 8`:** Provides sufficient parallel attention heads to capture different features of the input data effectively.
- **`num_layers = 4`:** Deep enough to capture complex relationships in the data while keeping the model manageable in terms of computational resources.
- **`dropout = 0.1`:** Helps in regularizing the model and preventing overfitting, especially useful given the depth of the transformer model.
- **`learning rate = 0.001`:** A common starting point for Adam optimizer; might require tuning based on specific training dynamics.
- **`betas = (0.9, 0.999)`:** Standard values for the Adam optimizer, ensuring stable training.
- **`eps = 1e-08`:** Prevents division by zero in the Adam optimizer.

The combination of these choices aims to balance model complexity, training efficiency, and performance on sentiment classification tasks.

![archhhh](https://github.com/user-attachments/assets/1cd30713-9965-488e-97f6-5bd2ade93a3b)

## Model Evaluation and result

I trained the transformer-based model for 3 epochs, achieving an accuracy of 86%. When I extended the training to 5 epochs, I noticed the model started to overfit. Despite trying to adjust the hyperparameters and layers, the issue persisted. Therefore, I believe the solutions are to perform data augmentation or to use a pretrained model such as BERT.

## Model deployment
Once the model is trained and saved, and all preprocessing steps are complete, I start the FastAPI server to serve predictions.This allows me to create an API that can handle real-time requests and provide sentiment analysis for texts. 
You can then build a front-end interface to visualize the results. Using ngrok, I expose the local FastAPI server to the public, creating a shareable URL to interact with the application from anywhere.
1. Create an `ngrok` Account and Get a Token
2. Set the Authentication Token:  ngrok.set_auth_token("token")
3. Choose a Tunnel on Port for example : 8000 

then anyone with the link will be able to interact with your FastAPI application

![Screenshot from 2024-09-02 21-09-18](https://github.com/user-attachments/assets/4472c004-14ad-438c-9c86-f74210210e39)

To expose your local FastAPI server to the internet using `ngrok`, you'll need to follow these steps:![Screenshot from 2024-09-02 21-09-26](https://github.com/user-attachments/assets/f4862a3d-d1fc-4c3b-b93b-c783ce28f1b9)


## Requirements

- Python 3.x
- PyTorch
- transformers
- BeautifulSoup4
- NLTK
- NumPy
- fastapi
- pyngrok
