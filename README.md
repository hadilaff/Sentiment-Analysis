# Sentiment-Analysis of MDB Movie Reviews
The main goal of this project is to develop model that can accurately classify the sentiment of movie reviews and implement a transformer-based model from scratch

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

### Architecture and Hyperparameters
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


