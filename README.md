# -Text-Generation-LSTM
COMPANY:CODTECH IT SOLUTIONS

NAME: SHAIK MUSKAAN

INTERN ID :CT06DZ629

DOMAIN:ARTIFICIAL INTELLIGENCE

DURATION:6 WEEKS

MENTOR:NEELA SANTHOSH
Description:
# Shakespearean Text Generation with LSTM

## Project Overview

This project demonstrates how to build and train a text generation model using a Long Short-Term Memory (LSTM) neural network with TensorFlow and Keras. The model is trained on a dataset of Shakespeare's works to generate new, coherent text in a similar style.

The specific model used is a Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN) well-suited for sequence data like text. The model was trained on the complete works of William Shakespeare, allowing it to learn the unique vocabulary, syntax, and sentence structure characteristic of the playwright.

## Detailed Methodology

### 1. Data Preprocessing and Preparation

-   The raw text from `tinyshakespeare.txt` is the foundation of this project. The notebook starts by loading this text.
-   Initial cleaning involves converting all text to lowercase to ensure consistency.
-   Punctuation and special characters are handled to simplify the vocabulary for the model.
-   A `Tokenizer` from Keras is employed to perform two key functions:
    -   Building a comprehensive vocabulary of every unique word found in the text.
    -   Mapping each unique word to a specific integer ID.
-   The entire body of text is transformed into a long sequence of these integer IDs. This numerical representation is what the neural network can actually process.
-   To prepare the data for training, this long sequence is divided into numerous shorter sequences. Each sequence consists of a fixed number of words (the input) and the single, subsequent word (the target). This creates thousands of training examples.
-   The target words are then converted into a one-hot encoded format. This is a vector of zeros with a `1` at the index corresponding to the target word's ID.
-   These data preparation steps are critical for training the LSTM model effectively.

### 2. Model Architecture and Design

-   The model is built using the `Sequential` API from Keras. This allows for a straightforward, layer-by-layer construction.
-   **Embedding Layer:** The first layer is an `Embedding` layer. This layer's role is to convert the integer-based inputs into dense, low-dimensional vectors. This is a crucial step as it allows the model to learn semantic relationships between words.
-   **Bidirectional LSTM Layer:** A `Bidirectional` wrapper is used around the first `LSTM` layer. This enables the model to process the sequence both forwards and backwards, giving it a more complete understanding of the context around each word.
-   **Dropout Layer:** A `Dropout` layer is strategically placed to prevent overfitting. During training, it randomly sets a fraction of the neuron outputs to zero, forcing the model to learn more robust features.
-   **Second LSTM Layer:** Another `LSTM` layer is added to further process the patterns learned from the previous layers.
-   **Dense Output Layer:** The final layer is a `Dense` layer with a `softmax` activation function. This layer produces a probability distribution over the entire vocabulary, which represents the model's prediction for the next word.

### 3. Model Training

-   The model is compiled using the `adam` optimizer, which is an efficient algorithm for gradient descent.
-   The `loss` function is set to `categorical_crossentropy`, which is the standard choice for classification tasks where the target is one-hot encoded.
-   The model is trained over multiple `epochs`. Each epoch represents one full pass through the entire training dataset. The training process involves adjusting the model's internal parameters to minimize the loss and improve accuracy.

### 4. Text Generation

-   The final step involves using the trained model to generate new text.
-   A custom function takes a `seed_text` (a starting phrase) as input.
-   The function prepares the `seed_text` by tokenizing and padding it to the correct length.
-   The model is then used to predict the next word in the sequence.
-   A key parameter, `temperature`, is used to control the randomness of the predictions.
    -   A lower temperature makes the model's output more predictable and less creative.
    -   A higher temperature makes the output more random, which can lead to surprising but sometimes less coherent results.
-   The newly predicted word is appended to the `seed_text`, and the process is repeated to generate a longer, continuous block of text.

### 5. Code and Environment

-   All code is written in Python within a Jupyter Notebook environment.
-   The project was developed and executed using Google Colab, leveraging its free GPU resources for accelerated training.
-   The key libraries used are TensorFlow, Keras, and NumPy.

### 6. Project Timeline and Challenges

-   The project began with setting up the Colab environment and ensuring all dependencies were installed.
-   Initial challenges included file loading errors due to incorrect file names and typos in the code, such as `texts_sequences` instead of `texts_to_sequences`.
-   A significant challenge was an `InvalidArgumentError` related to the CuDNN-optimized LSTM layer. This was resolved by forcing the model to use the more robust, non-CuDNN LSTM implementation.
-   Another challenge involved a Python `ImportError` where the `Input` layer was incorrectly imported from `tensorflow.keras.models` instead of `tensorflow.keras.layers`.

### 7. Generated Text Examples

Below are a few examples of text generated by the trained model.

---

**Example 1**

**Prompt:** "to be or not to be that is the question"

**Generated:** ...
***(Insert the generated text from your Colab notebook here)**
...

---

**Example 2**

**Prompt:** "once upon a time in a faraway land there"

**Generated:** ...
***(Insert the generated text from your Colab notebook here)**
...

---

**Example 3**

**Prompt:** "the quick brown fox jumps over the"

**Generated:** ...
***(Insert the generated text from your Colab notebook here)**
...

### 8. Final Thoughts and Future Work

-   This project successfully demonstrates the fundamentals of building a text generation model.
-   The quality of the generated text is directly tied to the size and quality of the training data.
-   Future improvements could include using a much larger training corpus, exploring more advanced architectures like GPT-2, and fine-tuning hyperparameters for better performance.


Result:
<img width="1360" height="768" alt="Image" src="https://github.com/user-attachments/assets/4c25bf29-557d-487b-b8f9-4256672acce2" />
