## Motivation
The goal of this work was to examine whether LLM-based toxic content classifiers genuinely surpass traditional neural network classifiers in terms of accuracy and by how much.

### What are Guardrails?
Guardrails are filtering mechanisms in LLM-based applications that safeguard against generating toxic, harmful, or otherwise undesired content. They act as essential tools to mitigate risks associated with LLM use, such as ethical concerns, data biases, privacy issues, and overall robustness.

As LLMs become more widespread, the potential for misuse has grown, with risks ranging from spreading misinformation to facilitating criminal activities [Goldstein et al., 2023](https://arxiv.org/pdf/2301.04246).

In simple terms, a guardrail is an algorithm that reviews the inputs and outputs of LLMs and determines whether they meet safety standards.

For example, if a user’s input relates to child exploitation, a guardrail could either prevent the input from being processed by the LLM or adapt the output to ensure it remains harmless. In this way, guardrails intercept potentially harmful queries and help prevent models from responding inappropriately.

Depending on the application, guardrails can be customized to block various types of content, including offensive language, hate speech, hallucinations, or areas of high uncertainty. They also help ensure compliance with ethical guidelines and specific policies, such as fairness, privacy, or copyright protections [Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1).


## Methodology
I evaluated the performance of three binary classifiers:
1. Llama3 8B with in-context-learning (ICL)
2. Two Layer Neural Network - a feed-forward neural network trained on [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) training dataset.

### Dataset
For this experiment, I used the [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) dataset, which includes toxic comments and hate speech from Wikipedia, with each comment labeled by humans as either 1 ("toxic") or 0 ("safe").

The training and validation datasets were used to train a neural network classifier.

For evaluating the performance of both classifiers, I used the Test dataset.

### Classifiers

For the experiment, I set up 2 different classifiers: 

#### Llama3 8B with ICL
I used Meta's Llama3 model to classify toxic content using Test Dataset. Using in-context-learning the LLM is tasked to classify `user comment` as toxic or safe by returning 0 for "safe" or 1 for "toxic" content. If LLM can not return the answer or does not know, it should return 2. I used similar prompt structure and toxic content categories as per [Inan et al., 2023](https://arxiv.org/pdf/2312.06674) paper. 



#### Feed-forward Neural Network
I train a simple 2 layer neural network with the following architecture:
- Input layer: 1024
- Hidden layer 1: 100
- Hidden layer 2 (with dropout): 25
- Output: 2

For each sample, I generated embeddings using the [mGTE](https://arxiv.org/pdf/2407.19669) sentence embedding model developed by Alibaba Group, which is accessible [here](https://arxiv.org/pdf/2407.19669).

I use Cross Entropy Loss and Stochastic Gradient Descent optimizaton. 

Below figure shows the training and validation loss for each epoch during training.

{{< figure src="/posts/guardrail/images/nn_1024_100_25_loss.png" attr="Training and validation loss during Neural Network training" align=center target="_blank" >}}

After the training, the performance of the neural network was evaluated on Test Dataset.

## Results

Here is the most interesting part.

{{< figure src="/posts/guardrail/images/shock.gif" align=center target="_blank" >}}


**Llama3 7B with ICL**

The LLM failed to classify 146 samples. I updated their labels to 1 assuming that we want a model with high recall score. To classify 3,000 test samples it took me more than an hour. 

Below is the summary of the model perfomance:
- Accuracy Score:  0.8
- Precision:  0.82
- Recall:  0.78
- F1 Score:  0.8

**Feed-forward Neural Network**

Neural network classified all 3,000 test samples and it took a few minutes.

Below is the summary of the model perfomance:
- Accuracy Score:  0.9
- Precision:  0.86
- Recall:  0.96
- F1 Score:  0.91


Below, are confusion matrices for both classifiers. 

{{< figure src="/posts/guardrail/images/cm.png" attr="Confusion matrices" align=center target="_blank" >}}

In this scenario, the neural network model significantly outperformed the LLM-based classifier across all evaluation metrics. 

For a toxic content classification system, achieving a high recall rate is essential to ensure maximum detection of harmful content. Consequently, the neural network—with a recall rate of 0.96—would be preferable to the LLM-based classifier, which achieved a recall rate of only 0.78.

Additionally, in a production environment, the neural network would offer faster processing speeds, taking less than a second per request, whereas the LLM requires approximately 2-3 seconds for each classification.

However, it is worth noting that the neural network may not generalize as effectively to novel, unseen content, where the LLM could potentially offer an advantage.


## Limitations and next steps
To implement the Llama3.1 8B model, I utilized the Ollama framework, a streamlined tool for running LLMs on local machines. Due to quantization, model performance may have been significantly affected. The next phase involves conducting the same experiment with the full Llama3.1 model on AWS Bedrock. Additionally, I plan to test the LlamaGuard model under similar conditions.

For further refinement, the neural network could be trained on a broader range of toxic content types, and alternative architectures and embedding models could be explored.

## 🔗 Blog post
You can read the full blog post [here](https://drjulija.github.io/posts/guardrail/).

