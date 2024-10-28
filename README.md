## Motivation
The goal of this work was to examine whether LLM-based toxic content classifiers genuinely surpass traditional neural network classifiers in terms of accuracy.

### What are Guardrails?
Guardrails are filtering mechanisms in LLM-based applications that safeguard against generating toxic, harmful, or otherwise undesired content. They act as essential tools to mitigate risks associated with LLM use, such as ethical concerns, data biases, privacy issues, and overall robustness.

As LLMs become more widespread, the potential for misuse has grown, with risks ranging from spreading misinformation to facilitating criminal activities [Goldstein et al., 2023](https://arxiv.org/pdf/2301.04246).

In simple terms, a guardrail is an algorithm that reviews the inputs and outputs of LLMs and determines whether they meet safety standards.

For example, if a user’s input contains hate speech, a guardrail could either prevent the input from being processed by the LLM or adapt the output to ensure it remains harmless. In this way, Guardrails intercept potentially harmful queries and help prevent models from responding inappropriately.

Depending on the application, Guardrails can be customized to block various types of content, including offensive language, hate speech, hallucinations, or areas of high uncertainty. They also help ensure compliance with ethical guidelines and specific policies, such as fairness, privacy, or copyright protections [Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1).


## Methodology
I evaluated the performance of two binary classifiers:
1. Llama3 8B with in-context-learning (ICL)
2. Two Layer Neural Network - a feed-forward neural network that I trained on the [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) training dataset.

### Dataset
For this experiment, I used the [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) dataset, which includes toxic comments and hate speech from Wikipedia, with each comment labeled by humans as either 1 ("toxic") or 0 ("safe").

To support training, validation, and testing, I divided the dataset into three balanced subsets that contains equal number of samples for each category 0 or 1:
1. Training Dataset - 25,868 samples
2. Validation Dataset - 6,582 samples
3. Test Dataset - 3,000 samples

The Training and Validation datasets were used to train a neural network classifier.

For evaluating the performance of the classifiers, I used the Test dataset.


### Classifiers

For the experiment, I set up two different classifiers: 

#### Llama3 8B with ICL
I used Meta's Llama3 model to classify toxic content using the Test Dataset. Using in-context-learning the LLM is tasked to classify `user comment` as toxic or safe by returning 0 for "safe" or 1 for "toxic" content. If the LLM can not return the answer or does not know, it should return 2. I used a similar prompt structure and toxic content categories as per [Inan et al., 2023](https://arxiv.org/pdf/2312.06674) paper.


#### Feed-forward Neural Network
I train a simple 2 layer neural network with the following architecture:
- Input layer: 1024
- Hidden layer 1: 100
- Hidden layer 2 (with dropout): 25
- Output: 2

For each sample, I generated embeddings using the [mGTE](https://arxiv.org/pdf/2407.19669) sentence embedding model developed by Alibaba Group, which is accessible [here](https://arxiv.org/pdf/2407.19669).

I use Cross Entropy Loss and Stochastic Gradient Descent optimizaton. 

After the training, the performance of the neural network was evaluated on the Test Dataset.

## Results

Here is the most interesting part.

The neural network model significantly outperformed the LLM-based classifier across all evaluation metrics. The LLM failed to classify 146 samples. I updated their labels to 1 assuming that we want a model with high recall score. 

See the results in the table below:

| Metric     | **Llama3 7B with ICL**   | **Neural Network**   |
| :--------- | :----------------------: | :------------------: |
| **Accuracy**   | 0.8                      | 0.9                  |
| **Precision**  | 0.82                     | 0.86                 |
| **Recall**     | 0.78                     | 0.96                 |
| **F1 Score**   | 0.8                      | 0.91                 | 

For a toxic content classification system, achieving a high recall rate is essential to ensure maximum detection of harmful content. High recall minimizes the risk of leaving harmful content undetected. For toxic content moderation, undetected content is a higher risk to user safety than occasional false positives (content wrongly flagged as toxic but isn’t harmful). Consequently, the neural network with a recall rate of 0.96 would be preferable to the LLM-based classifier, which achieved a recall rate of only 0.78.

## 🔗 Blog post
You can read the full blog post [here](https://drjulija.github.io/posts/guardrail/).

