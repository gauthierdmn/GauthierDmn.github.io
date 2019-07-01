---
layout: post
title:  "Question Generation using a Sequence-to-Sequence model with Attention"
date:   2019-06-20 00:00:00 -0700
categories: nlp
---

This blog post is the first part of a series of three posts on Question Generation using Deep Learning. It aims at intoducing the task and presenting a baseline to generate questions from a sentence or a paragraph. The second and third parts will address more advanced methods to improve on this baseline, namely the use of Pointer-Generator networks and Reinforcement Learning sub-objectives.

If you want to directly jump into the code, visit the [code repository](https://github.com/GauthierDmn/question_generation) of the project and follow the steps to train your own model! 

# Introduction

Question Generation is a machine learning task consisting in generating questions from a sentence or a paragraph automatically. With such a model, you could win the famous *Jeopardy!* TV game with no effort, where the candidates have to guess the question related to a given answer. Other applications range from the creation of learning materials for an Education purpose, generate FAQ rubrics for customer support, or even create datasets to train Question Answering models, the reverse task of Question Generation, which are very useful for plenty of other tasks such as conversational chatbots.

{% include image.html file="jeopardy.jpg" description="Jeopardy! TV Show" %}

In order to create such an algorithm, statistical NLP models were extensively studied. Most methods used a mix of Semantic Role Labeling, Part-Of-Speech tagging and a set of hand-crafted linguistic rules to transform a sentence into a question. However, this template-based question generation technique has some limitations. Among others, it needs a significant number of rules if you want to deal with complex sentences, and the questions created will always be very close to the source sentence, resulting in often too easy questions.

A good question is not only a syntactically and semantically correct sentence, often starting with an auxiliary or an interrogative word. It should also paraphrase a sentence or a paragraph, to be a minimum challenging. Another key dimension to consider is how usefulness the question is for the task.

# Towards an end-to-end approach with Deep Learning

An alternative to previous attempts to generate questions draws its inspiration from the recent progress in Machine Translation. In this respect, Recurrent Neural Networks (RNN) and in particular Long Short-Term Memory (LSTM) networks turned out to be very effective at modeling text sequences from a source language and decoding this source sentence into a target language. For Question Generation, we have the same paradigm: we have a source sentence or paragraph, and we want to generate a new sentence, the question.

An interesting difference between Machine Translation and Question Generation is that there is a big overlap between the source and the target vocabularies for the latter task. For Machine Translation, you would have a source vocabulary in French and a target vocabulary in English for example. Another difference is that for Question Generation, you often want to reuse words from the source sentence to create your question. This latter distinction makes the Question Generation task even closer to another popular NLP task: Text Summarization. To summarize a text, you also want to reuse words from the source text but not too often in order to balance between copying and paraphrasing. This is why this Question Generation implementation will not only draw ideas from MT but also Text Summarization state-of-the-art techniques.

Now let’s dig into the core of the subject and see how we can make the magic happen!

# The Sequence-to-Sequence model

First proposed by [Cho et al (2014)](https://www.aclweb.org/anthology/D14-1179) and [Sutskever et al (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) to get state-of-the-art results in Machine Translation, the **encoder-decoder** architecture turned out to be very effective for many tasks where you have sequences both as input and output, without necessary length alignment between the two sequences.

This model is divided into two parts. The first one, the **encoder**, processes an input sequence $$X = (x_1, ..., x_T)$$ and emits a context vector $$C$$, usually a linear transformation of its final state. Then the **decoder** takes as input the context vector and generates the target sequence $$Y = (y_1, ..., y_S)$$, in our case a sequence of words forming a question. This model models the conditional probability as follows:

\begin{equation}
P(Y|X) = P(y_1, ..., y_S | x_1, ..., x_T) = \prod_{t=1}^{S}P(y_t|y^{(0)}, y^{(1)}, ..., y^{(t-1)}, C)
\end{equation}

We train the encoder-decoder model on many sentence and question pairs drawn from a large dataset $$D$$ so as to maximize the log probability of a correct question generated $$Y$$ given a source sentence $$X$$, so the training objective is:

\begin{equation}
\frac{1}{|D|}\sum_{(X,Y) \in D}log P(Y|X)
\end{equation}

In practice, we reverse the problem and aim to minimize the negative log likelihood instead. Once the objective converges towards a minimum, we stop the training process and generate questions finding the most likely sequence of words according to our model, given a source sentence:

\begin{equation}
\hat{Y} = argmax_Y P(Y|X)
\end{equation}  

Finding the most likely sequence of words $$Y$$ to form a question is a very interesting topic by itself so I will dedicate a full article describing the most popular solutions.

# Encoder

The encoder I used is divided into two sequential modules: an **embedding layer** and a two-layer **bidirectional LSTM**.

The **embedding layer** takes sequences of words as input, the words in our paragraphs, together with sequences of answer features, which are indicator values encoding the position of the answer tokens in the source sentence (represented in green in the following image).

{% include image.html file="embedding.gif" description="Embedding Layer" %}

It first maps the input words and the answer features to a high dimension space, then concatenate them together. I used [GloVE](https://nlp.stanford.edu/projects/glove/) word embeddings as a pretrained representation of the word vectors. They offer a decent initialization as they encode both syntactic and semantic information, enabling the model to converge faster.

The embedding layer allowed us to convert our sequence of words into continuous high dimensional vectors, supposed to encode more information than a single integer. The next step is to process those values and extract meaningful information we will need to generate a question. 

This will be handled by the second module of the encoder, the **Long Short-Term Memory** network, a specific recurrent neural network. What makes LSTM a good candidate for this task is its ability to remember information over time, letting information flow to the following steps as new inputs are processed. It is similar to what you are doing now reading this article: as you read, your brain keeps information about previous words in the current sentence, but also previous sentences, so that you can get a full understanding of the idea presented.

The reason why LSTM networks remember information is related to their **recurrent behavior**. Each state is a function of the preceding states, and controlling gates let information pass – or not – depending on its utility for the task. At step $$t$$, we have this (simplified) relation: 

\begin{equation}
h^{(t)} = f(h^{(t-1)}, x^{(t-1)}; \theta) = f^t(h^{(0)}, x^{(0)}; \theta)
\end{equation}

where $$h^{(t)}$$ is called the hidden state of the system, $$x^{(t)}$$ the input vector and $$\theta$$ the parameters of the model. It means that at step $$t$$, the LSTM is emitting a vector encoding information about all previously computed outputs through the state $$h^{(t)}$$.

Here is a visualization of a simplified LSTM computation process. Contrary to the embedding layer presented before, each hidden state is emitted one by one due to the recurrent behavior of the LSTM. As a result, its computation can be intensive since we cannot take advantage of parallelism.

{% include image.html file="lstm.gif" description="LSTM Layer" %}

Finally, to improve the causal structure of the LSTM which only captures information from the past, we make it **bidirectional**, so that the hidden state at step $$t$$ is the concatenation of $$h^{(t)}$$ as previously described and $$h^{(-t)}$$, the hidden state at step $$t$$ of a second LSTM network processing the input sequence in reverse order.

# Decoder 

After the input sentence is encoded, we keep the last hidden state of the encoder $$h^{(T)}$$ as input for the decoder, and call it the context vector $$C$$.

We then feed this context vector into another LSTM, which is this time **unidirectional** since we cannot process future words before they are generated. The decoding LSTM layers are initialized with the weights of the encoding LSTM, and at each step $$t$$, it emits a hidden state $$s^{(t)}$$, exactly as described in the encoder section. Those hidden state vectors will be used to generate question words after we feed them into a dense layer, also called feed-forward neural network, which output size is the size of our vocabulary. Taking the softmax over the resulting vector, we get a normalized distribution over the output vocabulary $$V$$.

Then, a straightforward strategy to select our predicted question word at step $$t$$ is to sample the word with highest probability:

\begin{equation}
\hat{y}^{(t)} = argmax_{y \in V} softmax(g(s^{(t)}))
\end{equation}

with $$g$$ the dense layer.

What we feed the LSTM at step $$1$$ is the context vector $$C$$. At step $$t > 1$$, we take the hidden vector $$s^{(t-1)}$$ as new input for the LSTM, and keep on generating question words until the model emits an end-of-sentence token, $$<eos>$$. See the following illustration for more details:

{% include image.html file="decoder.gif" description="Decoder" %}

# Input Feeding and Teacher Forcing

A first drawback of the decoding strategy described above is that at each time step $$t$$, we don’t take into account previously generated words. If the first word we generate is the word "What", it could be very useful for the model to know it at step $$2$$ in order to avoid generating another interrogative word. As a result, it is common not only to feed the LSTM with the previous LSTM hidden state $$s^{(t-1)}$$, but also the previously generated word $$\hat{y}^{(t-1)}$$, mapped to its embedding representation. 

A second drawback we can have is the risk to propagate errors we made at previous time steps to future time steps. For example, if the first word generated is wrong, like "Cats" instead of "Who", then using "Cats" to generate future words will certainly create a sentence far from the question we expect.
To avoid this error propagation, a good practice is to **give the decoder at step $$t$$ the true word $$y^{(t-1)}$$** it should have generated at step $$t-1$$, instead of the predicted word $$\hat{y}^{(t-1)}$$: it is called **Teacher Forcing**. However, this strategy is only possible when training the model, since we have access to the true output question. To evaluate the model on unseen data, using the previously predicted word as input for the decoder is an option, but more advanced methods work better in practice, which I will let you read about in a separate blog post.

We end up with the following equation for the decoder:

\begin{equation}
\forall t \in [1, ..., S], y^{(t)} = argmax_{y^{(t)} \in V} P(y^{(t)}|{y^{(1)}, y^{(2)}, ..., y^{(t-1)}}, C)
\end{equation}

# Attention Mechanism

We are almost done with the implementation of this baseline encoder-decoder or Sequence-to-Sequence model for question generation. Now even with Input Feeding and Teacher Forcing, our decoder still suffers from a major limitation. Remember, we said that the input sentence was encoded or compressed into a fixed-size vector $$C$$ which "summarizes" the information of the input sentence. In practice, a loss of information occurs when dealing with long sequences. It is pretty unlikely that the context vector still remembers the dependencies of the beginning of the sentence, resulting in average decoding performances. The Machine Translation community rapidly introduced a trick to alleviate this issue, reversing the input sentence words when passing it through the encoder so that it shortens the path between the decoder and relevant information in the encoded sentence, as in [Sutskever et al (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). Similarly, feeding the input sentence twice was also considered to allow the model to better memorize information, see [Zaremba et al (2015)](https://arxiv.org/pdf/1410.4615.pdf). 

However, a very popular and effective solution was presented in [Luong et al (2015)](https://arxiv.org/pdf/1508.04025.pdf) and [Bahdanau et al (2016)](https://arxiv.org/pdf/1409.0473.pdf) as **Attention**. The central idea behind Attention is to use all the intermediate states of the encoder instead of only the last one $$h^{(T)}$$ and allow the decoder at each decoding step to "attend" to the relevant part of the input sentence it needs to generate a new word. In practice, our context vector $$C$$ is replaced with a weighted average of the hidden states $$h^{(t)}, t \in {1, ..., T}$$ of our encoder, where the weights $$\alpha^{(i)}$$, called **alignment weights**, are trainable weights and are computed at each decoding step $$i$$:

\begin{equation}
\forall i \in [1, ..., S],  C^{(i)} = \sum_{t=1}^{T}\alpha_{i,t}*h^{(t)}
\end{equation}
and 
\begin{equation}
\alpha_{i,t} = \frac{e_{i,t}}{\sum_{k=1}^{T} e_{i,k}}
\end{equation}
where 
\begin{equation}
e_{i,t} = a(s_i, h_t) 
\end{equation}
is an **alignment score** which scores how well the inputs around position $$t$$ and the output at position $$i$$ match.

Various functions $$a$$ can be considered to compute thos alignment scores, as long as it reflects a **similarity measure** between the hidden states $$s_i$$ and $$h_t$$ (see previously cited papers for more details).

# Results and Discussion

Alright, the time to play *Jeopardy!* with our model has finally arrived! But first, let's do a qualitative analysis of this baseline model after it was trained on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [NewsQA](https://datasets.maluuba.com/NewsQA) datasets, looking at some questions it generated. For a quantitative analysis of the results, I let you go through the *Results* section in the [code repository](https://github.com/GauthierDmn/question_generation).

Here are some pairs of Input Sentences and Questions generated by our model. The Answer to the question is represented in bold characters:

> **S**: Napoleon Bonaparte was a **French** statesman and military leader of Italian descent who rose to prominence during the French Revolution and led several successful campaigns during the French Revolutionary Wars.
<br/>**Q**: What nationality was Napoleon Bonaparte ?

> **S**: Napoleon dominated European and global affairs **for more than a decade** while leading France against a series of coalitions in the Napoleonic Wars.
<br/>**Q**: How long did Napoleon dominate European and global affairs ?

> **S**: He won most of these wars and the vast majority of his battles, building a large empire that ruled over much of continental Europe before its final collapse **in 1815**.
<br/>**Q**: When did Napoleon's empire end ?

> **S**: Nadal supports football clubs **Real Madrid and RCD Mallorca**.
<br/>**Q**: What are the names of the two main basketball clubs ?

> **S**: **Toni Nadal** is a former professional tennis player.
<br/>**Q**: Who is $$<unk>$$ Nadal ?

Well, it is not too bad. Right? The first three questions are grammatically correct and ask for the right information, so they are valid. However, the last two questions have some discrepancies that are worth discussing because they underline the flaws of our baseline model.

Question 4 is both grammatically and semantically correct, but fails to retrieve the right sport the input sentence is referring. It is funny because it seems that the difficult part was handled right, and the algorithm is making a stupid mistake making the question invalid at the end. This observation is in fact related to the design of the encoder-decoder approach we used. When the Text Summarization community started to use this method to summarize a text, they came up with this same observation that the model was performing pretty well but from time to time was **failing to generate factual information founded in the input text**. This is because the model is **abstractive** i.e. it does not extract parts of the input text to generate a summary or a question, which is something we definitely need to retrieve facts such as the right sport in our case.

Question 5 has a $$<unk>$$ token inside, which stands for "unknown". When training our model, we limited our vocabulary to the top 40,000 most frequent words for computational reasons, and replaced all unfrequent words by the $$<unk>$$ token. In this case, we guess that the word hiding behind the unknown token is "Toni", but since "Toni" is an **out-of-vocabulary word**, the model cannot retrieve this token when generating a question. A solution could be to increase the size of our vocabulary, and it definitely helps, but we will still have $$<unk>$$ tokens at the end because of the sparse property of the English language. A better solution would be to copy words from the input sentence so that it drastically reduces the out-of-vocabulary words without increasing the size of the training vocabulary.

Both problems of examples 4 and 5 seem to have the same solution, which is to implement a mechanism to **copy words from the source sentence** so as to extract facts and out-of-vocabulary words. This is what Abigail See et al. is doing in [*Get To the Point: Summarization with Pointer-Generator Network*](https://arxiv.org/pdf/1704.04368.pdf) (2017) where he introduces a mechanism to copy words from the input text with certain probability using the alignment scores of the Attention mechanism we talked about before. This method will be presented as a standalone blog post as an improvement to our baseline model, and we will also explore how we can use Reinforcement Learning sub-objectives to make this Question Generation model even stronger!
