---
layout: post
title:  "Neural Question Generation: Seq2Seq model with Attention"
date:   2019-06-20 00:00:00 -0700
categories: nlp
---

# Introduction

Neural Question Generation is a machine learning task consisting in creating questions from a sentence or a paragraph. With such a model, you could become very popular winning consistently the famous Jeopardy! TV game, where the candidates have to guess the question related to a given answer. Generating questions can also be very useful in Education in order to create learning materials on specific topics, or even generate FAQ rubrics for websites or chatbots.

In order to create such an algorithm, statistical NLP models were first studied. In [], ... used Semantic Role Labeling, Part-Of-Speech tagging and a set of linguistic rules like reversing the auxiliary and the verb to transform a sentence into a question. However, this template-based question generation technique has some limitations. Among others, it needs a significant number of rules if you want to deal with complex sentences, and the questions created will always be very close to the source sentence, resulting in often too easy questions.

A good question is not only a syntactically and semantically correct sentence, often starting with an auxiliary or an interrogative word. It should also paraphrase a sentence or a paragraph, to be a minimum challenging. Finally, another dimension to consider is the usefulness of a question. Yes, questions are not all created equal.

An alternative to previous attempts to generate question draws its inspiration from the recent progress in Machine Translation. In this respect, Artificial Neural Networks and in particular Long Short-Term Memory networks turned out to be very effective at modeling text sequences from a source language and decoding this source sentence into a target language. For Question Generation, we have the same paradigm: we have a source sentence or paragraph, and we want to generate a new sentence, the question.

An interesting difference between Machine Translation and Question Generation is that there is a big overlap between the source and the target vocabularies for the latter task. For MT, you would have a source vocabulary in French and a target vocabulary in English for example. Another difference is that for Question Generation, you often want to reuse words from the source sentence/paragraph to create your question. This latter distinction makes the Question Generation task even closer to another popular NLP task: Text Summarization. To summarize a text, you also want to reuse words from the source text but not too often in order to balance between copying and paraphrasing. This is why this Question Generation implementation will not only draw ideas from MT but also Text Summarization state-of-the-art techniques.

I decided to split the subject into 3 parts, starting with the presentation of a baseline yet very powerful model, followed by two more advanced parts where we will improve this baseline, correcting some of its limitations.

Now let’s dig into the core of the subject and see how we can make the magic happen!


{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

