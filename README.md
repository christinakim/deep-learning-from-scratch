# Deep Learning From Scratch
How to get started
- [ML resources](#ml-resources)
- [DL resources](#deep-learning-resources)
- [NLP resources](#nlp-resources)
- [Brushing up on math](#brushing-up-on-math)

Paper reading
- [How to manage papers](#how-to-manage-papers)
- [How to figure out what to read](#how-to-figure-out-what-to-read-check-out-these-sources)
- [How to read a paper](#how-to-read-a-paper)

[Reimplementations](#reimplementations)
- [Dropout](dropout.py)
- [LSTM](LSTM.py)
- [MADE](MADE.py)
- [Multilayer Perceptron (MLP)](mlp.py)
- [Self Attention](self_attention.py)
- [Transformer](transformer.py)
- [Variational Autoencoder (VAE)](VAE.py)

## How to get started
I spent some time learning classical ML first. 

### ML resources

- started off with a homemade ML in 10 weeks course:

    tl;dr, here's the course, using content primarily from Hands-On Machine Learning with Scikit-Learn and TensorFlow and Andrew Ng's Coursera course on ML:
    - Chapter 2 End-to-End Machine Learning Project
    - Chapter 3 Classification (precision/recall, multiclass)
    - Text feature extraction (from sklearn docs)
    - Chapter 4 Training Models (linear/logistic regression, regularization)
    - Advice for Applying Machine Learning
    - Chapter 5 SVMs (plus kernels)
    - Chapter 6 Decision Trees (basics)
    - Chapter 7 Ensemble Learning and Random Forests (xgboost, RandomForest)
    -  Chapter 8 Dimensionality Reduction (PCA, t-SNE, LDA)
    - Machine Learning System Design
    (Google) Best Practices for ML Engineering
    
    A group of friends and I worked through this content at a cadence of one meeting every other Wednesday starting late June 2018 wrapping up at the end of 2018. 

### Deep learning resources

- Neural Networks and Deep Learning by Michael Nielsen [http://neuralnetworksanddeeplearning.com/index.html](http://neuralnetworksanddeeplearning.com/index.html)
- fast.ai
    - Practical Deep Learning for Coders [https://course.fast.ai/videos/?lesson=1](https://course.fast.ai/videos/?lesson=1)
    - Part 2: Deep Learning from the Foundations [https://course.fast.ai/videos/?lesson=8](https://course.fast.ai/videos/?lesson=8)
- distill is a good resource for topics. ex:
    - [https://distill.pub/2017/momentum/](https://distill.pub/2017/momentum/)
    - [https://distill.pub/2016/augmented-rnns/](https://distill.pub/2016/augmented-rnns/)

### NLP resources

- Kyunghyun Cho's lecture notes on "Natural Language Processing with Representation Learning": https://github.com/nyu-dl/NLP_DL_Lecture_Note/blob/master/lecture_note.pdf
- Jacob Eisenstein's textbook on "Natural Language Processing" (https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)

### Brushing up on math 
It's easy to get intimated by the math in papers. I found that taking the time to relearn linear algebra and some calculus has had compounding returns!

- [Matrix Calculus by Terence Parr and Jeremy Howard](https://explained.ai/matrix-calculus/)
- backprop chapter in [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Matrix Algebra - Linear Algebra for Deep Learning](https://www.quantstart.com/articles/matrix-algebra-linear-algebra-for-deep-learning-part-2)
- [3blue1brown for practical and visual linear algebra](https://www.3blue1brown.com/essence-of-linear-algebra-page)
- [for theoretical linear algebra: Finite Dimensional Vector Spaces](https://www.amazon.com/Finite-Dimensional-Vector-Spaces-Paul-Halmos/dp/178139573X)

## Paper reading

Once you've understood common concepts, the best way to keep up to date with research and continue learning beyond courses is by reading and reimplementing papers.
#### How to manage papers
- track papers through [Zotero](https://www.zotero.org/) or [Mendeley](https://www.mendeley.com/). I started off using Zotero but switched Mendeley to share folders/papers in groups I was in
    
#### How to figure out what to read? Check out these sources:
 - twitter - follow 20+ practitioners/researchers you admire on twitter to find interesting papers
 - ML subreddit
 - AI/DL fb groups
 - arXiv - there's 10-20 new papers on arXiv every day for AI/computational linguistics so you could just browse arXiv every day for the latest papers in the topics you're most interested in
 - AI blogs
     - [Import AI](https://jack-clark.net/)
     - [NLP Newsletter](https://github.com/dair-ai/nlp_newsletter) 
 
#### How to read a paper:
 - your objective is to figure out quickly which papers NOT to read
- spend time in the conclusions 
- try to answer the question `what is novel`?
- create a reading group! Even just one other person can already save you 50% of the time.

## Reimplementations 
Purpose: to break down deep learning concepts and architecture into code using PyTorch! It's easy import from libraries and never really understand what something is doing. This repo is to reimplement common architectures and atomic concepts in deep learning in simpler code. 

- [Dropout](dropout.py)
- [LSTM](LSTM.py)
- [MADE](MADE.py)
- [Multilayer Perceptron (MLP)](mlp.py)
- [Self Attention](self_attention.py)
- [Transformer](transformer.py)
- [Variational Autoencoder (VAE)](VAE.py)
