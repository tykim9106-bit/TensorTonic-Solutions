# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Ty Kim's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/tykim9106.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| K-Means Centroid Update | Update K-means centroids as cluster means while applying the required behavior for empty clusters. | https://www.tensortonic.com/problems/k-means-centroid-update |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| One-Hot Encoding (Multi-class) | Convert multiclass integer labels into a NumPy one-hot matrix with one active column per sample. | https://www.tensortonic.com/problems/one-hot-encoding |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Detect Train-Serving Skew | Detect train-serving skew by comparing offline and online feature values under configured tolerances. | https://www.tensortonic.com/problems/train-serving-skew |
| Word Count Dictionary | Count token occurrences in text and return a dictionary mapping each distinct word to its frequency. | https://www.tensortonic.com/problems/word-count-dict |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| Class Token [CLS] | Prepend a learned classification token to each Vision Transformer patch sequence for image-level prediction. | https://www.tensortonic.com/research/vit/vit-class-token |
| ViT Encoder Block | Build a Vision Transformer encoder block with layer normalization, multi-head attention, MLP, and residual connections. | https://www.tensortonic.com/research/vit/vit-encoder-block |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |
| Position Embedding | Add learned positional embeddings to Vision Transformer patch-token sequences while preserving batch dimensions. | https://www.tensortonic.com/research/vit/vit-position-embedding |
| Negative Sampling Distribution | Build the Word2Vec negative-sampling distribution from unigram counts raised to the three-quarter power. | https://www.tensortonic.com/research/word2vec/word2vec-noise-dist |
| Skip-gram Pair Generation | Generate Word2Vec skip-gram training pairs by pairing each center token with words inside its context window. | https://www.tensortonic.com/research/word2vec/word2vec-skipgram-pairs |
| Frequent-Word Subsampling | Implement Word2Vec frequent-word subsampling by computing token retention probabilities from corpus frequencies. | https://www.tensortonic.com/research/word2vec/word2vec-subsampling |
| Estimate a Scalar Derivative | Estimate a scalar polynomial derivative with a forward finite difference using coefficients ordered by ascending power. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-finite-difference-derivative |
| Check a Product-Chain Gradient | Compare analytic gradients with forward-difference estimates for a two-operation scalar graph. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-gradient-check-product-chain |
| Apply a Gradient-Descent Step | Apply one NumPy gradient-descent update and compute the first-order predicted objective change without mutating inputs. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-gradient-descent-step |
| Measure Scalar Expression Partials | Estimate the three partial derivatives of a scalar expression by perturbing one input at a time. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-scalar-expression-partials |
| Build a Scalar Expression Graph | Build a scalar expression graph in forward order from leaf records and addition or multiplication operations. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-build-expression-graph |
| Create an Addition Value Node | Create a scalar addition node that stores its forward value, operation type, and ordered parent identifiers. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-value-addition-node |
| Create a Multiplication Value Node | Create a scalar multiplication node that stores its forward value, operation type, and ordered parent identifiers. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-value-multiplication-node |
| Apply Ranked BPE Merges | Apply learned byte-pair merge rules to UTF-8 byte IDs in their supplied priority order, then reconstruct text through the supplied vocabulary. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l01-apply-bpe-merge-ranks |
| Train a Deterministic BPE Vocabulary | Choose the highest count with a lexicographic byte-string tie break, assign the next token ID, and replace non-overlapping matches from left to right. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l01-train-byte-pair-encoding |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-einsum-attention-scores |
| Causal Grouped-Query Attention | Compute causal scaled dot-product attention in which each contiguous group of query heads shares one key/value head. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-causal-grouped-query-attention |
| Parameter-Matched SwiGLU Block | Choose a parameter-matched SwiGLU hidden width under an available-width limit, then evaluate the bias-free block. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-parameter-matched-swiglu |
| RMSNorm Forward Pass | Normalize each final-dimension vector by its root mean square and apply the learned scale without mean subtraction. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-rmsnorm-forward |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/tykim9106)
<!-- tensortonic:end -->
