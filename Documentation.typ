#set document(title: "Building the E-Brain: From Counting Letters to Neural Attention", author: "Tijil Das")
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  header: align(right)[_Building the E-Brain — Personal Engineering Journal_],
  footer: locate(loc => {
    let page_number = counter(page).at(loc).first()
    let total_pages = counter(page).final(loc).first()
    align(center)[#page_number of #total_pages]
  })
)
#set text(font: "Linux Libertine", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em)

#align(center)[
  #v(1cm)
  #text(size: 24pt, weight: "bold")[Building the E-Brain] \
  #v(0.3cm)
  #text(size: 14pt, style: "italic", fill: luma(80))[A Handcrafted Journey from Counting Letters to Neural Networks and Attention] \
  #v(0.5cm)
  #text(size: 11pt, weight: "medium")[Tijil Das] \
  #text(size: 9pt, fill: luma(120))[November 2026]
  #v(1cm)
]

#line(length: 100%, stroke: 0.5pt + luma(180))

#outline(indent: auto, depth: 3)

#v(1cm)
#line(length: 100%, stroke: 0.5pt + luma(180))

= Prologue: Training My Brain to Train the E-Brain

When I began this project, my goal was straightforward: build an AI model from scratch. Not by importing PyTorch, TensorFlow, or huggingface wrappers, but by writing pure, bare-metal Java. I wanted to see the numbers move, calculate the gradients myself, and touch every single weight.

The journey started with a simple counting trick (Bigram Model), ran straight into a terrifying exponential memory wall, transitioned into learned continuous tables, discovered word vectors, introduced non-linear hidden layers with squashing functions, confronted the reality of overfitting, and finally arrived at the doorstep of the Transformer's core engine: Attention.

This document is the unfiltered engineering journal of that journey.

---

= Stage 1: The Bigram Model and The First Wall

== The Intuition: Counting What Follows What
I started with the simplest possible idea of predicting text. If I read through a book, whenever character $A$ appears, what character $B$ comes immediately after it?

I loaded a story chunk (`Text.txt`, roughly 85,000 characters about a cartographer named Arthur Pendelton investigating a mystery in Maine). Across that text, there are 69 unique characters (letters, numbers, punctuation, and spaces).

I created a 2D table of counts:
$ "matrix"[i][j] = "number of times character " j " directly followed character " i $

```java
// Iterate through each character pair and increment the counter
for(int i = 0; i < text.length() - 1; i++) {
    char first_in_pair = text.charAt(i);
    char second_in_pair = text.charAt(i + 1);
    increment_1(first_in_pair, second_in_pair, matrix);
}
```

To generate text, I took the current character, looked at its row in the matrix, summed up the counts, picked a random number between 0 and that sum, and rolled the roulette wheel to pick the next character.

== What Happened
It worked! It generated text:
```text
llenin ornan, 191190 ofighes ackindent fomingn. Haland g be aty walithean thad...
```
It was mostly gibberish, but it felt like alien English. The character `'q'` almost always picked `'u'`. Vowels followed consonants. Spaces appeared at reasonable word-length intervals.

== The Wall: My Laptop Will Melt
Then came the realization that forced me to stop. A bigram only remembers *one* previous letter. If the current letter is `'t'`, it has no idea whether the preceding letter was `'a'` (as in "at"), `'i'` (as in "it"), or `'o'` (as in "ot"). It has zero context.

So what happens if we want the table to remember 2 letters? Or 3 letters? Or 10 letters?

I wrote down the math in my `Limit.txt`:
- 1 letter of memory: $V times V$ cells ($69 times 69 = 4,761$ cells).
- 2 letters of memory: $(V times V) times V = 69^2 times 69 = 328,509$ cells.
- 3 letters of memory: $V^3 times V = 69^3 times 69 approx 2.27 times 10^7$ cells.
- $x$ letters of memory: $V^x times V$ cells.

For 10 letters of memory:
$ 69^{10} times 69 approx 1.65 times 10^{20} " cells" $

Storing $1.65 times 10^{20}$ doubles requires over *1.2 billion Terabytes of RAM*. My laptop wouldn't just freeze; the memory requirement is physically impossible on Earth. Counting tables hit an absolute, impenetrable brick wall.

I had to abandon pure counting. There had to be a way to *learn* the numbers rather than count them.

---

= Stage 2: The Learned Table — Learning by Nudging

== The New Concept: Scores Instead of Counts
In Stage 2, I kept the exact same $V times V$ table, but with a radical shift:
1. The cells no longer hold integer counts. They hold continuous real-valued *scores* (which can be negative).
2. The table starts initialized with tiny random noise (between $-0.1$ and $+0.1$).
3. The scores are nudged using calculus (gradient descent) every time the model makes a prediction.

== Step-by-Step Architecture
=== 1. Softmax (Turning Scores into Probabilities)
Because scores can be negative or large, we convert a row of $V$ scores into valid probabilities that sum to 1.0. To prevent floating-point overflow ($e^{1000} = infinity$), we subtract the maximum score in the row:

$ p_j = frac(e^(s_j - max(s)), sum_(k=1)^V e^(s_k - max(s))) $

Before any learning, because scores are near zero, every character has probability roughly $1/V = 1/69 approx 0.0145$.

=== 2. The Loss Function (Measuring Wrongness)
If the actual next character in the book was $c$, and our model gave it probability $p_c$, how bad was the guess? We use cross-entropy loss:
$ "Loss" = - ln(p_c) $
- If $p_c = 1.0$ (certain and correct), $"Loss" = -ln(1) = 0$.
- At the start, with random guesses ($p_c approx 1/69$), the theoretical starting loss is:
$ - ln(1/69) = ln(69) approx 4.2341 $

=== 3. The Nudge (Gradient Descent)
For every column $j$ in the active row:
$ "gradient"_j = p_j - "target"_j quad ("where target" = 1 "if correct, else" 0) $
$ s_j arrow.l s_j - "learningRate" times "gradient"_j $

If the model gave a wrong letter a high probability $0.4$, its gradient is $+0.4$, so its score is pushed down. If the correct letter had probability $0.05$, its gradient is $0.05 - 1.0 = -0.95$, so its score is pushed up.

== What Happened
Running this in Java across 30 epochs with a learning rate of $0.1$:
- *Theoretical Initial Loss:* $4.2341$
- *Measured Initial Loss:* *4.2299* (near perfect match!)
- *Final Loss:* *2.4158* (the loss dropped steadily and flattened)
- *The 'q' test:* Probability of `'u'` following `'q'` reached *95.63%*!

The model rediscovered what counting had found purely by nudging. But it still only had 1 character of memory.

---

= Stage 3: Vectors — Two Characters of Memory Without an Explosion

== The Core Breakthrough: Continuous Embeddings
Instead of treating each character combination as a completely isolated discrete row, what if every character is described by a list of numbers—a *vector*?

If `'a'` is a list of 10 numbers, and `'e'` is a list of 10 numbers, the model can learn that both are vowels and share similar properties.

== The Architecture
- *Embedding Dimension ($D$):* 10
- *Embedding Table ($C$):* A $V times D$ matrix ($69 times 10$).
- *Context ($N = 2$):* Older character $c_1$, newer character $c_2$.
- *Input Vector ($x$):* Concatenate their two 10-dimensional vectors side-by-side:
  $ x = [ C[c_1], C[c_2] ] in RR^(20) $
- *Weight Matrix ($W$):* $20 times 69$
- *Bias Vector ($b$):* $69$ zeros

Scores are computed via matrix multiplication:
$ "score"_j = sum_(i=0)^(19) x_i W[i][j] + b[j] $

== The Backpropagation Chain
This required writing full backpropagation by hand:
1. Output gradient: $g_j = p_j - "target"_j$
2. Bias gradient: $g_j$
3. Weight gradient: $x_i times g_j$
4. Input gradient: $d x_i = sum_j W[i][j] times g_j$
5. Split $d x$: The first 10 numbers nudge $C[c_1]$; the second 10 numbers nudge $C[c_2]$.

== Verification & Results
Before training, I verified the math with a numerical gradient check:
$ frac("Loss"(W + epsilon) - "Loss"(W), epsilon) approx frac(partial "Loss", partial W) $
The analytic gradient matched the numerical gradient to *6 decimal places* (`-0.001067` vs `-0.001067`).

- *Starting Loss:* $4.2358$
- *Final Loss:* *2.2811* (comfortably broke through the Bigram limit of $2.4158$!)
- *Generated Text:* Real syllables and words began appearing: `wave`, `on`, `ale`, `tin`, `is`, `the`, `ovens`, `-ing`, `-'s`.

---

= Stage 4: The Hidden Layer — Learning Combinations

== The Linear Limitation
In Stage 3, the score was a pure linear sum:
$ "score"_j = f(c_1, j) + g(c_2, j) + b_j $
The older character and newer character only voted independently. The model couldn't express an interaction like:
#align(center)[_"If the last character is 'h' AND the one before it is 't', then 'e' is very likely."_]

== Adding a Hidden Layer ($H = 100$) with $tanh$
To capture non-linear combinations, we insert a hidden layer between the input and the output:
1. $z_k = sum_(i=0)^(39) x_i W_1[i][k] + b_1[k] quad (k = 0 dots 99)$
2. $h_k = tanh(z_k)$ (squashes activations between $-1$ and $+1$)
3. $"score"_j = sum_(k=0)^(99) h_k W_2[k][j] + b_2[j]$

Backpropagating through $tanh$ uses its derivative:
$ d z_k = d h_k times (1 - h_k^2) $

== Results
I tested this in two steps:
- *Change A (Longer memory, $N=4$ characters, linear):* Loss dropped to *2.2117*.
- *Change B (Hidden layer $H=100$ + $tanh$):* Loss plummeted to *1.8561*!

Sample generated text:
#quote[
  _"...who that left. Arthed rifed the gatching his op hat rusing latchel into understanding of fwat warked bastly into the engraw besaltide tabbe a shard of fwandequally from walked bee on the live sand..."_
]
Dialogue formatting, capitalizations, prepositions, and grammatical structures emerged naturally.

---

= Stage 5: The Reality Check — Learning vs. Memorizing

== Train vs. Validation Split
When loss falls to 1.85, is the model actually learning English grammar, or is it simply memorizing Arthur Pendelton's story in Aldersgate?

To find out, I cut the 85,447-character text into:
- *Training Set:* First 90% (76,902 characters) — used for nudging.
- *Validation Set:* Last 10% (8,545 characters) — strictly held aside; forward pass only.

== The Overfitting Discovery
Running 20 epochs revealed the truth:
#table(
  columns: (1.5fr, 2fr, 2fr, 2fr),
  align: center,
  table.header([*Epoch*], [*Train Loss*], [*Validation Loss*], [*Gap*]),
  [1], [2.5469], [2.2336], [-0.3133],
  [4], [2.0106], [2.0046], [-0.0060],
  [10], [1.8665], [1.9583], [+0.0918],
  [15], [1.8248], [1.9532], [+0.1284],
  [20], [1.8035], [1.9434], [+0.1399],
)

*What the numbers mean:*
- *Epochs 1 to 4:* Both train and validation losses plummeted together. The model was genuinely learning universal English patterns.
- *After Epoch 10:* Validation loss stalled near $1.94 - 1.95$, while training loss continued downward to $1.80$.
- The widening gap of $0.14$ was direct empirical proof of *overfitting*. The model was beginning to memorize specific phrases from the training set that did not exist in the validation set.

---

= Stage 6: Fixing Overfitting & The Scaling Limits

I tested four standard interventions to battle the overfitting gap:

#table(
  columns: (2.5fr, 1.8fr, 1.8fr, 2fr),
  align: center,
  table.header([*Intervention*], [*Train Loss*], [*Val Loss*], [*Gap / Result*]),
  [Baseline (Stage 5)], [1.8258], [1.9513], [0.1255],
  [1. More Text ($2 times$ size)], [1.7801], [*1.8582*], [*0.0781* (Cut in half!)],
  [2. Best-Epoch Saving], [1.8261], [1.9300], [Kept Epoch 14 peak],
  [3. Weight Decay ($0.9999$)], [2.2831], [2.2514], [-0.0316 (Over-regularized)],
  [4. LR Decay ($0.015 times 0.9$)], [1.7150], [*1.8605*], [0.1455],
)

== Takeaways from Stage 6
1. *More Text is King:* Doubling the data volume was the single most powerful remedy. It brought validation loss down to *1.8582* and cut the generalization gap nearly in half.
2. *Learning Rate Decay is the Best Algorithmic Fix:* Starting with larger exploratory steps and decaying gradually by $0.9$ allowed the weights to settle into sharper, cleaner minima ($1.8605$ on fixed text).

---

= Stage 7: The Attention Mechanism — Breaking the Fixed Slot Bottleneck

== The Core Idea: Dynamic Routing Instead of Fixed Slots
In Stages 4, 5, and 6, when we expanded the context to $N = 4$ or $N = 8$, the first hidden weight table $W_1$ was forced to have $(N times D)$ rows. This created three severe architectural flaws:
1. *Rigid Slot Weights:* The weights connecting "character 3 steps back" were completely distinct from "character 4 steps back." If the word `"the"` shifted by just one letter position in the context window, the network had to re-learn its weights from scratch.
2. *Equal Cost for Irrelevant Context:* The model paid the exact same parameter cost for distant characters whether they were critical verbs or irrelevant spaces.
3. *Linear Growth of Hidden Weights:* Scaling context to $N = 16$ or $N = 32$ would explode $W_1$ into tens of thousands of weights, triggering intense overfitting.

*Attention* completely flips this paradigm. Instead of fixed, static weight slots for each position, the newest character dynamically *queries* all previous characters in the context window, calculates how relevant each one is, and retrieves a weighted blend of their values. 

The hidden layer $W_1$ no longer needs to scale with $N$. It only receives the blended output of dimension $D$ (joined with the newest character). Context can scale to $N = 16, 32, 64$ with *zero growth in $W_1$*.

== Step-by-Step Architecture

=== 1. Position Vectors ($x_i = C[c_i] + P[i]$)
Because attention computes dot products across an unordered set, attention by itself is order-blind (it cannot distinguish between `"cat"` and `"act"`). To inject sequence order, we create a position table $P$ of size $N times D$ filled with learned weights.

At each context position $i in {0, dots, N-1}$:
$ x_i = C[c_i] + P[i] in RR^D $

=== 2. The Three Projections: Query, Key, and Value
We introduce three learned projection tables, each of size $D times D$:
- *Query ($W_q$):* The newest character ($x_(N-1)$) asks: _"What grammatical or contextual clues do I need to predict what comes next?"_
  $ q = x_(N-1) times W_q in RR^D $
- *Keys ($W_k$):* Every character in the context window ($x_i$) advertises: _"This is the feature identity I offer."_
  $ k_i = x_i times W_k in RR^D quad (i = 0 dots N-1) $
- *Values ($W_v$):* Every character in the context window ($x_i$) packages: _"This is the actual information I hand over if you choose to listen to me."_
  $ v_i = x_i times W_v in RR^D quad (i = 0 dots N-1) $

=== 3. Scaled Attention Scores
How relevant is position $i$ to the newest character? We take their dot product and scale it by $1/sqrt(D)$ (which prevents dot products from growing excessively large in higher dimensions and saturating the softmax):
$ "score"_i = frac(q dot k_i, sqrt(D)) = frac(sum_(m=0)^(D-1) q_m k_(i, m), sqrt(D)) $

=== 4. Softmax (Attention Weights)
We pass the $N$ raw scores through the same softmax function written in Stage 2 to obtain normalized attention weights $a_0, dots, a_(N-1)$ that strictly sum to $1.0$:
$ a_i = frac(e^("score"_i - max("score")), sum_(j=0)^(N-1) e^("score"_j - max("score"))) $

=== 5. Blend (The Attention Output)
We compute a weighted sum of the value vectors:
$ "att_out" = sum_(i=0)^(N-1) a_i v_i in RR^D $

=== 6. Prediction via Hidden Layer (Residual Connection)
We concatenate the retrieved context `"att_out"` ($D$ numbers) with the newest character's vector $x_(N-1)$ ($D$ numbers) to form a $2D$-dimensional vector:
$ "net_in" = [ "att_out", x_(N-1) ] in RR^(2D) $
This vector passes directly into our existing Stage 4 hidden layer:
$ z = "net_in" times W_1 + b_1 quad arrow.r quad h = tanh(z) $
$ "scores" = h times W_2 + b_2 quad arrow.r quad p = "softmax"("scores") $
$ "Loss" = - ln(p_"target") $

== The Hand Check (Paper vs. Code Verification)
Before training on text, I verified the forward pass using a hand-computed toy example with $D = 2, N = 3$ (skipping $sqrt(D)$):
- Query: $q = [1.0, 0.0]$
- Keys: $k_0 = [1.0, 0.0], k_1 = [0.0, 1.0], k_2 = [1.0, 1.0]$
- Values: $v_0 = [2.0, 0.0], v_1 = [0.0, 4.0], v_2 = [1.0, 1.0]$

*Manual Calculations:*
- *Scores ($q dot k_i$):*
  $ s_0 = 1 dot 1 + 0 dot 0 = 1.0 $
  $ s_1 = 1 dot 0 + 0 dot 1 = 0.0 $
  $ s_2 = 1 dot 1 + 0 dot 1 = 1.0 $
  $ "scores" = [1.0, 0.0, 1.0] $
- *Softmax Weights ($a_i$):*
  $ e^(1-1) = 1.0, quad e^(0-1) approx 0.3679, quad e^(1-1) = 1.0 quad arrow.r quad "Sum" approx 2.3679 $
  $ a_0 = 1.0 / 2.3679 approx 0.4223 $
  $ a_1 = 0.3679 / 2.3679 approx 0.1554 $
  $ a_2 = 1.0 / 2.3679 approx 0.4223 $
- *Blended Output ($sum a_i v_i$):*
  $ "out"_0 = (0.4223 times 2.0) + (0.1554 times 0.0) + (0.4223 times 1.0) = 0.8446 + 0.4223 = bold(1.2669) $
  $ "out"_1 = (0.4223 times 0.0) + (0.1554 times 4.0) + (0.4223 times 1.0) = 0.6216 + 0.4223 = bold(1.0439) $

*Java Code Output:*
```text
Scores:  [1.0, 0.0, 1.0]
Softmax: [0.422, 0.155, 0.422]  (Expected: ~0.422, ~0.155, ~0.422)
Output:  [1.267, 1.044]         (Expected: ~[1.266, 1.042])
```
The forward implementation was mathematically exact.

== The Attention Backward Pass (Deriving the Chain Rule)
To learn $W_q, W_k, W_v, P,$ and $C$, the gradients must flow backwards through the attention mechanism:

1. Let $d"Out"$ be the gradient arriving at `"att_out"` from the hidden layer.
2. *Gradient for Values ($v_i$):*
   $ d v_i = a_i times d"Out" $
3. *Gradient for Attention Weights ($a_i$):*
   $ d a_i = d"Out" dot v_i = sum_(m=0)^(D-1) d"Out"_m times v_(i, m) $
4. *Softmax Backward Formula:*
   Let $S = sum_(j=0)^(N-1) a_j times d a_j$. The derivative of softmax yields:
   $ d s_i = a_i times (d a_i - S) $
5. *Scaled Key & Query Gradients:*
   $ d k_i = frac(d s_i, sqrt(D)) times q $
   $ d q = sum_(i=0)^(N-1) frac(d s_i, sqrt(D)) times k_i $
6. *Weight Matrix Gradients:*
   $ "grad"_Wq = x_(N-1)^T times d q $
   $ "grad"_Wk = sum_(i=0)^(N-1) x_i^T times d k_i $
   $ "grad"_Wv = sum_(i=0)^(N-1) x_i^T times d v_i $
7. *Gradients for Inputs ($x_i$):*
   Gradients from $v_i$ and $k_i$ project back through $W_v^T$ and $W_k^T$. For position $N-1$, gradients from $q$ and the residual connection add in.
   Finally, $d x_i$ splits equally into the character embeddings $C[c_i]$ and position vectors $P[i]$.

*Rigorous Gradient Checking with Relative Error:*
Early in training, all dot-product scores are nearly equal, making attention distributions virtually uniform. As a result, perturbing $W_k$ barely changes the loss, producing deceptively small analytic gradients ($\sim 10^{-8}$). To rigorously verify the calculus, we trained the weights for 500 steps until attention became non-uniform, and evaluated gradients across $W_v$, $W_1$, and $W_k$ using the relative error formula:
$ "RelError" = frac(|"analytic" - "numerical"|, |"analytic"| + |"numerical"|) $
All checks met the strict numerical threshold of $< 0.001$:
- $W_v[2][3]$: Analytic = $-0.00003403$, Numerical = $-0.00003403$, Rel Error = $4.30 times 10^(-7)$ [PASS]
- $W_1[4][5]$: Analytic = $+0.00047317$, Numerical = $+0.00047317$, Rel Error = $5.15 times 10^(-9)$ [PASS]
- $W_k[1][2]$: Analytic = $+0.00000001$, Numerical = $+0.00000001$, Rel Error = $7.49 times 10^(-4)$ [PASS]

== Experimental Results & The Honest Diagnosis of Stage 7

#table(
  columns: (2fr, 2fr, 2fr, 2fr),
  align: center,
  table.header([*Configuration*], [*Final Train Loss*], [*Final Val Loss*], [*Best Val Loss / Gap*]),
  [Stage 6 Baseline ($N=4$, MLP)], [1.8340], [1.8624], [1.8624 (Gap: +0.0284)],
  [Attention ($N = 4$)], [2.0882], [2.1129], [2.1129 (Gap: +0.0247)],
  [Attention ($N = 16$)], [2.1013], [*2.1107*], [*2.0944* (Gap: +0.0095)],
)

=== The Honest Reading: Underfitting and the Information Bottleneck
At first glance, seeing a validation gap of only $+0.0095$ felt like a victory over overfitting. But an honest comparison against our Stage 6 baseline ($1.83$ train, $1.86$ val) reveals the real story:
*The model was not just regularized; it was severely underfitting.*

Why?
In the old Stage 4/6 model, the network retained 4 separate character vectors ($4 times 10 = 40$ numbers) fed into $W_1$. It knew with absolute clarity which character occupied which exact slot.
In Stage 7 Attention, we looked back across 16 characters, but compressed that entire history into *a single blended vector of 10 numbers*. A single 10-dimensional vector simply does not possess enough degrees of freedom to represent both "what the newest character is" and "what previous punctuation or prefixes occurred". Everything was averaged away.

Yet, $N = 16$ ($2.094$) still beat $N = 4$ ($2.113$), proving that longer context helps. The limit was not the attention concept—it was the narrow bottleneck we forced it through.

---

= Stage 8: Removing the Bottleneck — Dimension Scaling, Residuals & Multi-Head Attention

To shatter the information bottleneck, we systematically implemented and tested three architectural upgrades in order:
1. *Change 1: Vector Dimension Scaling ($D: 10 arrow.r 32$)*
2. *Change 2: Additive Residual Connection ($x_N + "att\_out"$)*
3. *Change 3: Multi-Head Attention ($H = 4$ heads of dimension $d_k = 8$)*

== Change 1: Expanding the Vector Highway ($D = 32, N = 16$)
Instead of squeezing the blended context into 10 numbers, we expanded $D$ to 32. Each character vector and position vector now holds 32 continuous values.
- Projection matrices $W_q, W_k, W_v$ expand from $10 times 10$ to $32 times 32$.
- The scaled dot-product factor becomes $sqrt(D) = sqrt(32) approx 5.6568$.
- The blended context vector `"att_out"` carries 32 numbers of rich linguistic nuance rather than 10.
- Concatenating `"att_out"` with the current character vector $x_(N-1)$ creates an input vector of length $2D = 64$ into the hidden layer ($W_1: 64 times 100$).

== Change 2: Additive Residual Connections ($x_N + "att\_out"$)
In Change 1, the model concatenated the blended attention output with $x_(N-1)$, requiring $2D$ inputs into $W_1$.
More importantly, in standard single-head attention, the model is constantly torn: should it spend its attention weights looking at the current character to remember it, or looking back into history?

The *Residual Connection* solves this elegantly by element-wise addition:
$ "net\_in" = x_(N-1) + "att\_out" $
By adding $x_(N-1)$ directly into the attention output:
1. *Direct Identity Highway:* The identity of the current character is provided to the hidden layer unconditionally and "for free".
2. *Freed Attention Capacity:* The attention weights no longer need to waste energy copying $x_(N-1)$. Every ounce of attention capacity can focus entirely on retrieving past context.
3. *Cleaner Gradient Flow:* Gradients flow uninterrupted straight back into $x_(N-1)$ via the addition rule ($frac(partial (A + B), partial A) = 1$), preventing gradient vanishing.
4. *Parameter Efficiency:* Input dimension into $W_1$ drops from $2D = 64$ back down to $D = 32$, reducing $W_1$ parameters by half ($3,200$ vs $6,400$).

== Change 3: Multi-Head Attention ($H = 4$ Heads, $d_k = 8$)
=== The Theoretical Motivation
With a single attention head, the model must produce a single probability distribution $a_0, dots, a_(N-1)$ over the context.
However, natural language requires asking multiple independent questions at the same time:
- *Head 1:* "Where was the last space or word boundary?"
- *Head 2:* "What vowel preceded this consonant?"
- *Head 3:* "Is there an opening quote or parenthesis in this clause?"
- *Head 4:* "What is the root verb or prefix?"

A single set of weights $W_q, W_k$ forces these distinct queries to average out into a compromised, blurry distribution.
*Multi-Head Attention* resolves this by splitting the representation into $H = 4$ independent subspaces:
- Total dimension: $D = 32$
- Number of heads: $H = 4$
- Dimension per head: $d_k = frac(D, H) = frac(32, 4) = 8$

=== Multi-Head Mathematical Formulation
For each head $h in {0, 1, 2, 3}$:
1. *Independent Projections:*
   $ q^((h)) = x_(N-1) times W_q^((h)), quad W_q^((h)) in RR^(D times d_k) $
   $ k_i^((h)) = x_i times W_k^((h)), quad W_k^((h)) in RR^(D times d_k) $
   $ v_i^((h)) = x_i times W_v^((h)), quad W_v^((h)) in RR^(D times d_k) $
2. *Scaled Dot-Product per Head:*
   $ s_i^((h)) = frac(q^((h)) dot k_i^((h)), sqrt(d_k)) = frac(sum_(m=0)^(d_k - 1) q_m^((h)) k_(i, m)^((h)), sqrt(8)) $
3. *Softmax Distribution per Head:*
   $ a_i^((h)) = frac(exp(s_i^((h)) - max_j s_j^((h))), sum_(j=0)^(N-1) exp(s_j^((h)) - max_j s_j^((h)))) $
4. *Head Output:*
   $ "head"_h = sum_(i=0)^(N-1) a_i^((h)) v_i^((h)) in RR^(d_k) $
5. *Concatenation & Residual Assembly:*
   The 4 head outputs (each 8 numbers) are concatenated side-by-side to reconstruct a 32-dimensional vector:
   $ "MultiHeadOut" = ["head"_0 \,||\, "head"_1 \,||\, "head"_2 \,||\, "head"_3] in RR^(32) $
   $ "net\_in" = "MultiHeadOut" + x_(N-1) $
6. *Feed-Forward Network:*
   $ h = tanh("net\_in" times W_1 + b_1), quad W_1 in RR^(32 times 100) $
   $ "scores" = h times W_2 + b_2, quad W_2 in RR^(100 times V) $

=== Backpropagation Through Multi-Head Attention
During the backward pass:
1. $d"net\_in"$ splits directly: $d x_(N-1) += d"net\_in"$, and $d"MultiHeadOut" = d"net\_in"$.
2. The gradient $d"MultiHeadOut"$ is sliced into $H$ chunks of length $d_k$, yielding $d"head"_h$ for each head.
3. For each head independently, the attention gradients ($d v_i^((h)), d a_i^((h)), d s_i^((h)), d k_i^((h)), d q^((h))$) are computed exactly as derived in Stage 7, scaling by $frac(1, sqrt(d_k))$.
4. Gradients for $W_q^((h)), W_k^((h)), W_v^((h))$ accumulate across sequences.
5. All heads project their input gradients back to the shared embeddings $x_i$, where they sum together before updating $C$ and $P$.

== Experimental Progression Across Stage 8

All three configurations were trained on the identical 90% train / 10% validation split of `Text.txt` with $N = 16$:

#table(
  columns: (2.5fr, 1.5fr, 1.5fr, 1.5fr, 2fr),
  align: center,
  table.header([*Architecture Configuration*], [*Epoch 1 (Tr/Val)*], [*Epoch 3 (Tr/Val)*], [*Epoch 5 (Tr/Val)*], [*Takeaway & Impact*]),
  [Stage 7: Single-Head ($D=10$, Joined)], [---], [---], [2.1013 / 2.1107], [Bottlenecked by narrow $D=10$],
  [Change 1: Bigger $D=32$ (Joined)], [2.7096 / 2.4951], [2.3343 / 2.2410], [2.1659 / 2.1070], [Carries $3.2 times$ more information],
  [Change 2: Additive Residual ($D=32$)], [2.6976 / 2.4385], [2.2848 / 2.2094], [2.1428 / 2.1080], [Frees attention from copying $x_N$],
  [Change 3: Multi-Head ($H=4, d_k=8$)], [2.7027 / 2.4934], [2.2834 / 2.1955], [*2.1141* / *2.0781*], [*Best Val Loss (2.0781)*; multi-query routing],
)

=== Critical Reflections & Diagnostics on Stage 8
1. *Why We Couldn't Compare to Baseline Yet:*
   Our Stage 6A MLP baseline achieved $approx 1.86$ after a long run of 15 to 20 epochs. In Stage 8, training stopped after only 5 epochs while losses were still dropping steadily by $approx 0.07$ per epoch. Evaluating attention fairly requires a matching long run.
2. *Why Validation Loss Below Train Loss is Not "Proof of Zero Overfitting":*
   During an epoch, train loss is computed online as a rolling average while the weights are continuously being updated and improved. By contrast, validation loss is measured in a separate evaluation pass using the final, most optimized weights of that epoch. In early epochs of rapid learning, validation loss naturally comes out lower than the lagging training average. This is an artifact of the evaluation timeline, not proof of generalization.
3. *Borderline Key Gradients:*
   The gradient check on $W_k$ yielded values around $10^(-8)$ and a relative error of $7.5 times 10^(-4)$, which sits right at the boundary of floating-point precision. Because initial dot products are small, the softmax distribution over 16 positions remains close to a uniform $1/16 = 0.0625$. When attention is uniform, perturbing keys barely changes the blended output, leaving key and query gradients tiny. To determine whether attention is actually specializing or merely averaging, we need direct diagnostics.

---

= Stage 9: Attention Diagnostics, The Fair Long Run & Tuning

== 9A. The Diagnostic: Seeing What Attention is Doing
To determine whether attention is truly earning its keep or merely performing a blurry uniform blur, we inspect the attention distribution $a_0, dots, a_(15)$ across each of the 4 heads for a concrete context window of 16 characters.
- *Uniform baseline:* $frac(1, 16) = 0.0625$ per position.
- If all weights linger near $0.06$, the heads have not specialized.
- If individual heads concentrate their probability mass (e.g., $0.30 - 0.50$) on specific positions (such as the recent 1--3 characters, word-boundary spaces, or specific syntactic markers), the heads have specialized.

== 9B. The Fair Long Run (30 Epochs, 2x Text, LR Decay, Best-Epoch Saving)
To place Multi-Head Attention on equal footing with our Stage 6A baseline:
- *Dataset:* Doubled corpus ($2 times$ text, $approx 171,000$ characters total, 90% train / 10% validation).
- *Context & Geometry:* $N = 16, D = 32, H = 4$ heads ($d_k = 8$), $H_"mlp" = 100$, additive residual connection.
- *Schedule:* 30 full epochs, starting learning rate $eta_0 = 0.015$ with exponential decay $times 0.95$ per epoch.
- *Best-Epoch Saving:* Tracking and saving the model weights that achieve the lowest validation loss across training.

=== Empirical Training Progression (30 Epochs)

#table(
  columns: (1fr, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
  align: center,
  table.header([*Epoch*], [*Train Loss*], [*Val Loss*], [*Gap (Val - Tr)*], [*Learning Rate*]),
  [Epoch 1], [2.4933], [2.2960], [-0.1972], [0.01500],
  [Epoch 5], [2.0096], [2.0493], [+0.0397], [0.01222],
  [Epoch 10], [1.9201], [1.9530], [+0.0329], [0.00945],
  [Epoch 15], [1.8345], [1.8810], [+0.0465], [0.00732],
  [Epoch 18], [1.7897], [1.8634], [+0.0737], [0.00627],
  [Epoch 22], [1.7358], [1.8191], [+0.0833], [0.00511],
  [Epoch 25], [1.6981], [1.7937], [+0.0956], [0.00438],
  [Epoch 29], [*1.6535*], [*1.7511*], [+0.0975], [0.00357],
  [Epoch 30], [1.6444], [1.7515], [+0.1071], [0.00339],
)

=== Baseline Comparison

#table(
  columns: (2.5fr, 1.5fr, 1.5fr, 2fr),
  align: center,
  table.header([*Model Architecture*], [*Train Loss*], [*Best Val Loss*], [*Status vs Baseline*]),
  [Stage 6A Baseline (MLP, $N=4$, LR Decay)], [1.8340], [1.8596], [Baseline Reference],
  [Stage 7 Attention ($N=16$, Single Head, $D=10$)], [2.1013], [2.0944], [Trails (+0.2348)],
  [Stage 8 Multi-Head ($N=16, D=32, 5$ Epochs)], [2.1141], [2.0781], [Trails (+0.2185)],
  [Stage 9 Multi-Head ($N=16, D=32, 30$ Epochs)], [*1.6535*], [*1.7511*], [*BEATS BASELINE by -0.1085!*],
)

*Conclusion:* Multi-Head Attention not only matches the MLP baseline, it decisively beats it by over *0.108 points of validation loss*. When given an equivalent 30-epoch training runway, the dynamic routing and 16-character receptive field vastly outperform fixed position weights.

== 9A. Attention Distribution Diagnostic

To confirm whether attention is specializing rather than defaulting to uniform averaging ($1/16 = 0.0625$), we inspect the attention weights $a_0, dots, a_(15)$ across all 4 heads for a validation sequence:

```text
Context Window: "ing him forward " -> Target: 'o'
Uniform baseline weight (1/16) = 0.0625

Position:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Character:   i   n   g       h   i   m       f   o   r   w   a   r   d    
Head 0:   0.00 0.00 0.01 0.06 0.01 0.00 0.01 0.10 0.01 0.01 0.01 0.02 0.07 0.06 0.01  0.61  -> Max 0.614 on ' ' (pos 15)
Head 1:   0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.05 0.02 0.02 0.00 0.08 0.06 0.02 0.05  0.65  -> Max 0.646 on ' ' (pos 15)
Head 2:   0.00 0.00 0.00 0.02 0.01 0.00 0.01 0.02 0.01 0.02 0.01 0.04 0.07 0.08 0.12  0.58  -> Max 0.576 on ' ' (pos 15)
Head 3:   0.00 0.00 0.00 0.05 0.01 0.01 0.01 0.06 0.00 0.01 0.01 0.00 0.05 0.03 0.17  0.58  -> Max 0.575 on ' ' (pos 15)
```

And in a second sequence with distinct punctuation boundaries:
```text
Context Window: "men stepped out " -> Target: 'f'
Position:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Character:   m   e   n       s   t   e   p   p   e   d       o   u   t    
Head 0:   0.01 0.00 0.00 0.06 0.00 0.00 0.01 0.01 0.01 0.01 0.01 0.11 0.08 0.04 0.02  0.61  -> Max 0.615 on ' ' (pos 15)
Head 1:   0.00 0.00 0.00 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.33 0.11 0.02 0.05  0.45  -> 0.33 on ' ' (pos 11), 0.45 on ' ' (pos 15)
Head 2:   0.00 0.00 0.00 0.02 0.00 0.00 0.01 0.01 0.02 0.01 0.01 0.16 0.10 0.15 0.05  0.46  -> 0.16 on ' ' (pos 11), 0.15 on 'u' (pos 13)
Head 3:   0.00 0.00 0.00 0.05 0.00 0.00 0.01 0.00 0.01 0.00 0.01 0.21 0.02 0.06 0.08  0.53  -> 0.21 on ' ' (pos 11), 0.53 on ' ' (pos 15)
```

=== Key Diagnostic Takeaways
1. *Attention has strongly specialized:* The weights are nowhere near the uniform $0.0625$ baseline.
2. *Word-Boundary Detection:* The heads have naturally learned to attend to word delimiters (spaces). Head 1 and Head 3 place significant attention mass ($0.33$ and $0.21$) on position 11 (the space after the word "stepped"), linking multi-word context across the 16-character window.
3. *Head Diversity:* Rather than collapsing into identical distributions, Head 1 focuses heavily on the previous word boundary, while Head 2 samples vowel phonemes ('o', 'u') within the current word. Attention has earned its place.

---

= Stage 10: The Complete Causal Transformer — Autoregressive Attention & Deep Block Stacking

Stage 10 transforms our multi-head attention network into a full, modern autoregressive Transformer. It introduces three core innovations:
1. *Predicting at Every Position Simultaneously (Sequence-to-Sequence Training):* Instead of sliding 16 characters to predict a single 17th character, each window of $T=16$ characters predicts all 16 target characters simultaneously ($c_(t+1)$ for each $c_t$).
2. *The Causal Lower-Triangular Mask:* Enforcing strict temporal directionality by masking future positions ($i > t$) with $-10^9$ before softmax.
3. *Deep Residual Transformer Blocks ($L = 1$ and $L = 2$):* Implementing the complete canonical Transformer block (Multi-Head Attention + Residual Addition + Position-wise Feed-Forward Network + Residual Addition), and stacking them.

== 10.1 Mathematical Formulation of Causal Attention

Given a sequence of character indices $c_0, c_1, dots, c_(T-1)$ with targets $c_1, c_2, dots, c_T$:
1. *Initial Embedding:*
   $ X_t^((0)) = C[c_t] + P[t] in RR^D, quad t in {0, dots, T-1} $
2. *Multi-Head Projections (Per Block $l$):*
   For each head $h in {0, dots, H-1}$ and each position $t$:
   $ q_t^((h)) = X_t W_q^((h)), quad k_t^((h)) = X_t W_k^((h)), quad v_t^((h)) = X_t W_v^((h)) in RR^(d_k) $
3. *Causal Score Matrix & Masking:*
   $ "score"[t][i]^((h)) = cases(
     frac(q_t^((h)) dot k_i^((h)), sqrt(d_k)) & "if" i <= t,
     -10^9 & "if" i > t " (future masked)"
   ) $
4. *Causal Softmax:*
   $ A[t][i]^((h)) = cases(
     frac(exp("score"[t][i]^((h)) - max_(j <= t) "score"[t][j]^((h))), sum_(j=0)^t exp("score"[t][j]^((h)) - max_(k <= t) "score"[t][k]^((h)))) & "if" i <= t,
     0.0 & "if" i > t
   ) $
5. *Attention Output & First Residual:*
   $ "head\_out"_t^((h)) = sum_(i=0)^t A[t][i]^((h)) v_i^((h)) in RR^(d_k) $
   $ "att\_out"_t = ["head\_out"_t^((0)) \,||\, dots \,||\, "head\_out"_t^((H-1))] in RR^D $
   $ Y_t = X_t + "att\_out"_t in RR^D $
6. *Position-wise Feed-Forward Network & Second Residual:*
   The same feed-forward layer is applied to every position independently:
   $ h_t = tanh(Y_t W_1 + b_1) in RR^(H_"mlp") $
   $ "ff\_out"_t = h_t W_2 + b_2 in RR^D $
   $ Z_t = Y_t + "ff\_out"_t in RR^D $
7. *Deep Stacking:*
   For a 2-block model (Stage 10B), $Z_t^((0))$ becomes the input $X_t^((1))$ for Block 1.
8. *Output Logits and Cross-Entropy Loss:*
   $ "scores"_t = Z_t^((L-1)) W_"out" + b_"out" in RR^V $
   $ p_t = "softmax"("scores"_t) $
   $ "Loss" = frac(1, T) sum_(t=0)^(T-1) (-ln(p_t[c_(t+1)])) $

== 10.2 Rigorous Verification Before Training

Before launching full training, the architecture underwent two formal sanity tests:

=== 1. Causality Invariance Test
We passed a 16-character sequence through the transformer, perturbed the character at position 10, and recomputed the output probabilities:
- *Positions 0 through 9:* Maximum absolute probability difference $= bold(0.00 times 10^0)$ (exact bitwise zero!).
- *Positions 10 through 15:* Non-zero probability change ($approx 0.0014$).
*Result:* Causal masking is airtight; no information leaks backwards from the future.

=== 2. Gradient Checks (Relative Error $< 0.001$)
Evaluated after 600 training steps on a single block ($L=1$):
- *Check $W_v[0][2][3]$:* Analytic = $-0.00019477$, Numerical = $-0.00019477$, Relative Error = $2.52 times 10^(-8)$ [PASS]
- *Check $W_1[4][5]$:* Analytic = $-0.00003141$, Numerical = $-0.00003141$, Relative Error = $1.76 times 10^(-7)$ [PASS]
- *Check $W_k[2][0][4]$:* Analytic = $+0.00000006$, Numerical = $+0.00000006$, Relative Error = $1.33 times 10^(-4)$ [PASS]

---

= Conclusion & Engineering Summary

Across nine handcrafted stages, built in bare-metal Java without external deep-learning frameworks:
1. *Stage 1:* Counting co-occurrences revealed the catastrophic $V^x$ exponential memory wall ($1.2$ billion Terabytes for 10 characters).
2. *Stage 2:* Continuous score matrices and gradient descent proved that probability distributions can be learned dynamically without counting.
3. *Stage 3:* Continuous character embedding vectors ($C in RR^(V times D)$) decoupled memory length from table dimensions.
4. *Stage 4:* Non-linear hidden layers with $\tanh$ activations enabled non-linear feature interaction modeling.
5. *Stage 5:* Held-out validation splits established an honest metric for separating genuine linguistic generalization from memorization.
6. *Stage 6:* Data scaling and learning rate decay demonstrated optimal regularizing regimes, establishing our reference benchmark of $approx 1.86$.
7. *Stage 7:* Query-Key-Value Self-Attention decoupled context length from parameter count, scaling context to 16 characters while revealing the 10-dimensional bottleneck.
8. *Stage 8:* Expanding $D to 32$, adding element-wise residual connections, and splitting attention across 4 heads eliminated the bottleneck.
9. *Stage 9:* In a rigorous 30-epoch long run, Multi-Head Attention achieved a validation loss of *1.7511*, decisively outperforming the fixed baseline by *-0.1085*, while empirical diagnostics proved that attention heads independently specialize in word boundaries and morphological structure.

We have completed the ground-up construction, mathematical verification, and empirical mastery of the neural language model.

