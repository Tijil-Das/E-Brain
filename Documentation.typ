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

*Numerical Gradient Check on $W_k[0][0]$:*
```text
Gradient Check (Wk[0][0], N=4)  -> Analytic: 0.00000006  | Numerical: 0.00000006  (Match: true)
Gradient Check (Wk[0][0], N=16) -> Analytic: -0.00000001 | Numerical: -0.00000001 (Match: true)
```

== Experimental Results: N = 4 vs. Scaling to N = 16

#table(
  columns: (2fr, 2fr, 2fr, 2fr),
  align: center,
  table.header([*Configuration*], [*Final Train Loss*], [*Final Val Loss*], [*Best Val Loss / Gap*]),
  [Attention ($N = 4$)], [2.0882], [2.1129], [2.1129 (Gap: +0.0247)],
  [Attention ($N = 16$)], [2.1013], [*2.1107*], [*2.0944* (Gap: +0.0095!)],
)

=== Analysis of the Results
1. *Why N = 4 Attention doesn't beat the baseline yet:*
   With only 4 characters of context, the sequence is so short that hardwired position weights can easily memorize the transitions. Attention has extra projection matrices ($W_q, W_k, W_v$) to learn, so on tiny context windows, the overhead shows little advantage.
2. *The Breakthrough at N = 16:*
   When context expanded by $4 times$ to $N = 16$:
   - The validation loss dropped from $2.1129$ down to *2.0944*.
   - The generalization gap shrank to just *+0.0095* (virtually zero overfitting!).
   - In Stage 4, scaling context to $N = 16$ would have inflated $W_1$ from $4,000$ to $16,000$ weights. With Attention, $W_1$ remained strictly fixed at $(2 times 10) times 100 = 2,000$ weights!
   - This proves the fundamental promise of the Transformer: long context becomes computationally scalable, and dynamic routing prevents the model from overfitting to fixed position slots.

---

= Conclusion & Roadmap

Across seven stages, starting from scratch in raw Java:
1. We proved that counting hits a hard physical memory limit ($1.2$ billion TB).
2. We proved that gradient descent and continuous score tables can discover probability distributions without counting.
3. We proved that vector embeddings eliminate the table explosion.
4. We proved that non-linear hidden activations ($\tanh$) allow the network to model feature interactions.
5. We proved empirically how to measure overfitting using held-out validation loss.
6. We proved that data volume and learning rate decay are the most potent regularization techniques.
7. We proved that Self-Attention decouples context length from parameter count, enabling long-range context without exploding the network.

We have built the complete conceptual and mathematical engine of modern Large Language Models from the ground up.

