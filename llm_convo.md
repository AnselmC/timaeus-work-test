User: Can you summarize this paper for me and point out its most important contributions and findings? 
```"Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression
Allan Raventós  Mansheej Paul∗  Feng Chen  Surya Ganguli
Stanford University {aravento, mansheej, fengc, sganguli}@stanford.edu
Equal Contribution. Code released at https://github.com/mansheej/icl-task-diversity
Abstract
Pretrained transformers exhibit the remarkable ability of in-context learning (ICL): they can learn tasks from just a few examples provided in the prompt without updating any weights. This raises a foundational question: can ICL solve fundamentally new tasks that are very different from those seen during pretraining? To probe this question, we examine ICL’s performance on linear regression while varying the diversity of tasks in the pretraining dataset. We empirically demonstrate a task diversity threshold for the emergence of ICL. Below this threshold, the pretrained transformer cannot solve unseen regression tasks, instead behaving like a Bayesian estimator with the non-diverse pretraining task distribution as the prior. Beyond this threshold, the transformer significantly outperforms this estimator; its behavior aligns with that of ridge regression, corresponding to a Gaussian prior over all tasks, including those not seen during pretraining. Thus, when pretrained on data with task diversity greater than the threshold, transformers can optimally solve fundamentally new tasks in-context. Importantly, this capability hinges on it deviating from the Bayes optimal estimator with the pretraining distribution as the prior. This study also explores the effect of regularization, model capacity and task structure and underscores, in a concrete example, the critical role of task diversity, alongside data and model scale, in the emergence of ICL.

1Introduction
Pretrained transformers (PTs) can learn new tasks from just a few examples provided in the prompt without taking any gradient steps on those examples [1]. This ability, called in-context learning (ICL), has unlocked the widspread use of language models by making it efficient to adapt general purpose models to bespoke tasks without explicit training. Though remarkable, what makes ICL mysterious, and potentially harmful [2], is that the learning algorithm implemented by the PT in its forward pass is not built into its architecture or training process; instead it emerges from pretraining on large-scale data with a next token prediction objective. This raises a foundational question: can ICL really solve fundamentally new tasks that are very different from those seen during pretraining? If so, what learning algorithm does ICL implement? To answer these questions, we need to better understand how the different ingredients that go into pretraining influence this ability.

Towards this end, we explore how the diversity of tasks in the pretraining data affects the emergence of ICL. Prior work [3] has proposed that ICL works by performing Bayesian inference. During pretraining, transformers learn a prior over latent tasks represented in the pretraining data. When prompted with examples at inference time, they "retrieve" relevant pretraining tasks and generate subsequent tokens from the posterior distribution conditioned on the query and inferred tasks. This suggests that ICL performance on a new task is influenced by its similarity to tasks implicitly learned during pretraining. However, the distribution of tasks in our pretraining data, 
𝒯
Pretrain
, is usually a limited and unrepresentative subsample of the ideal distribution of tasks, 
𝒯
True
, that we want our model to be capable of learning in-context. For instance, 
𝒯
True
 could be the set of all instructions we want an A.I. assistant to follow. But, large-scale language modeling datasets [4, 5] used to pretrain these models contain very few examples correctly formatted for ICL. Instruction finetuning (IFT) datasets [6, 7, 8, 9, 10, 11] designed to ameliorate this are expensive to collect and thus contain tasks from just a few domains. Under the Bayesian framework, this distribution mismatch would cause the Bayesian estimator with a prior over the limited pretraining tasks, 
𝒯
Pretrain
, to perform suboptimally on tasks that are very different from those seen during pretraining. This motivates our question: can a model pretrained on a dataset with low task diversity nevertheless learn new, unseen tasks?

For general purpose language modeling, the size and complexity of 
𝒯
Pretrain
 and the vague specification of 
𝒯
True
 make this question challenging to analyze. So, following recent work [12, 13, 14], we study ICL for linear regression. Here, a task is a linear regression problem with a given latent regression vector; the PT must predict the target for a new data point from examples of data-target pairs provided in the prompt. Prior work [13] has shown that transformers that see an unlimited number of latent regression vectors during pretraining learn to perform ridge regression with the Bayes optimal ridge parameter. We instead consider the case where the pretraining task distribution, 
𝒯
Pretrain
, contains a limited and finite set of latent regression vectors (see Section 2 for details). To evaluate its ability to learn new tasks, the PT is tested on the ideal task distribution, 
𝒯
True
, which is a Gaussian distribution over all latent regression vectors. Studying this setting has two advantages: first, we can directly vary the task diversity in 
𝒯
Pretrain
 by changing the number of unique latent regression vectors seen during pretraining. Second, we can calculate the optimal estimator that minimizes the pretraining loss—the Bayesian estimator with prior 
𝒯
Pretrain
—as well as the optimal estimator for all tasks—the Bayesian estimator with prior 
𝒯
True
. This allows us to interpret the behavior of the PT by comparing its predictions to those of the optimal estimators under either task distribution. In our work, we vary the pretraining task diversity and probe the PT’s ability to learn fundamentally new tasks in-context: does it behave like the optimal estimator for 
𝒯
Pretrain
 and perform suboptimally on tasks from 
𝒯
True
, or does it align with the optimal estimator for 
𝒯
True
 which can solve new, unseen tasks?

Contributions. Our contributions are as follows:

• We find that a transformer pretrained on data with low task diversity behaves like the Bayesian estimator with prior 
𝒯
Pretrain
; it performs optimally on pretraining tasks but cannot learn new tasks in-context. However, as pretraining task diversity increases, the PT deviates from this Bayesian estimator, significantly outperforming it on new tasks, and at a large but still finite number of pretraining tasks, the PT’s performance closely matches that of the optimal estimator on 
𝒯
True
.
• We identify a task diversity threshold for the emergence of ICL. Below this threshold, increasing the pretraining dataset size while keeping task diversity constant biases the PT towards the pretraining task distribution. Conversely, beyond this threshold, increasing the dataset size without increasing its task diversity improves the PT’s performance on new, unseen tasks. This suggests that the PT’s behavior undergoes a sharp algorithmic phase transition in the limit of many examples per task, aligning with the optimal estimators on 
𝒯
Pretrain
 before the threshold and on 
𝒯
True
 after it. We also examine this transition from the perspective of learning dynamics.
• We empirically show that increasing the task dimension at fixed SNR increases the task diversity threshold. However, the scaling of the PT’s error with dimension is vastly superior to that of the optimal Bayesian estimator for 
𝒯
Pretrain
; at a task diversity that is beyond the threshold at all dimensions we consider, the PT remains near-optimal with increasing dimension, whereas the optimal estimator for 
𝒯
Pretrain
 grows progressively less similar to the optimal estimator for 
𝒯
True
.
• We show that increasing weight decay significantly decreases the task diversity threshold while increasing number of layers or embedding size increases the task diversity threshold. This elucidates the effect of regularization and model capacity on the emergence of ICL.
Overall these contributions suggest that the emergence of in-context learning in pretrained transformers cannot be fully explained by a theory of Bayesian inference on the pretraining distribution.

Refer to caption
Figure 1: Schematic for ICL of linear regression. (Left) A task corresponds to a latent regression vector, 
𝐰
 (purple line). 
(
𝐱
1
,
y
1
,
…
,
𝐱
K
,
y
K
)
 (black circles) is a sequence of in-context examples for this task. (Right) The PT, 
f
θ
, takes this as input and generates 
K
 outputs. The 
k
th output, 
f
θ
​
(
S
k
)
, is the prediction for the target of 
𝐱
k
 and depends only on the context 
S
k
=
(
𝐱
1
,
y
1
,
…
,
𝐱
k
−
1
,
y
k
−
1
,
𝐱
k
)
.
2Problem setup
ICL of linear regression (schematic in Fig. 1). Each ICL task corresponds to a latent 
D
-dimensional regression vector, 
𝐰
∈
ℝ
D
. At inference time, the transformer takes as input a sequence of 
K
 data-target pairs, 
(
𝐱
1
,
y
1
,
…
,
𝐱
K
,
y
K
)
, which are the in-context examples corresponding to this single task 
𝐰
. For 
k
∈
{
1
,
…
,
K
}
, the data are drawn i.i.d. from a 
D
-dimensional standard normal distribution, 
𝐱
k
∼
𝒩
​
(
𝟎
,
𝐈
D
)
, and the targets are scalars given by 
y
k
=
𝐰
⊺
​
𝐱
k
+
ε
k
. The 
ε
k
’s are noise scalars drawn i.i.d. from a normal distribution with mean 0 and variance 
σ
2
, 
ε
k
∼
𝒩
​
(
0
,
σ
2
)
. Let 
f
θ
 be the PT with parameters 
θ
. Since we use a decoder-only transformer with a causal attention mask, for each 
k
∈
{
1
,
…
,
K
}
, the transformer sees the context 
S
k
=
(
𝐱
1
,
y
1
,
…
,
𝐱
k
−
1
,
y
k
−
1
,
𝐱
k
)
 and based on this context, it makes a prediction, 
f
θ
​
(
S
k
)
, for the target of 
𝐱
k
. Thus, in each forward pass, the PT solves 
K
 linear regression problems each with the same latent regression vector but an increasing number of in-context examples.

Pretraining. The transformer is pretrained to minimize the next token prediction mean squared error (MSE) on sequences of data and target pairs. The latent regression vector for each sequence is drawn from the pretraining task distribution, 
𝒯
Pretrain
. This distribution has limited diversity as it is the uniform distribution over a finite set of 
M
 tasks, 
𝒯
Pretrain
=
𝒰
​
{
𝐰
(
1
)
,
…
,
𝐰
(
M
)
}
. Each task in 
𝒯
Pretrain
 is drawn i.i.d from a 
D
-dimensional standard normal distribution, 
𝐰
(
i
)
∼
𝒩
​
(
𝟎
,
𝐈
D
)
,
i
∈
1
,
…
,
M
. By increasing the number of tasks, 
M
, in 
𝒯
Pretrain
, we can increase the diversity of the pretraining data. Since the transformer makes a prediction for every data point in the sequence, its loss, 
L
𝒯
Pretrain
, is just the MSE for each prediction, averaged over the predictions in the sequence:

L
𝒯
Pretrain
​
(
θ
)
=
𝔼
𝐰
∼
𝒯
Pretrain
𝐱
1
,
…
,
𝐱
K
∼
𝒩
​
(
𝟎
,
𝐈
D
)
ε
1
,
…
,
ε
K
∼
𝒩
​
(
0
,
σ
2
)
[
1
K
​
∑
k
=
1
K
(
f
θ
​
(
S
k
)
−
y
k
)
2
]
.
(1)
Evaluation. We evaluate the PT’s performance on tasks seen during pretraining by computing 
L
𝒯
Pretrain
 using Eq. 1 but with new samples of data and noise. Since these are new instances of the task with new in-context examples, this evaluation corresponds to the test error of the PT. For a PT to successfully perform ICL of linear regression on new tasks, it must accurately predict the targets from the in-context examples for any task drawn from an ideal task distribution, 
𝒯
True
, over all latent regression vectors; in our case 
𝒯
True
=
𝒩
​
(
𝟎
,
𝐈
D
)
. We evaluate the PT’s performance on new tasks by computing 
L
𝒯
True
, which follows Eq. 1 but where the tasks are sampled from the ideal task distribution: 
𝐰
∼
𝒯
True
 in the expectation.

Comparing the PT to optimal estimators. An advantage of studying ICL of linear regression is that we can calculate the ground truth optimal estimators that minimize the loss, 
L
𝒯
, in Eq. 1 for both task distributions, 
𝒯
Pretrain
 and 
𝒯
True
. The optimal estimator for the 
k
th prediction, 
y
^
k
𝒯
, that minimizes the 
k
th term in the sum in 
L
𝒯
 is the Bayesian estimator with 
𝒯
 as the prior. This is given by the posterior mean of 
y
k
 conditioned on the context: 
y
^
k
𝒯
=
𝔼
𝒯
,
ε
k
⁡
[
y
k
∣
S
k
]
, where the expectation is over the task distribution, 
𝒯
, and the noise, 
ε
k
 (Section A.1).

For task distribution 
𝒯
Pretrain
=
𝒰
​
{
𝐰
(
1
)
,
…
,
𝐰
(
M
)
}
, the discrete minimum mean squared error (dMMSE) estimator is optimal. It is given by 
y
^
k
dMMSE
=
(
𝐰
^
k
dMMSE
)
⊺
​
𝐱
k
 where 
𝐰
^
1
dMMSE
=
1
M
​
∑
i
=
1
M
𝐰
(
i
)
 and for 
k
∈
{
2
,
…
,
K
}
, (Section A.2)

𝐰
^
k
dMMSE
=
∑
i
=
1
M
exp
⁡
(
−
1
2
​
σ
2
​
∑
j
=
1
k
−
1
(
y
j
−
𝐰
(
i
)
⊺
​
𝐱
j
)
2
)
∑
l
=
1
M
exp
⁡
(
−
1
2
​
σ
2
​
∑
j
=
1
k
−
1
(
y
j
−
𝐰
(
l
)
⊺
​
𝐱
j
)
2
)
​
𝐰
(
i
)
.
(2)
Intuitively, 
𝐰
^
k
dMMSE
 is just a weighted sum of the pretraining 
𝐰
(
i
)
s with weight governed by the likelihood of observing targets 
{
y
1
,
…
,
y
k
−
1
}
 conditioned on inputs 
{
𝐱
1
,
…
,
𝐱
k
−
1
}
 and the task being 
𝐰
(
i
)
. A PT that minimizes the pretraining loss 
L
𝒯
Pretrain
 will behave like this estimator.

For task distribution 
𝒯
True
=
𝒩
​
(
𝟎
,
𝐈
D
)
, the Ridge regression estimator with the ridge parameter set to the noise scale 
σ
2
 is optimal: 
y
^
k
Ridge
=
(
𝐰
^
k
Ridge
)
⊺
​
𝐱
k
, where 
𝐰
^
1
Ridge
=
𝟎
 and for 
k
=
{
2
,
…
,
K
}
,

𝐰
^
k
Ridge
=
(
𝐗
⊺
​
𝐗
+
σ
2
​
𝐈
D
)
−
1
​
𝐗
⊺
​
𝐲
,
(3)
where 
𝐗
=
(
𝐱
1
⊺
,
…
,
𝐱
k
−
1
⊺
)
∈
ℝ
(
k
−
1
)
×
D
 and 
𝐲
=
(
y
1
,
…
,
y
k
−
1
)
 (Section A.3). A PT that performs optimally on new tasks will behave like this estimator. We can compare the behavior of the PT to that of the optimal estimators by computing the mean square difference of the predictions under a given task distribution 
𝒯
. We write this as

Δ
PT,Ridge/dMMSE
𝒯
=
𝔼
𝐰
∼
𝒯
𝐱
1
,
…
,
𝐱
K
∼
𝒩
​
(
𝟎
,
𝐈
D
)
ε
1
,
…
,
ε
K
∼
𝒩
​
(
0
,
σ
2
)
[
1
K
​
D
​
∑
k
=
1
K
(
f
θ
​
(
S
k
)
−
y
^
k
Ridge/dMMSE
)
2
]
.
(4)
Refer to caption
Figure 2: ICL emerges in PTs beyond a threshold pretraining task diversity. We show all results on both tasks seen during pretraining (top row) and on new tasks (bottom row). The left column compares the normalized loss of transformers pretrained with increasing task diversity to that of dMMSE and Ridge. When the pretraining task diversity is small, the PT’s performance matches that of dMMSE; it performs very well on tasks seen during pretraining but poorly on new tasks. As the pretraining task diversity increases, both dMMSE and PT approach Ridge. However, the PT approaches Ridge much faster, significantly outperforming dMMSE on new tasks (bottom left). In the middle and right columns, we compare the PT’s predictions to those of dMMSE and Ridge respectively (Eq. 4). We also increase the number of sequences per task at each level of task diversity by increasing the batch size while keeping total training steps fixed. This reveals a task diversity threshold between 
2
14
 and 
2
15
 pretraining tasks at which there is a phase transition in the behavior of the model. Below the threshold, increasing the dataset size leads to PTs with predictions more aligned with dMMSE on 
𝒯
Pretrain
 (top middle). However, beyond this threshold (indicated by gray shading), increasing the dataset size leads to PTs more aligned with Ridge on all tasks (right).
3Experiments and results
Unless specified otherwise, we study linear regression in 
D
=
8
 dimensions with up to 
K
=
16
 in-context examples and observation noise variance 
σ
2
=
0.25
. We use either a base transformer model with the GPT2 architecture [15] with 8 layers, 128-dimensional embeddings, and 2 attention heads or a small model with 4 layers, 64-dimensional embeddings, and 2 attention heads. We train with the Adam optimizer [16] and a one-cycle triangle learning rate schedule [17] with 50% warmup. The base model is trained with batch size 256 for 500K training steps, though these hyperparameters are varied in our experiments. We always sweep over a range of learning rates and choose the largest learning rate at which training is stable. For further details see Appendix B.

To pretrain a randomly initialized transformer on data with task diversity 
M
, we first construct the pretraining task distribution, 
𝒯
Pretrain
, as described in Section 2. We then minimize the objective 
L
𝒯
Pretrain
 in Eq. 1 using minibatch stochastic gradient descent. For each sequence in a minibatch, we sample a single task 
𝐰
 from 
𝒯
Pretrain
, as well as new samples of data, 
{
𝐱
i
}
i
=
1
K
, and noise, 
{
ε
i
}
i
=
1
K
, from their respective continuous distributions, to form a sequence 
(
𝐱
1
,
𝐰
⊺
​
𝐱
1
+
ε
1
,
…
,
𝐱
K
,
𝐰
⊺
​
𝐱
K
+
ε
K
)
. If we train for 
N
 steps at batch size 
B
, the transformer will see a total of 
N
​
B
 unique sequences and roughly 
N
​
B
M
 unique sequences for each latent task in 
𝒯
Pretrain
. By increasing either 
B
 or 
N
 at fixed 
M
, we can increase the total size of the pretraining dataset (or number of sequences per task) while keeping the dataset diversity—the number of unique 
𝐰
s in 
𝒯
Pretrain
—fixed.

Refer to caption
Figure 3: Increased pretraining steps reveals the same task diversity threshold for the emergence of ICL. Columns 1 and 2 in this figure are similar to the middle column in Fig. 2 and columns 3 and 4 correspond to the right column in Fig. 2, except here we increase the number of sequences per task by increasing the number of training steps while keeping batch size = 256. Both methods of increasing dataset size—increasing batch size in Fig. 2 and increasing training steps in this figure—reveal a transition in the behavior of the PT: beyond the task diversity threshold, ICL on new tasks emerges.
3.1Task diversity threshold for the emergence of in-context learning
For Fig. 2, we pretrain our base transformer on datasets with increasing task diversity (on the x-axis) while keeping the total number of sequences seen during pretraining fixed (
B
=
256
,
N
=
500
​
K
). We evaluate the PTs and both optimal estimators on tasks seen during pretraining drawn from 
𝒯
Pretrain
 (Fig. 2 top left) and on new tasks drawn from 
𝒯
True
 (Fig. 2 bottom left) and plot MSE normalized by task dimension—
L
𝒯
/
D
 from Eq. 1). Since dMMSE is optimal on tasks from 
𝒯
Pretrain
 (as discussed in Section 2), the green dMMSE markers denote the lowest possible loss the PT could achieve in this setting. In fact, the pretraining objective 
L
𝒯
Pretrain
 explicitly encourages the PT to match dMMSE performance. On the other hand, Ridge is optimal on tasks sampled from 
𝒯
True
 (Fig. 2 bottom left); the blue markers denote the lowest possible MSE the PT could attain on new tasks.

Low task diversity phase: the PT is Bayesian with respect to the pretraining distribution and cannot solve new tasks. At low pretraining task diversity—
M
 up to about 
2
6
—the PT’s MSE closely tracks that of dMMSE on tasks sampled from 
𝒯
Pretrain
 (Fig. 2 top left); the PT performs optimally on tasks seen during pretraining. But it significantly underperforms on new tasks sampled from 
𝒯
True
, indicated by the gap in MSE between the PT and Ridge (Fig. 2, bottom left). In this regime, it behaves like the Bayesian estimator with prior 
𝒯
Pretrain
.

High task diversity phase: the PT is non-Bayesian with respect to the pretraining task distribution and can solve new tasks. At higher task diversities—above 
2
6
 pretraining tasks—the PT’s MSE deviates from dMMSE and approaches Ridge under both 
𝒯
Pretrain
 and 
𝒯
True
. Crucially, the PT starts to significantly outperform dMMSE on unseen tasks sampled from 
𝒯
True
 (Fig. 2 bottom left) at the expense of not fully minimizing its training objective, 
L
𝒯
Pretrain
 (gap between PT and dMMSE under 
𝒯
Pretrain
, Fig. 2 top left). This suggests that, a PT trained on a finite but large number of pretraining tasks can learn fundamentally new tasks in-context and this ability depends on it deviating from the optimal Bayesian estimator on the pretraining task distribution.

Refer to caption
Refer to caption
Figure 4:Learning dynamics of small PTs shows a transition at the task diversity threshold. We plot 
Δ
PT,Ridge
𝒯
Pretrain
 vs training steps for small PTs. For the same 
M
, learning curves for short (500K steps, left) or long (2M steps, center) training durations are similar, and for 
M
>
M
∗
≈
2
11.5
 learning curves are similar to that of a model trained with infinite task diversity. Right: For 
M
≤
2
10
, 
t
∗
 (the training step at which 
Δ
PT,Ridge
𝒯
Pretrain
 is minimized) is well modeled by a scaling law 
t
∗
∝
M
α
. A linear fit of 
log
⁡
t
∗
 vs 
log
⁡
M
 (dashed red line) gives 
α
≈
0.47
. But for 
M
>
2
10
, 
Δ
PT,Ridge
𝒯
Pretrain
 decreases through training; 
t
∗
=
 2M, is larger than predicted by the scaling law. This sudden break in the scaling law suggests a fundamental difference in the learning dynamics of models on either side of the threshold.
Finite size scaling of training data suggests an algorithmic phase transition as task-diversity increases. The experiments in Fig. 2 (left column) suggest that, when tested on both task distributions 
𝒯
Pretrain
 and 
𝒯
True
, the ICL algorithm implemented by a PT exhibits a smooth crossover in performance from dMMSE to Ridge. We next examine how this transition changes as we increase the number of sequences per task seen over pretraining, at fixed task diversity. One might reasonably expect that, if the transformer sees more sequences per latent task in 
𝒯
Pretrain
, both its predictions and performance should become more similar to those of dMMSE, and less similar to those of Ridge, at all values of task diversity. Strikingly, this natural expectation is violated in a manner that facilitates ICL on 
𝒯
True
.

At each number of tasks, we increase the number of sequences per task by increasing batch size from 256 to 512 to 1024, while leaving the number of training steps fixed at 500K. We observe that 
Δ
PT, dMMSE
𝒯
Pretrain
, which quantifies how different the PT and dMMSE estimator’s predictions are when testing on tasks drawn from 
𝒯
Pretrain
, does in fact decrease for 
M
≤
2
10
 (Fig. 2 top center) as we train on more sequences per task. Moreover, for each 
M
∈
{
2
10
,
…
,
2
14
}
 the PT’s predictions also become less similar to those of Ridge, both on tasks from 
𝒯
Pretrain
 (Fig. 2, top right) and 
𝒯
True
 (Fig. 2, bottom right). Crucially, this movement in behavior of the PT towards dMMSE and away from Ridge, at least on tasks drawn from 
𝒯
Pretrain
, holds only up to a threshold number of tasks between 
2
14
 and 
2
15
. Beyond this threshold, pretraining on more sequences per task at a fixed task diversity actually makes the PT more like Ridge, in that both 
Δ
PT,Ridge
𝒯
Pretrain
 and 
Δ
PT,Ridge
𝒯
True
 decrease (Fig. 2, right top and right bottom respectively). This means that, beyond a task diversity threshold, the PT can not only optimally solve new tasks from 
𝒯
True
 by matching Ridge performance, but also the PT gets better at doing so if trained on more sequences per task, despite the limited set of tasks experienced in pretraining. Thus, in contrast to the natural expectation stated above, more sequences per task does not promote overspecialization of the PT to the 
𝒯
Pretrain
 at task diversities beyond the threshold.

Finally, the motion of the ICL algorithm implemented by PT towards (away) from Ridge above (below) a task diversity threshold (Fig. 2, right top and bottom) indicates that as one increases the number of sequences per task at fixed task diversity, the smooth cross over in performance of the PT between dMMSE and Ridge, shown in Fig. 2, left top and bottom, will become sharper and sharper in task diversity, ultimately exhibiting a sharp phase transition in the limit of infinite number of sequences per task. Remarkably, this phase transition in the ICL algorithm implemented by the PT appears at a moderate task diversity threshold below 
2
15
 pretraining tasks; even though dMMSE significantly underperforms relative to Ridge on 
𝒯
True
 at this task diversity, the PT nevertheless remains unimpaired by this limited task diversity and can optimally solve new tasks.

Increased training time at fixed batch size further supports an algorithmic phase transition. To confirm the above results, we also increase the number of sequences per task, at each task diversity, by increasing the number of training steps 
N
 from 500K to 1M while keeping batch size fixed at 256. We observe that doubling 
N
 (change from pale blue to red in Fig. 3) and doubling 
B
 (change from pale blue to red in Fig. 2) have very similar effects on 
Δ
PT,dMMSE
𝒯
 and 
Δ
PT,Ridge
𝒯
, for both 
𝒯
=
𝒯
True
 and 
𝒯
=
𝒯
Pretrain
. More importantly, the task diversity threshold, which we determined as the cross-over point in 
Δ
PT,Ridge
𝒯
True
 between batch sizes 256, 512, and 1024 at 500K training steps (Fig. 2 bottom right) happens at the same number of tasks as the crossover point between 500K and 1M steps at batch size 256 (Fig. 3, right). Given that our two approaches for training the baseline transformer on more data yield the same task diversity threshold, and that doubling batch size leads to significantly faster training times than doubling number of steps, from here onward we consider the task diversity threshold to be cross-over point in 
Δ
PT,Ridge
𝒯
True
 between batch sizes 256 and 512 when training for 500K steps. See Appendix D for more ablations of batch size and training steps that provide further evidence for how the number of sequences seen by the transformer is the key factor determining the similarity of its predictions to those of dMMSE and Ridge at each number of tasks.

Refer to caption
Figure 5: Transformers pretrained with high, but not low, task diversity can learn new tasks in-context. We compare the normalized loss of the PT to that of dMMSE and Ridge as we interpolate between tasks in the pretraining dataset. Left: At 
2
5
 tasks, well below the task diversity threshold, the PT performance matches that of the dMMSE estimator along interpolating paths, but under-performs Ridge on new tasks at the center. Middle: At 
2
10
 tasks, the PT outperforms dMMSE on new tasks at the center of the interpolation path, but is not yet as good as Ridge on new tasks. Right: At 
M
=
2
15
 tasks, just above the task diversity threshold, the PT performs as well as Ridge even on new tasks at the center. This demonstrates that, when pretrained on data with a finite but large number of unique tasks, the PT, unlike the Bayes optimal estimator for 
𝒯
Pretrain
, can learn new tasks in-context.
Learning dynamics and a break in the scaling of early stopping time further supports an algorithmic phase transition. To probe if the observed transition is merely an effect of under-fitting, we study the learning dynamics of small PTs in the very large number of steps regime. First, in Appendix E, we verify that the small PT also demonstrates an algorithmic phase transition but at lower task diversity threshold between 
2
11
 and 
2
12
 pretraining tasks. In Fig. 4 left, we visualize the learning curves (
Δ
PT,Ridge
𝒯
True
 vs training steps) of PTs trained for 500K steps at batch size 512 with pretraining task diversities, 
M
, below and above the task diversity threshold, 
M
∗
. For 
M
<
M
∗
, 
Δ
PT,Ridge
𝒯
True
 decreases early in training until it reaches a minimum at time step 
t
∗
, and then increases as the PT approaches dMMSE. We define 
t
∗
 as the early stopping time for Ridge. For 
M
>
M
∗
, 
Δ
PT,Ridge
𝒯
True
 decreases throughout training. To evaluate if, in the latter case, models are undertrained and 
t
∗
 is larger than the total training time, we extend training to 2M steps at batch size 512 (
4
×
 the training time, see Appendix B). Fig. 4 center, shows these learning curves along with that of the model trained with infinite task diversity; even in this long training regime, the task diversity threshold does not change. For both short and long training durations, models trained with the same 
M
 have similar qualitative behavior (whether distance to Ridge decreases then increases or monotonically decreases). Additionally, learning curves of the models with 
M
>
M
∗
 are very similar to the learning curves for models trained on infinite pretraining task diversities and they achieve similar final accuracy (dahed lines vs markers in Fig. 10), suggesting that these models are approaching the Ridge solution.

In Fig. 4 right, we study how 
t
∗
, scales with 
M
. For most 
M
<
M
∗
, 
t
∗
 obeys a simple scaling behavior 
t
∗
∝
M
α
, 
α
≈
0.47
. However, for 
M
>
2
10
, the distance to Ridge decreases monotonically through training and 
t
∗
=
 2M steps. Despite the caveat that our experiments are necessarily in the large but finite training step regime with a decayed learning rate schedule, this stark break in the scaling behavior of 
t
∗
 near the task diversity threshold suggests that the observed transition is not just caused by under-fitting but an underlying difference in the learning dynamics.

Refer to caption
Figure 6: The task diversity threshold increases with task dimension, and the PT’s ability to solve new tasks scales significantly better than dMMSE’s. We vary the dimension of the regression problem 
D
 (first three panels) while leaving the signal-to-noise ratio fixed. The task diversity threshold consistently increases with task dimension (gray shading denotes post threshold). At 
2
20
 tasks (right), the PT’s predictions are similar to those of Ridge at all 
D
 (orange 
Δ
PT,Ridge
𝒯
True
), whereas dMMSE grows progressively less similar to Ridge (blue 
Δ
Ridge,dMMSE
𝒯
True
).
The transition along interpolating paths. To obtain an additional description of the algorithmic transition in the PT from dMMSE to Ridge, we compute the ICL performance of the PT, and compare it to both dMMSE and Ridge, on a one parameter family of new tasks 
𝐰
α
 that interpolate between pairs of seen tasks 
𝐰
i
 and 
𝐰
j
 in the support of 
𝒯
Pretrain
. The interpolation path is given by

𝐰
α
=
1
2
​
(
‖
𝐰
i
‖
2
+
‖
𝐰
j
‖
2
)
​
α
​
𝐰
i
+
(
1
−
α
)
​
𝐰
j
‖
α
​
𝐰
i
+
(
1
−
α
)
​
𝐰
j
‖
2
 for 
​
α
∈
[
0
,
1
]
.
(5)
Here we fix the norm of the interpolated vector 
𝐰
α
 to the average of the two endpoint norms to avoid 
‖
𝐰
α
‖
 taking on very small values for 
α
∼
1
2
. Fig. 5 shows the results of this analysis for 
2
5
 (left, low task diversity regime), 
2
10
 (center, below task diversity threshold), and 
2
15
 (right, just above the task diversity threshold) tasks. At each value of 
α
, MSE is averaged over a large number of task pairs 
(
𝐰
i
,
𝐰
j
)
. Examination of the average performance at the center of many interpolation paths, corresponding to fundamentally new tasks far from tasks seen during pretraining, clearly reveals a transition in PT performance from dMMSE to Ridge, where new tasks can only be optimally learned above, but not below, the task diversity threshold. In contrast, unlike the PT, dMMSE cannot solve new tasks at any task diversity in the range considered.

The PT outperforms a smoothed dMMSE model. We have seen that at an intermediate task diversity the PT significantly outperforms dMMSE on new tasks in 
𝒯
True
. It is clear why dMMSE performs poorly on new tasks in 
𝒯
True
 at low task diversity: its prior over tasks concentrates on 
M
 unique tasks in 
𝒯
Pretrain
, while the prior over tasks in 
𝒯
True
 is Gaussian. A natural conjecture is that the PT cannot memorize all 
M
 tasks in 
𝒯
Pretrain
 for large enough 
M
. Therefore we also compare PT performance to a smoothed dMMSE estimator in which the discrete point prior over 
M
 tasks seen in pretraining is replaced with a mixture of 
M
 isotropic Gaussians with the same centers but with variance chosen to optimize performance on 
𝒯
True
 (see Appendix G for details). This smoothed dMMSE outperforms dMMSE as it has a prior over tasks closer to the Gaussian 
𝒯
True
. But remarkably, the PT still outperforms the smoothed dMMSE even with optimal smoothing (Fig. 12). This indicates the PT, even at moderate task diversity, implements a more sophisticated algorithm than a simple smoothed dMMSE arising from the PT’s inability to resolve the 
M
 pretraining tasks to high precision.

3.2The PT exhibits superior scaling of task diversity threshold with dimension than dMMSE.
We next explore the dependence of the task diversity threshold on the regression problem dimension 
D
. We vary 
D
∈
{
8
,
16
,
32
}
 while simultaneously scaling up maximal context length as 
K
=
2
​
D
, and increasing observation noise 
σ
2
 to match the SNR to that of 
D
=
8
 and 
σ
2
=
0.25
. We also train a larger model with 12 layers, 256-dimensional embeddings, and 4 attention heads that is sufficiently expressive to match Ridge performance at 
D
=
32
. Fig. 6, first 3 panels reveal that the task diversity threshold of the PT increases moderately (approximately linearly) with task dimension (i.e. roughly 
2
14
, 
2
15
, and 
2
16
 at 
D
=
8
,
16
, and 
32
 respectively). This linear scaling is remarkable considering the volume of all possible tasks scales exponentially with dimension due to the concentration of the Gaussian 
𝒯
True
 to a sphere for large 
D
. Thus we expect dMMSE performance to scale much more poorly with dimension 
D
 since the finite number of tasks in 
𝒯
Pretrain
 would need to cover a substantial portion of the sphere for dMMSE to approach Ridge. To test this hypothesis, for 
M
=
2
20
, which is the largest task diversity we consider, we explore how the similarity of PT and dMMSE predictions to Ridge on new tasks scales with 
D
 (Fig. 6, right panel). We see that 
Δ
dMMSE,Ridge
𝒯
True
 grows significantly as we increase 
D
, while remarkably 
Δ
PT,Ridge
𝒯
True
 is largely dimension independent. Overall this indicates that the scaling of PT error with dimension is vastly superior than that of dMMSE; PT remains near optimal and close to Ridge at all 
D
 for 
M
=
2
20
, while dMMSE departs from Ridge as 
D
 increases.

3.3Effect of Regularization and model capacity on the task diversity threshold.
We study the dependence of the task diversity threshold on various hyperparameters. First, adding explicit regularization in the form of weight decay (see Appendix B for details), and increasing its value over three orders of magnitude, consistently lowers the threshold task diversity (Fig. 7, left). Note however, the lower task diversity threshold also comes with worse performance (Figure 13, top). This suggests that various forms of implicit regularization could help drive the algorithmic transition in the PT without weight decay. We also explore the effect of model capacity on the task diversity threshold by either increasing the embedding dimension of both small and base PTs or increasing the depth of small PTs. Fig. 7 center shows that increasing embedding dimension over a reasonable range does not affect the task diversity threshold of base PT. However, for small PT, increasing either the embedding dimension (Fig. 7 center) or depth (Fig. 7 right) increases the task diversity threshold. Base PT has a much larger capacity then small PT and also has a larger threshold; we hypothesize that small PT is still in a regime where the threshold is sensitive to capacity while base PT is not. Together, these results suggest that model capacity plays an important role in the emergence of in-context learning: increasing capacity (up to a point) leads to an increase in the task-diversity threshold.

Refer to caption
Refer to caption
Refer to caption
Figure 7:Explicit regularization and model capacity affect the task diversity threshold. Increasing explicit regularization, in the form of weight decay, lowers the task diversity threshold in base PTs (left). Increasing embedding dimension, has no effect on the threshold for base PTs, but does increase the threshold for a small PT (middle). Increasing depth, while holding other hyperparameters fixed, increases the threshold for a small PT (right). See Figure 13 for plots of 
Δ
dMMSE,Ridge
𝒯
True
 vs 
M
.
4Related work
The Bayesian framework for ICL introduced by Xie et al. [3], which motivates our work, hypothesizes that PTs "locate" concepts learned during pretraining to solve ICL tasks. A series of empirical work in language models [18, 19, 20] use this framework to select better in-context examples while Min et al. [21] use it to study the robustness of latent task inference. Our work builds on this framework in the linear regression setting and validates it at low task diversities. However, we find a regime—large but finite number of pretraining tasks—in which the ability to learn new tasks in-context is an emergent phenomenon that cannot be fully explained by Bayesian inference.

Prior work [12, 13, 14] has also shown that transformers can do linear regression in-context. However, they pretrain with unlimited task diversity, sampling a completely new regression vector for each sequence. In contrast, our work considers pretraining datasets with limited task diversity where ICL on new tasks emerges even though the pretraining loss does not explicitly encode it. Another line of work hypothesizes that ICL performs gradient descent in the activations of the forward pass, providing explicit constructions for the weights of the PT to implement this for linear regression [13, 14] or exploring this hypothesis in language models [22]. However more experiments are required to test the hypothesis that trained transformers actually match proposed constructions. Instead of studying the explicit mechanism by which in-context learning is implemented, our work focuses on the impact of the pretraining task diversity. Similar questions pertaining to the role of task diversification have been explored in the meta-learning literature [23, 24, 25].

Kirsch et al. [26] also show the emergence of in-context learning with pretraining task diversity on a toy classification task. By studying this question in the controlled setting of linear regression, we can compare to the optimal estimators on 
𝒯
Pretrain
 and 
𝒯
True
. This allows us to establish that ICL at finite task diversity emerges because the PT departs from the optimal estimator on the pretraining task distribution, and is not just a consequence of the pretraining task distribution becoming similar to the ideal task distribution. Among other important perspectives on ICL, Chan et al. [27] identify, in a toy setting, several properties of the training distribution—burstiness and occurrence of rare classes—that are necessary for the emergence of ICL. Wei et al. [28] study how ICL in large language models is affected by semantic priors and input-label mappings, focusing on differences across model scale. Olsson et al. [29] study inductions heads—circuits responsible for completing patterns by copying tokens—as a mechanism for implementing ICL.

5Discussion
Overall, we have extensively explored the impact of pretraining task diversity on the emergence of in-context learning of fundamentally new tasks not seen during pretraining. We found several surprises by working in the controlled setting of linear regression, where we could compare the performance of the PT to Bayesian estimators that are optimal, either for the limited diversity pretraining task distribution 
𝒯
Pretrain
 (i.e. dMMSE), or for the diverse ideal task distribution 
𝒯
True
 (i.e. Ridge). These comparisons reveal an algorithmic phase transition in the PT from the former to the latter at an intermediate task diversity threshold; beyond this threshold, the PT solves fundamentally new tasks not seen during pretraining. Strikingly, this task diversity threshold scales moderately with task dimension, over the range of dimensions considered, despite the exponential growth in the volume of all possible tasks with dimension. Indeed this PT scaling vastly outperforms that of dMMSE. Overall, these results indicate that ICL of new tasks by PTs is an emergent phenomenon that cannot be explained by Bayesian inference on limited diversity pretraining task distributions. Moreover, our experiments suggest some form of implicit regularization in PTs allows them to break free of the pretraining task distribution to solve new tasks, given a moderate pretraining task diversity.

Remarkably, beyond the task diversity threshold, PTs learn the optimal estimator for the underlying generative model for pretraining tasks; this is the case for both Gaussian and Laplace priors over tasks (see Figure 14 for experiments with Laplace prior). This is true even though solutions with lower training loss exist; indeed when trained on more data at fixed diversity, PTs behave more like Ridge at the expense of higher training loss. Our experiments in Fig. 4 suggest that this algorithmic transition is due an underlying change in learning dynamics. We explore this hypothesis by probing the linear mode connectivity of the loss landscape [30, 31]. In Fig. 11 we find that PTs trained with large 
M
 inhabit the same loss basin as PTs trained with 
M
=
∞
: the training loss barrier between PTs trained with 
M
≳
2
13
 and PTs with 
M
=
∞
 is similar to two PTs trained with 
M
=
∞
. In contrast, there are large loss barriers between PTs trained with 
M
<
2
13
 and 
M
=
∞
. Additionally, PTs trained with 
M
=
∞
 are closer in weight space to PTs trained with large 
M
 than those trained with small 
M
 (see Appendix F). Overall, these experiments provide further evidence that PTs trained with task diversities beyond the threshold find solutions similar to the optimal model for 
𝒯
True
; we leave further exploration of these loss landscapes to future work.

An intriguing question is how these observations carry over to language. A key mystery about the efficacy of ICL in language tasks lies in how different the tasks learned in-context are from the pretraining distribution of large language corpora. It is also less clear how to categorize the contents of such corpora according to tasks and measure their resulting task diversity. Regardless, our observation in linear regression that a moderate threshold in pretraining task diversity can enable PTs to solve new tasks may imply that many language tasks that are quite different from the statistics of large language corpora can still nevertheless be solved in-context.

Our results also suggest that the scale of data alone does not lead to good ICL performance. In fact, below the task diversity threshold, increasing the size of the pretraining dataset without increasing task diversity hurts ICL performance. It is necessary to increase both the diversity and size of the dataset for ICL to emerge. Thus to improve ICL in language settings, our work motivates future studies into uncovering the relevant notion of tasks in language modeling and approaches to increase task diversity in language corpora. More generally, our empirical analysis of the impact of pretraining task diversity on ICL motivates further theoretical studies. Such studies will be key to understanding the mystery of why simple next token prediction during pretraining can lead to in-context learning of so many apparently different tasks.

Acknowledgements
The authors would like to thank the Google TPU Research Cloud (TRC) whose generous support made this project possible. SG thanks an NSF CAREER award and NTT Research for support.
```


Assistant: Here's a summary of the key contributions and findings from this paper:

Key Contributions:

1. The paper investigates how pretraining task diversity affects in-context learning (ICL) in transformers, specifically studying linear regression tasks.

2. It identifies a task diversity threshold for the emergence of ICL - below this threshold, transformers behave like Bayesian estimators limited to pretraining tasks, but above it, they can effectively learn new, unseen tasks.

3. The paper demonstrates that transformers can significantly outperform Bayesian estimators when pretrained on sufficiently diverse tasks, suggesting ICL cannot be fully explained by Bayesian inference.

Main Findings:

1. Task Diversity Threshold:
- Below the threshold: Transformers behave like Bayesian estimators with the pretraining distribution as prior
- Above the threshold: Transformers can optimally solve new tasks not seen during pretraining
- The threshold occurs at a moderate number of pretraining tasks (between 2^14 and 2^15)

2. Scaling Properties:
- The task diversity threshold scales approximately linearly with task dimension
- Transformer performance scales much better with dimension compared to Bayesian estimators
- Beyond the threshold, increasing dataset size improves performance on new tasks

3. Effects of Model Parameters:
- Increasing weight decay decreases the task diversity threshold
- Increasing model capacity (layers or embedding size) increases the threshold
- Model regularization plays an important role in the emergence of ICL

4. Learning Dynamics:
- The paper identifies a sharp algorithmic phase transition in transformer behavior at the threshold
- This transition is evident in both the model's performance and learning dynamics

Practical Implications:
- The findings suggest that both data scale AND task diversity are crucial for developing effective ICL
- The work provides insights into how to better design pretraining datasets to enable ICL
- The results may help explain why language models can perform ICL on tasks quite different from their pretraining distribution

This paper makes significant theoretical and empirical contributions to understanding how in-context learning emerges in transformers and what conditions are necessary for it to work effectively.

User: ok now could you do the same for this second paper (and set it in context with the first): 
```"The Developmental Landscape of In-Context Learning

 

Jesse Hoogland * 1  George Wang * 1  Matthew Farrugia-Roberts 2  Liam Carroll 2  Susan Wei 3  Daniel Murfet 3 

†
*Equal contribution 1Timaeus 2Independent 3School of Mathematics and Statistics, the University of Melbourne. Correspondence to: Jesse Hoogland <jesse@timaeus.co>.  
Under review.
Abstract
We show that in-context learning emerges in transformers in discrete developmental stages, when they are trained on either language modeling or linear regression tasks. We introduce two methods for detecting the milestones that separate these stages, by probing the geometry of the population loss in both parameter space and function space. We study the stages revealed by these new methods using a range of behavioral and structural metrics to establish their validity.

1Introduction
The configuration space of a cell is high-dimensional, the dynamics governing its development are complex, and no two cells are exactly alike. Nevertheless, cells of the same type develop in the same way: their development can be divided into recognizable stages ending with milestones common to all cells of the same type (Gilbert, 2006). In biology a popular visual metaphor for this dynamic is Waddington’s (1957) developmental landscape, in which milestones correspond to geometric structures that each cell must navigate on its way from a pluripotent form at the top of the landscape to its final functional form at the bottom.

The configuration space of a modern neural network is high-dimensional, the dynamics governing its training are complex, and due to randomness in initialization and the selection of minibatches for training, no two neural networks share exactly the same training trajectory. It is natural to wonder if there is an analogue of the developmental landscape for neural networks. More precisely, we can ask if there is an emergent logic which governs the development of these systems, and whether this logic can be understood in terms of developmental milestones and the stages that separate them. If such milestones exist, we can further ask how they relate to the geometry of the population loss.

We examine these questions for transformers trained in two settings: language modeling, in which a transformer with around 3M parameters is trained on a subset of the Pile (Gao et al., 2020) and linear regression, in which a transformer with around 50k parameters is trained on linear regression tasks following Garg et al. (2022). These settings are chosen because transformers learn to do In-Context Learning (ICL) in both. The emergence of ICL during training (Olsson et al., 2022) is the most conspicuous example of structural development in modern deep learning: a natural starting place for an investigation of the developmental landscape.

We introduce two methods for discovering stages of development in neural network training, both of which are based on analyzing the geometry of the population loss via statistics. The first method is the local learning coefficient (Lau et al., 2023) of Singular Learning Theory (SLT; Watanabe, 2009), which is a measure of the degeneracy of the local geometry of the population loss in parameter space. The second is trajectory PCA, referred to here as essential dynamics (Amadei et al., 1993), which analyzes the geometry of the developmental trajectory in function space. These methods are motivated by the role of geometry in SLT and the connection between milestones in the Waddington landscape and singularities introduced by Thom (1988).

Our methods reveal that, in both the language and linear regression settings, training can be divided into discrete developmental stages according to near plateaus in the local learning coefficient (see Figure 1) and geometric structures in the essential dynamics (see Figure 2).

Our main contribution is to introduce these two methods for discovering stages, and to validate that the stages they discover are genuine by conducting extensive setting-specific analyses to determine the behavioral and structural changes taking place within each stage.

We also initiate the study of forms of the developmental trajectory, which are remarkable geometric objects in function space that we discover occurring at (some) milestones. These forms seem to govern the large scale development of transformers in our two settings. We view the appearance of these structures as evidence in favor of the idea of a simple macroscopic developmental landscape in the sense of Waddington, which governs the microscopic details of the neural network training process.

Refer to caption
(a)Two-Layer Attention-only Language Transformer.
Refer to caption
(b)Linear Regression Transformers.
Figure 1:We train transformer models on both (a) natural language data (1 seed) and (b) synthetic in-context regression data (error bars over 5 seeds). In addition to test loss (top row), we track loss landscape degeneracy as quantified by the local learning coefficient (LLC) (bottom row, see Section 3.1). An automated procedure finds plateaus in the LLC, which mark boundaries between distinct developmental stages (ranges of training time indicated with different colors: orange hues for increasing LLC, blue for decreasing). These boundaries are developmental milestones, where certain structures and associated behaviors complete their formation.
2In-Context Learning
In-Context Learning (ICL), the ability to improve performance on new tasks without changing weights, underlies much of the success of modern transformers Brown et al. (2020). This makes a deeper understanding of ICL necessary to interpreting how and why transformers work. Additionally, ICL is perhaps the clearest example of an “emergent ability” Wei et al. (2022), that appears suddenly as a function of both model depth and training time Olsson et al. (2022). Taken together, this makes ICL uniquely well-motivated as a case study for investigating the learning process of transformers.

Let 
f
w
 be a transformer with parameters 
w
, which takes a context 
S
≤
k
=
(
t
1
,
…
,
t
k
)
 as input, where 
k
≤
K
. Let 
f
w
​
(
S
≤
k
)
 be the output logits at token 
t
k
, and let 
n
 be the number of length-
K
 sequences in the training data 
{
S
K
i
}
i
=
1
n
.

Given a per-token loss 
ℓ
n
,
k
 (cross-entropy for language or MSE for linear regression, see Sections 2.1 and 2.2 respectively), the training loss is

ℓ
n
​
(
w
)
=
1
K
​
∑
k
=
1
K
ℓ
n
,
k
​
(
w
)
,
(1)
with the test loss 
ℓ
^
 defined analogously on a held-out set of examples.

In general, we say that the trained transformer 
f
w
 performs ICL if the per-token loss 
ℓ
n
,
k
​
(
w
)
 improves consistently with 
k
 on an independent test set. More directly, we follow the experimental setup of Olsson et al. (2022) and, dropping the 
n
 index, define the ICL score as (see Section C.2.3)

ICL
k
1
:
k
2
⁡
(
w
)
=
ℓ
^
n
,
k
1
​
(
w
)
−
ℓ
^
n
,
k
2
​
(
w
)
.
(2)
Rather than studying ICL itself, our aim is to study how models develop the ability to perform ICL over the course of training, which we quantify by tracking the ICL score across checkpoints.

2.1In-Context Learning in Language Models
Following Elhage et al. (2021) and Olsson et al. (2022), we study ICL in autoregressive attention-only (no MLP layers) transformers (details in Section E.1). We consider the standard task of next-token prediction for sequences of tokens taken from a subset of the Pile (Gao et al., 2020; Xie et al., 2023). At the token level, the sample training loss for 
k
>
1
 is the cross-entropy, given by

ℓ
n
,
k
​
(
w
)
=
−
1
n
​
∑
i
=
1
n
log
⁡
(
softmax
​
(
f
w
​
(
S
≤
k
i
)
)
​
[
t
k
+
1
i
]
)
.
(3)
2.2In-Context Learning of Linear Functions
Following the framework of Garg et al. (2022), a number of recent works have explored the phenomenon of ICL in the stylized setting of learning simple function classes, such as linear functions. This setting is of interest because we have a precise understanding of theoretically optimal ICL, and because simple transformers are capable of good ICL performance in practice.

In the following we give a standard presentation on ICL of linear functions (full training details in Section E.2). A task is a vector 
\task
∈
ℝ
D
. Given a task 
\task
, we generate 
x
i
∈
ℝ
D
,
y
i
∈
ℝ
 iid for 
i
=
1
,
…
,
K
 according to the joint distribution 
q
​
(
x
,
y
|
\task
)
=
q
​
(
y
|
x
,
\task
)
​
q
​
(
x
)
, resulting in the following context,

S
K
=
(
x
1
,
y
1
,
…
,
x
K
−
1
,
y
K
−
1
,
x
K
)
,
with label 
y
K
. We limit our study to the setting where 
q
​
(
y
|
x
,
\task
)
=
𝒩
​
(
\task
T
​
x
,
σ
2
)
, 
q
​
(
x
)
=
𝒩
​
(
0
,
I
D
)
, and task distribution 
q
​
(
\task
)
=
𝒩
​
(
0
,
I
D
)
.

Consider a training dataset 
{
(
\task
​
[
i
]
,
S
K
i
,
y
K
i
)
}
i
=
1
n
 which consists of 
n
 iid samples drawn from 
q
​
(
y
|
x
,
\task
)
​
q
​
(
x
)
​
q
​
(
\task
)
. Upon running a context through the transformer, we obtain a prediction 
y
^
k
i
=
f
w
​
(
S
≤
k
i
)
 for each subsequence 
S
≤
k
i
, which leads to the per-token training loss

ℓ
n
,
k
​
(
w
)
=
1
n
​
∑
i
=
1
n
(
y
^
k
i
−
y
k
i
)
2
.
(4)
3Methodology
It might seem hopeless to extract geometrical information about a high-dimensional dynamical system like neural network training. But this is not so. It is well-understood that the local geometry of a potential governing a gradient system can have global effects; this has been the basis of deep connections between geometry and topology over the last century (Freed, 2021). In SLT we have a similar principle: the local geometry of the log likelihood has a strong influence on statistical observables, and therefore information about this geometry may be learned from samples.1
1
The main example is the learning coefficient, which is a natural quantity in Bayesian statistics shown by Watanabe to be equal to the Real Log Canonical Threshold (RLCT) of the log likelihood function Watanabe (2009), an invariant from algebraic geometry.

We use two probes of the geometry of the population loss to study development: the Local Learning Coefficient (LLC) and Essential Dynamics (ED). These indicators are applied to snapshots gathered during training to identify milestones that separate developmental stages. Once identified, we perform a targeted analysis of associated behavioral changes, meaning changes in the input-output behavior of the neural network, and structural changes, meaning changes in the way that the neural network computes internally.

3.1The Local Learning Coefficient
In Bayesian statistics given a model 
p
, true distribution 
q
, and parameter 
w
∗
, the Local Learning Coefficient (LLC), denoted 
λ
​
(
w
∗
)
, is a positive scalar which measures the degeneracy of the geometry of the population log likelihood function 
L
 near 
w
∗
. The more degenerate the geometry, the closer to zero the LLC will be.2
2
This quantity has some relation to the idea of counting effective parameters, but in general is not best thought of as counting anything discrete; see Section A.1 for some examples.
 The LLC is also a measure of model complexity: the larger the LLC, the more complex the model parametrized by 
w
∗
. The learning coefficient and its connection to geometry was introduced by Watanabe (2009) and the local setting was studied by Lau et al. (2023).

Conditions guaranteeing the existence and well-definedness of the theoretical LLC can be found in Lau et al. (2023). For our purposes here, it suffices to explain the LLC estimator proposed there.

The original LLC estimator requires the specification of a log likelihood over iid data, to which we can associate an empirical 
L
n
​
(
w
)
=
−
1
n
​
∑
i
=
1
n
log
⁡
p
​
(
z
i
|
w
)
 and a population 
L
​
(
w
)
=
−
E
z
​
log
⁡
p
​
(
z
|
w
)
. The LLC estimate 
λ
^
​
(
w
∗
)
, where 
w
∗
 is a local minimum of 
L
​
(
w
)
, is

λ
^
​
(
w
∗
)
=
n
​
β
​
[
𝔼
w
|
w
∗
,
γ
β
​
[
L
n
​
(
w
)
]
−
L
n
​
(
w
∗
)
]
,
with 
β
=
1
/
log
⁡
n
. The expectation is with respect to a tempered posterior distribution localized at 
w
∗
, i.e.,

𝔼
w
|
w
∗
,
γ
β
​
[
f
​
(
w
)
]
=
∫
f
​
(
w
)
​
p
​
(
w
;
w
∗
,
β
,
γ
,
z
1
:
n
)
​
𝑑
w
where

p
​
(
w
;
w
∗
,
β
,
γ
,
z
1
:
n
)
∝
exp
⁡
{
−
n
​
β
​
L
n
​
(
w
)
−
γ
​
‖
w
−
w
∗
‖
2
2
}
.
In this work, we generalize the usage of the original LLC estimator from likelihood-based3
3
In Section A.3 we describe how to properly define log likelihoods for the two settings introduced in Section 2.
 to loss-based. Given an average empirical loss 
ℓ
n
​
(
w
)
 over model parameters 
w
, the generalized LLC estimate 
λ
^
​
(
w
∗
)
, where 
w
∗
 is a local minimum of the corresponding population loss 
ℓ
​
(
w
)
, is

λ
^
​
(
w
∗
)
=
n
​
β
​
[
𝔼
w
|
w
∗
,
γ
β
​
[
ℓ
n
​
(
w
)
]
−
ℓ
n
​
(
w
∗
)
]
,
(5)
where now the expectation is with respect to the loss-based posterior (also known as a Gibbs posterior) given by

p
​
(
w
;
w
∗
,
β
,
γ
,
z
1
:
n
)
∝
exp
⁡
{
−
n
​
β
​
ℓ
n
​
(
w
)
−
γ
2
​
‖
w
−
w
∗
‖
2
2
}
.
(6)
where 
β
 is an inverse temperature which controls the contribution of the loss, 
γ
 is the localization strength which controls proximity to 
w
∗
.

Refer to caption
(a)Two-Layer Attention-only Language Transformer.
Refer to caption
(b)Linear Regression Transformers.
Figure 2: The training trajectory in a subspace of function space projected onto its top three principal components and smoothed, with osculating circles (gray). The evolute, the set of centers of osculating circles, is shown in black. Accentuated families of osculating circles correspond to cusps in the evolute. The corresponding point on the curve (where the evolute “points”) is called a vertex. If a cusp singularity or vertex occurs at the same time in all plots, we infer the presence of a form. (a) Forms 
1
,
2
 at the LM2–LM3, LM4–LM5 boundaries, respectively. Explained variances 
0.56
,
0.15
,
0.07
. (b) Forms 
1
, 
2
, and 
3
 respectively at the LR1–LR2, LR2–LR3, LR3–LR4 boundaries. Explained variances 
0.65
,
0.08
,
0.05
.
There are various important factors in the implementation of LLC estimation. The most important is the sampling method used to approximate the expectation over the localized posterior in 6. We employ SGLD and detail diagnostics for tuning its hyperparameters in Appendices A and E.3.

3.2Essential Dynamics
The parameter space of modern neural networks is usually high-dimensional. Therefore, to study the development of structure in neural networks over training, it is natural to apply methods of dimensionality reduction. Similar problems of data analysis are faced in many sciences, and the application of PCA to trajectories of systems undergoing development is common in neuroscience (Briggman et al., 2005; Cunningham & Yu, 2014) and molecular dynamics (Amadei et al., 1993; Meyer et al., 2006; Hayward & De Groot, 2008).

Following the terminology in molecular dynamics we refer our version of this method as Essential Dynamics (ED).

Suppose a stochastic process 
a
t
∈
ℝ
s
 is a function of the weight parameter 
w
t
 (and possibly other random variables) at a step 
t
 in training. Let 
A
t
1
,
…
,
A
t
T
 be samples from the stochastic process at steps 
t
1
,
…
,
t
T
. Let 
A
 denote the 
s
×
T
 data matrix with columns 
A
t
i
. We apply PCA to the matrix 
A
, that is, we compute the top 
v
 eigenvectors 
u
1
,
…
,
u
v
 of the (sample) covariance matrix 
A
​
A
T
 (the Principal Components or PCs). Let 
V
⊆
ℝ
s
 denote the subspace spanned by the PCs and 
π
 the orthogonal projection onto 
V
≅
ℝ
v
.

By the developmental trajectory we mean a curve 
γ
:
ℝ
→
V
 which approximates 
t
↦
π
​
(
a
t
)
. In practice we produce this curve by smoothing the datapoints (see Section B.8). By essential dynamics we mean the study of 
a
t
 by means of the geometry of this curve 
γ
 and its projections onto two-dimensional subspaces of 
V
 spanned by pairs of PCs.

We focus on what we call behavioral ED where 
a
t
 is a finite-dimensional projection of the neural network function. Let 
f
w
​
(
x
)
∈
ℝ
m
 denote the output of a neural network with parameter 
w
 and given input samples 
I
=
{
x
i
}
i
=
1
n
 define

a
t
=
(
f
w
t
​
(
x
i
)
)
i
=
1
n
∈
(
ℝ
m
)
I
≅
ℝ
m
​
n
.
(7)
Ideally, the 
v
 features discovered by ED would provide a low-dimensional interpretable presentation of the development of the system, just as in thermodynamics a small number of macroscopic observables are a sufficient description of heat engines Callen (1998). In practice, however, the principal components may not be directly interpretable.

Nonetheless, just as the time evolution of systems in thermodynamics is sometimes punctuated by sudden changes (phase transitions) so too we may see in ED signatures of changes in the mode of development.

3.2.1Geometric Features
We can infer geometric features of the developmental trajectory in function space from its two-dimensional projections. Given a pair 
u
i
,
u
j
 of PCs let 
π
i
,
j
​
(
f
)
=
(
u
i
T
​
f
,
u
j
T
​
f
)
 be the projection and 
γ
i
,
j
​
(
t
)
=
π
i
,
j
​
(
γ
​
(
t
)
)
 the corresponding plane curve. We are interested in singularities and vertices of these plane curves (Jimenez Rodriguez, 2018, §5.4).

A cusp singularity of 
γ
i
,
j
 occurs at 
t
=
t
0
 if 
γ
i
,
j
′
​
(
t
0
)
=
γ
i
,
j
′′
​
(
t
0
)
=
0
. At such a point the curve may make a sudden turn. For example, there is a cusp singularity at the LR2–LR3 boundary in 
γ
1
,
2
 in Figure 2(b).

A vertex of 
γ
i
,
j
 occurs at 
t
=
t
0
 if the curve behaves like it is in “orbit” around a fixed point 
P
. Technically, we require that 
γ
i
,
j
 has contact of order 
4
 with the squared-distance function from 
P
 (see Section B.1). To discover vertices in practice, we plot osculating circles for the 
γ
i
,
j
 and look for circles that are “accentuated”. These occur when many nearby points have nearly the same osculating circle. More systematically, we study the evolute which is the curve consisting of all centers of osculate circles to 
γ
i
,
j
. Cusps of the evolute correspond to vertices of the original curve (Bruce & Giblin, 1992, Prop 7.2).

A form of the developmental trajectory is a timestep 
t
=
t
0
 and function 
f
α
∗
∈
V
 such that for all pairs 
i
<
j
 and at 
t
=
t
0
 the projection 
π
i
,
j
​
(
f
α
∗
)
 is either a cusp singularity of 
γ
i
,
j
 or is at the center of the osculating circle for a vertex of 
γ
i
,
j
. Thus the form4
4
Our usage of this term follows Thom (1988), who used it to refer to critical points. We use it in a more general sense.
 is the unifying object in function space for simultaneous geometric phenomena across the PC plots. We explain this definition further in Section B.3.

For example, in Figure 2 at the LR2–LR3 boundary there is a cusp singularity in 
γ
1
,
2
 and vertices in 
γ
1
,
3
,
γ
2
,
3
. We therefore identify this as a form of the linear regression transformer development (see Table 2).

Table 1:Stages for Language Model.
Stage	End	Type	
Δ
​
ℓ
^
Δ
​
λ
^
Forms
LM1	900	A	
−
2.33
+
26.4
-
LM2	6.5k	A	
−
1.22
+
22.5
7k
LM3	8.5k	B	
−
0.18
−
1.57
-
LM4	17k	A	
−
0.40
+
8.62
17.7k
LM5	50k	A	
−
0.34
+
1.77
-
3.3Using LLC and ED to Detect Milestones
Our methodology to detect milestones is to apply LLC estimation to a selection of snapshots gathered during training and look for critical points (that is, plateaus, where the first derivative vanishes) in the 
λ
^
​
(
w
t
)
 curve.5
5
Plotting the LLC estimates across time involves evaluating it at points that need not be local minima of the population loss, on which see Section A.6.
 These critical points are declared to be milestones. Subject to a few choices of hyperparameters (Section A.5), this process can be automated. Once identified, we classify stages as Type A if the LLC is increasing during the stage (the model becomes more complex), or Type B if the LLC is decreasing (the model becomes simpler), see Section A.2. The periods between milestones are defined to be the stages.

Table 2:Stages for Regression Transformer (seed=1).
Stage	End	Type	
Δ
​
ℓ
^
Δ
​
λ
^
Forms
LR1	1k	A	
−
0.32
+
21.4
2.5k
LR2	40k	A	
−
2.21
+
149
34.5k
LR3	126k	B	
−
0.07
−
12.3
106.1k
LR4	320k	B	
−
0.05
−
44.1
168.9k
LR5	500k	A	
−
0.029
+
3.56
-
Next, using ED we identify forms of the developmental trajectory. If these occur near the end of a stage we view this as further evidence for the correctness of the boundary.

In short, we use critical points of the LLC curve to find milestones and forms discovered by ED to corroborate the resulting stages. The forms may also offer a starting point to understand what is happening in a stage.

The justification for this methodology, in the present paper, is empirical: using a range of other behavioral and structural indicators we can show that the stages thus discovered make sense. The theoretical justification for this empirical observation, in the absence of critical points of the governing potential (in this case the population loss), remains unclear. Our working hypothesis is that subsets of the model are at critical points for subsets of the data distribution at the milestones we identify (see Appendix B.5 and the toy model of forms in Section B.4).

4Results for Language Modeling
Plateaus in LLC estimates (Figures 1(a) and 1) reveal five developmental stages. We associate these stages with the development of bigrams (LM1), 
n
-grams (LM2), previous-token heads (LM3), and induction heads (LM4; Olsson et al. (2022)). Forms of the ED (Figure 2(a) and Figure 6(a)) corroborate the LM2–LM3 and LM4–LM5 boundaries. There may be other interesting developmental changes: we do not claim this list is exhaustive. We did not, for example, discover significant changes in stage LM5.

The two forms identified in this setting appear to be quite interpretable, and consistent with the description of the stages given below; an analysis can be found in Section C.1.3.

4.1Stage LM1 (0–900 steps)
Behavioral changes.
The model learns bigram statistics, which is optimal for single-token prediction. Figure 3 (top) shows that the average cross entropy between logits and empirical bigram frequencies (see Section C.2.1) is minimized at the LM1–LM2 boundary, with a value only 
.3
 nats above the entropy of the empirical bigram distribution.

Refer to caption
Figure 3:The language transformer learns bigram statistics in LM1 (top). At the start of LM2, the positional embedding suddenly becomes useful (middle), enabling behavioral changes such as the learning of common 
n
-grams (bottom).
4.2Stage LM2 (900–6.5k steps)
Behavioral changes.
A natural next step after bigrams are 
n
-grams, token sequences of length 
n
. We define an 
n
-gram score as the ratio of final-position token loss on (1) a baseline set of samples from a validation set truncated to 
n
 tokens and (2) a fixed set of common 
n
-grams (see Section C.2.2). We see a large improvement in 
n
-gram score for 
n
=
3
,
4
 in Figure 3 (bottom), rising to several times the baseline ratio (
1.0
). Although this is one natural next step for the learning process, we do not rule out other possible developmental changes for this stage, such as skip-trigrams.

Structural changes.
The positional embedding is necessary for learning 
n
-grams, and, as expected, the model becomes dependent on the positional embedding during LM2. This is apparent in comparing the test loss with and without the positional embedding zero-ablated in Figure 3 (middle) — the curves are indistinguishable at first but diverge at the LM1–LM2 boundary (see Section C.3.1). We also see a rise in previous-token attention among second layer attention heads in the background of Figure 4 (top), which we also suspect plays a role with 
n
-grams.

Interestingly, even before the heads that eventually become induction heads develop their characteristic attention patterns in stages LM3 and LM4, they begin to compose (that is, read and write from the same residual stream subspace) near the start of stage LM2 (see Figures 20 and C.3.2). This suggests that the foundations of the induction circuit model are laid well in advance of any measurable change in model outputs or attention activations.

4.3Stage LM3 (6.5k–8.5k steps)
Structural changes.
First-layer previous-token heads, the first half of induction circuits, begin to form Elhage et al. (2021). Figure 4 (top) shows that the fraction these heads (highlighted in blue) attend to the immediately preceding token begins to increase during this stage (see Section C.3.3).

During this stage the LLC decreases, suggesting an increase in degeneracy and decrease in model complexity, perhaps related to the interaction between heads. It would be interesting to study this further via mechanistic interpretability.

Refer to caption
Figure 4:Induction circuit formation begins with previous-token heads (LM3, top), followed by induction heads (LM4, middle), which leads to a drop in ICL score (LM4, bottom). The 
h
th attention head in layer 
l
 is indexed as 
l
:
h
.
4.4Stage LM4 (8.5k–17k steps)
Behavioral changes.
The model learns to perform ICL as studied by Olsson et al. (2022) (Figure 4 bottom).

Structural changes.
The second half of the induction circuits, second-layer induction heads, begin to develop. Given a sequence 
[
A
]
​
[
B
]
​
…
​
[
A
]
, the prefix-matching score of Olsson et al. (2022) measures attention to 
[
B
]
 from the latter 
[
A
]
 (see Section C.3.4). Figure 4 (middle) shows that the prefix-matching score begins to increase for the two heads that become induction heads (highlighted in blue).

5Results for Linear Regression
In the linear regression setting, plateaus in the LLC estimate (Figures 1(b) and 2) reveal five developmental stages, corresponding to learning the task prior (LR1), acquiring in-context learning (LR2), and “over-fitting” to the input distribution (LR3/LR4). Forms in the ED (Figure 2(b) and Figure 6(b)) corroborate the LR1–LR2, LR2–LR3 and LR3–LR4 boundaries. Near-plateaus in the LLC, an additional form in the ED and other metrics suggest that LR2, LR3, and LR4 can be divided into additional substages (Section D.1.4).

The positions of these milestones vary slightly between runs, but the overall ordering is preserved. In this section, we present results for the regression transformer trained from seed 
1
 (“
\lrt
​
1
”). For other training runs, see Appendix F.

5.1Stage LR1 (0–1k steps)
Behavioral changes.
Similar to bigrams in the language model setting, the model learns the optimal context-independent algorithm, which is to predict using the average task 
\task
¯
, which is zero for our regression setup. Figure 5 shows that the average square prediction for all tokens 
𝔼
​
[
‖
y
^
k
‖
2
]
 approaches zero during LR1, reaching a minimum slightly after the end of LR1of 
0.1
 (which is smaller than the target noise 
σ
2
=
0.125
).

5.2Stage LR2 (1k–40k steps)
Behavioral changes.
The model acquires in-context learning during this stage (Figure 5 bottom, with 
g
=
1
). This parallels stage LM4 in the the language model setting.

Structural changes.
The token and positional embedding begin to “collapse” towards the end of this stage, effectively losing singular values and aligning with the same activation subspace (Section D.3.1). At the same time, several attention heads form distinct and consistent patterns (Section D.3.5).

5.3Stages LR3 & LR4 (40k–126k & 126k–320k steps)
Behavioral changes.
The model begins to “overfit” to the input distribution. Performance continues to improve on typical samples, but deteriorates on extreme samples for which the norm of the inputs 
x
k
 is larger than encountered in during training.

Structural changes.
During these stages, the layer norm weights undergo a phenomenon we term layer norm collapse, in which a large fraction of the layer norm weights rapidly go to zero (Section D.3.4). This phenomenon is most pronounced in the unembedding layer norm, where it occurs in tandem with a similar collapse in the weights of the unembedding matrix (Section D.3.3). This results in the model learning to read its final prediction from a handful of privileged dimensions of the residual stream. These observations, which point to concrete examples of degeneracy in the network parameter, may explain part of the observed LLC decrease over these stages.

Stage LR4 differs from LR3 in the layer norm collapse expanding from the unembedding to earlier layer norms, particularly the layer norm before the first attention block. This affects a smaller fraction of weights than the unembedding collapse.

Refer to caption
Figure 5:Top: During LR1, the model learns to predict with the task prior, 
x
k
↦
y
^
k
=
0
. Bottom: ICL emerges during LR2. In LR3, the model becomes worse at ICL on out-of-distribution inputs 
x
k
∼
𝒩
​
(
0
,
g
​
I
D
)
.
6Discussion and Related Work
Refer to caption
(a)Two-Layer Attention-only Language Transformer.
Refer to caption
(b)Linear Regression Transformer.
Figure 6: The PC scores over time from ED for each setting, with forms marked by dashed lines at the timesteps given in Table 1, Table 2. This shows the alignment between stage boundaries set by LLC and the forms of the developmental trajectory (and suggests a possible substage in LR4, see Section D.1.4). Note that local extrema of the PC scores occur near the timesteps associated with forms, see Section B.6.
Stage-wise development.
The study of stage-wise development in artificial neural networks spans several decades (Raijmakers et al., 1996). It has long been known that in simple (linear) neural networks, these stages can be linked to “modes” of the data distribution, meaning eigenvectors of covariance matrices (Baldi & Hornik, 1989; Rogers & McClelland, 2004; Saxe et al., 2019). In our experiments some of the early stages, such as learning bigrams (LM1), can be understood as learning modes in this sense. With the observation of emergent phenomena in neural network training (Wei et al., 2022; Olsson et al., 2022; McGrath et al., 2022) these ideas have attracted renewed attention.

Developmental biology and bifurcations.
We have emphasized changes in geometry over a stage whereas in developmental biology the focus, in the mathematical framework of bifurcation theory, is more on the singular geometry at milestones (Rand et al., 2021; MacArthur, 2022; Sáez et al., 2022). The relationship between these two points of view is beyond the scope of this paper. For more on the relation between the points of view of Waddington and Thom, see (Franceschelli, 2010).

Loss landscape geometry.
Many authors have used one or two-dimensional slices of the loss landscape to visualize its geometry (Erhan et al., 2010; Goodfellow et al., 2014; Lipton, 2016; Li et al., 2018; Notsawo et al., 2023). These approaches are limited by the fact that a random slice is with high probability a quadratic form associated to nonzero eigenvalues of the Hessian and is thus biased against important geometric features, such as degeneracy. The LLC and ED are able to probe such degenerate geometry in a quantitative way.

Trajectory PCA.
The use of trajectory PCA to study structural development in transformers was initiated by Olsson et al. (2022), who were the first to observe the vertex in 
γ
1
,
2
 which we show is part of the first form in the language modeling setting (due to their using a small number of checkpoints, what we observe to be a vertex appears as a sharp point there). This feature was observed in transformers ranging in scale up to 13B parameters, suggesting the ED method can be scaled substantially.

Progress measures.
Barak et al. (2022) show in a special setting that there are progress measures for neural network learning that reflect hidden continuous progress invisible to loss and error metrics, and which may precede phase transitions. The structural and behavioral indicators used here resemble the progress measures investigated by Nanda et al. (2023) in the context of reverse-engineering transformers trained on modular arithmetic. The LLC and ED differ in not requiring prior knowledge of what is changing during a stage. As we’ve demonstrated, the developmental and mechanistic approaches are complementary.

Algorithmic compression.
It is natural to interpret a stage in which the LLC decreases as a simplification or compression of an existing algorithm. Our observation of Type B stages is consistent with the literature on grokking (Nanda et al., 2023). In the linear regression setting the collapse of layer norm (Section D.3.4), embedding (Section D.3.1), and attention patterns (Section D.3.5) seem to be involved, but at present the evidence for this is not conclusive.

Universality.
Olah et al. (2020) hypothesize that similar internal structures may form across a wide range of different neural network architectures, training runs, and datasets. Examples of such preserved structures include Gabor filters, induction heads, and “backup heads” (Olah et al., 2020; McGrath et al., 2023; Wang et al., 2022). In our linear regression setting, we observe a stronger form of universality, in which the order of developmental stages appears to be preserved. The abstract structure which jointly organizes the macroscopic similarities between these processes is what we refer to as the developmental landscape, following Waddington (1957). Though we do not expect all aspects of development to be universal6
6
For example, layer norm collapse requires layer norm. In the case of language models, we expect the strength of the bigram stage to depend on the size of the tokenizer.
, we conjecture that the developmental trajectory will remain macroscopically simple even for much larger models.

7Conclusion
Transformers undergo stage-wise development.
We identified five developmental stages in the setting of transformers trained on natural language modeling and five in synthetic linear regression tasks. These stages are distinguished by the local learning coefficient, geometric features of the developmental trajectory in essential dynamics, and a range of other data- and model-specific metrics.

Stages are detected by the LLC and ED, which are scalable data- and model-agnostic developmental indicators.
Many stages are not visible in the loss. While a stage may be easily isolated once you know the right data-specific metric (e.g. the cross entropy with bigram frequencies for LM1) these post-hoc metrics depend on knowing that the stage exists and some of its content. In contrast, the LLC and ED can be used to discover developmental stages.

Future work: developmental interpretability.
Studying neural network development helps to interpret the structure of the final trained network, since we can observe the context in which that structure emerges. We expect that LLC and ED, applied over the course of training in the manner pioneered in this paper, will aid in discovering and interpreting structure in other neural networks.

Acknowledgments
We thank Edmund Lau for advice on local learning coefficient estimation and Mansheej Paul for advice on training transformers in the linear regression setting. We thank Evan Hubinger and Simon Pepin Lehalleur for helpful conversations. We thank Andres Campero, Zach Furman, and Simon Pepin Lehalleur for helpful feedback on the manuscript.

We thank Google’s TPU Research Cloud program for supporting some of our experiments with Cloud TPUs.

LC’s work was supported by Lightspeed Grants."
```

Assistant: Here's a comparative summary of both papers and how they relate to each other:

Key Connections:

1. Both papers study the emergence of in-context learning (ICL) in transformers, but from different perspectives:
- The first paper (Raventós et al.) focuses on how pretraining task diversity affects ICL emergence
- The second paper (Hoogland et al.) examines how ICL develops through distinct developmental stages during training

2. Complementary Findings:
- Raventós shows there's a threshold of task diversity needed for ICL to emerge
- Hoogland demonstrates that this emergence happens through discrete developmental stages, with specific milestones

Key Differences in Approach:

1. Methodology:
- Raventós: Focuses on comparing transformer behavior to optimal Bayesian estimators
- Hoogland: Uses novel methods (LLC and ED) to detect developmental stages and milestones

2. Scope:
- Raventós: Primarily studies linear regression tasks
- Hoogland: Examines both language modeling and linear regression tasks

Major Contributions of Each:

Raventós et al.:
1. Identifies a task diversity threshold for ICL emergence
2. Shows that beyond this threshold, transformers can solve new tasks optimally
3. Demonstrates that ICL requires deviation from Bayesian inference on the pretraining distribution

Hoogland et al.:
1. Introduces new methods (LLC and ED) for detecting developmental stages
2. Identifies specific stages in both language and regression tasks
3. Shows that transformer development follows discrete, identifiable stages

Synthesis:

These papers provide complementary perspectives on ICL emergence:
- Raventós explains the conditions needed (task diversity)
- Hoogland describes how this capability develops over time

Together, they suggest that ICL:
1. Requires sufficient task diversity in pretraining
2. Develops through distinct stages rather than emerging suddenly
3. Involves both structural and behavioral changes in the transformer

This combined understanding could help inform better training strategies and architectural choices for transformers.

User: and here's a third paper: 
```"Dynamics of Transient Structure in In-Context Linear Regression Transformers

 
Liam Carroll 1 2  Jesse Hoogland 1  Matthew Farrugia-Roberts 3  Daniel Murfet 4 

†
1Timaeus 2Gradient Institute 3Department of Computer Science, University of Oxford 4School of Mathematics and Statistics, the University of Melbourne. Correspondence to: Liam Carroll <lemmykc@gmail.com>.  
Preprint, under review.
Abstract
Modern deep neural networks display striking examples of rich internal computational structure. Uncovering principles governing the development of such structure is a priority for the science of deep learning. In this paper, we explore the transient ridge phenomenon: when transformers are trained on in-context linear regression tasks with intermediate task diversity, they initially behave like ridge regression before specializing to the tasks in their training distribution. This transition from a general solution to a specialized solution is revealed by joint trajectory principal component analysis. Further, we draw on the theory of Bayesian internal model selection to suggest a general explanation for the phenomena of transient structure in transformers, based on an evolving tradeoff between loss and complexity. We empirically validate this explanation by measuring the model complexity of our transformers as defined by the local learning coefficient.

1Introduction
Why do neural networks transition between qualitatively different modes of computation during training? This phenomenon has been studied for decades in artificial and biological neural networks (Baldi & Hornik, 1989; Rogers & McClelland, 2004). Recent work on transformers has uncovered particularly salient examples of transitions between two well-characterized alternative ways of approximating the data distribution. For instance, Power et al. (2022) show a “grokking” transition from an initial memorizing solution to a generalizing solution while training transformers to perform modular arithmetic. Conversely, Singh et al. (2024) show a transition from a “transient” generalizing solution to a memorizing solution while training transformers for in-context classification.

In this paper, we study a similar transition from generalization to memorization in transformers trained for in-context linear regression. Following Raventós et al. (2023), we construct sequences with latent regression vectors (tasks) sampled uniformly from a fixed set of size 
M
 (the task diversity). In this setting, Raventós et al. (2023) showed that fully trained transformers may behaviorally approximate either of two distinct in-context learning algorithms:

1. Discrete minimum mean squared error (dMMSE): The posterior mean given a uniform prior over the 
M
 tasks (implies memorizing the 
M
 tasks in some fashion).
2. Ridge regression (ridge): The posterior mean given a Gaussian prior from which the fixed tasks were initially sampled (independent of 
M
, generalizes to new tasks).
Moreover, Panwar et al. (2024, §6.1) showed that for intermediate 
M
 values, the out-of-distribution loss of a given transformer is non-monotonic, suggesting that these transformers initially approach ridge before diverting towards dMMSE. We term this phenomenon transient ridge.

In this paper, we extend the brief analysis of Panwar et al. (2024, §6.1) and investigate the dynamics of transient ridge in detail, contributing the following.

• In Section˜4, we replicate transient ridge and we comparatively analyze the in-distribution function-space trajectories of our transformers using joint trajectory principal component analysis, revealing generalization–memorization as a principal axis of development and clarifying how the task diversity affects the dynamics.
• In Section˜5, we explain transient ridge as the transformer navigating a tradeoff between loss and complexity that evolves over training, akin to Bayesian internal model selection Watanabe, 2009, §7.6; Chen et al., 2023, and we validate this explanation by estimating the complexity of the competing solutions using the local learning coefficient (Lau et al., 2025).
These results expand our understanding of the transient ridge phenomenon and highlight the evolving loss/complexity tradeoff as a promising principle for understanding similar transience phenomena. Section˜6 discusses limitations and directions for further investigation.

2Related work
In this section, we review empirical and theoretical work on the topic of the emergence and transience of computational structure in deep learning.

Internal computational structure.
Modern deep learning has shown striking examples of the emergence of internal computational structure in deep neural networks, such as syntax trees in transformers trained on natural language (Hewitt & Manning, 2019), conceptual chess knowledge in AlphaZero (McGrath et al., 2022), and various results from mechanistic interpretability (e.g., Olah et al., 2020; Cammarata et al., 2020; Elhage et al., 2021).

It is known that properties of the data distribution influence the emergence of computational structure. For example, Chan et al. (2022) studied an in-context classification and identified data properties that are necessary for transformers to develop in-context learning abilities. Raventós et al. (2023) studied in-context linear regression (Garg et al., 2022; Akyürek et al., 2023; von Oswald et al., 2023; Bai et al., 2024) and showed that changing the task diversity of the training distribution can change the in-context learning algorithm approximated by the fully-trained transformer.

Transient structure.
In some cases, multiple interesting computational structures emerge throughout training, with different ones determining model outputs at different times. A well-known example is the “grokking” transition, in which transformers learning modular arithmetic initially memorize the mappings from the training set, before eventually generalizing to unseen examples (Power et al., 2022) using an internal addition algorithm (Nanda et al., 2023).

Conversely, Singh et al. (2024) showed that transformers trained for in-context classification (Chan et al., 2022) can gradually shift from predicting based on contextual examples to predicting memorized labels, losing the ability to generalize to new mappings. Singh et al. (2024) termed this phenomenon “transient in-context learning.”

Similarly, for in-context linear regression, Panwar et al. (2024, §6.1) observed transformers initially achieving low out-of-distribution generalization loss (indicating that they approximate ridge regression) before eventually specializing to a memorized set of tasks. In an attempt to unify terminology, we call this phenomenon “transient ridge.” Compared to Panwar et al. (2024, §6.1), our work is novel in that it offers a more in-depth empirical analysis of this phenomenon, and we also offer an explanation of the phenomenon.

Additional examples of “transient structure” have recently been observed in settings including language modeling (Hoogland et al., 2024), in-context Markovian sequence modeling (Edelman et al., 2024; Park et al., 2024), and in-context modular arithmetic (He et al., 2024).

Explaining transient in-context learning.
There have been attempts to explain transience in the in-context classification setting originally studied by Singh et al. (2024). Nguyen & Reddy (2024) offer a simplified model in which in-context learning is acquired more rapidly than in-weight learning, and targeted regularization of the induction mechanism can cause it to later give way to in-weight learning.

Chan et al. (2024) give an explanation based on regret bounds for in-context and in-weight learning. In their model, in-context learning emerges because it is initially more accurate than in-weight learning for rare classes. Once the model sees more data for a class, in-weight learning becomes more accurate than in-context learning, due to limitations in their proposed induction mechanism.

Compared to these models, we offer a higher-level explanation of the general phenomenon of transient structure in terms of principles governing the preference for one solution over another at different points in training. We study this explanation in the setting of in-context linear regression, but it is also applicable in other settings.

Explaining transient structure.
There have been several attempts to explain transient structure in more general terms. If the memorizing solution achieves lower loss than the transient generalizing solution, the ultimate preference for memorization is not surprising (Singh et al., 2024; Park et al., 2024). The question remains, why would a generalizing solution arise in the first place if it is not as accurate as the memorizing solution (Singh et al., 2024)?

Panwar et al. (2024) speculate that the initial emergence of the generalizing solution could be due to an inductive bias towards simplicity. However, this still leaves the question, given that a less-accurate generalizing solution does emerge, why would it then fade later in training (Singh et al., 2024)?

Our work integrates these two perspectives. Rather than prioritizing accuracy or simplicity, we postulate a tradeoff between accuracy and simplicity that evolves over training. This explains the emergence of a simpler, less accurate generalizing solution (ridge) and its subsequent replacement by a complex, more accurate memorizing solution (dMMSE).

Internal model selection in deep learning.
Recent work has studied the relevance of internal model selection in Bayesian inference to deep learning. Chen et al. (2023) showed that, when small autoencoders transition between different encoding schemes during training (Elhage et al., 2022), such transitions are consistent with Bayesian inference. Hoogland et al. (2024) and Wang et al. (2024) found that the same theory can be used to detect the formation of internal structure, such as induction circuits (Elhage et al., 2021; Olsson et al., 2022) in small language models. Ours is the first work to analyze a transition between two transformer solutions in detail from this perspective.

3In-context linear regression
In this section, we introduce the in-context linear regression setting and the idealized dMMSE and ridge solutions, largely following Raventós et al. (2023).

3.1Nested multi-task data distributions
Given a latent regression vector, or task, 
𝐭
∈
ℝ
D
, we define a conditional distribution 
q
​
(
S
|
𝐭
)
 of sequences of i.i.d. pairs

S
=
(
x
1
,
y
1
,
…
,
x
K
,
y
K
)
∈
(
ℝ
D
×
ℝ
)
K
where 
x
k
∼
q
​
(
x
)
=
𝒩
​
(
0
,
I
D
)
 and 
y
k
∼
q
​
(
y
|
x
k
,
𝐭
)
=
𝒩
​
(
𝐭
⊤
​
x
k
,
σ
2
)
. We set 
K
=
16
, 
D
=
8
, and 
σ
2
=
0.125
.

We then define an unconditional data distribution of sequences 
q
​
(
S
)
=
q
​
(
S
|
𝐭
)
​
q
​
(
𝐭
)
, where 
q
​
(
𝐭
)
 is one of several task distributions described below. We sample a dataset of size 
N
, 
𝒟
=
{
S
i
}
i
=
1
N
∼
q
​
(
S
)
, by first sampling 
𝐭
i
∼
q
​
(
𝐭
)
 and then sampling 
S
i
∼
q
​
(
S
|
𝐭
i
)
 for 
i
=
1
,
…
,
N
.

We define a task distribution 
q
M
​
(
𝐭
)
 for each task diversity 
M
∈
ℕ
∪
{
∞
}
 as follows. We fix an unbounded i.i.d. sequence 
𝐭
1
,
𝐭
2
,
…
∼
𝒩
​
(
0
,
I
D
)
. For 
M
∈
ℕ
 we define

𝒯
M
=
{
𝐭
1
,
…
,
𝐭
M
}
and
q
M
​
(
𝐭
)
=
Uniform
​
(
𝒯
M
)
.
We further define 
𝒯
∞
=
{
𝐭
1
,
…
}
 and 
q
∞
​
(
𝐭
)
=
𝒩
​
(
0
,
I
D
)
. We denote by 
q
M
​
(
S
)
 the data distribution formed from 
q
M
​
(
𝐭
)
, and by 
𝒟
(
M
)
 a corresponding dataset.

Note that, in a departure from Raventós et al. (2023), the task sets 
𝒯
1
⊆
𝒯
2
⊆
⋯
⊆
𝒯
∞
 are nested by construction. In particular, the root task 
𝐭
1
 is included at every 
M
, allowing us to compare all models by their behavior on 
q
​
(
S
|
𝐭
1
)
.

3.2Mean squared error objective
Given a sequence 
S
, denote by 
S
≤
k
 the context subsequence 
(
x
1
,
y
1
,
…
,
x
k
−
1
,
y
k
−
1
,
x
k
)
 with label 
y
k
. Let 
f
 be a function mapping contexts to predicted labels. Given a dataset 
{
S
i
}
i
=
1
N
∼
q
​
(
S
)
 we define the per-token empirical loss

ℓ
N
,
k
​
(
f
)
=
1
N
​
∑
i
=
1
N
(
f
​
(
S
≤
k
i
)
−
y
k
i
)
2
.
(1)
Averaging over context lengths we obtain the empirical loss

ℓ
N
​
(
f
)
=
1
K
​
∑
k
=
1
K
ℓ
N
,
k
​
(
f
)
.
(2)
The corresponding population loss 
ℓ
​
(
f
)
 is defined by taking the expectation over the data distribution 
q
​
(
S
)
,

ℓ
​
(
f
)
=
𝔼
S
∼
q
​
[
1
K
​
∑
k
=
1
K
(
f
​
(
S
≤
k
)
−
y
k
i
)
2
]
.
(3)
For a function 
f
​
(
⋅
,
w
)
 implemented by a transformer with parameter 
w
, we denote the losses 
ℓ
N
,
k
​
(
w
)
, 
ℓ
N
​
(
w
)
, and 
ℓ
​
(
w
)
. For task diversity 
M
 we use a superscript 
ℓ
M
.

3.3Idealized in-context linear regression predictors
Given a context 
S
≤
k
 there are many possible algorithms that could be chosen to predict 
y
^
k
. Raventós et al. (2023) studied the following two predictors:

Predictor 1 (dMMSE).
For 
M
∈
ℕ
 and 
k
=
1
,
…
,
K
, the discrete minimum mean squared error predictor, dMMSEM, is the function 
g
k
M
:
(
ℝ
D
×
ℝ
)
k
×
ℝ
D
→
ℝ
 such that

g
k
M
​
(
x
1
,
y
1
,
…
,
x
k
)
=
(
𝐭
^
k
M
)
⊤
​
x
k
(4)
where the dMMSEM task estimate 
𝐭
^
k
M
∈
ℝ
D
 is given by

𝐭
^
k
M
=
∑
m
=
1
M
exp
⁡
(
−
1
2
​
σ
2
​
∑
j
=
1
k
−
1
(
y
j
−
𝐭
m
⊤
​
x
j
)
2
)
​
𝐭
m
∑
m
=
1
M
exp
⁡
(
−
1
2
​
σ
2
​
∑
j
=
1
k
−
1
(
y
j
−
𝐭
m
⊤
​
x
j
)
2
)
.
Note that the dMMSEM task estimate and therefore the prediction explicitly depends on the tasks 
𝒯
M
=
{
𝐭
1
,
…
,
𝐭
M
}
.

Predictor 2 (ridge).
For 
k
=
1
,
…
,
K
, the ridge predictor 
g
k
∞
:
(
ℝ
D
×
ℝ
)
k
×
ℝ
D
→
ℝ
 is given by

g
k
∞
​
(
x
1
,
y
1
,
…
,
x
k
)
=
(
𝐭
^
k
∞
)
⊤
​
x
k
(5)
where if 
k
=
1
 the task estimate is 
𝐭
^
k
∞
=
𝟎
, otherwise the task estimate 
𝐭
^
k
∞
 is given by 
L
2
-regularized least-squares regression on the examples in the context with the regularization parameter set to 
σ
2
,

𝐭
^
k
∞
=
(
X
⊤
​
X
+
σ
2
​
I
D
)
−
1
​
X
⊤
​
Y
,
where 
X
=
(
x
1
⊤
,
…
,
x
k
−
1
⊤
)
 and 
Y
=
(
y
1
,
…
,
y
k
−
1
)
.

Optimality of predictors.
Raventós et al. (2023) showed that for finite task diversity 
M
∈
ℕ
, given data distribution 
q
M
​
(
S
)
, the minimum mean squared error predictions are given by equation ˜4, that is, dMMSEM, whereas for infinite task diversity, given data distribution 
q
∞
​
(
S
)
, the minimum mean squared error predictions are given by equation ˜5, that is, ridge.

Moreover, note that for a fixed context 
x
1
,
y
1
,
…
,
x
k
, we have that as 
M
→
∞
, 
𝐭
^
k
M
→
a.s.
𝐭
^
k
∞
. It follows that ridge is an approximately optimal predictor for 
q
M
​
(
S
)
 given a large finite task diversity 
M
. However, for all finite task diversities 
M
 it remains possible to reduce expected loss on 
q
M
​
(
S
)
 by specializing to the tasks in 
𝒯
M
 (at the cost of increased loss on sequences constructed from other tasks).

Consistency of predictors.
The task estimates 
𝐭
^
k
M
,
𝐭
^
k
∞
 are both asymptotically consistent assuming unbounded sequences drawn based on a realizable task 
𝐭
∈
𝒯
M
. However, the task estimates will differ for all 
k
 (due to the different priors, 
q
M
​
(
𝐭
)
 and 
q
∞
​
(
𝐭
)
), particularly for early tokens and especially in the under-determined regime 
k
≤
D
.

4The transient ridge phenomenon
In this section, we replicate the transient ridge phenomenon observed by Panwar et al. (2024, §6.1) by training transformers at a range of task diversity parameters and evaluating their performance on out-of-distribution (OOD) sequences.

We then apply the general technique of joint trajectory PCA: We use principal component analysis (PCA) to decompose the collective function-space trajectories of the transformers, producing a low-dimensional representation of their behavioral development. Without having to specify the idealized predictors, we recover the difference between dMMSE and ridge as correlated to the second principal component, and show that in the lead up to the task diversity threshold trajectories are increasingly drawn towards ridge.

4.1Transformer training
We train transformers on nested multi-task data distributions with varying task diversity (Section˜3.1) under the mean squared error objective (Section˜3.2) to see when they behaviorally approximate dMMSE or ridge (Section˜3.3). We use a 
2
-layer transformer with 
d
=
2.65
 million parameters (details in Appendix˜A; more architectures in Appendix˜I).

We train with each of a set 
ℳ
 of task diversities ranging from 
M
=
1
 to 
M
=
2
15
 and also including 
∞
. Each run generates a trajectory 
w
t
M
⊆
ℝ
d
 through parameter space for training steps 
t
=
0
,
…
,
T
, from which we subsample checkpoints 
𝒞
⊆
{
0
,
…
,
T
}
 using a union of linear and logarithmic intervals (Section˜A.4). For notational ease, we sometimes denote the function 
f
​
(
⋅
,
w
t
M
)
 as 
f
M
​
(
⋅
,
w
t
)
.

4.2Joint trajectory principal component analysis
An established method for studying the development of structure and function in a system is to analyze its trajectory in configuration space. Amadei et al. (1993) developed the technique of applying PCA to such trajectories, called essential dynamics or simply trajectory PCA. It is argued that important features of trajectories appear in the essential subspace spanned by the leading principal components (Briggman et al., 2005; Ahrens et al., 2012), though interpreting PCA of time series requires care cf., Shinn, 2023; Antognini & Sohl-Dickstein, 2018; also Section˜C.1.

Trajectory PCA has seen diverse applications in molecular biology (Amadei et al., 1993; Meyer et al., 2006; Hayward & De Groot, 2008), neuroscience (Briggman et al., 2005; Cunningham & Yu, 2014), and deep learning (Olsson et al., 2022; Mao et al., 2024). We adapt a multi-trajectory variant (Briggman et al., 2005) to study the collective behavioral dynamics of our family of transformer models trained with different task diversities. In particular, we simultaneously perform PCA on the trajectories of all models through a finite-dimensional subspace of function space. Our detailed methodology for this joint trajectory PCA is as follows.

Joint encoding of transformer trajectories.
Given a parameter 
w
, we can view the transformer as mapping each sequence 
S
 to a vector of predictions for its 
K
 subsequences,

f
​
(
S
,
w
)
=
(
f
​
(
S
≤
1
,
w
)
,
…
,
f
​
(
S
≤
K
,
w
)
)
∈
ℝ
K
.
We fix a finite dataset 
𝒟
(
1
)
=
{
S
i
}
i
=
1
B
∼
q
1
​
(
S
)
 of 
B
=
512
 input sequences (recalling that the root task 
𝐭
1
 is shared by all task sets, so is the natural task to use to compare in-distribution behavior). We concatenate the outputs of 
f
​
(
⋅
,
w
)
 for each input 
S
i
 into one long row vector,

f
​
(
𝒟
(
1
)
,
w
)
=
(
f
​
(
S
1
,
w
)
,
…
,
f
​
(
S
B
,
w
)
)
∈
ℝ
B
​
K
,
representing the function 
f
​
(
⋅
,
w
)
 as a point in a finite-dimensional subspace of function space.

We apply this construction to each transformer checkpoint 
{
w
t
M
}
M
∈
ℳ
,
t
∈
𝒞
. For each 
M
∈
ℳ
, we aggregate the row vectors from each checkpoint into a matrix 
F
M
∈
ℝ
|
𝒞
|
×
B
​
K
 and then stack each 
F
M
 vertically into 
F
ℳ
∈
ℝ
|
ℳ
|
​
|
𝒞
|
×
B
​
K
:

F
M
=
[
f
​
(
𝒟
(
1
)
,
w
t
1
M
)
f
​
(
𝒟
(
1
)
,
w
t
2
M
)
⋮
f
​
(
𝒟
(
1
)
,
w
t
|
𝒞
|
M
)
]
​
 for 
​
M
∈
ℳ
,
F
ℳ
=
[
F
1
⋮
F
2
15
F
∞
]
.
Principal component analysis.
We apply PCA to the joint matrix 
F
ℳ
. Supposing 
F
ℳ
 has been mean-centered, it has a singular value decomposition 
F
ℳ
=
U
​
Λ
​
V
⊤
 where 
U
∈
ℝ
|
ℳ
|
​
|
𝒞
|
×
|
ℳ
|
​
|
𝒞
|
 has left singular vectors as columns, 
Λ
∈
ℝ
|
ℳ
|
​
|
𝒞
|
×
B
​
K
 is a diagonal matrix of ordered positive singular values, and 
V
∈
ℝ
B
​
K
×
B
​
K
 has right singular vectors as its columns. For 
v
≤
B
​
K
, let 
V
v
 denote the loading matrix given by the first 
v
 columns of 
V
. The span of these columns forms the 
v
-dimensional (joint) essential subspace.

Projecting into the essential subspace.
The corresponding projection from feature space into the essential subspace is given by 
π
v
:
ℝ
B
​
K
→
ℝ
v
 where 
π
v
​
(
y
)
=
y
​
V
v
. The developmental trajectory of each model is then represented as a curve 
γ
M
:
𝒞
→
ℝ
v
 defined by

γ
M
​
(
t
)
=
π
v
​
(
f
​
(
𝒟
(
1
)
,
w
t
M
)
)
.
For any principal component dimension 
i
≤
v
 we call the 
i
th component function 
γ
M
i
:
𝒞
→
ℝ
 a PC-over-time curve.

The dMMSEM and ridge predictors defined in equations ˜4 and ˜5 can likewise be encoded as row vectors and projected into the essential subspace. For 
M
∈
ℕ
∪
{
∞
}
 let

G
M
=
(
g
1
M
​
(
S
≤
1
1
)
,
g
2
M
​
(
S
≤
2
1
)
,
…
,
g
K
M
​
(
S
≤
K
B
)
)
∈
ℝ
B
​
K
.
Then each predictor projects to a single point in the essential subspace, 
π
v
​
(
G
M
)
. Note that we do not include the points 
G
1
, …, 
G
∞
 in the data prior to performing PCA.

Refer to caption
Figure 1:Behavioral dynamics of the transient ridge phenomenon. (Top left): OOD loss over training on sequences sampled with a Gaussian task distribution for task diversities 
M
∈
ℳ
. For intermediate 
M
 we see non-monotonicity caused by the transient ridge phenomenon, or “forgetting” as observed by Panwar et al. (2024, §6.1). We define 
t
M
crit
 as the step at which the OOD loss is minimized for 
M
 (Section˜B.2). We mark this step with a circle in the other plots. (Right): We project each transformer’s trajectory 
{
f
​
(
⋅
,
w
t
M
)
}
t
∈
𝒞
 to a curve 
γ
M
​
(
t
)
 in the essential subspace computed by joint trajectory PCA. We project dMMSEM (diamonds) and ridge (square) into the same subspace. For intermediate task diversity 
M
, the development is deflected towards ridge on its way towards dMMSEM. (Bottom left): In-distribution function-space distances 
Δ
​
(
⋅
,
dMMSE
M
/Ridge
)
 clarify which fully-trained transformers (stars) approximate dMMSEM, and which transformers approximate ridge at 
t
M
crit
 (circles). (Note): loss and PC curves are lightly smoothed, see Appendix˜D for raw data.
4.3Experimental results
Figure˜1 shows OOD loss on a fixed test set 
𝒟
(
∞
)
∼
q
∞
​
(
S
)
 and the result of 2-dimensional joint trajectory PCA (
27.5
%
 explained variance). Appendix˜B shows in-distribution loss. Appendix˜C extends to 4-dimensional PCA, explores the effect of checkpoint distributions, and shows that results are insensitive to the choice of batch size 
B
≥
16
.

Essential subspace.
Strikingly, PC2 correlates with an axis of behavioral difference between dMMSEM (for increasing 
M
) and ridge. Appendix˜E confirms that the predictions on earlier tokens, where dMMSE and ridge differ more, load more heavily on PC2 than those for later tokens do. PC2 also correlates with OOD loss, while PC1 appears to correlate with loss on 
q
1
​
(
S
)
 and with a notion of “development time” (see Figure˜C.1 and Section˜C.2).

Task diversity threshold.
As in Raventós et al. (2023), at low task diversity (in our case 
M
≤
128
), fully-trained transformers behaviorally approximate dMMSEM, while above a task diversity threshold (
M
≥
362
), they converge to a point that behaviorally approximates ridge. Trajectories 
M
∈
{
182
,
256
}
 converge somewhere between.

Transient ridge.
Replicating Panwar et al. (2024, §6.1), we see that for intermediate 
M
 in the lead-up to the task diversity threshold, the OOD loss is non-monotonic. For 
M
∈
{
16
,
24
,
…
,
128
}
, loss decreases towards that of ridge, then increases to that of dMMSEM. We see a partial dip for 
M
∈
{
6
,
8
}
 and a partial rise for 
M
∈
{
182
,
256
}
.

Trajectory PCA reveals that this non-monotonicity coincides with changes in the development of in-distribution behavior. For low 
M
, the transformers proceed directly to dMMSEM in the essential subspace. As 
M
 increases (until the task diversity threshold), the trajectories are increasingly deflected from this straight path into one that transits via approximating ridge. Beyond the task diversity threshold, the trajectories proceed directly to ridge and do not depart.

This trajectory PCA result suggests that the presence of the approximate ridge solution in the optimization landscape is in some sense influencing the development of internal structures in the transformers. Moreover, as 
M
 increases, as the dMMSEM solution changes, the strength of the influence of the ridge solution increases. In the next section, we attempt to understand the nature of this influence.

5Evolving loss/complexity tradeoff
In this section, we model the transient ridge phenomenon as the result of the transformer navigating an evolving tradeoff between loss and complexity as it undergoes additional training. We draw on the theory of Bayesian internal model selection to qualitatively predict the nature of the tradeoff. We then empirically validate this model of the phenomenon by quantifying the complexity of the fully-trained transformers using the associated complexity measure.

5.1Learning solutions of increasing complexity
The learning dynamics of many systems follow a pattern of progressing from solutions of high loss but low complexity to solutions of low loss but high complexity. This pattern has been studied in detail in certain models including deep linear networks (e.g., Baldi & Hornik, 1989; Saxe et al., 2019; Gissin et al., 2020; Jacot et al., 2021), multi-index models (Abbe et al., 2023), and image models (Kalimeris et al., 2019), each with their own notion of “complexity.”

Unfortunately, we lack results describing how such a progression should play out, or what complexity measure to use, for general deep learning. Therefore, we turn to singular learning theory (SLT; Watanabe, 2009; 2018)—a framework for studying statistical models with degenerate information geometry, including neural networks (Hagiwara et al., 1993; Watanabe, 2007; Wei et al., 2023)—in which a similar loss/complexity tradeoff has been studied in general terms in the setting of Bayesian inference.

5.2Bayesian internal model selection
In Bayesian inference, SLT shows that the solutions around which the posterior concentrates are determined by a balance of loss and complexity. Moreover, the ideal balance changes as the number of samples increases, driving a progression from simple but inaccurate solutions to accurate but complex solutions Watanabe, 2009, §7.6; Chen et al., 2023. The leading-order complexity measure is the local learning coefficient (LLC; Lau et al., 2025), which can be understood as a degeneracy-aware effective parameter count. We outline this internal model selection principle below.

Bayesian posterior.
Consider a neural network parameter space 
𝒲
⊆
ℝ
d
. Let 
φ
 be a nonzero prior over 
𝒲
 and 
ℓ
n
:
𝒲
→
ℝ
 an empirical loss (the average negative log likelihood) on 
n
 samples. Then the Bayesian posterior probability of a neighborhood 
𝒰
⊆
𝒲
 given 
n
 samples is

p
n
​
(
𝒰
)
=
Z
n
​
(
𝒰
)
Z
n
​
(
𝒲
)
where 
Z
n
​
(
𝒳
)
 is the marginal likelihood of 
𝒳
⊆
𝒲
,

Z
n
​
(
𝒳
)
=
∫
𝒳
exp
⁡
(
−
n
​
ℓ
n
​
(
w
)
)
​
φ
​
(
w
)
​
𝑑
w
.
Bayesian posterior log-odds.
Consider two neighborhoods 
𝒰
,
𝒱
⊆
𝒲
. The preference of the Bayesian posterior for 
𝒰
 over 
𝒱
 can be summarized in the posterior log-odds,

log
⁡
p
n
​
(
𝒰
)
p
n
​
(
𝒱
)
=
log
⁡
Z
n
​
(
𝒰
)
−
log
⁡
Z
n
​
(
𝒱
)
,
(6)
which is positive to the extent that 
p
n
 prefers 
𝒰
 over 
𝒱
.

Watanabe’s free energy formula.
SLT gives an asymptotic expansion of the Bayesian local free energy 
−
log
⁡
Z
n
​
(
⋅
)
. Let 
u
∈
𝒲
 be a solution, that is, a local minimum of the expected negative log likelihood, and let 
𝒰
 be a closed ball around 
u
, in which 
u
 is a maximally degenerate global minimum. Then, under certain technical conditions on the model, we have the following asymptotic expansion in 
n
 Watanabe, 2018, Theorem 11; Lau et al., 2025:

−
log
⁡
Z
n
​
(
𝒰
)
=
ℓ
n
​
(
u
)
⋅
n
+
λ
​
(
u
)
⋅
log
⁡
n
+
O
p
​
(
log
⁡
log
⁡
n
)
(7)
where 
λ
​
(
u
)
 is the LLC and the lower-order terms include various other contributions, such as from the prior.

The loss/complexity tradeoff.
If 
v
∈
𝒱
 is a competing solution (with its own neighborhood), then ˜6 and ˜7 give

log
⁡
p
n
​
(
𝒰
)
p
n
​
(
𝒱
)
=
Δ
​
ℓ
n
⋅
n
+
Δ
​
λ
⋅
log
⁡
n
+
O
p
​
(
log
⁡
log
⁡
n
)
(8)
where 
Δ
​
ℓ
n
=
ℓ
n
​
(
v
)
−
ℓ
n
​
(
u
)
 and 
Δ
​
λ
=
λ
​
(
v
)
−
λ
​
(
u
)
.

Equation ˜8 describes an evolving tradeoff between loss and complexity as follows. Assume the lower-order terms from each free energy expansion cancel. Then if 
Δ
​
ℓ
n
<
0
 (
u
 has higher loss than 
v
) and 
Δ
​
λ
>
0
 (
u
 has lower LLC than 
v
), the sign of the log-odds depends on 
n
. The Bayesian posterior will prefer 
𝒰
 (around the simple but inaccurate solution) until 
log
⁡
(
n
)
/
n
<
(
−
Δ
​
ℓ
n
)
/
Δ
​
λ
, after which it will prefer 
𝒱
 (around the accurate but complex solution).

From Bayesian inference to deep learning.
Neural networks are typically trained by stochastic gradient-based optimization, not Bayesian inference. Nevertheless, as described in Section˜2, recent work suggests that some qualitatively similar evolving tradeoff governs the development of structure in deep learning over training time (Chen et al., 2023; Hoogland et al., 2024; Wang et al., 2024).

This suggests that some as-yet-unknown principle of “dynamic1
1
In the sense of nonlinear dynamics (cf., e.g., Strogatz, 1994), where it is well-established that degeneracy in the geometry of critical points of a governing potential influences system trajectories.
 internal model selection”—in which the loss and the LLC play leading roles—underpins the structure of the optimization landscape, in turn influencing the trajectories followed by stochastic gradient-based optimization. Based on this motivation, we apply equation ˜8 to qualitatively predict the transient ridge phenomenon in terms of the differences in loss and LLC of the transformers that approximately implement the dMMSEM and ridge predictors.

5.3Explaining the transient ridge phenomenon
We offer the following explanation for the dynamics of transformers trained on in-context linear regression data with task diversity 
M
∈
ℕ
. Let 
u
∞
∈
𝒰
 and 
v
M
∈
𝒱
M
 be transformer parameters approximately implementing ridge and dMMSEM respectively, along with their neighborhoods.

1. Low 
M
: We expect 
v
M
 to have much lower loss and LLC than 
u
∞
. As equation ˜8 never favors 
𝒰
, training should proceed directly to 
v
M
.
2. Intermediate 
M
: We expect 
v
M
 to have lower loss but higher LLC than 
u
∞
. As equation ˜8 initially favors 
𝒰
 but eventually favors 
𝒱
M
, training should proceed first towards 
u
∞
 before pivoting towards 
v
M
 (after a number of training steps that increases with 
M
).
3. High 
M
: We expect 
v
M
 to have slightly lower loss but much higher LLC than 
u
∞
. As equation ˜8 only favors 
𝒱
M
 at very high 
n
, trajectories should proceed to 
u
∞
 and should not depart by the end of training.
See Figure˜2 for a conceptual illustration. For 
M
 values that fall between these three prototypical cases, the posterior preference is less sharp. Therefore we expect to see gradual shifts the dynamics over the range of 
M
 values.

Refer to caption
Figure 2:Transient ridge in the loss landscape. Conceptual illustration of transient ridge arising as the result of an evolving tradeoff between loss and LLC (complexity, illustrated as sharpness). As 
M
 increases, we expect the loss gap between dMMSEM and ridge to shrink and the LLC of dMMSEM to grow, creating transience for intermediate 
M
.
5.4Empirical validation of the explanation
The above explanation is consistent with the findings of Section˜4. It remains to validate that the trends in the loss and LLC are as expected. In this section, we outline our experiments estimating the loss and LLC of 
u
∞
 and 
v
M
.

Estimating loss.
We first estimate the loss of 
u
∞
 and 
v
M
 by directly evaluating the idealized predictors ˜4 and ˜5. Alternatively, noting that the transformer cannot necessarily realize these predictors, we evaluate the end-of-training parameters (representing 
v
M
 for low 
M
 or 
u
 for high 
M
). Figure˜3(top) confirms that the loss gap between idealized predictors shrinks with increasing 
M
, and the transformers achieve similar loss to their respective algorithms.

Refer to caption
Figure 3:Loss and LLC estimates match predictions. (Top): Estimated loss with respect to data distribution 
q
M
​
(
S
)
 for the idealized predictors and fully-trained transformers. The gap between dMMSEM and ridge decreases with 
M
, and trained transformers approximate this loss on either side of the task diversity threshold (diamonds for dMMSEM, squares for ridge). (Bottom): Estimated LLC for fully-trained transformers. Large-
M
 LLCs converge to the LLC of ridge (dashed line). Small-
M
 LLCs, representing the LLC of dMMSEM, cross this line as 
M
 increases.
Estimating LLC.
The LLC is architecture-dependent, so we can’t meaningfully measure the LLC of the idealized predictors, only our fully-trained transformers (representing 
v
M
 for low 
M
 or 
u
∞
 for high 
M
). Following Lau et al. (2025), we estimate the LLC of a parameter 
w
∗
M
 as the average increase in empirical loss 
ℓ
n
 for nearby parameters,

λ
^
​
(
w
∗
M
)
=
n
​
β
​
(
𝔼
w
∣
w
∗
M
,
γ
β
​
[
ℓ
n
​
(
w
)
]
−
ℓ
n
​
(
w
∗
M
)
)
,
(9)
where 
n
 is a sample size, 
β
 is an inverse temperature, 
γ
 is a localization strength parameter, and 
𝔼
w
∣
w
∗
M
,
γ
β
 is expectation over the localized Gibbs posterior

p
​
(
w
;
w
∗
M
,
β
,
γ
)
∝
exp
⁡
(
−
n
​
β
​
ℓ
n
​
(
w
)
−
γ
2
​
‖
w
−
w
∗
M
‖
2
2
)
.
We sample from this posterior with stochastic gradient Langevin dynamics (SGLD; Welling & Teh, 2011). Appendix˜F gives further details on LLC estimation, sampling with SGLD, and hyperparameter calibration.

Figure˜3(bottom) shows LLC estimates 
λ
^
​
(
w
∗
M
)
 for fully-trained transformers. High-
M
 LLCs converge to 
λ
^
∞
, which we take as 
λ
​
(
u
∞
)
, the LLC of the ridge solution. For low-
M
 transformers that converge to dMMSEM, we take 
λ
^
​
(
w
∗
M
)
 as 
λ
​
(
v
M
)
. As expected, this LLC increases with 
M
, and crosses 
λ
^
∞
 during the onset of transience. Surprisingly, the estimated LLC of dMMSEM plateaus above 
M
=
32
, suggesting that the approximation achieved by the fully-trained transformers may be incomplete.

6Limitations and future work
Our findings support an understanding of the transient ridge phenomenon as driven by an evolving loss/complexity tradeoff, governed by principles that are yet to be fully discovered but qualitatively resemble Bayesian internal model selection. In this section, we enumerate remaining gaps in this understanding, representing future steps towards a comprehensive understanding of neural network development.

6.1Transformers versus idealized predictors
Our analysis is based primarily on in-distribution behavior, and it is not clear that our transformers can or do faithfully approximate the idealized predictors for all input sequences. Moreover, it is unclear whether the solutions governing training dynamics are necessarily the parameters to which transformers converge (we consider an alternative interpretation in Appendix˜G). Future work could seek a more detailed understanding of transformer solutions arising in practice, for example using mechanistic interpretability.

6.2The role of lower-order terms
In Section˜5.2, we make the simplifying assumption that the lower-order terms from each expansion cancel. However if these terms are not equal then their difference enters the posterior log odds, influencing the evolution of the posterior, especially for low 
n
. SLT has studied these terms (cf., e.g., Lau et al., 2025), but they are not as well-studied as the LLC. Future work could deepen our theoretical understanding of these lower-order terms or our empirical understanding of their role in internal model selection.

6.3Dynamic versus Bayesian internal model selection
Of course, our primary motivation is to study neural network development, rather than Bayesian internal model selection per se. While we have contributed further evidence that the loss and the LLC play a leading role in a principle of “dynamic internal model selection” that governs neural network development, the precise form of this principle and the precise roles of the loss and the LLC remain to be determined. Our work highlights this as a promising direction for future empirical and theoretical work.

6.4Why does transience stop?
Bayesian internal model selection suggests that the ridge solution should always eventually give way to a more complex but more accurate dMMSEM solution. In practice, replicating Raventós et al. (2023), we see a clear task diversity threshold above which transformers never leave ridge. This could be due to capacity constraints, under-training, neuroplasticity loss, or a concrete difference between Bayesian and “dynamic” internal model selection. Appendix˜H offers a preliminary analysis, but reaches no firm conclusion, leaving this an open question for future work.

6.5Beyond in-context linear regression
As outlined in Section˜2, the phenomenon of a transient generalizing solution giving way to a memorizing solution over training has now been observed in a range of sequence modeling settings beyond our setting of in-context linear regression Singh et al., 2024; Hoogland et al., 2024; Edelman et al., 2024; He et al., 2024; Park et al., 2024. There’s also the reverse phenomenon of a transition from a memorizing solution to an equal-loss but simpler generalizing solution (“grokking;” Power et al., 2022; Nanda et al., 2023).

These settings represent rich subjects for future empirical work investigating the principles governing the development neural networks. In the case of grokking, we note that the particular loss/complexity tradeoff outlined in Section˜5 does not account for transitions that decrease complexity, though such transitions can be described within the Bayesian internal model selection framework by taking lower-order terms into account (cf. Section˜6.2).

7Conclusion
This paper contributes an in-depth study of the training dynamics of transformers in the settings of in-context linear regression with variable task diversity. We adapt the technique of trajectory principal component analysis from molecular biology and neuroscience and deploy it to expand our empirical understanding of the developmental dynamics of our transformers, and the variation in these dynamics with task diversity, revealing the choice between memorization and generalization as a principal axis of development.

Moreover, we adopt the perspective of singular learning theory to offer an explanation of these dynamics as an evolving tradeoff between loss and complexity (as measured by the local learning coefficient), finding evidence that these elements play a leading role in governing the development of our transformers, akin to their role in governing the development of the posterior in Bayesian internal model selection. These findings open the door to future research aiming to uncover the true principles governing the development of internal structure in deep learning.

Impact statement
The emergence of computational structure in deep neural networks is not only a fascinating scientific and mathematical phenomenon. This structure determines a model’s out-of-distribution generalization behavior, and in turn its safety, robustness, and alignment properties. As society races ahead to develop ever more complex neural networks and integrate them ever more deeply into our digital and physical world, understanding the principles governing neural network development is a priority for the science of deep learning. This work aims to contribute towards improving our scientific understanding of neural network development, which is an integral part of (though not alone sufficient for) ensuring that future technological advances in the field of deep learning have robustly positive impact.

Acknowledgments
We thank Edmund Lau, George Wang, and Susan Wei for helpful conversations. Google’s TPU Research Cloud program supported some of our experiments with Cloud TPUs." Please summarize and put it into context of the first 2 papers.
```

Assistant: This third paper (Carroll et al.) builds directly on both previous papers and helps bridge their findings, providing a deeper explanation of the dynamics involved in ICL emergence. Here's how they relate:

Key Connections Across All Three Papers:

1. Building on Previous Work:
- Extends Raventós et al.'s work on task diversity threshold
- Complements Hoogland et al.'s developmental stages analysis
- Focuses specifically on the transition period around the task diversity threshold

2. Main Contribution:
The paper explains the "transient ridge phenomenon" - where transformers temporarily behave like ridge regression before specializing to their training tasks at intermediate task diversity levels.

Integration of Previous Findings:

1. With Raventós et al.:
- Confirms the task diversity threshold finding
- Explains why transformers behave differently above and below the threshold
- Provides deeper analysis of the transition period that Raventós observed

2. With Hoogland et al.:
- Uses similar methodological tools (trajectory analysis)
- Supports the developmental stages framework
- Adds complexity analysis to explain stage transitions

New Insights:

1. Loss/Complexity Tradeoff:
- Introduces a theoretical framework explaining why transformers transition between different behaviors
- Shows how the balance between loss and complexity evolves during training
- Uses Local Learning Coefficient (LLC) to measure complexity

2. Trajectory Analysis:
- Develops joint trajectory PCA to analyze how transformers develop
- Shows generalization-to-memorization as a principal axis of development
- Provides visualization of the transition dynamics

Synthesis of All Three Papers:

Together, these papers provide a comprehensive view of ICL emergence:
1. Conditions (Raventós): Task diversity threshold required
2. Stages (Hoogland): Discrete developmental stages in learning
3. Dynamics (Carroll): Explanation of transition mechanisms

This progression helps explain:
- Why ICL emerges (task diversity - Raventós)
- How it develops (stages - Hoogland)
- What happens during transitions (loss/complexity tradeoff - Carroll)

The three papers together provide a much richer understanding of ICL emergence than any one alone, moving from observation to characterization to mechanistic explanation.

User: Ok am I now given the following dataset:
```python
class LinearRegressionDataset(IterableDataset):
    """
    Dataset for in-context linear regression tasks.

    Noise is set to 0.
    Task magnitude is standardized to 1.
    Generates samples on the fly.
    """
    def __init__(
        self,
        num_tasks: int,
        seq_length: int,
        input_dim: int,
    ):
        """
        Initialize the LinearRegressionDataset.

        Args:
            num_tasks (int): Number of different tasks to generate.
            seq_length (int): Length of the input sequence (number of x, y pairs).
                              Note: this is half of the context length that the transformer will see.
            input_dim (int): Dimensionality of the input vectors x.
        """
        self.num_tasks = num_tasks
        self.seq_length = seq_length
        self.input_dim = input_dim

        # Generate random tasks
        self.tasks = torch.randn(num_tasks, input_dim)

        # Standardize tasks to have unit norm
        self.tasks = self.tasks / ((self.tasks ** 2).sum(dim=-1) ** 0.5).unsqueeze(-1)

    def __iter__(self):
        # This is called when an iterator is created from the dataset
        while True:
            # Generate input sequence
            x = torch.randn(self.seq_length, self.input_dim)

            # Randomly assign task
            task_index = torch.randint(0, self.num_tasks, (1,)).item()
            task = self.tasks[task_index]

            # Generate output
            y = torch.matmul(x, task)

            yield x, y, task
```
, the following model 
```python 
class ICLTransformer(nn.Module):
    """
    Transformer model for in-context learning of linear regression tasks.

    This is a wrapper around transformer_lens's HookedTransformer class.
    """

    def __init__(
        self,
        input_dim: int,
        seq_length: int = 16,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_head: int = 32,
        d_mlp: int = 128,
        act_fn: str = "relu",
    ):
        """
        Initialize the ICLTransformer.

        Args:
            input_dim (int): Dimensionality of the input vectors x.
            seq_length (int, optional): Length of the input sequence. Defaults to 16.
            d_model (int, optional): Dimensionality of the model's hidden states. Defaults to 64.
            n_layers (int, optional): Number of transformer layers. Defaults to 2.
            n_heads (int, optional): Number of attention heads. Defaults to 4.
            d_head (int, optional): Dimensionality of each attention head. Defaults to 32.
            d_mlp (int, optional): Dimensionality of the feedforward network. Defaults to 128.
            act_fn (str, optional): Activation function for the feedforward network. Defaults to "relu".
        """
        super().__init__()

        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer = HookedTransformer(
            HookedTransformerConfig(
              n_layers=n_layers,
              n_ctx = 2 * seq_length,
              d_model=d_model,
              d_head=d_head,
              n_heads=n_heads,
              d_mlp=d_mlp,
              act_fn=act_fn,
              d_vocab=input_dim + 1,  # Not using embeddings in typical way
              # d_vocab_out=1,
              use_split_qkv_input=True,
              use_attn_result=True,
              device=device
            )
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim + 1, d_model)  # + 1 for y value

        self.to(device)

    def output_proj(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project the transformer's output to the final prediction.

        Args:
            activations (torch.Tensor): Activations from the transformer.

        Returns:
            torch.Tensor: Predicted y values.
        """
        return activations[:,::2, -1]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ICLTransformer.

        Args:
            x (torch.Tensor): Input sequence of x values.
            y (torch.Tensor): Input sequence of y values.

        Returns:
            torch.Tensor: Predicted y values.
        """
        batch_size, seq_len, input_dim = x.shape

        # Interleave x and y into single input
        xy_combined = torch.zeros(batch_size, 2 * seq_len, input_dim + 1, device=self.device)
        xy_combined[:, ::2, :input_dim] = x
        xy_combined[:, 1::2, -1] = y

        # Project to d_model dimension
        hidden = self.input_proj(xy_combined)

        # Pass through transformer
        transformer_out = self.transformer(
            hidden,
            start_at_layer=0,  # Bypass embedding
            return_type="logits"  # Do not return loss (since our logits aren't really "logits")
        )

        predictions = self.output_proj(transformer_out)

        return predictions
        ```
        , the following training loop 
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_steps: int = 5000,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_logging: bool = True,
) -> list[dict]:
    """
    Train the ICLTransformer model.

    Args:
        model (nn.Module): The ICLTransformer model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        num_steps (int, optional): Number of training steps. Defaults to 5000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        lr_frac (float, optional): Fraction of training steps after which to apply learning rate decay. Defaults to 0.25.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.01.
        device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        wandb_logging (bool, optional): Whether to log to Weights & Biases. Defaults to True.

    Returns:
        list[dict]: List of model checkpoints.
    """

    if wandb_logging:
        wandb.init(project="icl-linear-regression")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()

    checkpoints = []

    pbar = tqdm.tqdm(enumerate(train_loader), total=num_steps)

    for step, (x, y, _) in pbar:
        if step >= num_steps:
            break

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        predictions = model(x, y)
        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, task in val_loader:
                    x, y = x.to(device), y.to(device)
                    predictions = model(x, y)
                    val_loss += criterion(predictions, y).item()

            val_loss /= len(val_loader)

            if wandb_logging:
                wandb.log({"batch_loss": loss.item(), "val_loss": val_loss})

            checkpoints.append(deepcopy(model.state_dict()))
            pbar.set_description(f"Batch Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

            model.train()

    return checkpoints
    ```
    , the following run definition 
    ```python
@dataclass
class Run:
    # Data
    num_tasks: int
    seq_length: int
    input_dim: int

    # Training
    num_steps: int = 5000
    learning_rate: float = 3e-3
    seed: int = 0
    wandb_logging: bool = True
    weight_decay: float = 0.0

    # Objects
    model: nn.Module = None
    train_loader: DataLoader = None
    val_loader: DataLoader = None
    checkpoints: list[dict] = field(default_factory=list)

    def __post_init__(self):
      torch.manual_seed(self.seed)

      self.model = ICLTransformer(input_dim=self.input_dim, seq_length=self.seq_length)
      dataset = LinearRegressionDataset(num_tasks=self.num_tasks, seq_length=self.seq_length, input_dim=self.input_dim)
      self.train_loader = DataLoader(dataset, batch_size=32)

      valset = [row for row in islice(iter(dataset), 0, 2048)]

      # Separate x, y, and task into lists
      x_val = [row[0] for row in valset]
      y_val = [row[1] for row in valset]
      task_val = [row[2] for row in valset]

      # Convert lists to tensors
      x_val = torch.stack(x_val)
      y_val = torch.stack(y_val)
      task_val = torch.stack(task_val)

      # Create TensorDataset and DataLoader for validation
      val_dataset = TensorDataset(x_val, y_val, task_val)
      self.val_loader = DataLoader(val_dataset, batch_size=32)

    def train(self):
        self.checkpoints = train_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.num_steps,
            self.learning_rate,
            wandb_logging=self.wandb_logging
        )
        return self.model
        ```
        and the challenge of understanding how the model is able to solve the prediction task for low task diversity settings of M = 1, 2. Given the time constraints (I have around 3 more hours to complete the task), I'd like to start by replicating some of the visualizations/experiments of the papers. I think a good starting point could be to visualize the developmental stages by plotting the loss vs training steps and the LLC. Do you think that makes sense? If so, how would you go about implementing this?

Assistant: Yes, that's a good starting point! To replicate the key visualizations from the papers, particularly focusing on low task diversity (M=1,2), we can start with:

1. First, let's create a function to run multiple experiments with different task diversities and track both training and validation loss:

```python
def run_experiments(task_diversities=[1, 2], seeds=5):
    results = {}
    for M in task_diversities:
        results[M] = []
        for seed in range(seeds):
            run = Run(
                num_tasks=M,
                seq_length=16,
                input_dim=8,
                num_steps=5000,
                seed=seed,
                wandb_logging=False  # Disable wandb for multiple runs
            )
            run.train()
            results[M].append(run)
    return results

# Modify the training loop to track losses at every step
def train_model(model, train_loader, val_loader, num_steps=5000, learning_rate=3e-3, 
                weight_decay=0., device="cuda" if torch.cuda.is_available() else "cpu",
                wandb_logging=True):
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=num_steps)
    
    for step, (x, y, _) in pbar:
        if step >= num_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(x, y)
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if step % 100 == 0:  # Increased frequency of validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_x, val_y, _ in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    val_predictions = model(val_x, val_y)
                    val_loss += criterion(val_predictions, val_y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            pbar.set_description(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            model.train()
    
    return train_losses, val_losses
```

2. Then, we can create a visualization function:

```python
import matplotlib.pyplot as plt

def plot_learning_curves(results):
    plt.figure(figsize=(12, 6))
    
    for M in results.keys():
        # Average losses across seeds
        avg_train_losses = np.mean([run.train_losses for run in results[M]], axis=0)
        avg_val_losses = np.mean([run.val_losses for run in results[M]], axis=0)
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(avg_train_losses, label=f'M={M}')
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        # Plot validation loss
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, len(avg_train_losses), 100), avg_val_losses, label=f'M={M}')
        plt.title('Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
```

3. For the LLC estimation, we can implement a simplified version following the papers:

```python
def estimate_llc(model, loader, n_samples=100, beta=1.0, gamma=1.0):
    """
    Estimate Local Learning Coefficient using SGLD sampling
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Initialize accumulator for LLC estimation
    loss_diff_sum = 0
    
    for _ in range(n_samples):
        # SGLD step
        noise_scale = np.sqrt(2.0 / (beta * gamma))
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
        
        # Compute loss at perturbed parameters
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        
        # Accumulate difference
        loss_diff_sum += perturbed_loss - criterion(model(x, y), y).item()
        
        # Restore original parameters
        model.load_state_dict(original_state)
    
    # Compute LLC estimate
    llc = (loss_diff_sum / n_samples) * len(loader.dataset) * beta
    
    return llc
```

4. We can then run and visualize:

```python
# Run experiments
results = run_experiments(task_diversities=[1, 2], seeds=3)

# Plot learning curves
plot_learning_curves(results)

# Estimate and plot LLC
llcs = {}
for M in results.keys():
    llcs[M] = []
    for run in results[M]:
        llc = estimate_llc(run.model, run.val_loader)
        llcs[M].append(llc)

plt.figure(figsize=(6, 4))
for M in llcs.keys():
    plt.scatter([M] * len(llcs[M]), llcs[M], label=f'M={M}')
plt.xlabel('Task Diversity (M)')
plt.ylabel('LLC Estimate')
plt.legend()
plt.show()
```

This will give us a basic replication of some key visualizations from the papers. We can then:
1. Compare the learning dynamics for M=1 vs M=2
2. Look for evidence of developmental stages in the loss curves
3. See if the LLC estimates align with the papers' findings about complexity differences

Would you like me to explain any part of this in more detail or should we proceed with implementation?


User: there's a bug in `plot_learning_curves`. The run does not have `train_losses` and `val_losses` attributes.

Assistant: Ah yes, you're right! We need to modify both the training loop and the plotting function since the losses are returned by `train_model`. Here's the corrected version:

```python
def run_experiments(task_diversities=[1, 2], seeds=5):
    results = {}
    for M in task_diversities:
        results[M] = []
        for seed in range(seeds):
            run = Run(
                num_tasks=M,
                seq_length=16,
                input_dim=8,
                num_steps=5000,
                seed=seed,
                wandb_logging=False  # Disable wandb for multiple runs
            )
            train_losses, val_losses = run.train()  # Now storing the returned losses
            results[M].append({
                'run': run,
                'train_losses': train_losses,
                'val_losses': val_losses
            })
    return results

def plot_learning_curves(results):
    plt.figure(figsize=(12, 6))
    
    for M in results.keys():
        # Average losses across seeds
        avg_train_losses = np.mean([run['train_losses'] for run in results[M]], axis=0)
        avg_val_losses = np.mean([run['val_losses'] for run in results[M]], axis=0)
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(avg_train_losses, label=f'M={M}')
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        # Plot validation loss
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, len(avg_train_losses), 100), avg_val_losses, label=f'M={M}')
        plt.title('Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
```

And we need to modify the Run class's train method to return the losses:

```python
def train(self):
    train_losses, val_losses = train_model(
        self.model,
        self.train_loader,
        self.val_loader,
        self.num_steps,
        self.learning_rate,
        wandb_logging=self.wandb_logging
    )
    return train_losses, val_losses
```

Now the plotting should work correctly. Would you like to try running this modified version?



User: right now we're just plotting a single LLC for a single seed. In the original paper, we tracked the LLC during training

Assistant: Ah yes, you're right! To track the LLC during training, we need to:
1. Save model checkpoints during training
2. Estimate LLC for each checkpoint
3. Plot the LLC evolution over time

Here's the modified code:

```python
def train_model(model, train_loader, val_loader, num_steps=5000, learning_rate=3e-3, 
                weight_decay=0., device="cuda" if torch.cuda.is_available() else "cpu",
                wandb_logging=True, checkpoint_freq=100):
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    checkpoints = []
    
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=num_steps)
    
    for step, (x, y, _) in pbar:
        if step >= num_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(x, y)
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if step % checkpoint_freq == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_x, val_y, _ in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    val_predictions = model(val_x, val_y)
                    val_loss += criterion(val_predictions, val_y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Save checkpoint
            checkpoints.append(deepcopy(model.state_dict()))
            
            pbar.set_description(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            model.train()
    
    return train_losses, val_losses, checkpoints

def run_experiments(task_diversities=[1, 2], seeds=5):
    results = {}
    for M in task_diversities:
        results[M] = []
        for seed in range(seeds):
            run = Run(
                num_tasks=M,
                seq_length=16,
                input_dim=8,
                num_steps=5000,
                seed=seed,
                wandb_logging=False
            )
            train_losses, val_losses, checkpoints = run.train()
            
            # Calculate LLC for each checkpoint
            llcs = []
            for checkpoint in checkpoints:
                run.model.load_state_dict(checkpoint)
                llc = estimate_llc(run.model, run.val_loader)
                llcs.append(llc)
            
            results[M].append({
                'run': run,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'llcs': llcs
            })
    return results

def plot_learning_curves_and_llc(results):
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    for M in results.keys():
        avg_train_losses = np.mean([run['train_losses'] for run in results[M]], axis=0)
        plt.plot(avg_train_losses, label=f'M={M}')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 3, 2)
    for M in results.keys():
        avg_val_losses = np.mean([run['val_losses'] for run in results[M]], axis=0)
        plt.plot(np.arange(0, len(avg_train_losses), 100), avg_val_losses, label=f'M={M}')
    plt.title('Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Plot LLC
    plt.subplot(1, 3, 3)
    for M in results.keys():
        avg_llcs = np.mean([run['llcs'] for run in results[M]], axis=0)
        std_llcs = np.std([run['llcs'] for run in results[M]], axis=0)
        steps = np.arange(0, len(avg_train_losses), 100)
        plt.plot(steps, avg_llcs, label=f'M={M}')
        plt.fill_between(steps, 
                        avg_llcs - std_llcs, 
                        avg_llcs + std_llcs, 
                        alpha=0.2)
    plt.title('Local Learning Coefficient')
    plt.xlabel('Steps')
    plt.ylabel('LLC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Run experiments and plot
results = run_experiments(task_diversities=[1, 2], seeds=3)
plot_learning_curves_and_llc(results)
```

This will:
1. Save model checkpoints every 100 steps during training
2. Calculate LLC for each checkpoint
3. Plot the evolution of training loss, validation loss, and LLC over time
4. Show error bars for LLC based on multiple seeds

The LLC plot should now show how the complexity of the learned solution evolves during training, similar to what was shown in the papers. Would you like to try running this?



User: why do I need to save and load checkpoints? Can't I just estimate LLC during training instead? Shouldn't this be faster?

Assistant: Yes, you're absolutely right! It would be more efficient to estimate the LLC during training rather than saving and loading checkpoints. Here's the modified version:

```python
def train_model(model, train_loader, val_loader, num_steps=5000, learning_rate=3e-3, 
                weight_decay=0., device="cuda" if torch.cuda.is_available() else "cpu",
                wandb_logging=True, checkpoint_freq=100):
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    llcs = []
    
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=num_steps)
    
    for step, (x, y, _) in pbar:
        if step >= num_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(x, y)
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if step % checkpoint_freq == 0:
            model.eval()
            # Compute validation loss
            val_loss = 0.0
            with torch.no_grad():
                for val_x, val_y, _ in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    val_predictions = model(val_x, val_y)
                    val_loss += criterion(val_predictions, val_y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Estimate LLC
            llc = estimate_llc(model, val_loader)
            llcs.append(llc)
            
            pbar.set_description(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LLC: {llc:.4f}")
            model.train()
    
    return train_losses, val_losses, llcs

def run_experiments(task_diversities=[1, 2], seeds=5):
    results = {}
    for M in task_diversities:
        results[M] = []
        for seed in range(seeds):
            run = Run(
                num_tasks=M,
                seq_length=16,
                input_dim=8,
                num_steps=5000,
                seed=seed,
                wandb_logging=False
            )
            train_losses, val_losses, llcs = run.train()
            results[M].append({
                'run': run,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'llcs': llcs
            })
    return results
```

The plotting function remains the same. This version should be more efficient since we're estimating the LLC in real-time during training rather than saving and reloading checkpoints. Would you like to try this version?



User: so currently the llc estimates take quite long. The way I understand it is that you are estimating the LLC by adding noise to the model params and then seeing how much that impacts the model's loss over the entire data? and then you do that for n_samples? Is my understanding correct? If so, shouldn't it be possible to estimate LLC on a subset of the data to speed things up? Also, it seems like you aren't reverting the model to it's original state. Finally, it seems like you aren't comparing the perturbed model to the original model but just the perturbed model to the perturbed model.

Assistant: You're absolutely right on all counts! Let me fix these issues and optimize the LLC estimation:

```python
def estimate_llc(model, loader, n_samples=100, beta=1.0, gamma=1.0, n_batches=10):
    """
    Estimate Local Learning Coefficient using SGLD sampling
    Args:
        model: the model to estimate LLC for
        loader: data loader
        n_samples: number of SGLD samples
        beta: inverse temperature
        gamma: localization strength
        n_batches: number of batches to use for estimation (speed up)
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Initialize accumulator for LLC estimation
    loss_diff_sum = 0
    
    # Get a subset of batches for faster estimation
    eval_batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        eval_batches.append(batch)
    
    # Get baseline loss with original parameters
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            x, y = x.to(device), y.to(device)
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    for _ in range(n_samples):
        # Load original state
        model.load_state_dict(original_state)
        
        # SGLD step
        noise_scale = np.sqrt(2.0 / (beta * gamma))
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
        
        # Compute loss at perturbed parameters
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                x, y = x.to(device), y.to(device)
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        # Accumulate difference from original loss
        loss_diff_sum += perturbed_loss - original_loss
    
    # Restore original parameters
    model.load_state_dict(original_state)
    
    # Compute LLC estimate
    # Scaling by total dataset size / subset size to account for using subset
    scale_factor = len(loader.dataset) / (len(eval_batches) * loader.batch_size)
    llc = (loss_diff_sum / n_samples) * scale_factor * beta
    
    return llc
```

The key changes are:
1. Using only a subset of batches for estimation
2. Computing the original loss first and comparing perturbed losses to it
3. Properly restoring the model state after each perturbation
4. Adding proper scaling to account for using a subset of data

This should be significantly faster while still giving reasonable LLC estimates. Would you like to try this optimized version?

Also, we could potentially further optimize by:
1. Reducing n_samples (currently 100)
2. Further reducing n_batches (currently 10)
3. Adjusting beta and gamma parameters

What do you think about these potential optimizations?


User: In the original paper, aren't the purturbations of each sample based off of the previous perturbations?

Assistant: Ah yes, you're right! In the original paper, they use Stochastic Gradient Langevin Dynamics (SGLD) where each perturbation builds on the previous one, creating a Markov chain. Here's the corrected version:

```python
def estimate_llc(model, loader, n_samples=100, beta=1.0, gamma=1.0, n_batches=10, sgld_lr=1e-4):
    """
    Estimate Local Learning Coefficient using SGLD sampling
    Args:
        model: the model to estimate LLC for
        loader: data loader
        n_samples: number of SGLD samples
        beta: inverse temperature
        gamma: localization strength
        n_batches: number of batches to use for estimation
        sgld_lr: learning rate for SGLD
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Initialize accumulator for LLC estimation
    loss_diff_sum = 0
    
    # Get a subset of batches for faster estimation
    eval_batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        eval_batches.append(batch)
    
    # Get baseline loss with original parameters
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            x, y = x.to(device), y.to(device)
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    # Initialize SGLD chain at original parameters
    current_state = deepcopy(original_state)
    model.load_state_dict(current_state)
    
    for _ in range(n_samples):
        # SGLD step
        # First compute gradient
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            x, y = x.to(device), y.to(device)
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        # Add L2 term for localization
        for param in model.parameters():
            orig_param = original_state[param.name]
            loss += (gamma/2) * ((param - orig_param)**2).sum()
            
        loss.backward()
        
        # Update parameters with gradient and noise
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
        
        # Compute perturbed loss
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                x, y = x.to(device), y.to(device)
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        # Accumulate difference from original loss
        loss_diff_sum += perturbed_loss - original_loss
        
        model.train()  # Set back to train for next SGLD step
    
    # Restore original parameters
    model.load_state_dict(original_state)
    
    # Compute LLC estimate
    scale_factor = len(loader.dataset) / (len(eval_batches) * loader.batch_size)
    llc = (loss_diff_sum / n_samples) * scale_factor * beta
    
    return llc
```

The key changes are:
1. Using SGLD to create a Markov chain of perturbations
2. Each step uses the gradient from the current state plus noise
3. Including the localization term in the loss
4. Parameters are updated sequentially rather than independently sampled

This should better match the method used in the papers. Would you like to try this version?

User: I'm having a bit of trouble interpreting the LLC. Given that a low LLC is supposed to correspond to a degenerate model (I'm interpreting this to mean that the model's weights are 'sensitive', i.e. a slight perturbation results in a large behavior change), then I'd expect that a small diff in loss between the original and perturbed model would mean a non-degenerate model. However, the function you wrote seems to indicate the opposite. Can you help me clear out my confusions?

Assistant: Let me help clarify this:

1. The LLC (λ) measures model complexity and degeneracy, where:
- Higher LLC = more complex, less degenerate model
- Lower LLC = simpler, more degenerate model

2. In the context of loss differences:
- If small perturbations cause large changes in loss → model is sensitive/unstable → more degenerate → lower LLC
- If small perturbations cause small changes in loss → model is stable → less degenerate → higher LLC

You're right - there seems to be an issue in our implementation. The current formula:
```python
llc = (loss_diff_sum / n_samples) * scale_factor * beta
```
suggests that larger loss differences lead to higher LLC, which is backwards.

Let's modify the implementation to reflect the correct relationship:

```python
def estimate_llc(model, loader, n_samples=100, beta=1.0, gamma=1.0, n_batches=10, sgld_lr=1e-4):
    """
    Estimate Local Learning Coefficient using SGLD sampling.
    Higher LLC indicates less degeneracy (more complex model).
    Lower LLC indicates more degeneracy (simpler model).
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Get a subset of batches for faster estimation
    eval_batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        eval_batches.append(batch)
    
    # Get baseline loss with original parameters
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            x, y = x.to(device), y.to(device)
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    # Initialize SGLD chain at original parameters
    current_state = deepcopy(original_state)
    model.load_state_dict(current_state)
    
    # Track squared loss differences (to measure stability)
    squared_diff_sum = 0
    
    for _ in range(n_samples):
        # SGLD step
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            x, y = x.to(device), y.to(device)
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        # Add L2 term for localization
        for param in model.parameters():
            orig_param = original_state[param.name]
            loss += (gamma/2) * ((param - orig_param)**2).sum()
            
        loss.backward()
        
        # Update parameters with gradient and noise
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
        
        # Compute perturbed loss
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                x, y = x.to(device), y.to(device)
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        # Accumulate squared difference from original loss
        diff = perturbed_loss - original_loss
        squared_diff_sum += diff * diff
        
        model.train()
    
    # Restore original parameters
    model.load_state_dict(original_state)
    
    # Compute LLC estimate
    # Higher variance in loss differences → lower LLC (more degenerate)
    # Lower variance in loss differences → higher LLC (less degenerate)
    scale_factor = len(loader.dataset) / (len(eval_batches) * loader.batch_size)
    avg_squared_diff = squared_diff_sum / n_samples
    llc = 1.0 / (avg_squared_diff * scale_factor * beta + 1e-6)  # add small constant to avoid division by zero
    
    return llc
```

The key changes are:
1. We now track squared differences in loss to measure variance/stability
2. We invert the relationship so that higher variance leads to lower LLC
3. Added a small constant to avoid division by zero

This should now better align with the theoretical interpretation where:
- High stability (small loss differences) → high LLC → complex model
- Low stability (large loss differences) → low LLC → degenerate model

Does this help clarify things? Let me know if you'd like me to explain any part in more detail!


User: thanks. I'm getting quite the zig-zag behavior when plotting the LLC which is in contrast to the original paper. Do you have any ideas what causes this?

Assistant: The zig-zag behavior could be due to several factors:

1. SGLD Chain Stability:
- We might need a longer burn-in period for the SGLD chain to stabilize
- The learning rate might be too high, causing unstable jumps
- We might need more samples to get a smoother estimate

2. Evaluation Consistency:
- Using different random batches each time we estimate LLC could introduce noise
- We might want to fix our evaluation batches at the start of training

Here's a modified version that might help:

```python
def estimate_llc(model, loader, n_samples=100, burn_in=50, beta=1.0, gamma=1.0, n_batches=10, sgld_lr=1e-5):
    """
    Estimate LLC with more stable SGLD sampling
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Get a subset of batches for faster estimation
    eval_batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        eval_batches.append([x.to(device) for x in batch])
    
    # Get baseline loss with original parameters
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    # Initialize SGLD chain
    current_state = deepcopy(original_state)
    model.load_state_dict(current_state)
    
    squared_diff_sum = 0
    
    # Burn-in period
    for _ in range(burn_in):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        for param in model.parameters():
            orig_param = original_state[param.name]
            loss += (gamma/2) * ((param - orig_param)**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
    
    # Actual sampling
    for _ in range(n_samples):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        for param in model.parameters():
            orig_param = original_state[param.name]
            loss += (gamma/2) * ((param - orig_param)**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
        
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        diff = perturbed_loss - original_loss
        squared_diff_sum += diff * diff
        
        model.train()
    
    model.load_state_dict(original_state)
    
    scale_factor = len(loader.dataset) / (len(eval_batches) * loader.batch_size)
    avg_squared_diff = squared_diff_sum / n_samples
    llc = 1.0 / (avg_squared_diff * scale_factor * beta + 1e-6)
    
    return llc

# In the training loop, we might want to fix the evaluation batches:
def train_model(model, train_loader, val_loader, num_steps=5000, learning_rate=3e-3, 
                weight_decay=0., device="cuda" if torch.cuda.is_available() else "cpu",
                wandb_logging=True, checkpoint_freq=100):
    
    # Fix evaluation batches for LLC estimation at start
    llc_eval_batches = []
    for i, batch in enumerate(val_loader):
        if i >= 10:  # n_batches
            break
        llc_eval_batches.append([x.to(device) for x in batch])
    
    # Rest of training loop...
    # When estimating LLC, pass the fixed eval_batches
```

Key changes:
1. Added burn-in period for SGLD chain
2. Reduced SGLD learning rate
3. Fixed evaluation batches
4. Moved device transfer of batches outside the LLC estimation loop

You might also want to try:
1. Increasing n_samples
2. Adjusting beta and gamma
3. Smoothing the LLC curve during plotting

Would you like to try these modifications?


User: yeah I think the fixed eval batches make sense. Can you adjust the code to incorporate this?

Assistant: Here's the modified code with fixed evaluation batches:

```python
def train_model(model, train_loader, val_loader, num_steps=5000, learning_rate=3e-3, 
                weight_decay=0., device="cuda" if torch.cuda.is_available() else "cpu",
                wandb_logging=True, checkpoint_freq=100):
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Fix evaluation batches for LLC estimation at start
    llc_eval_batches = []
    for i, batch in enumerate(val_loader):
        if i >= 10:  # n_batches
            break
        llc_eval_batches.append([x.to(device) for x in batch])
    
    train_losses = []
    val_losses = []
    llcs = []
    
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=num_steps)
    
    for step, (x, y, _) in pbar:
        if step >= num_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(x, y)
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if step % checkpoint_freq == 0:
            model.eval()
            # Compute validation loss
            val_loss = 0.0
            with torch.no_grad():
                for val_x, val_y, _ in val_loader:
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    val_predictions = model(val_x, val_y)
                    val_loss += criterion(val_predictions, val_y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Estimate LLC using fixed evaluation batches
            llc = estimate_llc(model, llc_eval_batches)
            llcs.append(llc)
            
            pbar.set_description(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LLC: {llc:.4f}")
            model.train()
    
    return train_losses, val_losses, llcs

def estimate_llc(model, eval_batches, n_samples=100, burn_in=50, beta=1.0, gamma=1.0, sgld_lr=1e-5):
    """
    Estimate LLC with fixed evaluation batches
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Get baseline loss with original parameters
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    # Initialize SGLD chain
    current_state = deepcopy(original_state)
    model.load_state_dict(current_state)
    
    squared_diff_sum = 0
    
    # Burn-in period
    for _ in range(burn_in):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        for param in model.parameters():
            orig_param = original_state[param.name]
            loss += (gamma/2) * ((param - orig_param)**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
    
    # Actual sampling
    for _ in range(n_samples):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        for param in model.parameters():
            orig_param = original_state[param.name]
            loss += (gamma/2) * ((param - orig_param)**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
        
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        diff = perturbed_loss - original_loss
        squared_diff_sum += diff * diff
        
        model.train()
    
    model.load_state_dict(original_state)
    
    scale_factor = len(eval_batches) * eval_batches[0][0].shape[0]  # batch_size * n_batches
    avg_squared_diff = squared_diff_sum / n_samples
    llc = 1.0 / (avg_squared_diff * scale_factor * beta + 1e-6)
    
    return llc
```

The main changes are:
1. Fixed evaluation batches are created at the start of training
2. LLC estimation function now takes these fixed batches directly
3. Simplified the scale factor calculation since we know the exact number of samples used

Would you like to try this version? We can also adjust the hyperparameters (n_samples, burn_in, beta, gamma, sgld_lr) if needed.



User: I'm getting the following error: 
```"----> 2 results = run_experiments(task_diversities=[1, 2], seeds=1, num_steps=2000)

3 frames
<ipython-input-59-faca4cf3ec77> in run_experiments(task_diversities, seeds, num_steps)
     12                 wandb_logging=False
     13             )
---> 14             train_losses, val_losses, llcs = run.train()
     15             results[M].append({
     16                 'run': run,

<ipython-input-47-d51134022bed> in train(self)
     43 
     44     def train(self):
---> 45         train_losses, val_losses, llcs = train_model(
     46             self.model,
     47             self.train_loader,

<ipython-input-58-0e5e6afc0d73> in train_model(model, train_loader, val_loader, num_steps, learning_rate, weight_decay, device, wandb_logging, checkpoint_freq)
     49 
     50             # Estimate LLC using fixed evaluation batches
---> 51             llc = estimate_llc(model, llc_eval_batches)
     52             llcs.append(llc)
     53 

<ipython-input-60-e23a85e9f228> in estimate_llc(model, eval_batches, n_samples, burn_in, beta, gamma, sgld_lr)
     34 
     35         for param in model.parameters():
---> 36             orig_param = original_state[param.name]
     37             loss += (gamma/2) * ((param - orig_param)**2).sum()
     38 

KeyError: None
"
```

Assistant: Ah, the error is because we're trying to access parameter names that don't exist. In PyTorch, parameters don't automatically have names. We need to use the state_dict keys instead. Here's the fix:

```python
def estimate_llc(model, eval_batches, n_samples=100, burn_in=50, beta=1.0, gamma=1.0, sgld_lr=1e-5):
    """
    Estimate LLC with fixed evaluation batches
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    # Store original parameters
    original_state = deepcopy(model.state_dict())
    
    # Get baseline loss with original parameters
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    # Initialize SGLD chain
    current_state = deepcopy(original_state)
    model.load_state_dict(current_state)
    
    squared_diff_sum = 0
    
    # Burn-in period
    for _ in range(burn_in):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        # Add L2 term for localization using state_dict
        for name, param in model.named_parameters():
            loss += (gamma/2) * ((param - original_state[name])**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
    
    # Actual sampling
    for _ in range(n_samples):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        # Add L2 term for localization using state_dict
        for name, param in model.named_parameters():
            loss += (gamma/2) * ((param - original_state[name])**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
        
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        diff = perturbed_loss - original_loss
        squared_diff_sum += diff * diff
        
        model.train()
    
    model.load_state_dict(original_state)
    
    scale_factor = len(eval_batches) * eval_batches[0][0].shape[0]  # batch_size * n_batches
    avg_squared_diff = squared_diff_sum / n_samples
    llc = 1.0 / (avg_squared_diff * scale_factor * beta + 1e-6)
    
    return llc
```

The key change is using `model.named_parameters()` to get both the parameter names and values, and then accessing the original state using those names. Would you like to try this version?

User: I'm still seeing spiky behavior, maybe it makes sense to smooth the curves? Although then the step changes might get lost. Any ideas? Could it be an implementation bug?

Assistant: The spiky behavior could be due to several factors. Let's try a few approaches:

1. First, let's add smoothing to our plotting function but plot both raw and smoothed curves to make sure we're not hiding important transitions:

```python
def plot_learning_curves_and_llc(results, window_size=5):
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    for M in results.keys():
        avg_train_losses = np.mean([run['train_losses'] for run in results[M]], axis=0)
        # Plot raw
        plt.plot(avg_train_losses, label=f'M={M} (raw)', alpha=0.3)
        # Plot smoothed
        smoothed = np.convolve(avg_train_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed, label=f'M={M} (smoothed)')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 3, 2)
    for M in results.keys():
        avg_val_losses = np.mean([run['val_losses'] for run in results[M]], axis=0)
        # Plot raw
        plt.plot(np.arange(0, len(avg_train_losses), 100), avg_val_losses, 
                label=f'M={M} (raw)', alpha=0.3)
        # Plot smoothed
        smoothed = np.convolve(avg_val_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(0, len(smoothed)*100, 100), smoothed, 
                label=f'M={M} (smoothed)')
    plt.title('Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Plot LLC
    plt.subplot(1, 3, 3)
    for M in results.keys():
        avg_llcs = np.mean([run['llcs'] for run in results[M]], axis=0)
        # Plot raw
        plt.plot(np.arange(0, len(avg_train_losses), 100), avg_llcs, 
                label=f'M={M} (raw)', alpha=0.3)
        # Plot smoothed
        smoothed = np.convolve(avg_llcs, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(0, len(smoothed)*100, 100), smoothed, 
                label=f'M={M} (smoothed)')
    plt.title('Local Learning Coefficient')
    plt.xlabel('Steps')
    plt.ylabel('LLC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

2. We could also try to stabilize the LLC estimation by:
- Increasing the number of samples and burn-in steps
- Adjusting the SGLD learning rate
- Using exponential moving average during training

Here's a modified LLC estimation:

```python
def estimate_llc(model, eval_batches, n_samples=200, burn_in=100, beta=1.0, gamma=1.0, sgld_lr=1e-6):
    """
    Estimate LLC with more stable parameters
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    
    original_state = deepcopy(model.state_dict())
    
    original_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in eval_batches:
            pred = model(x, y)
            original_loss += criterion(pred, y).item()
    original_loss /= len(eval_batches)
    
    # Initialize SGLD chain
    model.load_state_dict(original_state)
    
    squared_diff_sum = 0
    
    # Burn-in period with gradually increasing learning rate
    curr_lr = sgld_lr * 0.1
    for step in range(burn_in):
        curr_lr = min(sgld_lr, curr_lr * 1.05)  # Gradually increase lr
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        for name, param in model.named_parameters():
            loss += (gamma/2) * ((param - original_state[name])**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * curr_lr / beta)
                param.add_(-curr_lr * param.grad + noise)
    
    # Actual sampling with exponential moving average
    ema_squared_diff = None
    ema_alpha = 0.1
    
    for _ in range(n_samples):
        model.zero_grad()
        loss = 0
        for x, y, _ in eval_batches:
            pred = model(x, y)
            loss += criterion(pred, y)
        loss /= len(eval_batches)
        
        for name, param in model.named_parameters():
            loss += (gamma/2) * ((param - original_state[name])**2).sum()
            
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * sgld_lr / beta)
                param.add_(-sgld_lr * param.grad + noise)
        
        model.eval()
        perturbed_loss = 0
        with torch.no_grad():
            for x, y, _ in eval_batches:
                pred = model(x, y)
                perturbed_loss += criterion(pred, y).item()
        perturbed_loss /= len(eval_batches)
        
        diff = perturbed_loss - original_loss
        curr_squared_diff = diff * diff
        
        # Update exponential moving average
        if ema_squared_diff is None:
            ema_squared_diff = curr_squared_diff
        else:
            ema_squared_diff = (1 - ema_alpha) * ema_squared_diff + ema_alpha * curr_squared_diff
        
        model.train()
    
    model.load_state_dict(original_state)
    
    scale_factor = len(eval_batches) * eval_batches[0][0].shape[0]
    llc = 1.0 / (ema_squared_diff * scale_factor * beta + 1e-6)
    
    return llc
```

Key changes:
1. Increased n_samples and burn-in
2. Gradual learning rate warmup during burn-in
3. Added exponential moving average for squared differences
4. Reduced SGLD learning rate

Would you like to try these modifications? We can also investigate if there might be implementation bugs by:
1. Adding more logging/debugging statements
2. Verifying the SGLD chain is exploring properly
3. Checking if the LLC estimates are in a reasonable range
