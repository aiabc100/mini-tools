![image 1](<input (1)_images/imageFile1.png>)

![image 2](<input (1)_images/imageFile2.png>)

MEAP Edition
Manning Early Access Program

#### Build a Reasoning Model (From Scratch)

Version 8

Copyright 2026 Manning Publications

For more information on this and other Manning titles go to manning.com.

## welcome

Thank you for purchasing the MEAP for Build a Reasoning Model (From Scratch).

If you are like most people these days, LLMs are already part of your everyday toolkit.
Maybe you have asked an LLM to proofread an email, debug a tricky piece of code, or
explain a concept that sent you down an unexpected rabbit hole. Since 2022, and the
launch of ChatGPT, these models have moved from experimental novelties to essential tools
in our daily work and learning.

It’s been quite a journey to get here. The earliest GPT models, introduced in 2018, could
generate text that was more or less human-like, but they were primarily text-completion
models, largely unable to answer even simple queries, and the response quality was
nowhere near that of the LLMs we use today.

Then came instruction fine-tuning and alignment with human preferences, which
ChatGPT popularized in 2022. The techniques behind ChatGPT transformed LLMs into the
everyday problem solvers we use today. Currently, we are in the latest phase: developing
reasoning models. Reasoning is the ability for an LLM to tackle more complex problems
step-by-step.

Reasoning is one of the most exciting and important recent advances in improving LLMs,
but it’s also one of the easiest to misunderstand if you only hear the term reasoning and
read about it in theory. That’s why this book takes a hands-on approach. We’ll start with a
pre-trained base LLM and then add reasoning capabilities ourselves, step by step in code,
so you can see exactly how it works.

This book isn’t a “production deployment” manual, and we won’t use any third-party LLM
libraries. Instead, think of it as a behind-the-scenes tour where you get to develop the
machinery yourself.

By the end, you will not only understand what reasoning is and how it works, but you
will also have built it from scratch. That’s a perspective that will serve you well whether you
are using, developing, or planning to deploy LLMs in the future.

Please be sure to post any questions, comments, or suggestions you have about the
book in the liveBook discussion forum.

— Sebastian Raschka, PhD

### brief contents

- 1 Understanding reasoning models
- 2 Generating text with a pre-trained LLM
- 3 Evaluating reasoning models
- 4 Improving reasoning with inference-time scaling
- 5 Inference-time scaling via self-re nement
- 6 Training reasoning models with reinforcement learning
- 7 Improving GRPO for reinforcement learning
- 8 Distilling reasoning models for e cient reasoning


- Appendix A. References and further reading
- Appendix B. Exercise solutions
- Appendix C. Qwen3 LLM source code
- Appendix D. Using larger LLMs
- Appendix E. Batching and throughput-oriented execution
- Appendix F. Common approaches to model evaluation
- Appendix G. Building a chat interface


# 1 Understanding reasoning models

This chapter covers

What "reasoning" means for a Large Language Model

Reviewing the conventional pre-training and post-training stages of LLMs

Introducing key approaches to improving reasoning abilities in LLMs

How reasoning differs from pattern matching

Why we should build reasoning models from scratch

Welcome to the next stage of large language models (LLMs): reasoning. LLMs have
transformed how we process and generate text, but their success has been largely driven
by statistical pattern recognition. New advances in reasoning methods now enable LLMs to
tackle more complex tasks, such as solving logical puzzles and multi-step math problems.

Moreover, reasoning is an essential technique for making AI agents practical, for
example when an agent has to break a task into steps, use tools, and recover from
mistakes. Reasoning LLMs are already used in agent applications such as OpenClaw.

In Build a Reasoning Model (From Scratch), you will learn the inner workings of LLM
reasoning methods through a hands-on, code-first approach. We will start from a pre-
trained LLM and extend it step by step with reasoning capabilities. We implement these
reasoning components ourselves, from scratch, to see how these methods work in practice.

If you are curious about how LLMs themselves are built and trained, my earlier book
Build A Large Language Model (From Scratch) published by Manning (http://mng.bz/orYv)
provides a detailed coverage of these foundations, but it is not required for following along
here.

Even with a cursory knowledge of how LLMs are built, you will understand, by the end of
this book, how reasoning models work and be equipped to design, prototype, and evaluate
the main methods for improving reasoning in LLMs.

When working with reasoning models, it’s crucial to know how to check automatically
whether those answers are correct on tasks such as math, and how to turn a general-
purpose model into a smaller reasoning model.

We will use math examples throughout the book because they are easy to check
automatically, but advanced math knowledge is not required to follow along. Understanding
reasoning methods is the central focus of this book.

With its focus on practical applications and explanations, this book is written to speak to
LLM engineers, machine learning researchers, applied scientists, and software developers
alike. Understanding how reasoning models are set up helps us judge when a standard LLM
is sufficient and when extra reasoning machinery is worth the added complexity, cost, and
latency. It also helps us design better evaluations, debug failures on multi-step tasks, and
adapt reasoning techniques to real products and research workflows.

##### 1.1 Defining reasoning in the context of LLMs

In this book, I use the term reasoning in a practical engineering sense rather than a
philosophical one. In the context of LLMs, reasoning refers to a model generating
intermediate steps before producing its final answer.

These intermediate steps may appear as a visible step-by-step explanation, or they may
be enclosed in special tags such as <think>...</think> and hidden from the user. In
either case, the core idea remains the same, that is, the model is encouraged or trained to
spend tokens on intermediate problem solving instead of jumping directly to the conclusion.

Correspondingly, I use reasoning model to mean an LLM that has been improved, either
through training or prompting techniques, to produce such intermediate steps, which often
increases accuracy on complex tasks such as coding, logical puzzles, and math problems.

Before we get to the coding portions of this book in the upcoming chapters, I will briefly
discuss the main techniques that improve these reasoning behaviors and how they relate to
pattern matching and logical reasoning. This will lay the groundwork for further discussions
on how LLMs are currently built, how they handle reasoning tasks, and what they are good
and not so good at.

###### CHAIN-OF-THOUGHT (COT)

This style of intermediate-step generation is often referred to as chain-of-thought
(CoT). Researchers and engineers often say that the model "thinks" through the
problem step by step, meaning that it makes its intermediate reasoning process
explicit and easier to follow. In this book, I use the terms reasoning and thinking in
that common engineering sense. This does not imply that LLMs actually reason or
think in the same way humans do.

Figure 1.1 illustrates how a conventional LLM generates the answer to a user's question.

![image 3](<input (1)_images/imageFile3.png>)

- Figure 1.1 A simplified illustration of how a conventional, non-reasoning LLM might respond to a question
with a short answer.


As shown in figure 1.1, a conventional LLM might not show how it came up with its answer.
While the answer might be correct it doesn't help the user understand the process behind
the answer.

Figure 1.2 illustrates a simple example of multi-step (CoT) reasoning in an LLM.

![image 4](<input (1)_images/imageFile4.png>)

- Figure 1.2 A simplified illustration of how a reasoning LLM might tackle a multi-step reasoning task using a
chain-of-thought. Rather than just recalling a fact, the model combines several intermediate reasoning steps
to arrive at the correct conclusion. The intermediate reasoning steps may or may not be shown to the user,
depending on the implementation.


LLM-produced intermediate reasoning steps, as shown in figure 1.2, look very much like a
person articulating internal thoughts aloud. Yet how closely these methods (and the
resulting reasoning processes) mirror human reasoning remains an open research question,
one this book does not attempt to answer.

While figure 1.2 is a typical example of chain-of-thought reasoning, it is important to
emphasize that LLM reasoning differs from traditional, deterministic reasoning.

For instance, a symbolic logic engine or theorem prover follows strict, rule-based steps
that guarantee consistency and correctness. A symbolic logic engine works like following a
recipe where every step is fixed and must produce the same result each time. To use a
cooking analogy, if you follow the steps exactly, you always end up with the same dish.

In contrast to a symbolic logic engine, an LLM generates reasoning autoregressively,
which means that it is predicting one token at a time based on statistical patterns in its
training data. As a result, the LLM's "reasoning steps" are not guaranteed to be logically
sound, even if they look convincing.

This book focuses on explaining and implementing the fundamental techniques that
improve LLM-based reasoning and thus make LLMs better at handling complex tasks. My
hope is that by gaining hands-on experience with these methods, you will be better
prepared to understand and improve those reasoning methods being developed and maybe
even explore how they compare to human reasoning.

###### LLM VERSUS HUMAN REASONING

Reasoning processes in LLMs may superficially resemble human thought, particularly
in how intermediate steps are articulated. It is important to recognize a key
difference: humans can engage in deterministic reasoning by deliberately applying
rules of logic or by reasoning over an internal model of the world. Deterministic here
means that if we start with the same facts and follow the same steps, we will always
reach the same conclusion. In contrast, current LLM reasoning is probabilistic,
meaning that it generates one token at a time based on statistical patterns in
training data, without guarantees of logical consistency.

Humans often reason by consciously manipulating concepts, intuitively
understanding abstract relationships, or generalizing from a few examples. LLMs, by
contrast, work by picking up patterns from huge amounts of text rather than relying
on built-in reasoning rules or any kind of conscious thought.

In short, although the outputs of reasoning-enhanced LLMs can appear human-like,
the underlying mechanisms differ substantially and remain an active area of
exploration.

##### 1.2 Understanding the standard LLM training pipeline

Now, let’s briefly summarize how conventional (non-reasoning) LLMs are typically trained so
that we can understand where their limitations lie. This background will also help frame our
upcoming discussions on the differences between pattern matching and logical reasoning.

Before applying any specific reasoning methodology, conventional LLM training is usually
structured into two stages: pre-training and post-training, which are illustrated in figure

- 1.3.


Some recent papers also distinguish a mid-training stage between them, for example to
continue training a model on code, math, or long-context data, but to keep the terminology
simple in this book, I group that stage under pre-training.

![image 5](<input (1)_images/imageFile5.png>)

- Figure 1.3 Overview of a typical LLM training pipeline. The process begins with an initial model initialized with
random weights, followed by pre-training on large-scale text data to learn language patterns by predicting the
next token. Post-training then refines the model through instruction fine-tuning and preference fine-tuning,
which enables the LLM to follow human instructions better and align with human preferences.


In the pre-training stage of a typical LLM training pipeline, as shown in figure 1.3, LLMs are
trained on massive amounts of unlabeled text, often many terabytes of data or trillions of
tokens, which includes books, websites, research articles, and many other sources. The
pre-training objective (goal) for the LLM is to learn to predict the next word (i.e., token) in
these texts.

###### WORDS AND TOKENS

A token is a small unit of text that a language model processes. A token can be a full
word, part of a word, or even punctuation, depending on how the text is split by a
so-called tokenizer.

For example, the sentence "An LLM can be useful." might be broken into tokens like
"An", " L", "LM", " can", " be", " useful", and "." by a common tokenizer. These
tokens are then converted into numerical IDs that the model can ingest.

A tokenizer is a component that is not directly part of the LLM itself but is
nonetheless a critical component of the LLM text processing and generation pipeline.
We will see how tokenization works in practice in the next chapter.

LLMs become highly capable when pre-trained on massive datasets, which typically involves
several terabytes of text (equivalent to trillions of tokens). This training requires thousands
of GPUs running for many months and can cost millions of dollars. Here, "capable" means
that the LLMs begin to generate text that closely resembles human writing. Also, to some
extent, pre-trained LLMs will begin to exhibit so-called emergent properties, which means
that they will be able to perform tasks that they were not explicitly trained to do, including
translation, code generation, and so on.

These pre-trained models merely serve as base models for the post-training stage, which
uses two key techniques: supervised fine-tuning (often abbreviated as SFT in the literature
and also known as instruction tuning) and preference tuning (often implemented via a
technique called Reinforcement Learning with Human Feedback) to teach LLMs to respond
to user queries, which are illustrated in figure 1.4.

![image 6](<input (1)_images/imageFile6.png>)

- Figure 1.4 Example responses from a language model at different training stages. The prompt asks for a
summary of the relationship between sleep and health. The pre-trained LLM produces a relevant but
unfocused answer without directly following the instructions. The instruction-tuned LLM generates a concise
and accurate summary aligned with the prompt. The preference-tuned LLM further improves the response by
using a friendly tone and engaging language, which makes the answer more relatable and user-centered.


As shown in figure 1.4, instruction tuning improves an LLM's capabilities of personal
assistance-like tasks such as question-answering, summarizing and translating text, and
many more. The preference tuning stage then refines these capabilities. As the term
implies, preference tuning helps tailor responses to user preferences. Some readers may be
familiar with terms like Reinforcement Learning with Human Feedback or RLHF, which are
specific techniques to implement preference tuning.

In short, we can think of pre-training as "raw language prediction" (via next-token
prediction) that gives the LLM some basic properties and capabilities to produce coherent
texts. The post-training stage then improves the task understanding of LLMs via instruction
tuning and refines the LLM to create answers with preferred stylistic choices via preference
tuning.

It is worth noting that even an instruction-tuned model is not yet a "chatbot." A chat
interface adds another layer that guides the model's responses in an interactive, multi-turn
setting. This typically involves a system prompt, conversation history management, and
other orchestration (an example of this is implemented in appendix G).

NOTE These pre-training and post-training stages mentioned above are covered in my book "Build A
Large Language Model (From Scratch)" (http://mng.bz/orYv) published by Manning. The book you
are reading now does not require detailed knowledge of these stages. We will load a model that has
already undergone the expensive pre-training and post-training stages mentioned above, so that we
can focus on the methodology that is specific to reasoning models in the subsequent chapters.

- 1.3 Improving LLM reasoning with training and inference techniques


Reasoning in the context of LLMs became popular in the public eye with the announcement
of OpenAI's o1 in ChatGPT on September 12, 2024, which popularized the concept of
reasoning in LLMs. In the announcement article (https://openai.com/index/introducing-
openai-o1-preview/), OpenAI mentioned that "We've developed a new series of AI models
designed to spend more time thinking before they respond."

Furthermore, OpenAI wrote: "These enhanced reasoning capabilities may be particularly
useful if you're tackling complex problems in science, coding, math, and similar fields."

A few months later, in January 2025, DeepSeek released the DeepSeek-R1 model and
technical report (https://arxiv.org/abs/2501.12948), which details training methodologies
to develop reasoning models, which made big waves as they not only made freely and
openly available a model that competes with and exceeds the performance of the
proprietary o1 model but also shared a blueprint on how to train such a model.

We aim to understand how these methodologies used to develop reasoning models work
by implementing similar methods from scratch.

The different approaches to developing and improving an LLM's reasoning capabilities
can be grouped into three broad categories, as illustrated in figure 1.4.

![image 7](<input (1)_images/imageFile7.png>)

- Figure 1.5 Three approaches commonly used to improve reasoning capabilities in LLMs. These methods
(inference-compute scaling, reinforcement learning, and distillation) are typically applied after the
conventional training stages (initial model training, pre-training, and post-training with instruction and
preference tuning), but reasoning techniques can also be applied to the pre-trained base model.


As illustrated in figure 1.5, these methods are typically applied to LLMs that have
undergone the conventional pre-training and post-training phases, including instruction and
preference tuning. The following list provides a brief introduction to the three common
approaches that we will examine in more detail throughout the rest of the book.

- 1. Inference-time compute scaling (also often called inference-compute
scaling, test-time scaling, or other variations) includes methods that
improve model reasoning capabilities at inference time (when a user
prompts the model) without training or modifying the underlying model
weights. The core idea is to trade off increased computational resources
for improved performance, which helps make even fixed models more
capable through techniques such as chain-of-thought reasoning, and
various sampling procedures. This topic will be the focus of chapters 4
and 5.
- 2. Reinforcement learning (RL) refers to training methods that improve a
model's reasoning capabilities by encouraging it to take actions that lead
to high reward signals. These rewards can be broad, such as task success
or heuristic scores, or they can be narrowly defined and verifiable, such
as correct answers in math problems or coding tasks.


- Unlike Inference-time compute scaling, which can improve reasoning
performance without modifying the model, RL updates the model's
weights during training. This enables the model to learn and refine
reasoning strategies through trial and error, based on the feedback it
receives from the environment. We will explore RL in more detail in
chapters 6 and 7.
- 3. Distillation involves transferring complex reasoning patterns learned by
strong, larger models into smaller or more efficient models. Within the
context of LLMs, this typically means performing supervised fine-tuning
(SFT) using high-quality labeled instruction datasets generated by a
larger, more capable model. This technique is commonly referred to as
knowledge distillation or simply distillation in LLM literature. It's important
to note that this differs slightly from traditional knowledge distillation in
deep learning, where a smaller ("student") model typically learns from
both the outputs and the logits produced by a larger ("teacher") model.
This topic is discussed further in chapter 8.


###### REINFORCEMENT LEARNING FOR REASONING AND PREFERENCE TUNING

In the context of developing reasoning models, it is important to distinguish the RL
approach here from reinforcement learning with human feedback (RLHF), which is
used during preference tuning when developing a conventional LLM as illustrated
previously in figure 1.5.

Both settings use the same underlying process (RL) but they differ primarily in how
the reward is obtained and validated (human judgments for RLHF versus automated
verifiers or environments for reasoning RL).

RLHF incorporates explicit human evaluations or rankings of model outputs as reward
signals, directly guiding the model toward human-preferred behaviors. In contrast,
RL in the context of reasoning models typically relies on automated or environment-
based reward signals, which can be more objective but potentially less aligned with
human preferences. For instance, RL in a reasoning model development pipeline
might train a model to excel at mathematical proofs by providing explicit rewards for
correctness. In contrast, RLHF would involve human evaluators ranking various
responses to encourage outputs that align closely with human standards and
subjective preferences.

##### 1.4 Pattern matching versus logical reasoning

As mentioned in the previous section, during pre-training, LLMs are exposed to vast
quantities of text and learn to predict the next token by identifying and reproducing
statistical associations in that data. This process enables them to generate fluent and
coherent text, but it is fundamentally based on learned statistical regularities rather than
explicit rules or guaranteed logical inference.

LLMs respond to prompts by generating text continuations that are statistically
consistent with the patterns seen during training. This includes many premise-to-conclusion
patterns that can look like logical inference. The key difference is that the model does not
explicitly represent premises and apply formal rules, but instead predicts likely
continuations from similar examples in its training data.

Consider the following example:

###### Prompt

The capital of Germany is…

###### Response

Berlin.

An LLM producing the answer "Berlin" is not logically deducing the answer. Instead, it is
recalling a strong statistical association learned from training data. This behavior reflects
what we refer to as pattern matching, which means that the model completes text based
on learned correlations and not by applying structured reasoning steps.

But what about tasks that go beyond pattern recognition, i.e., tasks where a correct
answer depends on drawing conclusions from given facts? This brings us to a different kind
of capability: logical reasoning.

Logical reasoning involves systematically coming up with conclusions using rules. Unlike
pattern matching, it depends on intermediate reasoning steps and the ability to recognize
contradictions or draw implications based on formal relationships.

Consider the following prompt as an example:

Prompt: "All birds can fly. A penguin is a bird. Can a penguin fly?"

There are two ways to evaluate this.

First, in a closed-world (prompt only) setting, from the two premises (claims,
assumptions) in the prompt ("All birds can fly" and "A penguin is a bird"), the valid answer
is "Yes, a penguin can fly."

Second, in an open-world (with background knowledge) setting, if we also allow
background knowledge not included in the prompt (for example, that penguins cannot fly),
this external fact conflicts with the conclusion derived from the premises, as shown in
figure 1.6. A reasoning system should notice the inconsistency and either ask for
clarification or weaken the first statement (for example, "Most birds can fly, with exceptions
such as penguins").

![image 8](<input (1)_images/imageFile8.png>)

- Figure 1.6 Contradictory premises lead to a logical inconsistency. From "All birds can fly" and "A penguin is a
bird," we infer "Penguin can fly." This conclusion conflicts with the established fact "Penguin cannot fly,"
which results in a contradiction.


- Figure 1.6 shows how a system based on logical reasoning could process the previously
introduced "All birds can fly..." prompt.


In contrast, a statistical (pattern-matching) LLM does not explicitly track contradictions,
such as the one shown in figure 1.6, but instead predicts based on learned text
distributions. For instance, if information such as "All birds can fly" is reinforced strongly in
training data, the model may confidently answer: "Yes, penguins can fly."

In the next section, we will look at a concrete example of how an LLM handles this "All
birds can fly..." prompt.

###### LOGICAL REASONING AND RULE-BASED SYSTEMS

Why are explicit rule-based systems not more popular? Rule-based systems were
used widely in the '80s and '90s for medical diagnosis, legal decisions, and
engineering. They are still used in critical domains (medicine, law, aerospace), which
often require explicit inference and transparent decision processes. They are hard to
implement as they largely rely on human-crafted heuristics. (Heuristics are simple
decision rules that give a good-enough answer quickly without guaranteeing an
optimal one.) In contrast, deep neural networks, including LLMs, do not implement
hand-written rules; they learn decision patterns from data and can be highly flexible
when trained at scale.

##### 1.5 Simulating reasoning without explicit rules

We saw how contradictory premises can lead to logical inconsistencies. A conventional LLM
does not explicitly track contradictions but generates responses based on learned text
distributions.

Let's see a concrete example, shown in figure 1.7, of how a non-reasoning-enhanced
LLM like GPT-4o in OpenAI's ChatGPT responds to the "All birds can fly..." prompt discussed
in the previous section.

![image 9](<input (1)_images/imageFile9.png>)

- Figure 1.7 An illustrative example of how a language model (GPT-4o in ChatGPT) appears to "reason" about a
contradictory premise.


The example in figure 1.7 shows that GPT-4o appears to answer correctly even though this
model is not considered a reasoning model, unlike OpenAI's other offerings like o1, o3, o4-
mini, and more recent GPT-5, which have been explicitly developed with reasoning
methodology.

So, how did the 4o model generate its answer? Does this mean GPT-4o explicitly reasons
logically? No, not necessarily. At a minimum 4o is highly effective at simulating logical
reasoning in familiar contexts.

GPT-4o does not implement explicit contradiction-checking and instead generates
answers based on probability-weighted patterns. This approach works well enough if
training data includes many instances correcting the contradiction (e.g., text like "penguins
cannot fly") so that the model learns a statistical association between "penguins" and "not
flying." As we see in figure 1.7, this allows the model to answer correctly without explicitly
implementing rule-based or explicit logical reasoning methodologies.

In other words, the model recognizes the contradiction implicitly because it has
frequently encountered this exact reasoning scenario during training. This effectiveness
relies heavily on statistical associations built from a lot of exposure to reasoning-like
patterns in training data.

So, even when a conventional LLM seems to perform logical deduction as shown in figure

- 1.7, it's not executing explicit, rule-based logic but is instead leveraging patterns from its
vast training data.


Nonetheless, GPT-4o's success here is a great illustration of how powerful implicit
pattern matching can become when trained at a massive scale. These types of pattern-
based reasoning models usually struggle in scenarios where:

The logical scenario is novel (not previously encountered in training
data).

Reasoning complexity is high, involving intricate, multi-step logical
relationships.

###### LOGICAL REASONING AND CURRENT REASONING LLM OFFERINGS

While GPT-4o is not officially labeled as a reasoning model, OpenAI offers several
dedicated reasoning models, including o1, o3, o4-mini, and GPT-5. Moreover, other
companies have been developing LLMs with explicit reasoning capabilities. As of this
writing, popular examples include Anthropic's Claude 4, xAI's Grok 4, Google's
Gemini 2.5, DeepSeek's R1, Alibaba's Qwen3, and many more. The techniques
employed by these models are the focus of this book. As we will see, this is achieved
without implementing a rule-based reasoning pipeline (figure 1.6 illustrates the
general idea of rule-based reasoning). Instead, the LLM learns or improves its
reasoning capabilities as a result of the modified inference and training
methodologies.

We might say that LLMs simulate logical reasoning through learned patterns, and we can
improve it further with specific reasoning methods that include inference-compute scaling
and post-training strategies like reinforcement learning, but they are not explicitly
executing any rule-based logic internally.

Moreover, it's worth mentioning that reasoning in LLMs exists on a spectrum. This means
that even before the advent of dedicated reasoning models such as OpenAI's o1 and
DeepSeek-R1, LLMs were capable of simulating reasoning behavior. For instance, these
models exhibited behaviors aligning with our earlier definition, such as generating
intermediate steps to arrive at correct conclusions. What we now explicitly label a
"reasoning model" is essentially a more refined version of this capability. And these
improved reasoning capabilities are achieved by leveraging specific inference-compute
scaling techniques (chapters 4 and 5) and targeted post-training methods, such as
reinforcement learning (chapters 6 and 7), which are designed to improve and reinforce
reasoning-like behavior.

##### 1.6 Why build reasoning models from scratch?

Following the release of DeepSeek-R1 in January 2025, improving the reasoning abilities of
LLMs has become one of the hottest topics in AI, and for good reason. Stronger reasoning
skills allow LLMs to tackle more complex problems, making them more capable across
various tasks users care about.

This shift is also reflected in a February 12, 2025, statement from OpenAI's CEO:

"We will next ship GPT-4.5, the model we called Orion internally, as our last non-chain-
of-thought model. After that, a top goal for us is to unify o-series models and GPT-
series models by creating systems that can use all our tools, know when to think for a
long time or not, and generally be useful for a very wide range of tasks."

The quote above underlines the major shift from leading LLM providers towards reasoning
models, where "chain-of-thought" refers to a prompting technique that guides language
models to reason step-by-step to improve their reasoning capabilities, which we will cover
in more detail in chapters 4 and 5.

Also noteworthy is the mention of knowing "when to think for a long time or not." This
hints at an important design consideration: reasoning is not always necessary or desirable.

For instance, reasoning models are designed to be good at complex tasks such as
solving puzzles, advanced math problems, and challenging coding tasks. They are also
often useful in agent systems, especially for task decomposition, planning, and choosing
which tools to use. They are not necessary for simpler tasks like summarization,
translation, or knowledge-based question answering. In fact, using reasoning models for
everything can be inefficient and expensive. For instance, reasoning models are typically
more expensive to use, more verbose, and sometimes more prone to errors due to
"overthinking." Also, here, the simple rule applies: use the right tool (or type of LLM) for
the task.

Reasoning models are often more expensive than non-reasoning models for two reasons.
First, they tend to produce longer outputs because they include intermediate steps that

explain how an answer is derived. As figure 1.8 illustrates, LLMs generate text one token at
a time and each token requires a full forward pass. If a reasoning model's answer is twice
as long, generation involves roughly twice as many forward passes, which increases
compute costs.

![image 10](<input (1)_images/imageFile10.png>)

- Figure 1.8 Token-by-token generation in an LLM. At each step, the LLM takes the full sequence generated so
far and predicts the next token, which may represent a word, subword, or punctuation mark depending on the
tokenizer. The newly generated token is appended to the sequence and used as input for the next step. This
iterative decoding process is used in both standard language models and reasoning-focused models.


Second, many reasoning workflows require running the model several times for a single
task, for example to sample multiple candidate solutions, call tools, or run a verifier. These
additional calls multiply the total number of tokens processed and further increase cost
beyond the single-call behavior shown in figure 1.8.

That's exactly why it helps to implement these models and methods from scratch. It's
one of the best ways to understand how they work. And if we understand how LLMs and
these reasoning models work, we can better understand these trade-offs.

###### 1.7 A roadmap to reasoning models from scratch

Now that we have discussed reasoning in LLMs from a bird's-eye view, the subsequent
chapters will guide you through the process of coding and applying reasoning methods from
scratch. At a high level, the roadmap is simple. We start from a conventional LLM, learn
how to evaluate its reasoning behavior, and then explore two broad ways to improve it,
inference techniques and training techniques. Figure 1.9 summarizes the main steps.

![image 11](<input (1)_images/imageFile11.png>)

- Figure 1.9. A high-level roadmap of what we build in this book. We start with a conventional LLM, add
evaluation methods so that we can measure progress, and then explore two broad families of reasoning
improvements, namely, inference techniques and training techniques.


Stage 1 is an explicit step in figure 1.9 because, in practice, reasoning methods are usually
applied to an existing base LLM rather than trained from scratch. So, we begin with a
conventional LLM that has already been pre-trained, and we treat it as the baseline we
want to improve.

- Stage 2 then implements the evaluation tools needed to tell whether a method actually

helps. That is why stages 3 and 4 loop back to evaluation, since after improving reasoning
behavior through inference-time methods or additional training, we return to stage 2 to
measure whether the change actually helped.

- Stage 3 improves reasoning behavior at inference time without changing the model


weights, while stage 4 improves the model itself through additional training and turns the
pre-trained LLM into a dedicated reasoning model.

Figure 1.10 adds more details and summarizes the different substeps covered in each
chapter.

![image 12](<input (1)_images/imageFile12.png>)

- Figure 1.10 A detailed roadmap of the chapter-level substeps. After loading the base model, we cover
benchmark-based and judgment-based evaluation, then inference-time methods such as advanced text
generation and voting plus self-refinement, and finally training-time methods based on reinforcement learning
and distillation.


As summarized by figure 1.10, chapter 2 introduces the base model, chapter 3 establishes
the evaluation methods, chapters 4 and 5 cover the two inference-time techniques, and
chapters 6 to 8 cover two training-time techniques.

I am looking forward to the journey ahead and hope you are as well.

- 1.8 Summary


Conventional LLM training occurs in several stages:

Pre-training, where the model learns language patterns
from vast amounts of text.

Instruction fine-tuning, which improves the model's
responses to user prompts.

Preference tuning, which aligns model outputs with
human preferences.

Reasoning methods are applied on top of a conventional LLM.

Reasoning in LLMs refers to improving a model so that it explicitly
generates intermediate steps (chain-of-thought) before producing a final
answer, which often increases accuracy on multi-step tasks.

Reasoning in LLMs is different from rule-based reasoning and it also likely
works differently than human reasoning; currently, the common
consensus is that reasoning in LLMs relies on statistical pattern matching.

Pattern matching in LLMs relies purely on statistical associations learned
from data, which enables fluent text generation but lacks explicit logical
inference.

Improving reasoning in LLMs can be achieved through:

Inference-time compute scaling, enhancing reasoning
without retraining (e.g., chain-of-thought prompting).
Reinforcement learning, training models explicitly with
reward signals.

Supervised fine-tuning and distillation, using examples
from stronger reasoning models.

Building reasoning models from scratch provides practical insights into
LLM capabilities, limitations, and computational trade-offs.

# 2 Generating text with a pre-trained LLM

This chapter covers

Setting up the code environment for working with LLMs

Using a tokenizer to prepare input text for an LLM

The step-by-step process of text generation using a pre-trained LLM

Caching and compilation techniques for speeding up LLM text generation

In the previous chapter, we discussed the difference between conventional large language
models (LLMs) and reasoning models. Also, we introduced several techniques to improve
the reasoning capabilities of LLMs. These reasoning techniques are usually applied on top of
a conventional (base) LLM.

In this chapter, we will lay the groundwork for the upcoming chapters by loading a pre-
trained base model, as illustrated in figure 2.1. Previously, we discussed how reasoning
methods are often added after the usual post-training stages. However, starting from a
base model makes it easier to see which capabilities come from the reasoning methods
themselves. In other words, the conventional LLM in this chapter is a non-reasoning LLM,
and more specifically a pre-trained base model rather than an instruction- or preference-
tuned assistant.

![image 13](<input (1)_images/imageFile13.png>)

- Figure 2.1 A mental model depicting the four main stages of developing a reasoning model. This chapter
focuses on stage 1, loading a conventional LLM and implementing the text generation functionality.


In addition to setting up the coding environment and loading a pre-trained LLM, you will
learn how to use a tokenizer to prepare text input for the model. As illustrated in figure 2.1,
you will also implement a text generation function, enabling practical use of the LLM to
generate text. This functionality will be used and further improved in later chapters.

###### 2.1 Introduction to LLMs for text generation

In this chapter, we implement all the necessary LLM essentials, from setting up our coding
environment and loading a pre-trained LLM to generating text that we will reuse and build
upon in this book. In this sense, this chapter can be understood as a setup chapter.

Even though the LLM is a base model that has only undergone pre-training, it will
already be capable of generating coherent text and, in some cases, following basic
instructions, as illustrated in figure 2.2.

![image 14](<input (1)_images/imageFile14.png>)

- Figure 2.2 An overview depicting an LLM generating a response (output text) given a user query (input text)


- Figure 2.2 summarizes the components of an LLM text generation pipeline, and we will
discuss and implement these steps in more detail later in this chapter.


###### NOTE By convention, diagrams involving neural networks such as LLMs are drawn and read vertically from bottom (inputs) to top (outputs). Arrows indicate the flow of information upward through the model.

If you have not coded an LLM or used LLMs programmatically before, this chapter will teach
you how the text generation process works. In this chapter, we will not go deep into the
internals of an LLM, such as the attention mechanism and other architecture components;
this is the topic of my other book, Build a Large Language Model (from Scratch). Note that
understanding these internals are not required for this book, and, if you are curious, you
can learn about them after you finish reading this book.

This chapter will also be helpful if you have already read the earlier Build a Large
Language Model (From Scratch) book, since it adds new material on speeding up inference
and other practical optimizations.

Before we begin implementing the components shown in figure 2.2, including input
preparation, loading the LLM, and generating text, we first need to set up our coding
environment. This is the focus of the next section.

##### 2.2 Setting up the coding environment

This section describes two main ways to set up your Python coding environment to follow
along with the examples in this book: a straightforward pip-based setup and a uv-based
workflow. I recommend reading this section in its entirety before deciding which option is
best for you.

If you are reading this book, you have probably coded in Python before. In this case, the
simplest way to install dependencies, if you already have a Python environment set up
(with Python 3.10 or newer), is to use Python's package installer (pip) in your terminal.

If you have downloaded the code from the publisher's website, use the
requirements.txt file to install the required Python libraries used throughout this book:

pip install -r requirements.txt

Alternatively, to install the required packages directly without downloading the
requirements.txt file, use:

pip install -r https://raw.githubusercontent.com/\
rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt

###### PYTHON PACKAGES USED IN THIS CHAPTER

If you prefer to install only the packages used in this chapter, you can start with the
following command:

pip install torch>=2.10.0 tokenizers>=0.22.2 reasoning-from-scratch

However, PyTorch installation can be platform- and hardware-specific, especially if
you want GPU acceleration. If this command does not install the right build for your
system, I recommend using the installation selector on the official PyTorch website
(https://pytorch.org/get-started/locally/) first and then installing the remaining
packages separately.

torch refers to PyTorch, a widely used deep learning library that
provides tools for building and training neural networks.

tokenizers is a library that provides efficient tokenization algorithms,
used to prepare input data for LLMs.

reasoning-from-scratch is a custom library that I developed for this
book. It includes all the code examples implemented throughout the
chapters, along with additional utility functions we will be using.

While pip is the canonical way to install Python packages, my preferred way to use Python
is via the widely recommended uv Python package and project manager instead. Like many
others, I now recommend uv because it is significantly faster, and it handles dependency
resolution more reliably than pip. It also creates isolated environments automatically and
comes with its own Python executable (but will use the system Python if a compatible
version is already installed) so it is also a great option if you do not have Python installed
on your system yet.

Figure 2.3 outlines the 4-step process from installing uv to getting ready to execute the
code in this chapter, which we will cover in the remainder of this section.

![image 15](<input (1)_images/imageFile15.png>)

- Figure 2.3 Installing and using the uv Python package and project manager via the macOS terminal


Note that figure 2.3 steps through the uv installation and usage on a macOS terminal, but
uv is supported by Linux and Windows as well.

- 1) To install uv, run the installation for your OS from the official website: https://docs.

astral.sh/uv/getting-started/installation/

- 2) Next, clone the GitHub repo:


git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git

Here, the --depth 1 option tells git to perform a shallow clone, which means it only
downloads the latest version of the code without the full version history. This makes the
download faster and uses less space.

If you don't have git installed, you can also manually download the source code
repository from the publisher's website or by opening this link in your browser:
https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip (unzip it
after downloading).

- 3) Next, in the terminal, navigate to the reasoning-from-scratch folder.
- 4) Inside the reasoning-from-scratch folder, execute:


uv run jupyter lab

The command above will launch JupyterLab, where you can open a blank Jupyter notebook
to type and execute code or open the chapter 2 notebook that contains all the code covered
in this chapter. (You do not need to create or activate a virtual environment first.)

###### TIP Python script files can be executed via uv run script-name.py.

The above uv run... command also sets up a local virtual environment (usually inside an
invisible .venv/ folder) and installs all dependencies from the pyproject.toml file inside
the reasoning-from-scratch folder automatically. So, the manual installation of code
dependencies via the requirements file is not needed. However, if you plan to install
additional packages, you can use the following command:

uv add packagename

The supplementary code repository contains additional installation instructions and details
inside the ch02 subfolder if needed.

###### 2.3 Understanding hardware needs and recommendations

You may have heard that training LLMs is very expensive. For leading LLM companies, it is
not uncommon to spend anywhere between 1-10 million dollars on the small end and over
50 million dollars on the high end in terms of compute costs to train a new base model LLM
before even adding any reasoning techniques.

For example, the DeepSeek V3 model, which serves as the base checkpoint for the
DeepSeek R1 reasoning system, was trained on 2,048 Nvidia H800 GPUs for about 11
weeks with an estimated cost of 5.5 million USD. (DeepSeek V3 is one of the few more
recent models with a fully transparent compute disclosure.)

Furthermore, according to the technical report, the final training run used 14.8 million
GPU-hours of compute. The energy usage was roughly 620 MWh, which is roughly the
amount of electricity that an average American household uses in about 55 years.

This resource requirements and high price tag would make the development of an LLM
unfeasible for me and most readers. So, we are going to use a relatively small (but
capable) pre-trained LLM on top of which we implement reasoning techniques.

Note that this smaller LLM is a scaled-down version that otherwise follows the same
architecture as contemporary state-of-the-art models. And the reasoning methods that we
will apply are the same as those used by larger LLMs. The difference is that the smaller LLM
allows us to explore these methods in a budget-friendly way.

As an analogy, imagine you are curious to learn how cars work. If you are new to cars,
as a learning exercise, you probably wouldn't start out building an expensive Ferrari right
away. Instead, you would, for example, create a smaller car like a Volkswagen Beetle to
start with, which still teaches you a lot about how engines and the transmission work. On
the contrary, I would even say that working on a smaller car helps you better understand
how the engine and transmission work because it gets complicated refinements and other
details out of the way.

While we will use a relatively small model for these educational purposes in this book,
the usage, development, and application of the reasoning techniques are still
computationally intensive, and later chapters, such as chapters 5-8, will benefit from using
a GPU.

If you followed the previous section, you should have PyTorch installed, which you can
use to see if your computer has a PyTorch-supported GPU by executing the following
PyTorch code in Python:

import torch

print(f"PyTorch version {torch.__version__}")

if torch.cuda.is_available():
print(f"CUDA/ROCm GPU: {torch.cuda.get_device_name(0)}")

elif torch.xpu.is_available():
print(f"Intel GPU: {torch.xpu.get_device_name(0)}")

elif torch.backends.mps.is_available():
print("Apple Silicon GPU")

else:
print("Only CPU")

Depending on your machine, the code may return:

PyTorch version 2.10.0
Only CPU

Don't worry if your machine does not have a GPU to run the code. Chapters 2-5 can be
executed in a reasonable time on a CPU.

###### NOTE If you want GPU acceleration and are unsure which PyTorch build to install, use the official PyTorch install selector (https://pytorch.org/get-started) to choose the correct CPU, CUDA, or ROCm build for your system. For this book, any recent supported PyTorch CUDA build is fine.

Depending on the chapter, the code will automatically use an NVIDIA GPU if available,
otherwise run on the CPU (or Apple Silicon GPU if recommended for a particular section or
chapter). II will provide more information in the respective sections and chapters.

Like many other AI researchers who work on and with LLMs daily, I don't have a machine
with the necessary GPU hardware to train LLMs at home and use cloud resources instead. If
you are looking for cloud provider options, my personal preference is Lightning AI Studio
(https://lightning.ai/), due to its ease of use and feature support, as shown in figure 2.4.
Alternatively, Google Colab (https://colab.research.google.com/) is another good choice.

![image 16](<input (1)_images/imageFile16.png>)

- Figure 2.4 An overview of the Lightning AI GPU cloud platform in a web browser. The interface supports
Python scripts, Jupyter notebooks, terminal access, and lets users switch between CPU and various GPU types
based on their compute needs.


As of this writing, Lightning AI also offers users free compute credits after the sign-up and
verification process, which can be used for the different GPU choices shown in figure 2.4.
(As mentioned before, a GPU is not needed for this chapter; however, if you want to use a
GPU, the L4 GPU is more than sufficient for this chapter.

NOTE For disclosure, I helped build and launch the Lightning AI platform in 2023 but no longer have
any financial interest in the company. I am not sponsored to recommend it and pay for it myself. I
use it because I find it the most convenient option. It supports multiple types of GPUs, allows easy
switching between them and back to CPU to save costs, and lets me pause or resume environments
without redoing the setup.

The supplementary code repository contains additional GPU platform recommendations
inside the ch02 subfolder if needed.

###### USING PYTORCH

In this section, we imported and used the PyTorch library, which is currently the most
widely used general-purpose library. We will use it throughout this book to run and
train LLMs, including the reasoning methods we will develop. If you are new to
PyTorch, to get the most out of this book, I recommend reading through my PyTorch
in One Hour: From Tensors to Training Neural Networks on Multiple GPUs tutorial,
which is freely available on my website at https://sebastianraschka.com/teaching/
pytorch-1h.

###### 2.4 Preparing input texts for LLMs

In this section, we explore how to use a tokenizer to process input and output text for an
LLM, as shown in figure 2.5, which expands on the input and output preparation steps
shown earlier in figure 2.2 to provide a more detailed view of the tokenization pipeline.

This matters throughout the rest of the book, because every prompt, reasoning trace,
and generated answer ultimately has to be represented as token IDs before the model can
process it.

![image 17](<input (1)_images/imageFile17.png>)

- Figure 2.5 A simplified illustration of how an LLM receives input data and generates output. The user-provided
text is tokenized into IDs using the tokenizer's encode method, which are then processed by the LLM to
generate output token IDs. These are decoded back into human-readable text using the tokenizer's decode
method.


To see how this works in practice, we will begin by loading a tokenizer from this book's
reasoning-from-scratch Python package, which should have been installed according to
the instructions in section 2.2.

To download the tokenizer files (corresponding to the Qwen3 0.6B base LLM, which we
will introduce in the next section), run:

from reasoning_from_scratch.qwen3 import download_qwen3_small
download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")

This will display a progress bar similar to:

tokenizer-base.json: 100% (6 MiB / 6 MiB)

The command downloads the tokenizer-base.json file (approximately 6 megabytes in
size) and saves it in a qwen3 subdirectory.

Now, we can load the tokenizer settings from the tokenizer file into the Qwen3Tokenizer:

from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer

tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

Since we have not loaded the LLM yet (the central component shown in figure 2.5), we will
first do a simpler dry run using just the tokenizer. Specifically, we will do a tokenization
round-trip, that is, we will encode a text into token IDs and then decode those IDs back
into text, as illustrated in figure 2.6.

![image 18](<input (1)_images/imageFile18.png>)

- Figure 2.6 A demonstration of the round-trip tokenization process using a tokenizer. The user-provided input
text is first converted into token IDs using the encode method, and then accurately reconstructed back into
the original text using the decode method.


The following code snippet implements the encoding process shown at the bottom of figure
2.6:

prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)

And the following code implements the decoding process, converting the token IDs back
into text, shown at the top of figure 2.6:

text = tokenizer.decode(input_token_ids_list)
print(text)

Based on the printed results, we can see that the tokenizer reconstructed the original input
prompt from the token IDs:

'Explain large language models.'

Before we move on to the LLM, let's take a look at the token IDs that were generated by
the encode method. The following code prints each token ID and its corresponding decoded
string to help illustrate how the tokenizer works:

for i in input_token_ids_list:
print(f"{i} --> {tokenizer.decode([i])}")

The output is as follows:

840 --> Ex
20772 --> plain
3460 --> large
4128 --> language
4119 --> models
13 --> .

As shown in the output, the original text is split into six token IDs. Each token represents a
word or subword, depending on how the tokenizer segments the input.

For example, "Explain" was split into two separate tokens, "Ex" and "plain". This is
because the tokenizer algorithm uses a subword-based method based on Byte Pair
Encoding (BPE). BPE can represent both common and rare words using a mix of full words
and subword units. Spaces are also often included in tokens (for example, " large"), which
helps the LLM detect word boundaries.

The Qwen3Tokenizer has a vocabulary of about 151,000 tokens, which is considered
relatively large as of this writing (for comparison, the early GPT-2 has a vocabulary size of
approximately 50,000 tokens, and Llama 3 has a vocabulary size of approximately 128,000
tokens).

A larger vocabulary in a language model increases model size because the embedding
and output layers must store more token representations, and it can also increase the per-
token compute cost of producing the next-token probabilities. A larger vocabulary also
allows more words to be represented as single tokens rather than being split into subword
components. This can reduce sequence length. For example, splitting a word like "Explain"
into "Ex" and "plain" results in more input tokens. More tokens lead to longer input
sequences, which increases processing time and resource usage. So the tradeoff is between
a larger vocabulary with somewhat higher per-token cost and a smaller vocabulary that
often produces longer token sequences. For instance, doubling the number of tokens in a
sequence can roughly double the computational cost of running the model, since the model
has to process and generate more tokens overall.

Unfortunately, a detailed coverage and from-scratch implementation of a tokenizer is
outside the scope of this book. Interested readers can find additional resources, including
my from-scratch implementation, in the further resources and reading sections in appendix
A.

###### EXERCISE 2.1: ENCODING UNKNOWN WORDS

Experiment with the tokenizer to see if and how it handles unknown words. For this,
get creative and make up words that don't exist. Also, if you speak multiple
languages, try to encode words in a different language than English.

##### 2.5 Loading pre-trained models

In the previous section, we loaded and familiarized ourselves with the tokenizer that
prepares the input data for an LLM and converts LLM outputs back into a human-readable
text representation. In this section, we will load the LLM itself, as shown in the overview in
figure 2.7.

Once this base model is in place, the later chapters will build directly on top of it by
evaluating it, prompting it in different ways, and eventually training it into a stronger
reasoning model.

![image 19](<input (1)_images/imageFile19.png>)

- Figure 2.7 An overview of the four key stages in developing a reasoning model in this book. This section
focuses on loading pre-trained LLM in Stage 1.


As mentioned previously, this book uses Qwen3 0.6B as a pre-trained base model. In this
section, we load its pre-trained weights, as shown in figure 2.7. The "0.6B" in the model
name indicates that the model contains approximately 0.6 billion weight parameters.

Why Qwen3? After carefully evaluating several open-weight base models, I chose Qwen3
0.6B for the following reasons:

For this book, we want a small yet capable open-weight model (open-
weight models are models whose trained weights are publicly
downloadable) that can run on consumer hardware.

Qwen3 offers both a base model (the focus of our reasoning model
development) and an official reasoning variant that we can use as a
reference for evaluation purposes.

###### NOTE The canonical spelling of "Qwen3" does not include whitespace, whereas, for example, "Llama 3" does.

In line with the spirit of building things "from scratch," this book uses a custom
reimplementation of Qwen3 that I wrote in pure PyTorch, which is entirely independent of
external LLM libraries. The emphasis of this reimplementation is on code readability and
tweakability, in case you want to modify it later for your own experiments. Despite being
built from scratch, this implementation remains fully compatible with the original pre-
trained Qwen3 model weights.

This book does not cover the Qwen3 code implementation in depth. This topic alone
would fill an entire separate book, similar to my other book, Build A Large Language Model
(From Scratch). Instead, this Build A Reasoning Model (From Scratch) book specifically
focuses on implementing reasoning methods on top of a base model, in this case, Qwen3.

NOTE This reimplemented Qwen3 LLM runs entirely locally, just like any other neural network
implemented in PyTorch. There are no server-side components or external API calls involved. All
model usage happens on your own machine, and your data stays on your device. If you are
concerned about privacy, the setup we are using ensures full control over both the LLM inputs and
outputs.

For those interested in additional details about Qwen3, as well as the model code, please
see appendix C.

Before we load the model, we can specify the device we are going to use, namely, CPU
or GPU. The following code in listing 2.1 will select the best-available device automatically:

- Listing 2.1 Get device automatically


def get_device(enable_tensor_cores=True):

if torch.cuda.is_available():
device = torch.device("cuda")
print("Using NVIDIA CUDA GPU")

if enable_tensor_cores:
major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) >= (2, 9):

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

else:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

elif torch.backends.mps.is_available():
device = torch.device("mps")
print("Using Apple Silicon GPU (MPS)")

elif torch.xpu.is_available():
device = torch.device("xpu")
print("Using Intel GPU")

else:
device = torch.device("cpu")
print("Using CPU")

return device

Note that if you have a modern NVIDIA GPU (based on the Ampere architecture or newer),
the get_device() function automatically enables Tensor Cores for faster matrix
multiplications when enable_tensor_cores=True. This can slightly change floating-point
rounding but does not noticeably affect results in this book, and the speed advantage on
newer cards is worth it. On non-NVIDIA devices, these settings are ignored.

Using the code from listing 2.1, we can then obtain the device as follows:

device = get_device()

While GPUs generally provide substantial speed and performance improvements, it can be
helpful to initially run the remaining code in this chapter using the CPU for compatibility and
debugging purposes. You can temporarily override the automatic selection by explicitly
setting:

device = torch.device("cpu")

After finishing the chapter and verifying the code works properly on the CPU, remove or
comment out the manual override and rerun the code. If your system has a GPU, you
should then observe improved performance.

###### NOTE The code in the remainder of this chapter was executed on a Mac Mini with an Apple M4 CPU. Performance comparisons with the Apple Silicon M4 GPU and the NVIDIA H100 GPU are included at the end of the chapter.

Before we load the model and put it onto the selected device, we first need to download
the weights for Qwen3 0.6B. These files are required to initialize the pre-trained model
correctly. In the previous section, we called the same function with tokenizer_only=True
to download only the tokenizer files. Here, we set tokenizer_only=False so that the
model weights are downloaded as well:

download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")

The output is as follows:

qwen3-0.6B-base.pth: 100% (1433 MiB / 1433 MiB)

✓ qwen3/tokenizer-base.json already up-to-date

(There is a checkmark in front of the tokenizer because we already downloaded it in the
previous section.)

After downloading the model weights via the previous step, we can now instantiate a
Qwen3Model class into which we load the pre-trained weights via PyTorch's
load_state_dict method:

from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B) #A
model.load_state_dict(torch.load(model_path)) #B
model.to(device) #C

- #A Instantiate a Qwen3 model with random weights as placeholders
- #B Load the pre-trained weights into the model
- #C Transfer the model to the designated device (e.g., "cuda")


Note that if the device setting is "cpu", the model.to(device) operation will be skipped
because the model already sits in CPU memory by default.

After executing the code above, you should see the following output (if you are not
running the code in an interactive environment like a Jupyter notebook, you have to run
print(model) to see the output):

Qwen3Model(
(tok_emb): Embedding(151936, 1024)
(trf_blocks): ModuleList(

(0-27): 28 x TransformerBlock(

(att): GroupedQueryAttention(
(W_query): Linear(in_features=1024, out_features=2048, bias=False)
(W_key): Linear(in_features=1024, out_features=1024, bias=False)
(W_value): Linear(in_features=1024, out_features=1024, bias=False)
(out_proj): Linear(in_features=2048, out_features=1024, bias=False)
(q_norm): RMSNorm()
(k_norm): RMSNorm()

)
(ff): FeedForward(

(fc1): Linear(in_features=1024, out_features=3072, bias=False)
(fc2): Linear(in_features=1024, out_features=3072, bias=False)
(fc3): Linear(in_features=3072, out_features=1024, bias=False)

)
(norm1): RMSNorm()
(norm2): RMSNorm()

)

)
(final_norm): RMSNorm()
(out_head): Linear(in_features=1024, out_features=151936, bias=False)

)

This output is a summary of the Qwen3 0.6B base model architecture, as printed by
PyTorch. It highlights the model's core components: an embedding layer, a stack of 28
transformer blocks, and a final linear projection head. Each transformer block includes a
grouped-query attention mechanism and a multi-layer feedforward network, along with
normalization layers throughout.

These components are also illustrated visually in figure 2.8 for readers familiar with LLM
architectures. A detailed understanding of this architecture is not required for this book.
Since we are not modifying the base model itself, but rather building reasoning methods on
top of it, you can safely treat the architecture as a black box for now. However, interested
readers can optionally find more information on these components, such as RMSNorm, in
appendix C.

![image 20](<input (1)_images/imageFile20.png>)

- Figure 2.8 Overview of the Qwen3 0.6B model architecture, as instantiated earlier. Input text is tokenized and
passed through an embedding layer, followed by 28 repeated transformer blocks. Each block contains
grouped-query attention, feedforward layers, and RMS normalization. The model ends with a final
normalization and linear output layer. Arrows show the data flow through the model.


The key takeaway from this section is that we have now loaded a pre-trained model, with
its architecture shown in figure 2.8, that should be capable of generating coherent text. In
the next section, we will code a text generation function that feeds tokenized data into the
model and returns the response in a human-readable format.

##### 2.6 Understanding the sequential LLM text generation process

After loading a pre-trained LLM, our goal is to write a function that leverages the LLM to
generate text. This function forms the foundation for reasoning-improving methods that we
will implement later in the book, as shown in figure 2.9.

![image 21](<input (1)_images/imageFile21.png>)

- Figure 2.9 An overview of the four key stages in developing a reasoning model in this book. This section
explains the main concept behind text generation in LLMs, which allows us to implement a text generation
function for using the pre-trained LLM in the remainder of this chapter.


Before we get to implement this text generation function that we will use in this and
upcoming chapters (as shown in figure 2.9), let's go over the basic concepts behind text
generation in LLMs.

You may already know that text generation in LLMs is a sequential process where LLMs
generate one word (or token) at a time. This is often also called autoregressive text
generation and is shown in figure 2.10

![image 22](<input (1)_images/imageFile22.png>)

- Figure 2.10 An illustration of the sequential (autoregressive) text generation in LLMs. At each iteration, the
model generates the next token based on the input and previously generated tokens, which are cumulatively
fed back into the model to produce coherent output.


Note that the sequential text generation process shown in figure 2.10 is a broad overview.
The figure shows one generated output token (top row) at each step, when feeding it with
an input prompt. This is done for simplicity to explain the main concept behind LLM-based
text generation.

###### CHATBOT INTERFACES

While the diagram in figure 2.10 shows the underlying next-token prediction process,
a chat interface simply wraps this mechanism in a conversational loop. The model
still predicts one token at a time, but the system feeds the entire dialogue history
back into the model so that each new reply feels context-aware and coherent from
turn to turn. This book focuses on single-turn conversations, where the current
answer is independent of previous answers. However, interested readers can find an
implementation of a chat interface with answer history in appendix G.

Now, if we look at one of these iterations more closely, an LLM processes the full input
sequence in one forward pass and produces one prediction for each input position. This
means that if we have six input tokens, the model returns six corresponding predictions, as
illustrated in figure 2.11. The figure is arranged to make the position-by-position structure
easy to see, but the model outputs are not, in general, just the input prompt shifted to the
right. Rather, each position produces a distribution over possible next tokens. For text
generation, we only use the prediction at the last position, because that is the one that tells
us which new token to append next.

![image 23](<input (1)_images/imageFile23.png>)

- Figure 2.11 The LLM processes the full input sequence and produces one next-token prediction per input
position. These predictions are conceptually shifted one step ahead, because each position predicts the
following token. The illustration is position-aligned for intuition, but the outputs are not literally the input
prompt shifted to the right. During generation, we keep only the final prediction and use it to continue the
input prompt one token at a time.


Before implementing a text-generation function that uses the concept shown in figure 2.11
for each iteration to implement the autoregressive text generation process shown in figure
2.10, let's take a look at a code example to illustrate figure 2.11 further by reusing the
"Explain large language models." example prompt from section 2.4:

prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"Number of input tokens: {len(input_token_ids_list)}")

input_tensor = torch.tensor(input_token_ids_list) #A
input_tensor_fmt = input_tensor.unsqueeze(0) #B
input_tensor_fmt = input_tensor_fmt.to(device)

with torch.inference_mode():

output_tensor = model(input_tensor_fmt) #C
output_tensor_fmt = output_tensor.squeeze(0) #D
print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")

- #A Convert Python list into PyTorch tensor
- #B Add an additional dimension
- #C Generate the output
- #D Remove the extra dimension


###### SQUEEZING AND UNSQUEEZING TENSORS

Tensors are generalized matrices of n dimensions. Many PyTorch functions and model
components expect tensors with specific dimensions, so being able to add or remove
dimensions is important for making data compatible with these operations.

The .squeeze() and .unsqueeze() operations in PyTorch are used to change the
shape of a tensor by removing or adding dimensions of size 1. This is often useful for
reshaping a tensor to match what a model expects. For example, a model might
expect input tensors with two dimensions (e.g., rows and columns) so it can process
batches of inputs (see appendix E). But if the input is just a row vector, we can use
.unsqueeze(0) to add an extra dimension and make it compatible:

example = torch.tensor([1, 2, 3])
print(example)
print(example.unsqueeze(0))

This returns:

tensor([1, 2, 3])
tensor([[1, 2, 3]])

Here, .unsqueeze(0) adds a new dimension at position 0, turning a 1D tensor into a
2D tensor with shape (1, 3). Conversely, .squeeze(0) removes a dimension of size
1 from position 0:

example = torch.tensor([[1, 2, 3]])
print(example)
print(example.squeeze(0))

This returns:

tensor([[1, 2, 3]])
tensor([1, 2, 3])

This is useful when you want to remove extra dimensions that are not needed.

The output from the previous code example is follows:

Number of input tokens: 6
Formatted Output tensor shape: torch.Size([6, 151936])

- As we can see, we feed six input tokens into the model, which returns a 6×151,936-
dimensional matrix. The 6 in this matrix corresponds to the six input tokens. The second
dimension, 151,936, corresponds to the vocabulary size that the model supports. For
instance, each of the six tokens is represented by a vector with 151,936 values. We can
think of the values in these vectors as scores for each possible word in the vocabulary,
where the highest score corresponds to the most likely word or subword (in the 151,936-
entry vocabulary) to be chosen as the generated token.


So, to get the next generated word, we extract the last row of this 6×151,936-
dimensional matrix, find the token ID corresponding to the largest score value in this row,
and convert this token ID back into text via the tokenizer, as illustrated in figure 2.12.

![image 24](<input (1)_images/imageFile24.png>)

- Figure 2.12 A closer look at how the raw scores output by an LLM, in a single text generation iteration, are
converted into a token ID and its corresponding text representation.


Let's see how we can convert the LLM output matrix into the generated text token (shown
in figure 2.12) in code.

Note that for text generation we run the model in inference mode rather than training
mode, so PyTorch does not need to track gradients. As shown in figure 2.11, the earlier
positions correspond to predictions for tokens that are already inside the prompt. Only the
last position predicts the next unseen token that we want to append, which we can obtain
via the [-1] index:

last_token = output_tensor_fmt[-1]
print(last_token)

Using torch.inference_mode() in the previous step avoids storing gradient information
that we do not need during generation. This reduces memory usage and usually improves
speed.

This prints the 151,936 values corresponding to the last token:

tensor([ 7.3750, 2.0312, 8.0000, ..., -2.5469, -2.5469, -2.5469],
dtype=torch.bfloat16)

The dtype=torch.bfloat16 suffix indicates that these scores are stored in bfloat16, a
reduced-precision format commonly used to lower memory usage and improve efficiency in
LLM inference. Depending on your hardware and settings, you may instead see a different
dtype.

Then, we can use the argmax function to obtain the position with the largest value score
(value) in this tensor:

print(torch.argmax(last_token, dim=-1, keepdim=True))

The result is:

tensor([20286])

This returned integer value is the position of the largest value in this vector, and it also
corresponds to the token ID of the generated token (last_token), which we can translate
back into text via the tokenizer:

print(tokenizer.decode([20286]))

This prints the generated token:

Large

###### MAX VERSUS ARGMAX

It is helpful to briefly recall how max and argmax work and how they differ, since we
will use torch.argmax() later when we select the next token when implementing the
text generation function. In this chapter, using argmax corresponds to greedy
decoding, where we always pick the single highest-scoring next token. We will
discuss alternatives to this later in the book.

The torch.max() function returns the largest value in a tensor, whereas
torch.argmax() returns the index of that value. For example:

example = torch.tensor([-2, 1, 3, 1])
print(torch.max(example))
print(torch.argmax(example))

This returns:

tensor(3)
tensor(2)

The maximum value is 3, and it first appears at index 2.

We can also use keepdim=True with torch.argmax() to keep the output shape
consistent by retaining the reduced dimension:

print(torch.argmax(example, keepdim=True))

This returns:

tensor([2])

Here, keepdim=True keeps the result as a 1D tensor with the same number of
dimensions as the input, which can be helpful for keeping the shape required by the
tokenizer and for concatenation later on in our text generation function.

To recap, figure 2.10 illustrated the iterative (autoregressive) text generation process in an
LLM. Then, figure 2.11 zooms in on one of the iterations in this process. Figure 2.12 then
further zoomed into this one iteration and shows how the score matrix (output by an LLM),
gets converted into a token ID (and its corresponding text representation).

While we have seen how to use the LLM to generate a single token, in the next section,
we will put these concepts to action and implement a function that applies this concept
sequentially to generate coherent output text.

##### 2.7 Coding a minimal text generation function

The previous section explained a single iteration in the basic, sequential text generation
process in LLMs. In this section, building on that concept, we will implement a text
generation function that uses the pre-trained LLM to generate coherent text following a
user prompt, as illustrated in Figure 2.13 in the chapter overview.

![image 25](<input (1)_images/imageFile25.png>)

- Figure 2.13 An overview of the four key stages in developing a reasoning model in this book. In this section
we implement a text generation function for the pre-trained LLM.


This text generation function, mentioned in figure 2.13, works by first converting the input
prompt into token IDs that the model can process. The model then predicts the next most
likely token, appends it to the sequence, and reprocesses the extended sequence to
generate the next token. This iterative process continues until a stopping condition is met,
and the generated token IDs are then decoded back into text.

Figure 2.14 shows this process step by step, with both the generated token IDs and their
corresponding text at each stage. (This figure is similar to figure 2.10 shown at the
beginning of the previous section, except it shows the generated token ID alongside their
text representation.)

![image 26](<input (1)_images/imageFile26.png>)

- Figure 2.14 An illustration of sequential (autoregressive) text generation in large language models (LLMs),
with token IDs shown explicitly. At each iteration, the model generates the next token based on the original
input and all previously generated tokens. The predicted token is added to the sequence in both its textual
and token ID form.


The generate_text_basic_stream function in listing 2.2 below implements the sequential
text generation process (figure 2.14) using the argmax function introduced in the previous
section:

- Listing 2.2 A basic text generation function


@torch.inference_mode() #A
def generate_text_basic_stream(

model,
token_ids,
max_new_tokens,
eos_token_id=None

):

model.eval() #B

for _ in range(max_new_tokens):
out = model(token_ids)[:, -1] #C
next_token = torch.argmax(out, dim=-1, keepdim=True)

if (eos_token_id is not None #D

and torch.all(next_token == eos_token_id)):
break

yield next_token #E

token_ids = torch.cat([token_ids, next_token], dim=1) #F

- #A Disable gradient tracking for speed and memory efficiency
- #B Switch model to evaluation mode to enable deterministic behavior (best practice)
- #C Get the scores of the last token
- #D Stop if all sequences in the batch have generated EOS
- #E Yield each token as soon as it's generated
- #F Append the newly predicted token to the sequence


In essence, the generate_text_basic_stream function listing 2.2 applies the argmax-
based token ID extraction via a for-loop for a user-specified number of iterations
(max_new_tokens). It returns the generated token IDs, similar to what's shown in figure
2.14, which we can then convert back into text.

Let's use the function to generate a 100-token response to a simple "Explain large
language models in a single sentence." prompt to make sure that the Qwen3Model
and generate_text_basic_stream function work (we get to the reasoning task examples
in later chapters).

Please note that the following code will be slow and can take 1-3 minutes to complete,
depending on your computer (we will speed it up in later sections):

prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(

tokenizer.encode(prompt),
device=device #A
).unsqueeze(0)

max_new_tokens = 100 #B

for token in generate_text_basic_stream(
model=model,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,

):

token_id = token.squeeze(0).tolist() #C
print(

tokenizer.decode(token_id),
end="",
flush=True #D

)

- #A Transfer the input token IDs onto the same device (CPU, GPU) where the model is located
- #B Let the model generate up to 100 new tokens
- #C Convert output token IDs from PyTorch tensor to Python list
- #D Deactivates buffering so tokens are printed live


The generated output text is as follows:

Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.<|endoftext|>Human language
is a complex and dynamic system that has evolved over millions of
years to enable effective communication and social interaction. It is
composed of a vast array of symbols, including letters, numbers, and
words, which are used to convey meaning and express thoughts and
ideas. The evolution of language has

Note that the output was generated on a CPU. Depending on the device (e.g., CPU versus
GPU), the exact wording may vary slightly due to differences in floating-point behavior on
different hardware.

As we can see based on the output above, the model follows the instruction quite well by
producing a single, clear sentence in response to the prompt. It continues generating
additional, off-topic text after the special token <|endoftext|>. This token is used during
training to mark the end of a document and separate different samples.

TIP The leading whitespace in " Large" (the first output word) is expected because many
tokenizers encode words with a preceding space when they follow earlier text. In listing 2.2, tokens
are streamed one by one, following the input text, so this leading space appears naturally in the first
emitted token. If we want a cleaner output, we can call .lstrip() on the first token or the final
assembled string. .

When using the model for inference (generating text after training), we typically want it to
stop as soon as it produces the special token <|endoftext|>. This token is represented by
the ID 151643, which we can confirm using:

print(tokenizer.encode("<|endoftext|>"))

For convenience, this token ID is also saved via the tokenizer.eos_token_id attribute. We
can pass this ID to the generate_text_basic_stream function to signal when generation
should stop:

for token in generate_text_basic_stream(
model=model,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id #A

):

token_id = token.squeeze(0).tolist()
print(

tokenizer.decode(token_id),
end="",
flush=True

)

#A Pass end-of-sequence (eos) token ID

The output looks like this:

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

If we compare the response to the previous response, we can see that the text generation
stopped once the end-of-sequence token was encountered.

You may have noticed that generating the response is relatively slow and might take
several seconds up to multiple minutes, depending on the hardware.

Before we wrap up and learn how to speed up this function substantially, let's implement
a simple utility function in listing 2.3 that measures the runtime of the text generation
process:

- Listing 2.3 Token generation speed and memory usage


import warnings

def generate_stats(output_token_ids, tokenizer, start_time,

end_time):
total_time = end_time - start_time
print(f"\n\nTime: {total_time:.2f} sec")
print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")

for name, backend in (("CUDA", getattr(torch, "cuda", None)),

("XPU", getattr(torch, "xpu", None))):
if backend is not None and backend.is_available():

- #A
device_type = output_token_ids.device.type
if device_type != name.lower():

warnings.warn(
f"{name} is available but tensors are on "
f"{device_type}. Memory stats may be 0."

)

- #B
if hasattr(backend, "synchronize"):


backend.synchronize()

max_mem_bytes = backend.max_memory_allocated()
max_mem_gb = max_mem_bytes / (1024 ** 3)
print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")

backend.reset_peak_memory_stats()

- #A Check whether we are actually using this backend
- #B Synchronize if supported (important for async backends)


The generate_stats function in listing 2.3 will calculate the total runtime, given a start and
end time stamp, the generation speed in terms of tokens per second (tokens/sec), and the
GPU memory used. Note that the GPU memory usage is currently only computed for CUDA-
supported GPUs (and some newer Intel GPUs via XPU), as PyTorch lacks similar utility
functions for CPUs and Apple Silicon GPUs.

To apply the generate_stats function, we obtain a start_time and end_time stamp
immediately before and after running the generate_text_basic_stream function via
Python's time module:

import time

start_time = time.time()
generated_ids = []

for token in generate_text_basic_stream(
model=model,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id

):

token_id = token.squeeze(0).tolist()
print(

tokenizer.decode(token_id),
end="",
flush=True

)
next_token_id = token.squeeze(0)
generated_ids.append(next_token_id) #A

end_time = time.time()
output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

#A Collect generated tokens

The output, on a Mac Mini M4 CPU, is as follows:

- Time: 7.94 sec


- 5 tokens/sec
Large language models are artificial intelligence systems that can


understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

- At 5 tokens per second, the generation speed is relatively slow. In the next section, we will
implement a caching technique that speeds up the generation process 5-6 fold.


###### TEXT GENERATION AND INFERENCE TERMINOLOGY

When reading LLM literature or software documentation, you will often see the term
inference used where you might expect text generation. In a neural network context,
inference means something very specific, namely, taking a model whose parameters
are already learned and fixed, running a forward pass, and producing a prediction
(for example, generating the next token). Nothing is being estimated or learned
during this stage. In the forward pass, we are simply applying a function.

This is different from inference in statistics, where the goal is to learn unknown
information from data. Statistical inference involves estimating parameters,
quantifying uncertainty, or testing hypotheses about a population or data-generating
process.

Note, that while in large language model contexts, the model is computing the
next-token distribution, and people sometimes say the model is "estimating" or
"inferring" the next token, this is not estimation in the statistical sense. At this stage,
the model is a fixed function and its parameters are already learned.

So when we call the generate_text_basic_stream function, we are not
performing statistical inference. We are performing neural network inference, which
is just the forward application of a trained model to produce the next-token
distribution and select the next generated token from this distribution.

###### 2.8 Faster inference via KV caching

So now that we have a basic text generation function in place, we can turn our attention to
what happens when we actually run it in practice. As you may have noticed, the text
generation in the previous section can be a bit slow. That slowdown points us to a key
concern: performance during inference.

When running inference with LLMs, which in this context means generating text from a
prompt, runtime performance (efficiency) quickly becomes important, especially for long
sequences. While the code in this book emphasizes clarity over speed, real-world systems
often use engineering tricks to make inference more efficient.

This is useful for the later chapters as well, because evaluation and reasoning workflows
often require generating many tokens or many candidate answers, so small speedups here
compound quickly.

In the remaining two sections, we will cover two fundamental techniques, KV caching
and model compilation, as shown in the overview in figure 2.15, to speed up the text
generation.

![image 27](<input (1)_images/imageFile27.png>)

- Figure 2.15 An overview of the four key stages in developing a reasoning model in this book. This section
builds on pre-trained LLM and the basic text generation function we coded earlier and applies KV caching to
speed up execution.


As mentioned in figure 2.15, one engineering trick that increases the text generation speed
is KV caching, where KV refers to the keys and values used in the model's attention
mechanism. If you are not familiar with these terms, that's okay. The key idea is that we
can cache certain intermediate values and reuse them at each step of text generation, as
shown in figure 2.16, which helps speed up inference.

![image 28](<input (1)_images/imageFile28.png>)

- Figure 2.16 Illustration of how a KV cache improves efficiency during autoregressive text generation. Instead
of reprocessing the entire input sequence at each step, the KV cache stores intermediate representations so
that the LLM can reuse them to generate the next token. This eliminates the need to concatenate the
generated token with prior inputs in each subsequent iteration.


The key idea of KV caching, as shown in figure 2.16, is to store intermediate values
computed in each iteration in a cache. Previously, each new token generated by the
network was concatenated to the entire input sequence and fed back into the model
repeatedly (indicated by crossed-out boxes in the diagram). This approach was inefficient
because all tokens, except the newly generated one, remain identical in subsequent
iterations. By using a KV cache, we avoid redundant computation and instead directly
retrieve stored intermediate representations.

In rough terms, generating n new tokens without KV caching requires repeatedly
recomputing work over an increasingly long sequence, which adds up to about O(n2) total
decoding work. Here, O(n2) means the total work grows roughly with the square of the
output length. With KV caching, after the initial pass, each new step processes only the
newly added token, reducing this to roughly O(n), which means the total work grows
approximately in direct proportion to the number of generated tokens.

As mentioned earlier, the non-reasoning focused LLM details like KV caching, which we
used to improve the token generation speed, are outside the scope of this book, and they
are not required for the topics covered later in this book. However, interested readers can
find more information on the mechanics of KV caching in my freely available article:
Understanding and Coding the KV Cache in LLMs from Scratch (https://magazine.
sebastianraschka.com/p/coding-the-kv-cache-in-llms).

Below is a modified version of the generate_text_basic_stream function that
incorporates a KV cache, which is almost identical to the basic text generation function in
listing 2.2, except for the KV cache-related change highlighted via the comments:

- Listing 2.4 A basic text generation function with KV cache


from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_basic_stream_cache(

model,
token_ids,
max_new_tokens,
eos_token_id=None

):

model.eval()
cache = KVCache(n_layers=model.cfg["n_layers"]) #A
model.reset_kv_cache() #A

out = model(token_ids, cache=cache)[:, -1] #B
for _ in range(max_new_tokens):

next_token = torch.argmax(out, dim=-1, keepdim=True)

if (eos_token_id is not None

and torch.all(next_token == eos_token_id)):
break

yield next_token
out = model(next_token, cache=cache)[:, -1] #C

- #A Initialize the KV cache
- #B In the first round, the whole input is provided to the model as before
- #C Consequent iterations only feed the next_token to the input


The generate_text_basic_stream_cache function in listing 2.4 differs only slightly from
the generate_text_basic_stream function in listing 2.2. The main difference is the
introduction of a KVCache object.

During the first iteration, the model is given the full input token sequence as before,
using model(token_ids, cache=cache). Behind the scenes, the KV cache stores the
attention keys and values computed for all these input tokens, so they do not need to be
recomputed in later iterations.

In the following iterations, we no longer need to pass the entire sequence. Instead, we
only provide the next_token to the model using model(next_token, cache=cache). The
model then retrieves the necessary context from the previously stored KV cache.

Let's time this function to see whether it provides any performance benefits:

start_time = time.time()
generated_ids = []

for token in generate_text_basic_stream_cache(
model=model,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id

):

token_id = token.squeeze(0).tolist()
print(

tokenizer.decode(token_id),
end="",
flush=True

)
next_token_id = token.squeeze(0)
generated_ids.append(next_token_id)

end_time = time.time()

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

The output is:

Time: 1.40 sec
29 tokens/sec

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

As we can see, this approach is significantly faster, generating 29 tokens per second
compared to just 5 tokens per second previously (measured on a Mac Mini M4 CPU).

Importantly, we also see that the generated text is the same as before, which is an
important sanity check to ensure that the KV cache is implemented and used correctly.

In the next section, we will learn about another technique we can use to further improve
the generation speed, which will come in handy when we evaluate the model in the
upcoming chapters. Faster generation allows us to run more evaluations in less time and
makes it easier to compare different models or settings efficiently.

##### 2.9 Faster inference via PyTorch model compilation

In the previous section, we covered KV caching as a technique to improve runtime
efficiency as shown in the overview in figure 2.17.

![image 29](<input (1)_images/imageFile29.png>)

- Figure 2.17 An overview of the four key stages in developing a reasoning model in this book. This section
builds on pre-trained LLM and the basic text generation function we coded earlier, including KV caching, and
adds model compilation to speed up the execution speed even further.


As shown in figure 2.17, in this remaining section of this chapter, we will apply another
technique that can substantially speed up model inference: model compilation using
torch.compile. In simple terms, torch.compile analyzes the model's internal
computation graph and tries to turn groups of operations into more optimized kernels,
which reduces Python overhead and other execution inefficiencies. This can improve
runtime performance during text generation, especially when we call the same model code
repeatedly in a loop.

That will matter later when we run larger evaluations and more elaborate reasoning
workflows, where even modest per-step speedups can save substantial total runtime.

major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) >= (2, 8):

# This avoids retriggering model recompilations
# in PyTorch 2.8 and newer
# if the model contains code like self.pos = self.pos + 1
torch._dynamo.config.allow_unspec_int_on_nn_module = True

model_compiled = torch.compile(model)

If you are using a Mac with Apple Silicon and encounter an InductorError, please make
sure to use PyTorch 2.9 or newer.

It is worth noting that the first execution using the compiled model may be slower than
usual due to the initial compilation and optimization steps. To better measure the
performance improvement, we will repeat the text generation process multiple times.

To begin, we will test this using the non-cached version of the generation function. The
code in listing 2.5 is similar to what we used before except that we run it three times in a
row. The code execution may take a few minutes to finish, depending on the system:

- Listing 2.5 Generating text with the compiled model


for i in range(3): #A

start_time = time.time()

generated_ids = []

for token in generate_text_basic_stream(
model=model_compiled,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id

):

token_id = token.squeeze(0).tolist()
print(

tokenizer.decode(token_id),
end="",
flush=True

)

next_token_id = token.squeeze(0)
generated_ids.append(next_token_id)

end_time = time.time()

if i == 0: #B

print("\n\nWarm-up run") #B
else:

print(f"\n\nTimed run {i}:")

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

print(f"\n{30*'-'}\n")

- #A We run the token generation three times
- #B The first run is labeled as "Warm-up run"


The output is as follows:

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Warm-up run

Time: 11.68 sec
3 tokens/sec

------------------------------

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

- Timed run 1:

Time: 6.78 sec
6 tokens/sec

------------------------------

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

- Timed run 2:


Time: 6.80 sec

- 6 tokens/sec


------------------------------

As we can see from the results above, the compiled model achieves a slight improvement in
speed, with around 6 tokens per second compared to the previous 5 tokens per second.

Next, let's see how the KV cache version performs in comparison, using the same code
as before except for swapping generate_text_basic_stream with
generate_text_basic_stream_cache:

Listing 2.6 Generating text with the compiled model using a KV cache

for i in range(3):
start_time = time.time()
generated_ids = []

for token in generate_text_basic_stream_cache(
model=model_compiled,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id

):

token_id = token.squeeze(0).tolist()
print(

tokenizer.decode(token_id),
end="",
flush=True

)

next_token_id = token.squeeze(0)
generated_ids.append(next_token_id)

end_time = time.time()

if i == 0:

print("\n\nWarm-up run")
else:

print(f"\n\nTimed run {i}:")

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(

output_token_ids_tensor, tokenizer, start_time, end_time
)

print(f"\n{30*'-'}\n")

The output is as follows:

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Warm-up run

- Time: 8.07 sec
5 tokens/sec


------------------------------

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Timed run 1:

Time: 0.60 sec
68 tokens/sec

------------------------------

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

Timed run 2:

Time: 0.60 sec
68 tokens/sec

------------------------------

As we can see based on the outputs above, the model generation speed improved from 29
tokens per second for the uncompiled model with KV cache to 68 tokens per second when
the same model is compiled (on a Mac Mini M4 CPU), which is more than a 2-fold speed-up.

If you don't see any improvement, try running torch.compile with the "max-autotune"
mode instead of the default settings. For instance, replace

model = torch.compile(model)

With

model = torch.compile(model, mode="max-autotune")

###### EXERCISE 2.2: RERUN CODE ON NON-CPU DEVICES

If you have access to a GPU, rerun the code in this chapter on a GPU device and
compare the runtimes to the CPU runtimes.

In case you are curious, how the different model configurations compare on an Apple
Silicon GPU and a high-end NVIDIA GPU, see table 2.1.

Table 2.1 Token generation speeds and GPU memory usage for different model configurations on different
hardware

|Mode|Hardware|Tokens/sec|GPU<br>memory|
|---|---|---|---|
|Regular|Mac Mini M4 CPU|5|-|
|Regular compiled|Mac Mini M4 CPU|6|-|
|KV cache|Mac Mini M4 CPU|28|-|
|KV cache compiled|Mac Mini M4 CPU|68|-|
| | | | |
|Regular|Mac Mini M4 GPU|27|-|
|Regular compiled|Mac Mini M4 GPU|43|-|
|KV cache|Mac Mini M4 GPU|41|-|
|KV cache compiled|Mac Mini M4 GPU|71|-|
| | | | |
|Regular|NVIDIA H100 GPU|51|1.55 GB|
|Regular compiled|NVIDIA H100 GPU|164|1.81 GB|
|KV cache|NVIDIA H100 GPU|48|1.52 GB|
|KV cache compiled|NVIDIA H100 GPU|141|1.81 GB|


As shown in the table above, the NVIDIA GPU delivers the best performance, which is
expected. The CPU still performs surprisingly well once a KV cache and a compiled model
are enabled. However, there are a few important details that explain why the GPU gains are
not larger for this particular setup.

First, the model in this chapter is not optimized for GPUs. The implementation aims for a
good balance between memory usage and compute speed because memory is the main
bottleneck for most readers. A GPU-optimized variant would pre-allocate the full K and V
tensors up to the maximum context length (in this case, 40 thousand tokens). This pre-
allocation avoids repeated concatenation via torch.cat, but it also increases memory
consumption.

With a large context size, pre-allocating, for example, 40 thousand entries for both K
and V adds a noticeable footprint. Another approach would be to let users specify a fixed
context size at construction time, but that introduces extra configuration overhead. The KV
cache here grows on demand via torch.cat, which is simpler and more memory friendly,
although concatenating non-preallocated tensors is a bit slower on GPUs.

I included a GPU-optimized version in the bonus materials, where the KV cache variant is
slightly faster than the regular version, but it uses more memory (see https://github.
com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM).

Second, the model used for benchmarking is small. Larger models benefit much more
from KV caching and from GPU-optimized memory layouts. The same holds for batched
inference, where GPUs can better saturate their compute units. With a larger model or a
GPU-focused implementation, the performance gap in favor of the NVIDIA GPU would be
more pronounced.

All examples were run using a single prompt (i.e., a batch size of 1). For readers
interested in how performance scales with multiple inputs, batched inference is discussed in
appendix E.

- 2.10 Summary


Using LLMs to generate text involves multiple key steps:

Setting up the coding environment to run LLM code and
install necessary dependencies.

Loading a pre-trained base LLM (such as Qwen3 0.6B),
which will be extended with reasoning capabilities in later
chapters.

Initializing and using a tokenizer, which converts text
input into token IDs and decodes output back to human-
readable form.

Text generation in LLMs follows a sequential (autoregressive) process,
where the model generates one token at a time by predicting the next
most likely token.

The speed and efficiency of text generation can be improved through:

KV caching, which stores intermediate states to avoid
recomputing previously encountered input tokens at each
step.

Model compilation using torch.compile, which optimizes
runtime performance.

This chapter lays the technical foundation for reasoning capabilities in
upcoming chapters by implementing a functional, efficient text generation
pipeline using a pre-trained base LLM.

# 3 Evaluating reasoning models

This chapter covers

Extracting final answers reliably from an LLM response

Verifying answer correctness by comparing an LLM's output to the reference
solution using a symbolic math solver

Running a full evaluation pipeline by loading a pre-trained model, generating
outputs, and grading them against a dataset

Evaluation is what lets us distinguish between LLMs that merely sound convincing and those
that can solve problems correctly. LLM evaluation techniques span a broad range of
approaches, from measuring task accuracy to making sure that LLMs adhere to specific
safety standards. In this chapter, "evaluate" means quantitatively testing a model on many
examples and scoring its outputs against reference answers.

More specifically, we focus on implementing a verification-based method that checks
whether an LLM can solve math problems accurately by comparing its own answers against
reference solutions using a calculator-like implementation.

This verifier is particularly useful because it not only evaluates performance on math
tasks but also introduces the principle of verifiable rewards, which is the foundation of the
reinforcement learning approach to reasoning models that we will implement later in
chapter 6. (Interested readers can find additional evaluation methods in appendix F.)

![image 30](<input (1)_images/imageFile30.png>)

- Figure 3.1 A mental model of the topics covered in this book. This chapter covers evaluation methods (stage
2), with a special focus on implementing verifiers, which we will later reuse as the basis for verifiable rewards
in chapter 6.


###### 3.1 Building a math verifier

There are four common ways of evaluating trained LLMs in practice: multiple choice,
verifiers, leaderboards, and LLM judges, as shown in figure 3.1. These methods are widely
used across research papers, technical reports, marketing materials, and model cards
(documents that summarize how a model was trained, evaluated, and intended to be used),
and results often draw from more than one category.

As figure 3.1 illustrates, these evaluation approaches can be grouped into two broader
types: benchmark-based evaluation and judgment-based evaluation. Roughly speaking, the
former is usually more quantitative, whereas the latter relies more on qualitative
judgments. All four evaluation methods are useful in different contexts, but verifiers are
especially relevant for reasoning models.

Math problems provide a natural example: depending on the problem complexity, math
problems benefit from step-by-step reasoning to solve, yet evaluation is straightforward
because the final answer can be checked against a correct answer. In this setting, the
verifier approach provides a simple and reliable way to measure whether a model's
reasoning steps lead to the correct outcome.

In this chapter, we focus on verifiers as a benchmark-based approach for measuring
answer correctness in math problems, as illustrated in figure 3.2.

###### NOTE For readers interested in going further, appendix F covers other evaluation methods such as multiple-choice benchmarks, preference-based leaderboards, and LLM-as-a-judge approaches.

![image 31](<input (1)_images/imageFile31.png>)

- Figure 3.2 Evaluating an LLM with a verification-based method in free-form question answering. The model
generates a free-form answer (which may include multiple steps) and a final boxed answer, which is extracted
and compared against the correct answer from the dataset.


Verifiers compare the extracted answer with the reference solution, as shown in figure 3.2,
often by relying on external tools such as code interpreters or calculator programs.

While our immediate focus is evaluation, verifiers will reappear later in this book. They
not only serve as a way to measure performance but also provide the feedback signal used
in reinforcement learning methods for training reasoning models, which we will explore in
chapter 6.

The downside is that verifier methods can only be applied to domains that can be easily
(and ideally deterministically) verified, such as math and code. Also, this approach can
introduce additional complexity and dependencies, and it may shift part of the evaluation
burden from the model itself to the external tool.

Because math problem solving can be generated in unlimited variations
programmatically and benefits from step-by-step reasoning, this task has become a
cornerstone of reasoning model evaluation and development.

In the remainder of this chapter, we will build a math verifier step by step, following the
8 steps shown in figure 3.3.

![image 32](<input (1)_images/imageFile32.png>)

- Figure 3.3 A step-by-step workflow for building and applying a math verifier. Starting with a pre-trained LLM,
we generate answers, extract and normalize them, and then compare them against the ground-truth
solutions. Verified answers are then graded, and the process is repeated across a dataset (MATH-500) to
evaluate overall model performance.


The next section will start with steps 1 and 2 shown in figure 3.3, namely, loading the pre-
trained LLM introduced in the previous chapter and setting it up to generate answers.

###### 3.2 Loading a pre-trained model to generate text

In this section, we begin implementing the verifier by following steps 1 and 2 of the
workflow in figure 3.3. Specifically, we will load the pre-trained LLM introduced in the
previous chapter and configure it to generate answers. This provides the foundation for the
later steps, where we will extract, normalize, and verify these answers.

It also sets up a reusable model-loading path for the later chapters, where we will
compare different reasoning methods and need a consistent way to measure whether they
actually improve the model.

###### Specifically, we use the same pre-trained base model that we used in chapter 2. For our convenience, and for reuse in future chapters, we wrap the model loading logic in a load_model_and_tokenizer function call as shown in listing 3.1:

- Listing 3.1 Loading a pre-trained model


from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
get_device

)
from reasoning_from_scratch.qwen3 import (

download_qwen3_small,
Qwen3Tokenizer,
Qwen3Model,
QWEN_CONFIG_06_B

)

def load_model_and_tokenizer(

which_model, device, use_compile, local_dir="qwen3"
):

if which_model == "base":

download_qwen3_small(

kind="base", tokenizer_only=False, out_dir=local_dir
)

tokenizer_path = Path(local_dir) / "tokenizer-base.json"
model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

elif which_model == "reasoning":

download_qwen3_small(

kind="reasoning", tokenizer_only=False, out_dir=local_dir
)

tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
tokenizer = Qwen3Tokenizer(

tokenizer_file_path=tokenizer_path,
apply_chat_template=True,
add_generation_prompt=True,
add_thinking=True,

)

else:
raise ValueError(f"Invalid choice: which_model={which_model}")

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))

model.to(device)

if use_compile: #A
torch._dynamo.config.allow_unspec_int_on_nn_module = True
model = torch.compile(model)

return model, tokenizer

WHICH_MODEL = "base" #B
device = get_device()
# device = torch.device("cpu") #C

model, tokenizer = load_model_and_tokenizer(
which_model=WHICH_MODEL,
device=device,
use_compile=False

)

#A Optionally set to true to enable model compilation
#B Uses the base model, similar to chapter 2, by default
#C Uncomment this line if you have compatibility issues with your device

By default, listing 3.1 loads the base model, just as in chapter 2. An optional variant is the
official Qwen3 reasoning model, which the Qwen3 team trained on top of the base model
using reasoning-specific methods. This is not the custom reasoning model that we will build
ourselves later in the book. Instead, we use it here as a reference point. It can be loaded
by setting WHICH_MODEL = "reasoning" in listing 3.1 so that we can later compare its
evaluation results with those of the base model.

Now that we have loaded the model, we can use the text generation function from
chapter 2 to generate text, as is shown in listing 3.2.

- Listing 3.2 Generating model outputs


from reasoning_from_scratch.ch02 import (

generate_text_basic_stream_cache
)

prompt = ( #A
r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
r"what is the value of $a^2+b^2$?"

)

input_token_ids_tensor = torch.tensor( #B
tokenizer.encode(prompt),
device=device

).unsqueeze(0) #C

all_token_ids = []

for token in generate_text_basic_stream_cache( #D
model=model,
token_ids=input_token_ids_tensor,
max_new_tokens=2048,
eos_token_id=tokenizer.eos_token_id

):

token_id = token.squeeze(0) #E
decoded_id = tokenizer.decode(token_id.tolist())
print( #F

decoded_id,
end="",
flush=True

)
all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids) #G

- #A Define the math problem as a string prompt
- #B Convert the prompt into token IDs that the model can process
- #C Add batch dimension
- #D Generate output tokens from the model, one at a time
- #E Remove batch dimension
- #F Print token as it is generated
- #G Decode the full generated sequence into text


In listing 3.2, we start by encoding a simple math problem into token IDs that the model
can process. The model then generates tokens one by one in a streaming fashion, which we
print immediately as they appear so we can read the output while it's being generated. At
the same time, we collect the generated tokens into a list so that we can later decode them
into the complete final answer string. This pattern of both streaming and collecting tokens
is handy because it lets us monitor the generation live while still storing the full answer text
(all_tokens) that we can process later.

The response, generated by the code in listing 3.2, is as follows:

To find the value of \( a^2 + b^2 \) given that \( a + b = 3 \)
and \( ab = \frac{13}{6} \), we can use the following algebraic identity:

\[
a^2 + b^2 = (a + b)^2 - 2ab
\]

**Step 1:** Substitute the given values into the equation.

\[
a^2 + b^2 = (3)^2 - 2 \left( \frac{13}{6} \right)
\]

[...] #A

**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]

#A Shortened for brevity

As we can see, based on this answer, even though it is a base model, it provides a
reasoning model-like explanation. This is likely because the Qwen3 team included chain-of-
thought data during the pre-training stages, as stated in their technical report. Even though
the model has some reasoning-model-like behavior, adding additional reasoning methods
can further improve these capabilities. (Note that the response may differ depending on
whether you executed the code on a CPU, CUDA, or MPS device.)

Furthermore, if you are unfamiliar with the LaTeX syntax that is commonly used for
mathematics, the response above can be very hard to decipher. If this is the case, you can
use IPython's Latex class to render it, as shown below:

from IPython.display import Latex, display
display(Latex(all_tokens))

Executing the code above in a code notebook will render the response as shown in figure
3.4.

![image 33](<input (1)_images/imageFile33.png>)

- Figure 3.4 Rendered response with step-by-step calculations and the final boxed answer.


Note that the final answer given in figure 3.4,

![image 34](<input (1)_images/imageFile34.png>)

is indeed the correct answer to this problem.

##### 3.3 Implementing a wrapper for easier text generation

In the previous section, we loaded the pre-trained LLM and set up the text generation
functionality (as illustrated in figure 3.5), which are the first two steps of the evaluation
process covered in the remainder of this chapter.

![image 35](<input (1)_images/imageFile35.png>)

- Figure 3.5 Illustration of steps 1 and 2 from the verifier workflow. A pre-trained LLM is loaded and prompted
with a math problem, producing an output in raw LaTeX syntax. The answer is also shown in the rendered and
more readable form.


For additional convenience in later sections, we create a wrapper (listing 3.3) for the text
generation function so that we only have to pass in the model, tokenizer, and prompt, along
with some additional settings instead of repeating the tokenization and input preparation
steps each time:

- Listing 3.3 A wrapper for streamed text generation


def generate_text_stream_concat(
model, tokenizer, prompt, device, max_new_tokens,
verbose=False,

):

input_ids = torch.tensor( #A
tokenizer.encode(prompt), device=device
).unsqueeze(0)

generated_ids = []
for token in generate_text_basic_stream_cache( #B

model=model,
token_ids=input_ids,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id,

):

next_token_id = token.squeeze(0)
generated_ids.append(next_token_id.item())

if verbose: #C
print(
tokenizer.decode(next_token_id.tolist()),
end="",
flush=True

)

return tokenizer.decode(generated_ids) #D

- #A Encode prompt text into token IDs and place on device
- #B Stream tokens one by one using cached generation
- #C Optionally print tokens as they are generated
- #D Decode all generated IDs into final text string


This wrapper function in listing 3.3 handles the full cycle of text generation: it tokenizes the
input prompt, streams new tokens from the model, and then decodes the results into a final
string. And, as mentioned before, the optional verbose flag allows us to see tokens as they
are generated in real time. The function can be used as follows:

generated_text = generate_text_stream_concat(
model, tokenizer, prompt, device,
max_new_tokens=2048,
verbose=True #A

)

#A Using False will suppress the live token-by-token printing

This prints the exact same response as in section 3.2:

[...] #A

**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]

#A Shortened for brevity

##### 3.4 Extracting the final answer box

Now that we have the model loaded and ready, we can get to the chapter-specific and
interesting parts: evaluating the model.

Before we get started, it is worth recalling that this book takes a from-scratch approach,
which naturally includes some detailed, sometimes tedious steps. This is intentional, since
we want to build the evaluation pipeline ourselves to better understand how it works, rather
than just calling predefined wrapper functions and building blocks.

In the previous section, we saw that the model returned the final answer in an answer
box (written as r"\boxed{\dfrac{14}{3}}" in raw text). We did not explicitly enforce that
format yet, but the model likely produced it because boxed answers are a common
convention in math benchmarks and training data. This behavior is not necessarily proof of
overfitting to the evaluation tasks we will run later. But it reflects that pre-trained models
often encounter many problem formats online and learn to reproduce those stylistic
conventions.

Ht's also true that as a general rule, it is fair to assume that any information available on
the internet when a model was trained has been part of the training data.

Although it was not necessary here, when we evaluate the model in the MATH-500
dataset later on, we will add a specific prompt that instructs the model to return answers in
this boxed form, as it is a common convention that makes the evaluation more consistent
across different models and makes data extraction easier.

Although this step is mechanical, it will keep reappearing later whenever we need to turn
a model response into something that can be scored automatically at scale. We will now
write code that performs this extraction of the boxed answer content as illustrated in figure
3.6.

![image 36](<input (1)_images/imageFile36.png>)

- Figure 3.6 An illustration of how the boxed result from the LLM output is extracted.


Specifically, this section implements step 3 shown in figure 3.6. The next section will
implement the normalization method for step 4.

Since your model may produce slightly different responses (depending on your
hardware) than those shown above, we will work with a hard-coded answer for the time
being (pretending that this answer was generated by the model). In later sections, we will
revisit the model and have it generate answers for the tasks in the MATH-500 dataset.

model_answer = (
r"""... some explanation...

**Final Answer:**

\[ #A
\boxed{\dfrac{14}{3}} #A
\] #A
""")

#A The answer box we want to extract

###### NOTE We are using a raw string (r"""...""" instead of a regular string """..."""). Raw strings make it easier to handle the \ characters, which would otherwise be treated as escape sequences and require doubling each backslash.

Next, let's define a function in listing 3.4 to extract the answer box from the model_answer.

- Listing 3.4 Extracting answer boxes


def get_last_boxed(text):
boxed_start_idx = text.rfind(r"\boxed") #A
if boxed_start_idx == -1:

return None

current_idx = boxed_start_idx + len(r"\boxed") #B

#C
while current_idx < len(text) and text[current_idx].isspace():

current_idx += 1

- #D
if current_idx >= len(text) or text[current_idx] != "{":

return None

current_idx += 1
brace_depth = 1
content_start_idx = current_idx

- #E
while current_idx < len(text) and brace_depth > 0:


char = text[current_idx]
if char == "{":

brace_depth += 1
elif char == "}":

brace_depth -= 1
current_idx += 1

if brace_depth != 0: #F
return None

return text[content_start_idx:current_idx-1] #G

#A Find the last occurrence of "\boxed"
#B Get position after "\boxed"
#C Skip any whitespace after "\boxed"
#D Expect an opening brace "{"
#E Parse the braces with nesting
#F Account for unbalanced braces
#G Extract content inside the outermost braces

The get_last_box helper utility function in listing 3.4 extracts out the content of the last
\boxed{...} expression from a model's output. More specifically, it scans for the final
\boxed, skips over whitespace, checks for braces, and handles any nesting so that we
capture the intended answer string.

While it may look a bit tedious, having this parser in place will pay off when we run
evaluations on datasets like MATH-500, where extracting the correct final answer is the first
step toward measuring a model's reasoning ability. (MATH-500 is a curated collection of 500
problems that is widely used as a reasoning model benchmark dataset, which we will use
later in this chapter.)

Now, let's test it on the model answer:

extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)

The output of this function call is "\dfrac{14}{3}", which is the boxed answer we wanted
to extract.

###### RENDERING MATH FORMULAS

We can render math formulas via the Latex class we introduced earlier. Alternatively,
for single math formulas that are not accompanied by answer text, we can also use
the simpler Math class:

from IPython.display import Math
display(Math(r"\dfrac{14}{3}"))

This renders the fraction as

![image 37](<input (1)_images/imageFile37.png>)

While the previous get_last_boxed function correctly extracted the text, we will make the
answer extraction a bit more robust to account for cases where a final answer box is either
missing or incomplete via the extract_final_candidate function in listing 3.5:

- Listing 3.5 Extracting the final answer candidate


import re

RE_NUMBER = re.compile( #A

r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def extract_final_candidate(text, fallback="number_then_full"):

result = "" #B

if text: #C
boxed = get_last_boxed(text.strip())
if boxed:

result = boxed.strip().strip("$ ")

#D
elif fallback in ("number_then_full", "number_only"):

m = RE_NUMBER.findall(text)
if m:

result = m[-1] #E
elif fallback == "number_then_full":

result = text #F
return result

#A Regular expression for extracting numeric values from the text
#B Default return value if nothing matches
#C Prefer the last boxed expression if present
#D If no boxed expression, try fallback
#E Use last number
#F Else return full text if no number found

The extract_final_candidate function in listing 3.5 provides fallback settings in case no
boxed answer can be found, which are as follows:

"number_then_full" (default): pick the last simple number, else the
whole text;

"number_only": pick the last simple number, else return an empty string
"";

"none": extract only boxed content, else return empty string "".

For the fallback setting, the code in listing 3.5 uses regular expressions (regex for short)
via Python's re library. Regexes are a way to search for patterns in text. In our case, the
regex pattern is designed to recognize numbers, including fractions, decimals, and scientific
notation. While the regex syntax looks intimidating, you don't need to worry about the
exact syntax here. What matters is that this gives us a reliable tool to extract the last
numeric candidate from the model's output when no boxed answer is available.

Let's try it on our model answer:

print(extract_final_candidate(model_answer))

This correctly returns "\dfrac{14}{3}". Next, let's try some additional examples. First,
another boxed candidate:

print(extract_final_candidate(r"\boxed{ 14/3. }"))

This correctly returns "14/3.", stripping the extra whitespace but not the punctuation. The
punctuation character will be handled correctly by the equality check we implement later.

Next, let's try a candidate without a box, which should trigger the fallback setting, and
see what happens:

print(extract_final_candidate("abc < > 14/3 abc"))

Thanks to the default fallback setting, it will find the last number in the answer and also
correctly return "14/3".

In this section, we defined utility functions to extract the LLM's answer from within its
answer text context. This brings us one step closer to achieving the overall goal of verifying
whether this answer is indeed correct. In the next section, we will normalize the response
into a more general, canonical form before we implement the checking functionality.

###### WHY NOT USE AN LLM FOR THE ANSWER EXTRACTION?

We could use an LLM itself to extract the boxed answer. This would introduce
unnecessary complexity and potential errors. Extraction is a simple, mechanical task,
assuming that the LLM outputs the answer in a specific format: we just need to
locate the last boxed expression or, if that is missing, fall back to a number or the
raw text.

Regular expressions may look complicated at first, but in the end, we have a
small, reusable utility function that is cheap to execute and handles the extraction
deterministically and reproducibly, without depending on the variability of another
model's output.

##### 3.5 Normalizing the extracted answer

Previously, we extracted the boxed answer "\dfrac{14}{3}" from the model's response.
Models may print the same value in many ways, such as "\frac{14}{3}", "14/3",
"$14/3$", or "(14)/(3)". To implement and use a robust checking system that can check
whether the answer is correct, we first need a consistent method of comparing results.

In this section, we implement a normalization (or canonicalization) pass (step 4 in figure
3.7) that strips formatting and standardizes the answer.

![image 38](<input (1)_images/imageFile38.png>)

- Figure 3.7 An illustration of how the boxed result from the LLM output is extracted and converted into a
canonical plain form. This normalized answer is then later used for verification against the correct answer.


The normalization step shown in figure 3.7 is implemented via the normalize_text
function in listing 3.6.

- Listing 3.6 Normalizing extracted answers


LATEX_FIXES = [ #A
(r"\\left\s*", ""),
(r"\\right\s*", ""),
(r"\\,|\\!|\\;|\\:", ""),
(r"\\cdot", "*"),
(r"\u00B7|\u00D7", "*"),
(r"\\\^\\circ", ""),
(r"\\dfrac", r"\\frac"),
(r"\\tfrac", r"\\frac"),
(r"°", ""),

]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>") #B
SUPERSCRIPT_MAP = {

"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", #C
"⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", #C
"⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")", #C

}

def normalize_text(text):
if not text:
return ""
text = RE_SPECIAL.sub("", text).strip()

#D
match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
if match:

text = match.group(1)

text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text) #D
text = re.sub(r"\^\s*\\circ", "", text) #E
text = text.replace("°", "") #E

match = re.match(r"^\\text\{(?P<x>.+?)\}$", text) #F
if match:

text = match.group("x")

text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text) #G

for pat, rep in LATEX_FIXES: #H

text = re.sub(pat, rep, text)

def convert_superscripts(s, base=None):

converted = "".join(
SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
for ch in s

)
if base is None:

return converted
return f"{base}**{converted}"

text = re.sub(
r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
lambda m: convert_superscripts(m.group(2), base=m.group(1)),
text,

)
text = convert_superscripts(text)

- #I
text = text.replace("\\%", "%").replace("$", "").replace("%", "")
text = re.sub(

r"\\sqrt\s*\{([^}]*)\}",
lambda match: f"sqrt({match.group(1)})",
text,

)
text = re.sub(

r"\\sqrt\s+([^\\\s{}]+)",
lambda match: f"sqrt({match.group(1)})",
text,

)

- #J
text = re.sub(


r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
lambda match: f"({match.group(1)})/({match.group(2)})",
text,

)
text = re.sub(

r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
lambda match: f"({match.group(1)})/({match.group(2)})",
text,

)

- #K
text = text.replace("^", "**")
text = re.sub(

r"(?<=\d)\s+(\d+/\d+)",
lambda match: "+" + match.group(1),
text,

)

- #L
text = re.sub(


r"(?<=\d),(?=\d\d\d(\D|$))",
"",
text,

)

return text.replace("{", "").replace("}", "").strip().lower()

- #A LaTeX formatting to be replaced (left: original value, right: new value)
- #B Strip chat special tokens like "<|assistant|>"
- #C Dictionary mapping to convert unicode superscripts to plaintext superscripts
- #D Strip leading multiple-choice labels (e.g., like "c. 3" -> 3)
- #E Remove angle-degree markers
- #F Unwrap "\text{...}" if the whole string is wrapped
- #G Strip inline/display math wrappers: \( \) \[ \]
- #H LaTeX canonicalization
- #I Normalize number and root expressions
- #J Convert LaTeX fractions into division form
- #K Handle exponents and mixed numbers
- #L Remove thousands separators in numbers


The normalize_text function in listing 3.6 takes an extracted answer string and rewrites it
into a standardized format that we can reliably compare against reference solutions. It first
strips away special tokens and unnecessary LaTeX clutter, such as \left, \right, or degree
symbols. It then unwraps cases like \text{...}, removes inline math markers, and
rewrites common structures into a calculator-style form. For example, it turns \sqrt{a}
into sqrt(a) and \frac{a}{b} into (a)/(b). Finally, it normalizes exponents, mixed
numbers, and thousands separators and cleans up braces and casing. In short, the function
transforms differently formatted LaTeX outputs into a clean, standardized string
representation.

Let's now try the normalize_text function on our model answer:

print(normalize_text(extract_final_candidate(model_answer)))

As a result, instead of printing the answer with LaTeX formatting (r"\dfrac{14}{3}"), it
returns the answer in a standardized, LaTeX-free form:

"(14)/(3)"

Next, let's try a differently formatted answer:

print(normalize_text(r"\text{\[\frac{14}{3}\]}"))

This also returns, "(14)/(3)", as intended.

We now have a robust method to extract answer texts from an LLM's response. The next
task, covered in the next section, is to implement a function to compare the LLM answer to
a correct reference answer.

##### 3.6 Verifying mathematical equivalence

So far, in this chapter, we implemented steps to ask an LLM to generate an answer, extract
the relevant portion, and normalize it. The next step, as illustrated in figure 3.8, is to
compare the extracted answer to a correct reference answer, which is, in technical
contexts, referred to as ground truth.

This is one of the main reasons for building the verifier carefully here: in chapter 6, the
same basic idea will reappear as a verifiable reward signal, and in later chapters we will use
it to check whether our training changes improved the model.

![image 39](<input (1)_images/imageFile39.png>)

- Figure 3.8 An illustration of how an LLM-generated answer is checked against the correct reference answer
(ground truth). The final boxed answer is extracted and normalized, then compared to the correct answer
provided in the dataset. If both match, the response is graded as correct.


Note that if we want to implement the equality check shown in figure 3.8, a direct
comparison using Python's == operator is not sufficient, since expressions like "14/3" and "
(14)/(3)" would not match, and equivalent but unnormalized fractions such as "
(28)/(6)" and "(14)/(3)" would also be treated as unequal.

As part of our equality check, we implement an additional intermediate step: parsing the
extracted and normalized answer using a symbolic math engine.

For this, we use the SymPy open-source math library (https://sympy.org), which has
been developed and tested for two decades and has become a staple of scientific computing
in Python. The parsing function is implemented in listing 3.7.

###### NOTE If you haven't installed the dependencies in chapter 2, you can manually install SymPy via uv pip install sympy (or uv add sympy).

- Listing 3.7 SymPy parser for mathematical equality check


from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError

def sympy_parser(expr):
if expr is None or len(expr) > 2000: #A
return None

try:

return spp.parse_expr(
expr,
transformations=(

*spp.standard_transformations, #B
#C
spp.implicit_multiplication_application,

),

evaluate=True, #D
)

except (SympifyError, SyntaxError, TypeError, AttributeError,

IndexError, TokenError, ValueError, PolynomialError):
return None

- #A To avoid crashing on long garbage responses
- #B Standard transformations like handling parentheses
- #C Allow omitted multiplication symbols (e.g., 2y -> 2*y)
- #D Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)


The sympy_parser function in listing 3.7 takes an input expression, such as the normalized
answers we extract from the LLM response, and converts it into a SymPy object that can be
reliably compared for mathematical equivalence. To do so, it applies SymPy's standard
parsing rules, supports implicit multiplication like (2y instead of 2*y), and also simplifies
basic arithmetic (so 2+3 becomes 5).

###### NOTE The sympy_parser takes into account what looks like an excessive amount of error cases, but these are all errors that I encountered when evaluating the model on all 500 MATH-500 problems, as the model does not always generate perfectly formatted outputs.

Let's see it in action and apply it to the normalized answer candidate:

print(sympy_parser(normalize_text(

extract_final_candidate(model_answer)
)))

This returns the fraction 14/3. Next, let's try an unnormalized fraction:

print(sympy_parser("28/6"))

Similarly, this returns 14/3.

Using the sympy_parser, we can now implement the equality check function in listing
3.8:

- Listing 3.8 Equality check function using SymPy


from sympy import simplify

def equality_check(expr_gtruth, expr_pred):
if expr_gtruth == expr_pred: #A
return True

#B
gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

if gtruth is not None and pred is not None: #C
try:

return simplify(gtruth - pred) == 0 #D
except (SympifyError, TypeError):

pass

return False

- #A First, check if the two expressions are exactly the same string
- #B Parse both expressions into SymPy objects (returns None if parsing fails)
- #C If both expressions were parsed successfully, try symbolic comparison
- #D If the difference is 0, they are equivalent


The equality_check function in listing 3.8 determines whether a model's answer matches
the ground-truth solution. It first looks for an exact string match, which is the simplest
case. If the strings differ, it parses both expressions into SymPy objects (via the
sympy_parser function we implemented in listing 3.7) and checks whether their difference
simplifies to zero. This allows us to recognize answers that may look different on the
surface (for example, 14/3 and 28/6) but are mathematically the same.

Let's try the equality checker from listing 3.8 on an example:

print(equality_check(
normalize_text("13/4."),
normalize_text(r"(13)/(4)")

))

As intended, this ignores the formatting and returns True. Next, let's try a more
challenging example and see whether the symbolic math parser recognizes that 0.5 is the
same as 1/2:

print(equality_check(
normalize_text("0.5"),
normalize_text(r"(1)/(2)")

))

This also returns True. Now, let's try a negative example:

print(equality_check(

- normalize_text("14/3"),
- normalize_text("15/3")


))

This returns False since the expressions are different.

So far, so good. Based on the encouraging results above, we may conclude that we now
have a robust equality checker that we can use to evaluate the LLM on a math benchmark
dataset. Tmake sure that it's ready for prime time, let's try one more example:

print(equality_check(
normalize_text("(14/3, 2/3)"),
normalize_text("(14/3, 4/6)")

))

In this case, we are comparing two tuples. Since 2/3 and 4/6 are mathematically
equivalent, we would expect the result to be True. Instead, the function returns False,
because it currently only handles simple expressions, not tuples. We will address this
limitation in the next section.

##### 3.7 Grading answers

Now, we will build upon the mathematical equality checking function from the previous
section to implement a robust grading function that can also handle tuple-like expressions,
such as correctly comparing expressions like "(14/3, 2/3)" and "(14/3, 4/6)".

First, we implement a Python helper function that splits such tuple-like expressions into
individual subparts via listing 3.9.

- Listing 3.9 Helper function to split tuple-like expressions


def split_into_parts(text):
result = [text]

if text: #A
if (

len(text) >= 2
and text[0] in "([" and text[-1] in ")]"
and "," in text[1:-1]

):

items = [p.strip() for p in text[1:-1].split(",")] #B
if all(items):

result = items
else: #C

result = []

return result

- #A Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
- #B Split on commas inside brackets and strip whitespace
- #C If text is empty, return an empty list


The split_into_parts function in listing 3.9 helps us handle answers with multiple
components. If the input looks like a tuple or list, such as (a, b) or [a, b], it splits the
content on commas and returns the individual pieces. (If the string is empty, it simply
returns an empty list.) In essence, this function breaks down multi-part answers into
smaller parts that can be checked one by one.

Before we implement the grading function next, let's take the split_into_parts for a
test drive and try it on the tuple-like expression from earlier:

split_into_parts(normalize_text(r"(14/3, 2/3)"))

This returns ['14/3', '2/3'], as desired.

Now, we can implement the grade_answer function (listing 3.10), which splits tuple-like
expressions (if present) into subparts, and then uses the equality_check function from the
previous section to compare a generated answer to a reference (ground truth) answer.

- Listing 3.10 Function to grade predicted answers against ground truth


def grade_answer(pred_text, gt_text):
result = False #A
if pred_text is not None and gt_text is not None: #B

gt_parts = split_into_parts(
normalize_text(gt_text)

)
pred_parts = split_into_parts(

normalize_text(pred_text)
)

if (gt_parts and pred_parts #C
and len(gt_parts) == len(pred_parts)): #C

result = all(
equality_check(gt, pred)
for gt, pred in zip(gt_parts, pred_parts)

) #D

return result #E

- #A Default outcome if checks fail
- #B Only continue if both inputs are non-empty strings
- #C Ensure both sides have same number of valid parts
- #D Check each part for mathematical equivalence
- #E True only if all checks passed


The implementation of the grade_answer function in listing 3.10 first assumes the
prediction is incorrect (False) and only continues if both prediction and ground truth are
non-empty. It then normalizes each side and splits them into subparts (for example,
breaking "(14/3, 2/3)" into ["14/3", "2/3"]). If the number of subparts matches, it
compares them one by one using equality_check. The result is returned as correct (True)
only if all pairs match mathematically.

We can think of the grade_answer function as an advanced version of the
equality_check function from the previous section. The grade_answer function can split
tuple-like expressions and normalize the answers before applying the equality_check
function.

On simple expressions, it works similarly to the equality_check, returning True if two
expressions are mathematically equivalent:

grade_answer("14/3", r"\frac{14}{3}")

In addition, as described above, it now also returns True in case of two mathematically
equivalent tuple-like expressions:

grade_answer(r"(14/3, 2/3)", "(14/3, 4/6)")

To check the grade_answer function more comprehensively, the code in listing 3.11
contains more diverse test cases.

- Listing 3.11 Test cases and demo function to test the grader


tests = [ #A

- ("check_1", "3/4", r"\frac{3}{4}", True),
- ("check_2", "(3)/(4)", r"3/4", True),
- ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
- ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
- ("check_5", "(1, 2)", r"(1,2)", True),
- ("check_6", "(2, 1)", "(1, 2)", False),
- ("check_7", "(1, 2, 3)", "(1, 2)", False),
- ("check_8", "0.5", "1/2", True),
- ("check_9", "0.3333333333", "1/3", False),
- ("check_10", "1,234/2", "617", True),
- ("check_11", r"\text{2/3}", "2/3", True),
- ("check_12", "50%", "1/2", False),
- ("check_13", r"2\cdot 3/4", "3/2", True),
- ("check_14", r"90^\circ", "90", True),
- ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
- ("check_16", r"2²", "2**2", True),


]

def run_demos_table(tests):
header = ("Test", "Expect", "Got", "Status")
rows = []
for name, pred, gtruth, expect in tests:

got = grade_answer(pred, gtruth) #B
status = "PASS" if got == expect else "FAIL"
rows.append((name, str(expect), str(got), status))

data = [header] + rows

col_widths = [ #C
max(len(row[i]) for row in data)
for i in range(len(header))

]

for row in data: #D
line = " | ".join(

row[i].ljust(col_widths[i])
for i in range(len(header))

)
print(line)

passed = sum(r[3] == "PASS" for r in rows) #E
print(f"\nPassed {passed}/{len(rows)}") #E

- #A Define test cases: (name, prediction, ground truth, expected result)
- #B Run equality check
- #C Compute max width for each column to align table nicely
- #D Print table row by row
- #E Print summary of passed tests


The code in listing 3.11 is a simple test suite that takes in a selection of tests to check
whether the grade_answer function works as intended. The tests list contains tuples that
cover a selection of fractions, LaTeX notations, tuple inputs, decimals, percentages, and
other tricky formats.

The run_demos_table function then runs each test by calling grade_answer, collects the
outcomes, and organizes the results into a formatted table.

Calling the run_demos_table(tests) function in listing 3.11 prints the following:

Test | Expect | Got | Status

- check_1 | True | True | PASS
- check_2 | True | True | PASS
- check_3 | True | True | PASS
- check_4 | True | True | PASS
- check_5 | True | True | PASS
- check_6 | False | False | PASS
- check_7 | False | False | PASS
- check_8 | True | True | PASS
- check_9 | False | False | PASS
- check_10 | True | True | PASS
- check_11 | True | True | PASS
- check_12 | False | False | PASS
- check_13 | True | True | PASS
- check_14 | True | True | PASS
check_16 | True | True | PASS


Passed 16/16

As we can see based on the PASS results above, the grade_answer function is relatively
robust and capable of handling a variety of differently formatted expressions.

###### EXERCISE 3.1: ADDING MORE TEST CASES

Try to think of additional test cases, ideally challenging ones, and add them to the
run_demos_table() function. Can you find cases where the check fails incorrectly?

With the grade_function implemented, we now have the building blocks in place to
evaluate the LLM. In the next section, we will load a math dataset on which we will evaluate
the LLM.

##### 3.8 Loading the evaluation dataset

As we have seen in the chapter so far, implementing a robust verification pipeline can be a
tedious task. Fortunately, we now have all the pieces in place, from answer extraction to
grading, and are ready to evaluate the LLM on a benchmark dataset. For this, as illustrated
in figure 3.9, we will use the MATH-500 dataset (https://huggingface.co/datasets/
HuggingFaceH4/MATH-500), a widely used benchmark for reasoning models. It is a curated
collection of 500 problems sampled from the original MATH dataset.

![image 40](<input (1)_images/imageFile40.png>)

- Figure 3.9 Loading the evaluation dataset. After completing steps 2–6 on individual problems (generate,
extract, normalize, verify, and grade answers) in the previous sections, the two remaining steps are to load the
full dataset (step 7) and apply the same procedure across all problems to evaluate the model (step 8).


This choice of using MATH-500 is practical for three reasons. First, MATH-500 is large
enough to be meaningful but still small enough to make repeated evaluations manageable
throughout the book, whereas running the full MATH dataset each time would be much
slower. Second, in later chapters we use a non-overlapping training subset derived from the
original MATH dataset, so keeping MATH-500 as a held-out evaluation set gives us a clean
reference point. Third, MATH-500 is a common benchmark dataset in the reasoning-model
literature, which makes the results in this book easier to compare with prior work. We also
prefer MATH-500 over simpler datasets such as GSM8K here because it is more challenging
for modern reasoning models and better matches the type of multi-step symbolic
verification pipeline we are building in this chapter.

We will load the MATH-500 dataset (step 7 in figure 3.9) using the following code:

- Listing 3.12 Loading the MATH-500 dataset


import json
import requests

def load_math500_test(local_path="math500_test.json", save_copy=True):
local_path = Path(local_path)
url = (

"https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
"main/ch03/01_main-chapter-code/math500_test.json"

)

if local_path.exists():
with local_path.open("r", encoding="utf-8") as f:
data = json.load(f)

else:
r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()

if save_copy: # Saves a local copy
with local_path.open("w", encoding="utf-8") as f:
json.dump(data, f, indent=2)

return data

math_data = load_math500_test()
print("Number of entries:", len(math_data))

This prints:

Number of entries: 500

###### LOADING THE DATASET FROM HUGGING FACE MODEL HUB

The following information and code example is optional and provided for reference,
and you don't need to run the code below.

The MATH-500 dataset split was originally proposed in the PRM800K repository
(https://github.com/openai/prm800k/tree/main?tab=readme-ov-file#math-splits)
and is also available on the Hugging Face Hub (https://huggingface.co/datasets/
HuggingFaceH4/MATH-500). Here, we load a copy from the code repository to ensure
reproducibility in case the external sources change.

If you prefer to download the dataset directly from Hugging Face, you can use the
following code. Note that this requires the datasets library, which can be installed
via pip install datasets or uv add datasets:

from datasets import load_dataset
dset = load_dataset("HuggingFaceH4/MATH-500", split="test")

Before we jump to the next section to implement the model evaluation pipeline, let's take a
closer look at the structure of the dataset by printing its first entry (we use the built-in
pprint library for nicer formatting):

from pprint import pprint
pprint(math_data[0])

This produces the following output:

{'answer': '\\left( 3, \\frac{\\pi}{2} \\right)',
'level': 2,
'problem': 'Convert the point $(0,3)$ in rectangular coordinates to polar '

'coordinates. Enter your answer in the form $(r,\\theta),$ where '
'$r > 0$ and $0 \\le \\theta < 2 \\pi.$',

'solution': 'We have that $r = \\sqrt{0^2 + 3^2} = 3.$ Also, if we draw the '
'line connecting the origin and $(0,3),$ this line makes an angle '
'of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n'
'\n'
'[asy]\n'
'unitsize(0.8 cm);\n'
'\n'
'draw((-0.5,0)--(3.5,0));\n'
'draw((0,-0.5)--(0,3.5));\n'
'draw(arc((0,0),3,0,90),red,Arrow(6));\n'
'\n'
'dot((0,3), red);\n'
'label("$(0,3)$", (0,3), W);\n'
'dot((3,0), red);\n'
'[/asy]\n'
'\n'
'Therefore, the polar coordinates are $\\boxed{\\left( 3, '
'\\frac{\\pi}{2} \\right)}.$',

'subject': 'Precalculus',
'unique_id': 'test/precalculus/807.json'}

As we can see, the dataset entry is formatted as a Python dictionary with keys and values.
The relevant keys are

"problem": the math question or problem for the LLM to solve,

"answer": the correct (ground truth) answer we want to compare the
LLM answer against,

"solution": a worked-out, step-by-step explanation of the problem (not
used in this chapter, but useful for training or analysis).

(Note that the output contains the keys sorted in alphabetical order: "answer", "problem",
"solution", but the bulleted uses the more logical ordering for readability.)

Now that we have a pre-trained LLM, evaluation functions, and a benchmark dataset to
work with, we can implement the model evaluation.

##### 3.9 Evaluating the model

In this section, we put the LLM text generation and evaluation tools from steps 2–6 in
figure 3.10 into practice and apply them to the MATH-500 dataset (step 8 in figure 3.10),
which we loaded in the previous section.

This full pipeline is important beyond this chapter as well, because it becomes the main
way we will measure progress later when we compare prompting methods and training-
based improvements.

![image 41](<input (1)_images/imageFile41.png>)

- Figure 3.10 The complete evaluation pipeline on the MATH-500 dataset. After loading the dataset (step 7),
steps 2–6 are applied systematically across all problems to obtain the final model evaluation (step 8).


As you may recall from section 3.4 (Extracting the final answer box), our answer checking
pipeline expects that the model returns the answer in boxed form, which is a common
convention when evaluating reasoning models on math problems. To increase the likelihood
that the model adheres to this format, we can format the prompt as shown in listing 3.13:

- Listing 3.13 Function to render a prompt template for math evaluation


def render_prompt(prompt):

template = (
"You are a helpful math assistant.\n"
"Answer the question and write the final result on a new line as:\n"
"\\boxed{ANSWER}\n\n"
f"Question:\n{prompt}\n\nAnswer:"

)
return template

Let's now apply the prompt template from listing 3.13 to the example prompt we
introduced earlier in this chapter (section 3.2). For convenience, we redefine the example
prompt here:

prompt = (
r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
r"what is the value of $a^2+b^2$?"

)
prompt_fmt = render_prompt(prompt)
print(prompt_fmt)

The formatted prompt is now as follows:

You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
If $a+b=3$ and $ab=\tfrac{13}{6}$, what is the value of $a^2+b^2$?

Answer:

Next, we pass the prompt to the text generation wrapper function we defined in listing 3.3
in section 3.3 to recap the text generation process before we construct the model
evaluation function:

generated_text = generate_text_stream_concat(
model, tokenizer, prompt_fmt, device,
max_new_tokens=2048,
verbose=True

)

Using this prompt example, the model responds with a relatively brief answer:
"\boxed{10}". (Note that the generated response may differ depending on whether you
executed the code on a CPU, CUDA, or MPS device.)

While brevity can speed up generation by reducing the number of tokens, the response
is incorrect. In contrast, in section 3.3, without a prompt template, the model produced a
longer response, which led to the correct answer, 14/3.

Whether a prompt template is well-suited for a given model and task ideally needs to be
determined on a larger set of examples before we can draw any conclusions, for instance,
the MATH-500 dataset we will evaluate the model on later in this section.

###### PROMPT TEMPLATE CHOICES

The prompt template in listing 3.13 is used here to demonstrate how a model
evaluation pipeline can be implemented with answers that are automatically checked
for correctness. The chosen template encourages short outputs, which lets you work
through this chapter efficiently on a first read. Afterward, I recommend revisiting the
chapter with alternative settings to optimize accuracy on the official Qwen3
reasoning model variant that we use here as a reference model.

As it turns out, using no prompt template boosts the base model performance by
50%, but it reduces the accuracy of the reasoning model by 40%.

Additionally, we may also experiment with alternative prompt templates. For
instance, the common standard prompt for the MATH-500 benchmark is the following
variant that swaps "Question:" with "Problem:" in listing 3.13. This seemingly minor
change improves the base model’s accuracy by approximately 20%, likely because it
better matches the memorized training data (assuming the MATH-500 test set was
included in the training corpus). While the base model benefits from this change, the
accuracy of the reasoning model variant drops by 30%.

Earlier, we mentioned that answer extraction is a simple, mechanical task that can
be solved with deterministic code rather than employing another LLM for the answer
extraction. Based on the accuracy change due to different prompting templates, it
looks like our extraction method may not be reliable. This is not necessarily the case.

Also, it's not necessarily the case that the LLM generates misformatted answers.
It can just be incorrect, and switching to another LLM for extraction would not solve
that. Smaller base models are often quite sensitive to prompt phrasing. In the next
chapter, we will see how this becomes even more apparent once we introduce
additional prompt variations.

Next, before we implement the final model evaluation function, let us test our model
evaluation pipeline end to end on a smaller example via the demo function in listing 3.14:

- Listing 3.14 Demo function to run the evaluation pipeline


def mini_eval_demo(model, tokenizer, device):

ex = { #A
"problem": "Compute 1/2 + 1/6.",
"answer": "2/3"

}
prompt = render_prompt(ex["problem"]) #B
gen_text = generate_text_stream_concat( #C

model, tokenizer, prompt, device, #C
max_new_tokens=64, #C

) #C
pred_answer = extract_final_candidate(gen_text) #D
is_correct = grade_answer( #E

pred_answer, ex["answer"] #E
) #E

print(f"Device: {device}")
print(f"Prediction: {pred_answer}")
print(f"Ground truth: {ex['answer']}")
print(f"Correct: {is_correct}")

- #A Test example with "problem" and "answer" fields
- #B 1. Apply prompt template
- #C 2. Generate response
- #D 3. Extract and normalize answer
- #E 4. Grade answer


The mini_eval_demo function in listing 3.14 combines all the aspects we have covered so
far in this chapter:

- 1. Applying a prompt template
- 2. Feeding the formatted prompt to the LLM to generate an answer
- 3. Extracting and normalizing the answer
- 4. Grading the answer


This mini_eval_demo function essentially connects the evaluation components together into
a small function that we can use to test the code before coding the final evaluation pipeline
for the MATH-500 dataset. The code starts from a toy example (ex), renders the problem
into the prompt template (prompt), and streams a response from the model
(generate_text_stream_concat). It then parses the model output to a final candidate
answer (pred_answer) and grades it against the ground truth with grade_answer. Lastly, it
prints the results for us to evaluate.

Calling the mini_eval_demo(model, tokenizer, device) function results in the
following output:

Device: mps
Prediction: 1/3
Ground truth: 2/3
Correct: False

We can see that the generated answer ("1/3") was correctly extracted, but it doesn't
match the correct answer ("2/3"), and hence the check returns False. (Note that the
results may differ depending on whether you execute the code on a CPU, CUDA, or MPS
device.)

Now that we have tested our workflow on a simpler example, let's implement it to run on
the MATH-500 dataset.

- Listing 3.15 End-to-end model evaluation pipeline for MATH-500 dataset


import time

def eta_progress_message( #A
processed,
total,
start_time,
show_eta=False,
label="Progress",

):

progress = f"{label}: {processed}/{total}"
pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
if not show_eta or processed <= 0:

return progress.ljust(pad_width)

elapsed = time.time() - start_time
if elapsed <= 0:

return progress.ljust(pad_width)

remaining = max(total - processed, 0)

if processed:
avg_time = elapsed / processed
eta_seconds = avg_time * remaining

else:
eta_seconds = 0

eta_seconds = max(int(round(eta_seconds)), 0)
minutes, rem_seconds = divmod(eta_seconds, 60)
hours, minutes = divmod(minutes, 60)
if hours:

eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
elif minutes:

eta = f"{minutes:02d}m {rem_seconds:02d}s"
else:

eta = f"{rem_seconds:02d}s"

message = f"{progress} | ETA: {eta}"
return message.ljust(pad_width)

def evaluate_math500_stream(

model,
tokenizer,
device,
math_data,
out_path=None,
max_new_tokens=512,
verbose=False,

):

if out_path is None:
dev_name = str(device).replace(":", "-") #B
out_path = Path(f"math500-{dev_name}.jsonl")

num_examples = len(math_data)
num_correct = 0
start_time = time.time()

with open(out_path, "w", encoding="utf-8") as f: #C
for i, row in enumerate(math_data, start=1):

prompt = render_prompt(row["problem"]) #D
gen_text = generate_text_stream_concat( #E

model, tokenizer, prompt, device,
max_new_tokens=max_new_tokens,
verbose=verbose,

)

extracted = extract_final_candidate( #F
gen_text

)
is_correct = grade_answer( #G

extracted, row["answer"]

)
num_correct += int(is_correct)

record = { #H
"index": i,
"problem": row["problem"],
"gtruth_answer": row["answer"],
"generated_text": gen_text,
"extracted": extracted,
"correct": bool(is_correct),

}
f.write(json.dumps(record, ensure_ascii=False) + "\n")

progress_msg = eta_progress_message(
processed=i,
total=num_examples,
start_time=start_time,
show_eta=True,
label="MATH-500",

)
print(progress_msg, end="\r", flush=True)

if verbose: #I

print(
f"\n\n{'='*50}\n{progress_msg}\n"
f"{'='*50}\nExtracted: {extracted}\n"
f"Expected: {row['answer']}\n"
f"Correct so far: {num_correct}\n{'-'*50}"

)

seconds_elapsed = time.time() - start_time
acc = num_correct / num_examples if num_examples else 0.0
print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
print(f"Total time: {seconds_elapsed/60:.1f} min")
print(f"Logs written to: {out_path}")
return num_correct, num_examples, acc

- #A Helper function to print progress with optional ETA (estimated time to arrival)
- #B Make filename compatible with Windows
- #C Save results for inspection
- #D 1. Apply prompt template
- #E 2. Generate response
- #F 3. Extract and normalize answer
- #G 4. Grade answer
- #H Record to be saved for inspection
- #I Print responses during generation


The evaluate_math500_stream function in listing 3.15 uses the same main steps as the
smaller demo function from listing 3.14: for each problem, it renders the prompt, streams a
model response, extracts the answer candidate, and grades it against the reference answer.

In addition to iterating over a dataset with multiple entries, it adds some additional bells
and whistles. For instance, it saves the generated responses to a JSON file in a Python
dictionary-like format for record keeping and closer inspection.

Let's now run this function on a subset, the first 10 examples of MATH-500, which takes
about 0.7 minutes on a Mac Mini with an M4 chip. (Evaluating the reasoning model variant
takes about 7 min as it generates longer responses.)

print("Model:", WHICH_MODEL)
num_correct, num_examples, acc = evaluate_math500_stream(

model, tokenizer, device,
math_data=math_data[:10], #A
max_new_tokens=2048,
verbose=False #B

)

- #A Only evaluate on the first 10 examples
- #B Set to true to read the responses as they are being generated


In the code example above, we set max_new_tokens to a generous 2048, since the
reasoning model variant, per design, tends to generate much longer responses, and we
don't want to cut it off prematurely. This leads to much longer evaluation times, where it
may appear that the generation is stuck. Optionally, you could set verbose=True to see the
response being generated live, token by token.

The result of running the evaluate_math500_stream function is as follows:

Model: base
Device: mps
MATH-500: 10/10 | ETA: 00s
Accuracy: 30.0% (3/10)
Total time: 0.4 min

(Note that the results may differ depending on whether you execute the code on a CPU,
CUDA, or MPS device.)

As we can see, the model achieves a relatively low accuracy of 30%. We can open the
math500_base-mps.jsonl file in a text editor to analyze the results, together with the
generated response. For instance, we find that the answers, in all cases, have been
successfully extracted, but they are plain wrong, which indicates that the model does not
have very strong math problem-solving capabilities (yet). This is expected since it's merely
a base model.

###### LOADING THE .JSONL FILE PROGRAMMATICALLY

The .jsonl file suffix is a convention used for JSON files with one data entry per
row. You can view it in your favorite text editor. Optionally, we can load the .jsonl file
created during the evaluation in Python using the following code:

dev_name = str(device).replace(":", "-")
local_path = f"math500_{WHICH_MODEL}-{dev_name}.jsonl"
results = []
with open(local_path, "r") as f:

for line in f:
if line.strip():
results.append(json.loads(line))

The reasoning model variant, which you can enable by setting WHICH_MODEL =
"reasoning" in listing 3.1 in section 3.2, performs much better and achieves a 90%
accuracy on the same 10-sample subset and 50.8% on the complete 500-sample dataset,
as shown in table 3.1.

Table 3.1 MATH-500 task accuracy on different devices

|Mode|Device|Accuracy|MATH-500<br>size|
|---|---|---|---|
|Base|CPU|30%|10|
|Base|CUDA|30%|10|
|Base|MPS|30%|10|
|Reasoning|CPU|90%|10|
|Reasoning|CUDA|90%|10|
|Reasoning|MPS|80%|10|
| | | | |
|Base|CUDA|15.3%|500|
|Reasoning|CUDA|50.8%|500|


As shown in table 3.1, the reasoning variant, with its longer responses, has a drastically
improved accuracy, but also increases the compute intensity and answer generation time
substantially (from 0.4 min for the base model to 7 min for the reasoning model on a Mac
Mini with M4 chip on the 10-sample subset, and from 13.3 min to 185.4 min on an H100 on
the 500-sample dataset), which highlights one of the trade-offs of using reasoning models.
Please note that these numbers were obtained in PyTorch 2.8 and can differ in different
versions of PyTorch.

TIP The code repository contains a bonus script (https://github.com/rasbt/reasoning-from-scratch/
blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py) that runs the code in
this chapter in batched mode. This means it processes multiple examples per forward pass to
accelerate the evaluation while requiring more RAM. With a batch size of 128, this reduces the
runtime of the base model, when evaluating all 500 samples, from 13.3 min to 3.3 min on an H100
GPU. Similarly, it reduces the runtime of the reasoning model from 185.4 min to 14.6 min. Note that
the H100 is used as an example, and the script is compatible with other GPUs as well.

###### EXERCISE 3.2: CALCULATING THE AVERAGE RESPONSE LENGTH

Try to modify the code in this chapter to also report the average response length in
the evaluate_math500_stream function in listing 3.15. Instead of modifying the
function directly, you could also compute the response length from the generated
JSON report files.

###### EXERCISE 3.3: EXTENDING OR CHANGING THE EVALUATION DATASET

We choose a subset of only 10 examples for computational efficiency. Readers are
encouraged to consider running the code on larger or different portions of the
dataset to observe whether the 10-sample subset is representative. Ideally, you
could also experiment with your own data. (For reference, evaluating the base model
on the complete MATH-500 dataset takes about 13.3 min for the base model and
185.4 min for the reasoning model on an H100.)

###### EXERCISE 3.4: EXPERIMENTING WITH DIFFERENT PROMPT TEMPLATES

Models can be sensitive to different prompt templates. Experiment with different
prompt templates in listing 3.13 to see how it affects the results. Also, while the
Qwen3 team recommends using the base model without an additional chat template,
you can additionally enable the apply_chat_template=True setting in the tokenizer
(listing 3.1) and observe whether it improves the base model performance.

Note that this concludes our chapter on implementing a verification-based approach for
math tasks (figure 3.11). We chose math because it is both natural to implement and
widely used in reasoning-specific training, particularly reinforcement learning with verifiable
rewards, which we will cover in chapter 6. The same concept can be extended to other
domains, such as code, although we did not explore that here since executing code would
require additional setup of a secure virtual environment.

Before we move on, it is worth noting that evaluation also comes in many other flavors.
In this chapter we focused on verification-based accuracy for math problems, as it is a
popular method, and because we will reuse the same verifier as part of the reinforcement
learning pipeline in chapters 6 and 7.

For a broader overview, appendix F walks through other common evaluation strategies,
such as multiple-choice benchmarks, verifiers, leaderboards, and LLM-as-judge setups. This
appendix provides an overview with hands-on examples if you want a quick tour of how
these methods work in practice.

![image 42](<input (1)_images/imageFile42.png>)

- Figure 3.11 Mental model of the topics covered in this book. This chapter implemented a verifier-based
evaluation pipeline. In the next chapter, we will improve the reasoning capabilities of the LLM via more
advanced inference techniques.


Now, with an evaluation framework in place, the next chapter, as shown in figure 3.11,
focuses on improving the reasoning capabilities through more advanced inference (text
generation) techniques.

- 3.10 Summary


There are four main evaluation methods for LLMs: multiple choice,
verifiers, leaderboards, and LLM judges

Verification-based evaluation methods allow free-form answers and use
external tools to check correctness

This chapter focuses on verification-based evaluation by building a math
verifier that extracts, normalizes, and checks answers with SymPy

The verification pipeline involves several core steps from
loading the LLM to running the evaluation on a dataset

As part of the verification pipeline, answer extraction uses
string parsing to locate boxed content (with fallback
mechanisms for missing boxes)

Another step implements normalization, which
standardizes diverse answer formats by stripping LaTeX
and converting mathematical notation

Finally, the pipeline uses mathematical equivalence checking (via SymPy)
to compare expressions symbolically

The MATH-500 dataset provides 500 curated math problems for
evaluation

Prompt templates significantly impact model performance

The reasoning model achieves higher accuracy than the base model, but
requires much longer runtime

# 4 Improving reasoning with inference-time scaling

This chapter covers

Prompting an LLM to explain its reasoning to improve answer accuracy

Modifying the text generation function to produce diverse responses

Improving reasoning reliability by sampling multiple responses

Reasoning performance and answer accuracy can be improved without retraining or
modifying the model itself. These methods operate during inference, when the model
generates text. As shown in the overview in figure 4.1, in this chapter, we cover two
inference-time scaling methods. As we will see later in this chapter, both methods more
than double the accuracy of the base model we used in previous chapters.

![image 43](<input (1)_images/imageFile43.png>)

- Figure 4.1 A mental model of the topics covered in this book. This chapter focuses on techniques that improve
reasoning without additional training (stage 3). In particular, it extends the text-generation function and
implements a voting-based method to improve answer accuracy. The next chapter then introduces an
inference-time scaling approach where the model iteratively refines its own answers.


The next section provides a general introduction to inference-time scaling before discussing
the inference methods that are shown in figure 4.1 in more detail.

###### 4.1 Introduction to inference-time scaling

In general, there are two main strategies to improve reasoning:

- 1. Increasing training compute and
- 2. increasing inference compute (also known as inference-time scaling or
test-time scaling).


(In machine learning and AI, "compute" refers to the computational resources required to
train or run a model.) These two approaches are illustrated in figure 4.2.

![image 44](<input (1)_images/imageFile44.png>)

- Figure 4.2 Comparing inference-time scaling (this chapter) and training-time scaling (after chapter 5). Both
improve accuracy by using more compute, but inference-time scaling does this on the fly, without changing
the model's weight parameters. The plots are inspired by OpenAI's article introducing their first reasoning
model (https://openai. com/index/ learning- to-reason- with-llms/).


The plots shown in figure 4.2 make it look like we improve reasoning either by increasing
training-time compute or inference-time compute. LLMs are usually designed to improve
reasoning by combining heavy training-time compute (a topic of future chapters) and
increased inference-time compute (the topic of this chapter).

Inference-time compute scaling (also called inference-time scaling and test-time
compute scaling) means spending additional computation at answer-generation time, after
the model has already been trained, to improve the quality of its response. Instead of
changing the model's weights, we let the same trained model do more work per question,
for example by generating more tokens and "thinking" longer, sampling multiple answers,
or successively refining its answer.

In this book, we focus on three practical and foundational inference-time techniques
(figure 4.3):

- Method 1: Extending the chain-of-thought response to prompt the model
to explain its reasoning. This is a simple technique that can substantially
improve accuracy.

- Method 2: Parallel sampling via self-consistency, where the model
generates multiple responses and selects the most frequent one.

- Method 3: Iterative self-refinement, where the model reviews and
improves its own reasoning and answers across multiple steps. (This topic
is implemented and covered in more detail in the next chapter.)


![image 45](<input (1)_images/imageFile45.png>)

- Figure 4.3 Overview of three inference-time methods to improve reasoning covered in this book. The first
modifies the prompt to encourage step-by-step reasoning, and the second samples multiple answers and
selects the most frequent one. Both are discussed in this chapter. The third method, in which the model
iteratively refines its own response, is introduced in the next chapter.


The methods shown in figure 4.3 fall under the category of inference-time scaling because
they cause the model to generate more tokens, which increases compute resources during
inference. In other words, these methods achieve better accuracy while making the
inference process more expensive, which is a common theme of inference-time scaling
techniques.

The first method in this chapter improves answer accuracy by prompting the model to
explain its reasoning, a simple yet highly effective approach.

Next, we extend the text generation function introduced earlier to enable sampling
multiple responses for the same input (the 2nd method shown in figure 4.3). Using this
modified function, we implement self-consistency, a voting-based inference-time scaling
technique that increases answer accuracy by generating multiple answers and selecting the
most frequent one.

In total, I evaluated more than ten different inference-time scaling techniques across
thousands of experiments. The three methods illustrated in figure 4.3 were chosen for this
book because they combine three desirable properties. They deliver strong accuracy
improvements on the model used here, they represent the main practical paradigms of
inference-time scaling, and they are simple enough to implement from scratch without
introducing too much extra machinery. Concretely, they cover the two main patterns of
generating longer responses and generating multiple responses.

Longer responses that include reasoning and explanations (methods 1 and 3) are what
we typically expect from reasoning models, as discussed in previous chapters. And parallel
sampling (method 2) is also widely used in production systems, such as Claude 4 (as
described in https://www.anthropic.com/news/claude-4).

###### 4.2 Loading a pre-trained model

Before we begin implementing the inference-time scaling methods described in the previous
section, we load the pre-trained base model we used in the previous sections.

- Listing 4.1 Load tokenizer and base model


import torch

- from reasoning_from_scratch.ch02 import get_device
- from reasoning_from_scratch.ch03 import (
load_model_and_tokenizer


)

device = get_device()
device = torch.device("cpu") #A

model, tokenizer = load_model_and_tokenizer(
which_model="base",
device=device,
use_compile=False

)

#A Delete this line to run the code on a GPU (if supported by your machine)

The code in listing 4.1 loads the model and tokenizer we are using in this chapter. It is
similar to the code we used in previous chapters.

Since the code in this chapter is cheap to run, I recommend running this chapter on the
"cpu" device for the first time to get the same results as shown in this chapter (running
this chapter on a GPU can subtly alter the results.)

Let's try the model on a prompt from the MATH-500 dataset, which we worked with in
the previous chapter:

from reasoning_from_scratch.ch03 import render_prompt

raw_prompt = (
"Half the value of $3x-9$ is $x+37$. "
"What is the value of $x$?"

)
prompt = render_prompt(raw_prompt)
print(prompt)

The formatted prompt is as follows:

You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
Half the value of $3x-9$ is $x+37$. What is the value of $x$?

Answer:

We can use the prompt above as input to the generate_text_stream_concat we defined in
the previous chapter. We want to compare several inference-time scaling strategies while
keeping the rest of the code unchanged. For that reason, we make a small modification to
the text generation wrapper so that we can swap in different generation functions and
settings without rewriting the surrounding prompt-handling and output code. The changes
are highlighted via the code comments in listing 4.2:

- Listing 4.2 Modified generate_text_stream_concat function


from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache

def generate_text_stream_concat_flex(
model, tokenizer, prompt, device, max_new_tokens,
verbose=False,
generate_func=None, #A
**generate_kwargs #A

):

if generate_func is None: #B
generate_func = generate_text_basic_stream_cache

input_ids = torch.tensor(
tokenizer.encode(prompt), device=device
).unsqueeze(0)

generated_ids = []
for token in generate_func(

model=model,
token_ids=input_ids,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id,

**generate_kwargs, #C
):

next_token_id = token.squeeze(0)
generated_ids.append(next_token_id.item())

if verbose:

print(
tokenizer.decode(next_token_id.tolist()),
end="",
flush=True

)
return tokenizer.decode(generated_ids)

- #A We add parameters to accept a text generation function and additional arguments
- #B If the text generation function is undefined, we use generate_text_basic_stream_cache similar to chapter 3
- #C We can pass additional arguments to the text generation function if needed


In short, the generate_text_stream_concat_flex function above is similar to the
generate_text_stream_concat function from the previous chapter, except that we can
now pass in the text generation function (like generate_text_basic_stream_cache) as a
function argument instead of hard-coding it. The practical benefit is that the outer wrapper
can keep handling prompt encoding, streaming, and decoding in one place, while the inner
generation step can be replaced with different decoding strategies. This makes it easier to
compare methods fairly, since we change only the generation logic rather than the entire
pipeline. In future sections, we will swap the generate_text_basic_stream_cache function
with more advanced functions.

The usage is also similar to before, except that we can now pass the text generator
function (for example, generate_text_basic_stream_cache) explicitly:

response = generate_text_stream_concat_flex(
model, tokenizer, prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_basic_stream_cache

)

The generated output is:

\boxed{20}

Note that the answer is wrong, and the correct solution is 83. In the remainder of this
chapter, and the next chapter, we will implement inference-time scaling methods to get the
model to generate the correct answer.

##### 4.3 Generating better responses with chain-of-thought prompting

After loading the pre-trained base model in the previous section and setting up the text
generation function, this section focuses on improving the model output via so-called chain-
of-thought prompting.

Chain-of-thought prompting is a classic, simple, and effective technique that modifies
the input prompt to encourage the LLM to generate an explanation or so-called chain-of-
thought (also called reasoning chain), as illustrated in figure 4.4.

![image 46](<input (1)_images/imageFile46.png>)

- Figure 4.4 The first inference-time method, chain-of-thought prompting, modifies the prompt to encourage the
model to explain its reasoning step by step before producing a final answer.


The simplest way to try chain-of-thought prompting is to append an extra instruction that
asks the model to reason step by step. There are multiple ways to phrase this. For
example, the original zero-shot chain-of-thought paper used "Let's think step by
step." (https://arxiv.org/abs/2205.11916). This is just one example, and different
phrasings may work better in practice depending on the model and task. Here, we use
"Explain step by step." because it worked well in my experiments for this chapter and
keeps the example simple, as shown below:

prompt_cot = prompt + " \n\nExplain step by step."

response_cot = generate_text_stream_concat_flex(
model, tokenizer, prompt_cot, device,
max_new_tokens=2048, verbose=True,

)

The response is now as follows:

To solve the problem, we need to find the value of \( x \) such that
half the value of \( 3x - 9 \) is equal to \( x + 37 \).

# ... #A

### Step 3: Solve for \( x \)
Subtract \( 2x \) from both sides to isolate \( x \):
\[

- 3x - 2x - 9 = 74
\]
Simplify:
\[
x - 9 = 74
\]
Add 9 to both sides to solve for \( x \):
\[
x = 74 + 9
\]
\[
x = 83
\]


### Final Answer:
\[
\boxed{83}
\]

#A The response was truncated to save space.

As we can see, the model now writes a lengthy step-by-step explanation and, in this case,
now arrives at the correct answer.

This simple chain-of-thought prompting is a good demonstration of the inference-time
scaling trade-off. While the model now answers correctly, it expends many more tokens
than before. As discussed in chapter 2, LLMs generate text one token at a time, and each
additional token requires another forward pass through the model. So these intermediate
reasoning steps do not just make the answer longer, they also directly increase latency,
compute cost, and often API cost in practice.

Note that while the model generates the correct answer in this case, not all problems
benefit from chain-of-thought prompting. On simple problems, it can even sometimes
degrade the model's performance, as the model might sometimes generate erroneous
explanations and mislead itself. This phenomenon is also known as "overthinking."

Lastly, not every model benefits from the "Explain step by step" (or similar) instruction.
In this case, we use a simple base model that doesn't always generate explanations, so
chain-of-thought prompting can clearly help. Trained reasoning models, such as the
"reasoning" variant of the Qwen3 model we used in the exercises of the previous chapter,
already write explanations alongside their responses and don't need or benefit from this
type of chain-of-thought prompting.

###### WHY CHAIN-OF-THOUGHT CAN IMPROVE ACCURACY

Chain-of-thought prompting asks the model to write out the intermediate steps that
lead to a final answer. This helps in two practical ways.

First, walking through the steps gives the model more opportunities to correct
itself.

Second, step-by-step reasoning matches how many training examples are written.
For instance, large math and logic datasets often contain detailed solutions, so
asking for a chain of thought aligns the model with patterns it has already learned.

At the same time, chains-of-thought are not a guarantee for correctness. It can
still produce wrong reasoning, and for very simple problems it may even introduce
unnecessary steps that lead to more mistakes. In other words, chains-of-thought can
improve accuracy on many reasoning tasks, but it is not universally beneficial.

Overall, chain-of-thought answering does not provide the model with new
knowledge, but it changes how the model uses its existing knowledge. Often, this
shift can lead to more reliable answers. This is especially true for math, code, logic
problems, and other sorts of multi-step problems.

###### EXERCISE 4.1: USE CHAIN-OF-THOUGHT PROMPTING ON MATH-500

Modify the evaluate_math500_stream function in section 3.9 of chapter 3 to see if
chain-of-thought prompting improves the MATH-500 accuracy of the base model.

##### 4.4 Controlling output diversity with temperature scaling

The previous section gave a brief taste of inference-time scaling by extending the model's
answer via chain-of-thought prompting. Chain-of-thought prompting can be seen as a
sequential technique as we extend the number of next-token prediction steps.

In the remainder of this chapter, we will implement a technique that generates multiple
answers, as illustrated in figure 4.5. Since the answers are independent of each other, this
can be implemented as a parallel sampling (if we have the necessary resources, for
example, using multiple GPUs), which in this case wouldn't increase the wait time for a user
to get the answer.

(Note that chain-of-thought prompting can also be combined with this technique, but
more on that in a later section.)

![image 47](<input (1)_images/imageFile47.png>)

- Figure 4.5 The second inference-time method, self-consistency sampling, generates multiple answers and
selects the most frequent one. This method relies on temperature scaling, covered in this section, which
influences how the model samples its next token.


The self-consistency technique, illustrated in figure 4.5 and covered in this chapter, is also
called self-consistency sampling and was formally described in the Google Research paper
Self-Consistency Improves Chain-of-Thought Reasoning in Language Models (https://arxiv.
org/abs/2203.11171).

Before implementing self-consistency sampling (step 5 in figure 4.5), we first need to
extend the text generation function so that it can produce multiple different answers for the
same prompt. To achieve this, we will implement two techniques, temperature scaling (step
3) and top-p filtering (step 4), which allow the model to sample different responses.

Temperature scaling is the main focus of this section, because it becomes one of the key
controls for output diversity in the rest of this chapter and also forms the basis for the top-
p filtering method that follows. Before we get to temperature scaling, the next subsection
provides a brief overview of how the next token is sampled in an LLM.

These low-level sampling controls may seem technical in isolation, but they are exactly
what lets us generate diverse candidate answers for self-consistency later in this chapter.

- 4.4.1 Understanding the process of selecting the next token


This subsection gives a closer look at the text generation process we implemented so far
and explains how the next-token selection process works under the hood. This information
will help you understand the motivation behind temperature scaling.

For instance, suppose we have the following simple prompt:

ex_prompt = "The capital of Germany is"

response = generate_text_stream_concat_flex(
model, tokenizer, ex_prompt, device,
max_new_tokens=1, verbose=True

)

The model's response is " Berlin".

While this looks relatively simple, there are multiple steps happening under the hood, as
illustrated in figure 4.6.

![image 48](<input (1)_images/imageFile48.png>)

- Figure 4.6 How an LLM generates the next token. As in the other process diagrams in this book, the flow runs
from bottom to top. The model converts the input into token IDs, computes scores for all possible next tokens,
and selects the one with the highest score as the next output.


When generating text, as discussed in chapter 2, the inputs are first converted into token
IDs:

input_token_ids = torch.tensor(

tokenizer.encode(ex_prompt), device=device
).unsqueeze(0)
print(input_token_ids)

In this case, the token IDs are

tensor([[ 785, 6722, 315, 9856, 374]])

In the second step (step 2 in figure 4.6), we get the scores for the output token we want to
generate. These model output scores are also called logits. Note that an LLM generates one
output token for each input token, but we are only interested in the last token, which we
select via the [:, -1] tensor indexing. This last token corresponds to the token we want to
generate:

with torch.inference_mode():

next_token_logits = model(input_token_ids)[:, -1]
print(next_token_logits.shape)

The printed output shape is [1, 151936], where 151936 is the vocabulary size of this
tokenizer and model. The vocabulary size contains all the unique tokens the tokenizer can
handle and the LLM can generate.

To actually obtain the next generated token (here: " Berlin"), we have to find a
vocabulary entry that is associated with the largest score (step 3 in figure 4.6):

max_token_id = torch.argmax(next_token_logits)
print(f"Token ID: {max_token_id}")
print(f"Decoded token: '{tokenizer.decode([max_token_id])}'")

The output is:

Token ID: 19846
Decoded token: ' Berlin'

Above, we covered the three main steps in generating the next token. Before we go to the
next section, let us take a closer look at the score distribution of the next_token_logits
tensor we passed into torch.argmax function to obtain the token IDs, and plot them in
matplotlib:

- Listing 4.3 Plotting the next-token logit scores


import matplotlib.pyplot as plt

def plot_scores_bar(
next_token_logits, start=19_800, end=19_900,
arrow=True, ylabel="Logit value"

):

x = torch.arange(start, end) #A
logits_section = next_token_logits[0, start:end].float().cpu() #B

plt.bar(x, logits_section) #C
plt.xlabel("Vocabulary index")
plt.ylabel(ylabel)

if arrow: #D
max_idx = torch.argmax(logits_section)
plt.annotate(

"Berlin",
xy=(x[max_idx], logits_section[max_idx]),
xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
arrowprops={

"facecolor": "black", "arrowstyle": "->", "lw": 1.5

},
fontsize=10,

)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plot_scores_bar(next_token_logits)

#A Select vocabulary subsection
#B .cpu() is a shortcut for to(torch.device("cpu"))
#C Plot the logits (scores) within the selected range
#D Draw an arrow to highlight the largest score

Note that we restrict the plot to the 100 vocabulary index tokens between 19,800 and
19,900, rather than plotting the scores for all 151,936 entries, which would make the plot
too crowded. (I selected this specific range such that it contains the entry with the highest
score.) The resulting plot is shown in figure 4.7.

![image 49](<input (1)_images/imageFile49.png>)

- Figure 4.7 Example of next-token logits for a 100-token slice of a language model's much larger vocabulary.
Each bar represents one possible token's score within this slice, with "Berlin" having the highest logit value
and being selected as the next token.


The plot in figure 4.7 shows all 100 logit values (scores) for vocabulary indices 19,800-
19,899. The values range approximately from -8 to 20, where 20 corresponds to the
vocabulary index for the token " Berlin".

- 4.4.2 Rescaling token scores (logits) via a temperature parameter


Now that we have walked through how the model selects its next token, we can introduce
the concept of temperature. Temperature, or rather a chosen temperature parameter,
changes how sharp or spread out the logits (token scores) are, which in turn affects how
the next token is selected.

As shown in figure 4.8, this section focuses on rescaling the next-token logits with a
temperature parameter before using them for sampling. Rescaling here means adjusting
the magnitude of the scores so the sampling step becomes more or less sensitive to the
score differences.

![image 50](<input (1)_images/imageFile50.png>)

- Figure 4.8 In this section, we implement the core part of temperature scaling (step 3.2), which adjusts the
next-token scores. This allows us to control how confidently the model selects its next token in later steps.


The code for implementing the temperature rescaling step, step 3.2 in figure 4.8, is
relatively short and simple, as shown in listing 4.4 below.

- Listing 4.4 Rescaling next-token scores via temperature scaling


def scale_logits_by_temperature(logits, temperature):
if temperature <= 0:

raise ValueError("Temperature must be positive")
return logits / temperature

In essence, the code in listing 4.4 rescales the logit values before converting them to
probabilities in the next section.

In practice, temperature values are expected to be positive numbers. A temperature of
1.0 means no change, since dividing a number by 1 is the number itself.

We will add additional temperature scaling safeguards later when we add temperature
scaling to our text generation function.

For all other values, the logits are divided by the temperature. A temperature lower than
1.0 makes the distribution sharper (which will make the model more confident when we
select the next token in the upcoming sections). Temperatures higher than 1.0 flatten the
logits, which can make the sampling (step 3.4 in figure 4.8) more diverse. In other words,
higher temperatures reduce the probability gap between the top token and lower-ranked
tokens, which increases the chance that sampling will pick a non-maximum token.

###### Let's see this scale_logits_by_temperature function in action and try it out with relatively extreme temperature values 0.5 and 5.0 for a stronger effect:

- Listing 4.5 Plotting the temperature-rescaled logits


def plot_logits_with_temperature(
next_token_logits, start=19_800, end=19_900,
temps=(0.5, 5.0),

):

x = torch.arange(start, end)
logits_orig = next_token_logits[0, start:end].float().cpu()

logits_scaled = [ #A

scale_logits_by_temperature(logits_orig, T) for T in temps #A
] #A

plt.plot(x, logits_orig, label="Original logits", lw=2) #B
plt.plot( #B

x, logits_scaled[0], #B

- label=f"T={temps[0]} (sharper)", ls="--", lw=1 #B

) #B
plt.plot( #B

x, logits_scaled[1], #B

- label=f"T={temps[1]} (flatter)", ls=":", lw=3 #B


) #B

# Highlight max logit
max_idx = torch.argmax(logits_orig) #C
plt.annotate( #C

"Berlin",
xy=(x[max_idx], logits_orig[max_idx]),
xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
fontsize=12,

)

plt.xlabel("Vocabulary index")
plt.ylabel("Logit value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plot_logits_with_temperature(
next_token_logits,

temps=(0.5, 5.0) #D
)

- #A Apply temperature scaling
- #B Plot the logits (scores) within the selected range
- #C Draw an arrow to highlight the largest score
- #D Run the rescaling and plotting with temperatures 0.5 and 5


As shown in the plot in figure 4.9, the larger temperature (5.0) yields a much flatter score
distribution, whereas the smaller temperature (0.5) yields a much sharper one.

![image 51](<input (1)_images/imageFile51.png>)

- Figure 4.9 The effect of temperature scaling on logits. Lower temperatures make the distribution sharper,
while higher temperatures flatten it. (Please note that this visualization is shown as a line plot for readability,
though a bar plot would more accurately represent the discrete vocabulary scores.)


Note that the plotting code in listing 4.5 above looks very similar to the plotting code we
used in the previous section (listing 4.3). In addition to applying the temperature scaling,
we now use a line plot (plt.plot) instead of a bar plot (plt.bar). While a bar plot is
technically a better choice for the discrete vocabulary indices on the x-axis, the line plot
makes it easier to visualize and compare the rescaled logits in this case.

The important point is not the absolute height of one logit by itself, but how the gaps
between logits change. If a token such as "Berlin" stands out more strongly than the
alternatives, it will become more likely to be selected later. If the gaps shrink, lower-ranked
tokens become more likely to be selected compared to before. The next section illustrates
this by converting the logits into probabilities.

###### WHY TEMPERATURE?

The term temperature comes from physics, where temperature controls how much
randomness or movement there is in a system. In LLMs, we use the same idea to
control how confidently or creatively the model chooses its next token, as we will see
in the next two sections.

- 4.4.3 Sampling the next token from a probability distribution


The previous section rescaled the logit values using different temperature values. The
purpose of this is that it lets us (later) influence how the model selects the next token.

Before we get to the next-token sampling portion in the next section, this section adds
one more intermediate step: converting the rescaled logits into probability scores, as shown
in figure 4.10.

![image 52](<input (1)_images/imageFile52.png>)

- Figure 4.10 Overview of the sampling process for generating tokens. In this section, we focus on steps 3.3
and 3.4, where the next-token scores are converted into a probability distribution, and the next token is
sampled based on that distribution.


To demonstrate how rescaled logits are converted into probability scores, as described in
step 3.3 of figure 4.10, we will use a temperature of 5.0. This makes it easier to visualize
the resulting probabilities in a plot. The " Berlin" token, for example, has such a high
logit value that it would otherwise dominate the scale and make it difficult to see the
probabilities of the surrounding tokens.

The conversion from rescaled logits to probability scores can be done with a single
function call, torch.softmax, as shown in listing 4.6 below.

- Listing 4.6 Sampling the next-token from a probability distribution


rescaled_logits = scale_logits_by_temperature(next_token_logits, 5.0) #A

next_token_probas = torch.softmax( #B

rescaled_logits, dim=-1
)

#A Step 3.2: Rescale next-token scores
#B Step 3.3 Convert rescaled logits into probability scores

The torch.softmax function in listing 4.6 normalizes the logit values into values in the
range between 0 and 1, and such that the values sum to 1, which we can confirm via the
following code:

print("Probability sum:", torch.sum(next_token_probas))

Additionally, let's visualize the converted scores by reusing the plot_scores_bar function
from listing 4.3 earlier:

plot_scores_bar(

next_token_probas, arrow=False, ylabel="Probability value"
)

The resulting plot, now with the probability values on the y-axis, is shown in figure 4.11.

![image 53](<input (1)_images/imageFile53.png>)

- Figure 4.11 Token probabilities obtained by applying the softmax function to the rescaled logits. The token of
the highest probability (corresponding to " Berlin", but with the label omitted for code simplicity) is
selected as the next output.


In the plot in figure 4.11, we can see that the token ID 19,846 (" Berlin") has the highest
value in this selected vocabulary range. The probability is 0.0003, which we can confirm via
the following code:

print("Token ID 19,846 probability:", next_token_probas[:, 19846])

Note that while this 0.0003 value is relatively small, there is no other value larger than this
outside the selected vocabulary range in this plot. For instance, using the following code,
we can confirm that 0.0003 is indeed the largest value:

print("Highest probability:", max(next_token_probas.squeeze(0)))

We can interpret this score as the model's confidence. This means that the model is more
confident in " Berlin" (19,846) as the next token than in other tokens.

The reason the value is so small is that we are using a large temperature, which makes
it easier to plot this value next to the other token values. If we change the temperature
from 5 to 0.5, the probability score increases from 0.0003 to 0.3398, and the probability
scores of the other tokens will be even closer to 0.

###### THE SOFTMAX FUNCTION UNDER THE HOOD

The softmax function converts a vector of raw scores (logits) into a probability
distribution where each value lies between 0 and 1, and all values sum to 1. This
conversion makes it easier to interpret them and sample from them later.

If you are familiar with mathematical notation, the formula behind the softmax
function is

softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

Here, z is a a vector of real-valued inputs

z = [z₁, z₂, …, zₙ],

where

n is the number of elements in the vector,

- i is the index of the current element (1 ≤ i ≤ n),

- j is the index used to sum over all elements (1 ≤ j ≤ n),


This produces a normalized probability for each element zᵢ, such that

Σᵢ softmax(zᵢ) = 1

In code, the softmax function can be implemented with a simple 3-liner:

def softmax_with_temperature(logits, temperature):
scaled_logits = logits / temperature
return torch.softmax(scaled_logits, dim=0)

In practice, we prefer the torch.softmax function because it has some additional
numerical stability improvements to handle very small and very large values more
reliably.

The purpose of converting the logits into these probability scores is that the probability
scores are somewhat more interpretable, and we can now sample from them using a
torch.multinomial function.

For instance, if we draw one sample from this probability distribution, we have a 0.03%
chance of getting " Berlin" with our temperature setting of 5.0 (and a 33.98% chance of
getting " Berlin" with a temperature of 0.5).

The sampling process, corresponding to step 3.4 in the figure at the beginning of this
section, can be implemented as follows:

torch.manual_seed(123)
print(

"Sampled token:",
torch.multinomial(next_token_probas.cpu(), num_samples=1)

)

This code returns token ID 65,094, which corresponds to the word " mistress". Note that
the word doesn't make sense in our context, "The capital of Germany is", and it was
selected randomly in this case and influenced by the high-temperature setting, which
encourages tokens other than " Berlin" to be sampled.

The torch.multinomial function samples the vocabulary indices proportional to their
probabilities. In other words, vocabulary indices with higher probabilities are more likely to
be sampled. If we repeated the sampling a very large number of times, we would sample
the vocabulary index corresponding to the token " Berlin" with 0.03% probability, given a
temperature of 5.

Before we look at some additional examples, note that we specified a random seed
above to make the code in this chapter reproducible. The torch.multinomial() function
may still yield different results on "cuda" and "mps" devices, and may even crash when we
draw larger numbers of samples (I observed this issue in PyTorch 2.9 on both "cuda" and
"mps" devices), which is why we run the sampling on the CPU via .cpu().

Let's now sample more next-token candidates to get a more representative sample.

- Listing 4.7 Sampling multiple next-token candidates


def count_samples(probas, num_samples=1000, threshold=1, tokenizer=None):
samples = torch.multinomial( #A
probas.cpu(), num_samples=num_samples, replacement=True

)
counts = torch.bincount(samples.squeeze(0), minlength=1) #B

for i, c in enumerate(counts):
if c > threshold: #C

if tokenizer is None:

print(f"Vocab index {i}: {c.item()}x")
else:

print(f"'{tokenizer.decode([i])}': {c.item()}x")

- #A Draw samples according to probabilities
- #B Count how often each index was selected
- #C Print frequently sampled vocabulary indices (entries)


This count_samples function in listing 4.7 samples token indices from a probability
distribution and counts how often each token is drawn. It uses torch.multinomial to
randomly select num_samples indices proportional to their probabilities. The
replacement=True setting allows us to draw the same token multiple times.

Then, torch.bincount counts how often each index appears. Finally, it prints only
tokens that occur more than a specified threshold, so that it doesn't clutter the output with
more infrequently drawn tokens.

###### MULTINOMIAL SAMPLING

Multinomial sampling is the procedure used to pick the next token given a probability
distribution over the vocabulary, like the softmax probability scores. In multinomial
sampling, instead of always selecting the most likely token (known as greedy
decoding), we draw one token at random, where tokens with higher probability are
more likely to be selected. This randomness is important for generating diverse
responses, which we later use in self-consistency.

To illustrate it further we can think of the probability scores as a set of weighted
choices. For example, a token with probability 0.40 is four times as likely to appear
as one with probability 0.10, but both remain possible outcomes.

Suppose the model assigns the following probabilities:

"Berlin": 0.70

"Munich": 0.20

"Hamburg": 0.10

Greedy decoding would always return "Berlin."

Multinomial sampling instead draws one token according to these weights. Across
a very large number of draws, "Berlin" will appear most often (70% of the time),
"Munich" sometimes (20% of the time), and "Hamburg" only occasionally (10% of
the time).

This variability is what allows us to generate multiple candidate answers for self-
consistency in later sections.

Please note that the count_samples function is meant for illustration only. It draws many
samples from the distribution so you can see how often each token is selected. In real text
generation we only draw one token at a time, but taking a large number of samples here
makes the underlying probabilities easier to visualize and understand.

First, let's run the count_samples function on the probability scores that we obtained by
applying a temperature of 5:

torch.manual_seed(123)
count_samples(next_token_probas, tokenizer=tokenizer)

The output is as follows:

'}': 2x
' </': 2x
' represent': 2x
' Inf': 2x
'()*': 2x
' beside': 2x
' Kob': 2x
' ': 2x

As we can see, even though the default sample size is 1000, none of the sampled tokens
appear more than 2 times. Also, these are all nonsense tokens in the context of the "The
capital of Germany is" query. The reason for these nonsensical results is that we used a
temperature value that is much too high.

Let's try a lower temperature value of 0.35, which makes the score distribution sharper,
and which makes it more likely to select a meaningful next token:

torch.manual_seed(123)
probas_lowT = torch.softmax(

scale_logits_by_temperature(next_token_logits, 0.35), dim=-1
)

In this case, we see the following output:

' __': 158x
' Berlin': 435x
' ____': 169x
' ______': 209x
' Munich': 3x
' Hamburg': 3x
' _____': 18x

The output makes a lot more sense as next-token candidates for our "The capital of
Germany is" query. Out of the 1000 samples, the token " Berlin" was drawn 435 times.
You can check via print(probas_lowT[0, 19_846]) that the probability of drawing this
token is approximately 42% given the temperature value 0.35. To make it even more likely
that " Berlin" is sampled, we could further reduce the temperature.

Notice that some of the other rarer candidates also make sense, for example, both "
Munich" and " Hamburg" are big cities in Germany, so they are not completely unrelated to
the query. The underscore responses (" ____") are likely due to the model having seen
text in the form of a quiz with a placeholder, e.g., "The capital of Germany is ____".

You may wonder, if " Berlin" is the correct answer, what's the point in making the
model occasionally give the wrong answer by adding this temperature-rescaling and
multinomial sampling?

In general, for different kinds of queries, introducing randomness during sampling helps
the model explore alternative responses instead of always choosing the single most likely
token. This variability can be useful for creative or open-ended tasks, where there may be
multiple valid completions.

Specifically, in reasoning tasks, we can leverage this sampling diversity through
techniques such as self-consistency (section 4.6), which generate multiple candidate
answers and compare them to improve answer accuracy.

- 4.4.4 Adding temperature scaling to the text generation function


Before we move to the next section and introduce another improvement to the text
generation process by adding a token-probability selection filter, let's add the temperature
scaling modification to the text generation function so we can use it more readily when
generating new tokens via the model./

- Listing 4.8 Text generation with temperature scaling


from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_temp_stream_cache(

model,
token_ids,
max_new_tokens,
eos_token_id=None,
temperature=0.

):

model.eval()
cache = KVCache(n_layers=model.cfg["n_layers"])
model.reset_kv_cache()

out = model(token_ids, cache=cache)[:, -1] #A
for _ in range(max_new_tokens):

########################################
# NEW:
orig_device = token_ids.device

if temperature is None or temperature == 1.0:
next_token = torch.argmax(out, dim=-1, keepdim=True)

else:
logits = scale_logits_by_temperature(out, temperature) #B
probas = torch.softmax(logits, dim=-1) #C
next_token = torch.multinomial(probas.cpu(), num_samples=1) #D
next_token = next_token.to(orig_device)

#########################################
if (eos_token_id is not None

and torch.all(next_token == eos_token_id)):
break

yield next_token
out = model(next_token, cache=cache)[:, -1]

- #A Step 3.1: Get logits
- #B Step 3.2: Apply temperature scaling on logits
- #C Step 3.3: Convert to probabilities
- #D Step 3.4: Sample token according to probabilities


The generate_text_temp_stream_cache in listing 4.8 is similar to the
generate_text_stream_cache function from the chapter 2 exercises, which we also used in
chapter 3. What's new is that we have now inserted the temperature rescaling and
sampling. The new parts are below the # New comment indicator in the code.

torch.manual_seed(123)
response = generate_text_stream_concat_flex(

model, tokenizer, prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_temp_stream_cache, #A
temperature=1.1

)

#A Use the new temperature scaling-based text generation function

The output is \boxed{$x = \frac{90}{7}$}. The correct answer is 83, so the model is still
wrong, but this was more meant as a demonstration that we can tweak the answer by
using temperature scaling and sampling.

###### CHOOSING TEMPERATURE SETTINGS

In practice, temperature selection depends on the goal. A temperature of 0.0
corresponds to greedy decoding, where we always pick the highest-probability token.
Small nonzero values such as 0.3-0.8 are often useful when we want a bit more
diversity without making the output too erratic. Much higher values make the model
explore more broadly, which can be useful for creative generation or broad search,
but often hurts reliability on tasks where we want the single most likely answer.

In the next section, we will learn how to improve the sampling process.

##### 4.5 Balancing diversity and coherence with top-p sampling

In the previous section, we saw how temperature scaling and sampling (via
torch.multinomial) can increase the diversity of the LLM responses, for better or for
worse. Specifically, we saw that using the approach described in the previous section, we
may end up sampling "weird" tokens that are unrelated to the user query.

In this section, we improve the sampling process by adding a top-p filter (figure 4.12)
such that very low-confidence tokens are not sampled by accident. The top-p sampling
process described in this section is also known as nucleus sampling.

![image 54](<input (1)_images/imageFile54.png>)

- Figure 4.12 Overview of the top-p filtering process. The filter keeps only the highest-probability tokens by
sorting them, applying a cumulative cutoff, selecting the top-p subset, and renormalizing the result.


- Figure 4.12 summarizes the four steps that make up the top-p filter: sorting the token
probabilities (4.1), computing their cumulative sum (4.2), selecting the subset that satisfies
the top-p cutoff (4.3), and renormalizing the remaining probabilities so that they again
form a valid distribution (4.4).


The renormalization step is necessary because, once we remove all low-probability
tokens, the remaining probabilities no longer sum to one. To sample correctly, we rescale
these remaining values so they represent a proper probability distribution.

Having outlined the full top-p filtering pipeline in figure 4.12, the next subsections walk
through each step in detail. We will start with a brief recap of temperature scaling and then
implement steps 4.1 to 4.4 that are shown in figure 4.12.

- 4.5.1 Selecting a subset of top-p tokens


In this section, we implement the top-p filter illustrated in figure 4.12 that we will use to
improve the text generation function, which we plan to use for self-consistency sampling.

###### NOTE The purpose of top-p filtering in this section is to drop low-probability tokens so that only the most plausible options remain during sampling. This reduces the chance of producing tokens that do not fit the context.

Before we implement the top-p filter, let us briefly recap the temperature-scaling and
sampling process with a simpler toy dataset, which makes it easier to illustrate the process.
For instance, let's assume the model's and tokenizer's vocabularies have only 10 entries,
rather than 151,936.

- Listing 4.9 Recap of temperature scaling and sampling with toy data


toy_logits = torch.tensor( #A

[-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]
)

toy_logits_scaled = scale_logits_by_temperature(toy_logits, 1.0) #B
toy_probas = torch.softmax(toy_logits_scaled, dim=-1) #C

plt.bar( #D
torch.arange(len(toy_probas)), toy_probas,
alpha=0.5

)

plt.ylim([0, 1])
plt.xlabel("Vocabulary index")
plt.ylabel("Probability")
plt.show()

- #A Step 3.1: Get logits (here: use toy logits for 10 tokens)
- #B Step 3.2: Apply temperature scaling (we use 1.0 as a placeholder)
- #C Step 3.3: Convert to probabilities
- #D Plot probabilities in a bar plot


Please note that the toy_logits variable holds the example values that we would get for
the next-token logit scores via the model(token_ids, cache=cache)[:, -1] call in the
previous section, assuming that the vocabulary size is 10. The resulting plot is shown in
figure 4.13.

![image 55](<input (1)_images/imageFile55.png>)

- Figure 4.13 Example of token probabilities before top-p filtering. The distribution includes many low-
probability tokens, which will later be truncated by applying a cumulative probability threshold.


The bar plot in figure 4.13 shows the next-token logit scores after rescaling them to a
probability score. So far, this is a recap of the previous section. Next, we will add the first
two top-p filtering steps (steps 4.1 and 4.2) illustrated in figure 4.12. This involves sorting
the probability scores in descending order and computing the cumulative sum.

- Listing 4.10 Compute cumulative probability sum


sorted_probas, sorted_idx = torch.sort(toy_probas, descending=True) #A
cumsum = torch.cumsum(sorted_probas, dim=-1) #B

plt.bar(
torch.arange(len(sorted_probas)), sorted_probas,
alpha=0.5

)
plt.step(

torch.arange(len(cumsum)), cumsum,
where="mid", color="C1", label="Cumulative sum"

)

plt.ylim([0, 1])
plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.show()

#A Step 4.1: Sort by descending probability
#B Step 4.2: Compute cumulative sum

The torch.cumsum function used in listing 4.10 above computes the cumulative sum of
elements along a given dimension. In this example, it takes the sorted token probabilities
and adds them up step by step, so each position in torch.cumsum represents the total
probability accumulated up to that token.

For instance, the first element equals the highest probability, the second equals the sum
of the top two, and so on, until the final value reaches 1. This is best explained by looking
at the cumulative step plot, produced by the code in listing 4.10 and shown in figure 4.14.

![image 56](<input (1)_images/imageFile56.png>)

- Figure 4.14 Visualization of sorted token probabilities and their cumulative sum. This step prepares for top-p
filtering by showing how probabilities accumulate when ordered from highest to lowest, which helps
determine where to set the cutoff threshold.


Now that we have the cumulative probability sum, we can implement the core top-p
filtering step. The "p" in top-p stands for probability, and top-p can be translated to "keep
the smallest set of tokens whose cumulative probability stays below or equal to p." A simple
code implementation of top-p filtering may look as follows:

- Listing 4.11 A simple-top-p filtering rule


top_p = 0.8
keep_mask = cumsum <= top_p
n_kept = torch.sum(keep_mask).item()
print("Cumulative sum:", cumsum)
print("Tokens kept:", n_kept)

In code, this is implemented as keep_mask = cumsum <= top_p, which marks all tokens
whose cumulative probability mass does not yet exceed the threshold p (here, defined via
top_p=0.8). The number of retained tokens is then computed and assigned to the variable
n_kept. The output is

Cumulative sum: tensor([0.4538, 0.7290, 0.8119, 0.8798,
0.9170, 0.9475, 0.9701, 0.9886, 0.9969, 1.0000])
Tokens kept: 2

Looking at the returned cumulative probabilities, the second entry (0.7290) is just below
the top-p threshold of 0.8, and the third entry (0.8119) exceeds it, which is why only the
first 2 tokens are kept.

Now, a more common variant of top-p filtering includes the token that exceeds the
threshold:

- Listing 4.12 A common top-p filtering variant


keep_mask = (cumsum - sorted_probas) < top_p
n_kept = keep_mask.sum().item()
print("Tokens kept:", n_kept)

This code now returns 3 as the number of tokens kept. This follows the definition of top-p
filtering that the smallest set of tokens is kept such that the cumulative probability mass is
at least p.

Let's illustrate this top filtering with a plot using the code below.

- Listing 4.13 Visualizing top-p filtering


plt.bar(
torch.arange(len(sorted_probas)), sorted_probas,
alpha=0.5, label="Sorted probabilities"

)
plt.step(

torch.arange(len(cumsum)), cumsum, where="mid",
color="darkorange", label="Cumulative sum"

)

#A
plt.axhline(

top_p, color="red", linestyle="--",
label=f"top_p = {top_p}"

)
plt.axvline(

n_kept - 0.5, color="gray", linestyle=":",
label=f"Top-p cutoff at {n_kept} tokens"

)

plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0, 1.05)
plt.show()

#A Highlight cutoff

The plot produced from executing the code in listing 4.13 results is shown in figure 4.15.

![image 57](<input (1)_images/imageFile57.png>)

- Figure 4.15 Top-p (nucleus) filtering. Tokens are sorted by probability, and the smallest subset whose
cumulative probability exceeds the threshold (p = 0.8) is kept for sampling.


The plot in figure 4.15 shows the top_p = 0.8 threshold value (dashed horizontal line) that
defines which tokens are kept. In this case, the cumulative sum of the first two tokens is
below the threshold, hence we exclude all other tokens to the right of the cut-off (vertical
dotted line).

To implement this threshold cutoff, we can use the following code, which first zeroes out
all values to the right side of the cutoff (vertical dashed line in figure 4.15) and restores the
original sorting order.

- Listing 4.14 Applying top-p filtering


kept_sorted = torch.where(
keep_mask, sorted_probas,
torch.zeros_like(sorted_probas)

)
filtered = torch.zeros_like(toy_probas).scatter(0, sorted_idx, kept_sorted)
print(filtered)

The resulting tensor looks as follows:

tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.4538, 0.0000, 0.0000, 0.0000, 0.0829,
0.2752])

We can see that all values except the ones at index positions 4, 8 and 9 are zeroed out.
This means that if we now use the multinomial function only those three tokens will be
considered. Finally, we renormalize these values so that they sum up to one again:

denom = torch.sum(filtered).clamp_min(1e-12)
renormalized = filtered / denom
print(renormalized)

The resulting, normalized tensor is:

tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.5589, 0.0000, 0.0000, 0.0000,

- 0.1021, 0.3390])


The goal of top-p filtering, which we implemented in this section, is to remove tokens with
low probabilities to avoid them from being sampled later. This helps reduce nonsensical
token responses in given contexts.

- 4.5.2 Adding a top-p filter to the text generation function


In the previous section, we walked through the top-p filtering steps using a simple toy
example with a vocabulary size 10 to be able to visualize the procedure in a bar plot. In this
section, we are adding the four main top-p filtering steps (steps 4.1-4.4 in figure 4.16) to
the existing text generation function.

![image 58](<input (1)_images/imageFile58.png>)

- Figure 4.16 Integrating top-p filtering with temperature scaling. After rescaling the next-token scores, top-p
filtering is applied between steps 3.3 and 3.4 to limit sampling to the most probable tokens.


Given our previous text generation function, as illustrated in figure 4.16, we add the top-p
filtering between the probability conversion and the sampling we implemented earlier in the
temperature scaling section.

For this, we will first put together the top-p filtering steps from the previous section into
a single, convenient function we can call.

- Listing 4.15 Top-p filtering function


def top_p_filter(probas, top_p):
if top_p is None or top_p >= 1.0:
return probas

sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True) #A
cumprobas = torch.cumsum(sorted_probas, dim=1) #B

prefix = cumprobas - sorted_probas #C
keep = prefix < top_p #C
keep[:, 0] = True #D

kept_sorted = torch.where( #E
keep, sorted_probas, #E
torch.zeros_like(sorted_probas) #E

)

- #F
filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
- #G
denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
return filtered / denom


- #A Step 4.1: Sort by descending probability
- #B Step 4.2: Cumulative sum
- #C Step 4.3.1: Keep tokens where prefix cumulative mass (before each token) is < top_p
- #D For top_p <= 0, only the highest-probability token is guaranteed to be kept as a fallback
- #E Step 4.3.2: Zero out beyond cutoff
- #F Step 4.3.3: Map back to original order
- #G Step 4.4: Renormalize to sum to 1


To briefly recap what's happening in top-p filtering, the top_p_filter function in listing

- 4.15 first sorts token probabilities and computes their cumulative sum. It then keeps
tokens whose prefix cumulative mass (the cumulative probability before each token) is
below the top_p threshold, which includes the token that first crosses the threshold. It
zeroes out the rest, maps the kept values back to their original order, and renormalizes the
remaining probabilities so they sum to 1 again.


Before we add this top_p_filter function to our text generation functions, let's give it a
try and see how it works with the previous temperature-scaling approach. First, we get the
logits:

with torch.inference_mode():

next_token_logits = model(input_token_ids)[:, -1]
print(next_token_logits.shape)

The code above prints the dimensions [1, 151936] for the next_token_logits tensor,
since we are now working with the real data and full vocabulary, which has 151,936 entries.

Next, we rescale the logits into probability scores and apply the temperature scaling with
a temperature of 0.35, similar to before:

torch.manual_seed(123)
probas_lowT = torch.softmax(

scale_logits_by_temperature(next_token_logits, 0.35), dim=-1

)
count_samples(probas_lowT, threshold=1, tokenizer=tokenizer)

This code is, so far, similar to what we used in the previous temperature scaling examples
and prints the following sampled outputs:

' __': 158x
' Berlin': 435x
' ____': 169x
' ______': 209x
' Munich': 3x
' Hamburg': 3x
' _____': 18x

Now, let's add the top-p filter and see how it changes the results:

torch.manual_seed(123)
probas_lowT = torch.softmax(

scale_logits_by_temperature(next_token_logits, 0.35), dim=-1

)
probas_lowT_filtered = top_p_filter(probas_lowT, top_p=0.8)
count_samples(probas_lowT_filtered, threshold=1, tokenizer=tokenizer)

With a top_p threshold of 0.8, which is a typical value, we get rid of the 20% lowest
probability tokens, and the sampled outputs now look as follows:

' Berlin': 534x
' ____': 217x
' ______': 249x

As we can see, we now either select the correct city (and remove " Munich" and "
Hamburg" as sampled options) or print the underscore token, which the model might use to
format the text as a quiz question with the ' ______' as a placeholder.

Now that we walked through the top-p filtering process using a simple toy example, let's
return to our math query and add the top-p filter to the
generate_text_temp_stream_cache function we coded earlier. The updated function, now
called generate_text_top_p_stream_cache, is shown in listing 4.16 below.

- Listing 4.16 Text generation with top-p filtering


@torch.inference_mode()
def generate_text_top_p_stream_cache(

model,
token_ids,
max_new_tokens,
eos_token_id=None,
temperature=0.,
top_p=None

):

model.eval()
cache = KVCache(n_layers=model.cfg["n_layers"])
model.reset_kv_cache()

out = model(token_ids, cache=cache)[:, -1] #A
for _ in range(max_new_tokens):

orig_device = token_ids.device

if temperature is None or temperature == 0.0:
next_token = torch.argmax(out, dim=-1, keepdim=True)

else:
logits = scale_logits_by_temperature(out, temperature) #B
probas = torch.softmax(logits, dim=-1) #C

probas = top_p_filter(probas, top_p) #D

next_token = torch.multinomial(probas.cpu(), num_samples=1) #E
next_token = next_token.to(orig_device)

if (eos_token_id is not None

and torch.all(next_token == eos_token_id)):
break

yield next_token
out = model(next_token, cache=cache)[:, -1]

- #A Step 3.1: Get logits
- #B Step 3.2: Apply temperature scaling on logits
- #C Step 3.3: Convert to probabilities
- #D (New) Step 4: Apply top-p filter to probabilities
- #E Step 3.4: Sample token according to probabilities


To conclude this section, let's plug this new text generation function into the
generate_text_stream_concat_flex, similar to what we have done with the temperature-
scaling version of the text generation function before:

torch.manual_seed(123)
response = generate_text_stream_concat_flex(

model, tokenizer, prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.5,
top_p=0.8,

)

The output is " \boxed{18}", which is still not correct. Note that everything we have done
up to this point is mainly for the purpose of being able to sample different outputs. In the
next section, we will use the generate_text_stream_concat_flex with our augmented
text generation function (generate_text_top_p_stream_cache) to sample different
outputs when implementing the self-consistency inference-time scaling technique.

###### TOP-K FILTERING

Top-k filtering is another way to limit the set of candidate next tokens during
sampling.

Instead of keeping all tokens whose cumulative probability stays below a
threshold (as in top-p), top-k keeps only the k most likely tokens based on the
model's logits.

After sorting the vocabulary by probability, everything past the first k entries is

removed.
The remaining k tokens are then renormalized and sampled from.
In short, top-k keeps a fixed number of the most likely tokens, whereas top-p

keeps a variable number of tokens depending on their cumulative mass.

Top-k is simpler to implement, and I covered it in my Build a Large Language
Model (From Scratch) book. Top-p sampling has become more popular recently.

###### EXERCISE 4.2: USE TEMPERATURE SCALING AND TOP-P FILTERING ON MATH-500

Modify the evaluate_math500_stream function in section 3.9 of chapter 3 to see if
adding temperature scaling and top p sampling changes the MATH-500 accuracy of
the base model. You can use a setting of 0.9 for both temperature scaling and top-p.

##### 4.6 Improving response accuracy with self-consistency

After all the previous work on coding temperature scaling and top-p filtering, we are now
ready to implement the second inference-time scaling technique in this chapter: self-
consistency sampling.

Conceptually, this section ties together several pieces from earlier chapters: we reuse
the generation code from chapter 2, the answer extraction logic from chapter 3, and the
sampling controls from this chapter to build a complete voting-based inference-time
pipeline.

The idea behind self-consistency sampling was formally introduced in the Google
Research paper Self-Consistency Improves Chain-of-Thought Reasoning in Language
Models (https://arxiv.org/abs/2203.11171). Despite the fancy name, it's essentially a form
of simple majority voting, where we use temperature scaling and top-p filtering to generate
multiple answers and then select the most frequent one, as shown in figure 4.17.

![image 59](<input (1)_images/imageFile59.png>)

- Figure 4.17 The self-consistency sampling method generates multiple responses from the LLM and selects
the most frequent answer, which improves answer accuracy through majority voting across these sampled
responses.


The self-consistency sampling technique shown in figure 4.17 counts as an inference-time
scaling technique, since we don't update the model itself and we expend more compute
resources to improve response accuracy (more on the accuracy later, after we implement
and try this technique).

The self-consistency code implementation, thanks to the
generate_text_stream_concat_flex function, is relatively straightforward, and the main
procedure can be summarized in three main steps as illustrated in figure 4.18.

![image 60](<input (1)_images/imageFile60.png>)

- Figure 4.18 The three main steps for implementing self-consistency sampling. First, we generate multiple
answers for the same prompt using a temperature greater than zero and top-p filtering to generate different
answers. Second, we extract the final boxed answer from each generated solution. Third, we select the most
frequently extracted answer as the final prediction.


To implement the three steps illustrated in figure 4.18, in listing 4.15 below, we are simply
calling generate_text_stream_concat_flex repeatedly to generate multiple answers and
reuse the extract_final_candidate function from chapter 3. The only new code is the
majority voting based on the answer frequency.

- Listing 4.17 Text generation with top-p filtering


from reasoning_from_scratch.ch03 import extract_final_candidate
from collections import Counter

def self_consistency_vote(
model, tokenizer, prompt, device,
num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
show_progress=True, show_long_answer=False, seed=None,

):

full_answers, short_answers = [], []

for i in range(num_samples): #A
if seed is not None:
torch.manual_seed(seed + i + 1)

answer = generate_text_stream_concat_flex(
model=model, tokenizer=tokenizer, prompt=prompt, device=device,
max_new_tokens=max_new_tokens, verbose=show_long_answer,
generate_func=generate_text_top_p_stream_cache,
temperature=temperature, top_p=top_p,

)

short = extract_final_candidate( #B

answer, fallback="number_then_full" #B
) #B
full_answers.append(answer)
short_answers.append(short)
if show_progress:

print(f"[Sample {i+1}/{num_samples}] → {short!r}")

counts = Counter(short_answers)
groups = {s: [] for s in counts}
for idx, s in enumerate(short_answers):

groups[s].append(idx)

mc = counts.most_common() #C
if not mc:

majority_winners, final_answer = [], None

else:
top_freq = mc[0][1]
majority_winners = [s for s, f in mc if f == top_freq]
final_answer = mc[0][0] if len(majority_winners) == 1 else None

return {
"full_answers": full_answers,
"short_answers": short_answers,
"counts": dict(counts),
"groups": groups,
"majority_winners": majority_winners,
"final_answer": final_answer,

}

- #A 1) Sample multiple answers
- #B 2) Extract the final (short) answer from each answer
- #C 3) Choose the most frequent final answer (self-consistency vote)


In short, the self-consistency method in listing 4.16 above

- 1. Samples multiple answers with temperature greater than 0 and top-p
- 2. Extracts the final boxed answer from each answer
- 3. Chooses the most frequent final answer


Note that in our implementation, we use a for-loop to generate these answers sequentially.
In practice, it is also common to generate the answers on different devices so that the
sampling can be parallelized.

Furthermore, in the code above, we set the random seed for each round individually:

if seed is not None:
torch.manual_seed(seed + i + 1)

Technically, this is not necessary and seeing the random seed once should be sufficient to
generate diverse samples. This explicit seeding is useful if we want to individually rerun
some of the rounds.

Let's try out this function in practice and see what we get:

results = self_consistency_vote(
model,
tokenizer,
prompt,
device=device,
num_samples=5,
temperature=0.8,
top_p=0.9,
max_new_tokens=2048,
seed=123,
show_progress=True,

)

The printed results are shown below:

- [Sample 1/5] → '83'
- [Sample 2/5] → '22'
- [Sample 3/5] → '54'
- [Sample 4/5] → '83'
- [Sample 5/5] → '61'


And the final answer, since 83 is the most frequent answer, is 83 (we can access this
number programmatically via results["final_answer"]). If you want to read the
explanation and full answer, you can access one of the correct solutions returning 83, for
example print(results["full_answers"][0]), which prints:

To find the value of x, let's solve the equation step by step.

- 1. **Given Equation:**
\[
\frac{1}{2} \times (3x - 9) = x + 37
\]


# ...

**Final Answer:**
\[
\boxed{83}
\]

Finally, with this self-consistency approach, we were able to generate the correct answer.
(Note that the results may vary when executing the code on an "mps" or "cuda" device.)

The self_consistency_vote function currently also does not handle ties, so if multiple
samples have the same frequency, it returns None as the final answer. We will implement a
scoring method in the next chapter to calculate the confidence of a given answer, which can
be used as a tie-breaker.

###### EXERCISE 4.3: USE SELF-CONSISTENCY SAMPLING ON MATH-500

Modify the evaluate_math500_stream function in section 3.9 of chapter 3 to
evaluate whether self-consistency sampling improves the MATH-500 accuracy of the
base model. Use a sample size of 3 and set both the temperature and top-p value to
0.9.

As part of this exercise, implement tie-breaking so that ties are resolved by the
first answer appearing in the sample list. For example, if the samples answers are
13, 15, 13, 15, 16, the selected answers should be 13.

Note that you don't have to modify the self_consistency_vote itself to
implement this tie-breaking, but you can work with the results dictionary returned
by the function to implement this simple tie-breaking rule.

###### EXERCISE 4.4: EARLY STOPPING IN SELF-CONSISTENCY SAMPLING

To improve computational efficiency, implement an early-stopping version of self-
consistency that ends sampling once more than half of the answers agree.

###### CHOOSING TEMPERATURE AND TOP-P SETTINGS

Since we already discussed temperature earlier in the chapter, the practical question
here is how much randomness to introduce for self-consistency sampling. The goal is
not to make the model as random as possible, but to generate answers that are
diverse enough for majority voting to help while still keeping most samples sensible.
In practice, temperature values around 0.5 to 0.9 and top-p values around 0.7 to 0.9
are reasonable starting points.

As a rule of thumb, if all (long) answers look nearly identical, it is a good idea to
gently increase the temperature or top-p value to encourage more diversity. If the
(long) answers start to look off or nonsensical, the settings are likely too aggressive
and should be reduced, usually by lowering the temperature first. Here, "long"
answer means the full answer before extracting the final boxed result. You can
inspect the long answers in the results dictionary returned by the
self_consistency_vote function or run the function with the
show_long_answer=True setting.

You may recall that the aforementioned paper title that described this technique was Self-
Consistency Improves Chain-of-Thought Reasoning in Language Models. Where or how
does the "chain-of-thought reasoning" aspect factor into all of this?

Chain-of-thought reasoning here simply refers to the chain-of-thought prompting we
introduced earlier in this chapter, when we modified the prompt via "\n\n Explain step
by step." so that the LLM generates longer responses.

So, let's combine chain-of-thought prompting with self-consistency sampling:

results = self_consistency_vote(
model,
tokenizer,
prompt + "\n\nExplain step by step.",
device=device,
num_samples=5,
temperature=0.8,
top_p=0.9,
max_new_tokens=2048,
seed=123,
show_progress=True,

)

In this case, all 5 answers are 83 (correct), likely because the question is relatively simple
for the LLM when using a chain-of-thought.

Instead, it would be more interesting to see how well the model performs on the entire
MATH-500 dataset from the previous chapter. The results from various experiments are
shown in table 4.1.

Table 4.1 MATH-500 task accuracy for different methods

| |Method|Model|Accuracy|Time|
|---|---|---|---|---|
|1|Baseline (chapter 3), greedy decoding|Base|15.2%|10.1 min|
|2|Baseline (chapter 3), greedy decoding|Reasoning|48.2%|182.1 min|
|3|Chain-of-thought prompting ("CoT")|Base|40.6%|84.5 min|
| | | | | |
|4|Temperature and top-p ("Top-p")|Base|17.8%|30.7 min|
|5|"Top-p" + Self-consistency (n=3)|Base|29.6%|97.6 min|
|6|"Top-p" + Self-consistency (n=5)|Base|27.8%|116.8 min|
|7|"Top-p" + Self-consistency (n=10)|Base|31.6%|300.4 min|
| | | | | |
|8|"Top-p" + "CoT"|Base|33.4%|129.2 min|
|9|Self-consistency (n=3) + "Top-p" + "CoT"|Base|42.2%|211.6 min|
|10|Self-consistency (n=5) + "Top-p" + "CoT"|Base|48.0%|452.9 min|
|11|Self-consistency (n=10) + "Top-p" + "CoT"|Base|52.0%|862.6 min|
|12|Self-consistency (n=3) + "Top-p" + "CoT"|Reasoning|55.2%|544.4 min|


For brevity, methods labeled "Top-p" in table 4.1 use both temperature scaling and top-p
sampling.

The accuracy values shown in table 4.1 were computed on all 500 samples in the MATH-
500 test set using a "cuda" GPU (DGX Spark). Let's go through the results one by one. The
n=3 abbreviation means that we used a sample size of 3 in self-consistency sampling.

Rows 1 and 2 show the results of the base and reasoning variants using the code from
chapter 3, meaning we use the text generation function without temperature scaling or top-
p filtering. This text function simply selects the token with the highest score at each step,
which is also known as greedy decoding. We can see that the reasoning variant has
approximately 3 times the accuracy but also substantially increased runtime, since it
generates more tokens.

Next, in row 3, we see the results for the chain-of-thought prompting approach via the
"\n\nExplain step by step." prompt modification. As we can see, this boosts the base
model's accuracy from approximately 15% to 40%.

Row 4 shows what happens when we add temperature-scaling and top-p sampling to the
base model. (All experiments involving temperature scaling and top-p sampling in table 4.1
used a setting of 0.9 for both.) As we can see, the accuracy over the base model row 1 is
only moderately improved from 15.2% to 17.8%. This is expected because temperature-
scaling and top-p scaling merely help us to control the sampling diversity but are not
inference-time scaling techniques themselves. We can also see that the runtime increased
from 10.1 min to 30.7 min. This is not due to the sampling code overhead, but rather
because the model now generates longer responses in some cases.

Rows 5-7 show the self-consistency scaling results. We can see that increasing the
number of samples from 3 to 10 further improves accuracy to 31.6%, but it also
significantly increases runtime. In this case, there is almost no accuracy advantage to using
10 instead of 3 samples.

Row 8 shows temperature-scaling and top-p sampling combined with chain-of-thought
prompting. In this case, the sampling makes the chain-of-thought results worse (33.4%
compared to the 40.6% in row 3). When combining chain-of-thought prompting with self-
consistency, we can see that accuracy improves to 52% with a sample size of 10, but this
also substantially increases runtime to a staggering 862.9 min.

Note that in practice, if you have access to multiple GPUs, the different samples in self-
consistency sampling can be computed in parallel rather than sequentially. This would still
use the same amount of compute, but it could be distributed and parallelized to generate
the results faster.

Lastly, in row 12, we can see that the reasoning variant benefits from self-consistency
sampling as well, taking the performance of the reasoning variant (row 2) from 48.2% to
55.2% accuracy. Again, this comes at an increased runtime.

The takeaway is that the results in table 4.1 nicely highlight the trade-off in inference-
time scaling, where we trade better accuracy for more compute.

Note that one major downside of the self-consistency sampling approach in this chapter
is that it relies on a final boxed answer that we can extract for majority voting. This method
is tricker to apply to problems that don't have numeric or short final answers.

In the next chapter, as illustrated in figure 4.19, we will implement a different and more
versatile inference-time scaling method called self-refinement, in which the model
iteratively improves its own answers.

![image 61](<input (1)_images/imageFile61.png>)

- Figure 4.19 Summary of this chapter's focus on inference-time techniques. Here, the text generation function
was extended with a voting-based method to improve answer accuracy. The next chapter introduces self-
refinement, in which the model iteratively improves its responses.


- 4.7 Summary


Reasoning abilities and answer accuracy can be improved without
retraining the model by increasing compute at inference time (inference-
time scaling).

This chapter focuses on two such techniques: chain-of-thought prompting
and self-consistency; a third method, self-refinement, which was briefly
described, will be covered in for the next chapter.

A flexible text generation wrapper
(generate_text_stream_concat_flex) that uses different sampling
strategies that can be plugged in without changing the surrounding code.

Next tokens are produced from logits via softmax

Temperature scaling changes logits to control the diversity of the
generated text.

Top-p (nucleus) sampling filters out low-probability tokens to reduce the
chance of generating nonsensical answers

Chain-of-thought prompting (like "Explain step by step." or similar) often
yields more accurate answers by encouraging the model to write out
intermediate reasoning, though it increases the number of generated
tokens and thus increases the runtime cost.

Self-consistency sampling generates multiple answers, extracts the final
boxed result from each, and selects the most frequent answer via
majority vote to improve the answer accuracy.

Experiments on the MATH-500 dataset show that combining chain-of-
thought prompting with self-consistency can substantially boost accuracy
compared to the baseline without sampling, at the cost of much longer
runtimes.

The central trade-off of inference-time scaling: higher accuracy in
exchange for more compute.

# 5 Inference-time scaling via self-refinement

This chapter covers

Scoring LLM answers with a simple rule-based scorer

Computing an LLM's own confidence in its answers

Coding a self-refinement loop where the LLM iteratively improves its answers

The previous chapter introduced the concept of inference-time scaling (inference scaling for
short), which improves the model response accuracy without further training the model. In
particular, the focus of the previous chapter was on self-consistency, where the model
generates multiple answers, and the final answer is chosen by majority vote.

As outlined in figure 5.1, this chapter moves beyond the simple majority voting for
inference scaling and covers another popular and useful inference-scaling technique, self-
refinement. Instead of generating multiple answers to choose from, self-refinement focuses
on iteratively refining a single answer to correct potential mistakes.

![image 62](<input (1)_images/imageFile62.png>)

- Figure 5.1 A mental model of the topics covered in this book. This chapter continues stage 3 and focuses on
inference-time techniques for improving reasoning without additional training. This chapter introduces self-
refinement, where the model iteratively critiques and improves its own answers.


###### 5.1 Scoring and iteratively improving model responses

As discussed in the previous chapter, inference scaling provides a way to trade additional
compute for better accuracy. We also covered two inference scaling techniques, chain-of-
thought prompting and self-consistency.

Chain-of-thought prompting, as illustrated in figure 5.2, modifies the prompt, for
example, by adding the phrase "Explain step by step.", which can trigger a base model
to write longer explanations which can in turn result in better answer accuracy. This method
is particularly useful for base models that don't naturally provide reasoning-like
explanations. Models trained as reasoning models usually don't benefit from this type of
inference scaling, since they already explain their answers.

![image 63](<input (1)_images/imageFile63.png>)

- Figure 5.2 Three inference-time methods to improve reasoning covered in this book. The first two methods
were covered in the previous chapter. This chapter covers the third method, self-refinement, where the model
iteratively improves its own answers.


Self-consistency, the second method shown in figure 5.2, lets the model produce several
answers in parallel. We then pick the final answer by taking a simple majority vote over
these candidates.

Even though self-consistency is quite simple, it often yields large gains in answer
accuracy, which is why it has become a common choice in LLM applications where accuracy
is a higher priority than latency. Recent examples include DeepSeekMath-V2 and Google's
Gemini 3 Deep Think mode (see appendix A for references).

One downside of self-consistency is that majority voting requires short answers that can
be compared.

In this chapter, we implement a more versatile technique, self-refinement, in which the
LLM learns to improve its own answers iteratively (method 3 in figure 5.2).

But before we implement the self-refinement technique, we will first implement scoring
functions that we will use to compare and rank different answers, as outlined in figure 5.3.

![image 64](<input (1)_images/imageFile64.png>)

- Figure 5.3 We build a simple rule-based score, compute token probabilities and log-probabilities, and then use
these scores as part of a self-refinement method where the model iteratively improves its own answers.


After loading the pre-trained LLM, as in previous chapters (step 1 in figure 5.3), we will
start this chapter with a simple rule-based scoring function to illustrate the concept of
scoring (step 2). Then, we will go over the concepts of token probabilities (step 3) and
token log-probabilities (step 4), which we will need to implement the logprob scoring
method (step 5) for our self-refinement loop (step 6).

We will primarily use the logprob scoring function (step 5 in figure 5.3) to track the
progress within the self-refinement loop. These scoring functions can also be used to break
ties in self-consistency or select the best response instead of relying on a majority vote.

The chapter overview in figure 5.3, leading up to self-refinement, looks relatively short
and straightforward. The topics of token probability and log-probability are somewhat
complex and will also be relevant in the next chapter, where we implement reinforcement
learning methods to train the LLM. So, this chapter will spend a significant portion on
explaining the concept of log-probability scoring.

###### 5.2 Loading a pre-trained model

As in the previous chapter, we begin by loading the model used throughout this chapter.

- Listing 5.1 Load tokenizer and base model


import torch

- from reasoning_from_scratch.ch02 import get_device
- from reasoning_from_scratch.ch03 import (
load_model_and_tokenizer


)

device = get_device()
device = torch.device("cpu") #A

model, tokenizer = load_model_and_tokenizer(
which_model="base",
device=device,
use_compile=False

)

#A Delete this line to run the code on a GPU (if supported by your machine)

As in previous chapters, the code in listing 5.1 loads the model and tokenizer used
throughout this chapter.

Note that the code above runs on the CPU by default to ensure results that are generally
more consistent with those shown in this chapter. While small numerical differences can still
occur on the CPU depending on the operating system and machine, these differences are
typically smaller and more predictable than those observed across different accelerator
backends.

Later sections (sections 5.4 and 5.5) use computations involving very small numbers
with many digits after the decimal point, which are more sensitive to device choice and low-
level implementation details. This is not an issue in practice, but such mismatches can be
confusing on a first read-through. For that reason, I recommend starting with the CPU
device and considering MPS or CUDA devices later.

Next, to ensure that the model is loaded correctly, let's use it together with the
temperature and top-p sampler code from the previous chapter on a MATH-500 prompt:

- Listing 5.2 Generating text with temperature scaling and top-p sampling


- from reasoning_from_scratch.ch03 import render_prompt
- from reasoning_from_scratch.ch04 import (
generate_text_stream_concat_flex,
generate_text_top_p_stream_cache


)

raw_prompt = (
"Half the value of $3x-9$ is $x+37$. "
"What is the value of $x$?"

)
prompt = render_prompt(raw_prompt)
prompt_cot = prompt + "\n\nExplain step by step."

torch.manual_seed(0)

- response_1 = generate_text_stream_concat_flex(
model, tokenizer, prompt_cot, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.9,
top_p=0.9


)

The model produces the following response:

The problem states that half the value of 3x−9 is equal to x+37. We need to find
the value of \(x\).

### Step 2: Translate the problem into an equation

... #A

### Final Answer

\[
\boxed{83}
\]

#A Response truncated to preserve space

Because we are using temperature scaling and top-p sampling, changing the random seed
produces a different response:

torch.manual_seed(3)

- response_2 = generate_text_stream_concat_flex(
model, tokenizer, prompt_cot, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.9,
top_p=0.9,


)

This time, the model responds as follows:

We start with the given equation:
\[
\frac{1}{2} \times (3x - 9) = x + 37
\]

... #A
Final Answer:
\[
\boxed{83}
\]

#A Response truncated to preserve space

In both cases, the model produces the correct final answer (83). The second response is
much shorter, which we can confirm by printing the number of characters or tokens in each
response:

print("Response 1 characters:", len(response_1))

- print("Response 1 tokens:", len(tokenizer.encode(response_1)))
print("\nResponse 2 characters:", len(response_2))
- print("Response 2 tokens:", len(tokenizer.encode(response_2)))


The result is:

- Response 1 characters: 1419

- Response 1 tokens: 534
- Response 2 characters: 533


- Response 2 tokens: 231


A shorter response is not necessarily better. If two responses reach the same correct
answer, judging which one is better is not straightforward. It often depends on human
preferences regarding clarity, usefulness, and the partial correctness of the intermediate
steps.

Scoring intermediate steps of a response remains an active research area (see appendix
A), and methods such as process reward models, which evaluate the reasoning itself, do
not always yield better outputs in practice.

If the qualitative value of two responses is comparable, one thing is certain: shorter
responses are cheaper because they require generating fewer tokens and are therefore
preferred.

##### 5.3 Scoring LLM responses with a rule-based score

In the previous section, the LLM generated two correct responses. In this section, we
develop a simple rule-based scoring function to compare them (figure 5.4).

![image 65](<input (1)_images/imageFile65.png>)

- Figure 5.4 This section implements a rule-based scorer to rank different answers generated by the pre-trained
LLM.


The rule-based scoring function assigns a score to each of two LLM responses (figure 5.5),
which allows us to rank them and select the better one. Here, "better" refers to format and
brevity, not correctness.

![image 66](<input (1)_images/imageFile66.png>)

- Figure 5.5 Two generated responses reach the same correct answer but differ in their explanations. A scorer
evaluates the responses and assigns a score to each response.


There are several ways to implement a scoring function (scorer) for evaluating responses,
as shown in figure 5.5. The scorer can be a heuristic, meaning a simple rule-based method.
It can also be another LLM that rates the answers, often referred to as LLM-as-a-judge (see
appendix F.5). Or it can rely on internal probability scores or likelihoods, which we will
explore later in this chapter.

In this section, we begin with a heuristic, that is, a rule-based scorer, which we
implement in listing 5.3. We begin with this heuristic version not because it is the most
sophisticated option, but because it gives us a simple baseline that makes the later
probability-based scorers easier to compare and reason about.

- Listing 5.3 A simple rule-based scorer


from reasoning_from_scratch.ch03 import extract_final_candidate
import math

def heuristic_score(
answer,
prompt=None, #A
brevity_bonus=500.0,
boxed_bonus=2.0,
extract_bonus=1.0,
fulltext_bonus=0.0,

):

score = 0.0

- #B
cand = extract_final_candidate(answer, fallback="none")
if cand:

score += boxed_bonus

- #C
else:

cand = extract_final_candidate(answer, fallback="number_only")
if cand:

score += extract_bonus
else:

cand = extract_final_candidate(
answer, fallback="number_then_full"

)
if cand:

score += fulltext_bonus

- #D
score += 1.5 * math.exp(-len(answer) / brevity_bonus)
return score


#A A placeholder that we ignore in this section
#B Reward answers that have a final boxed value
#C Give weaker rewards if answer doesn't have a boxed value
#D Add a brevity reward that decays with text length

This heuristic_score assigns a numerical score to an LLM answer based on how cleanly
the answer can be extracted and the length of the answer. For instance, we award a bonus
if the answer has a \boxed{} response (boxed_bonus). Otherwise, we give a smaller bonus
if it at least contains a number (extract_bonus). The brevity_bonus assigns points based
on how short the answer is.

The absolute values are not very important. What matters more is their relative scale.
Here, boxed_bonus=2.0 is intentionally larger than extract_bonus=1.0 so that a cleanly
boxed final answer is preferred over one where we can only extract a number. The brevity
term contributes at most 1.5 additional points, so it mainly helps break ties between
answers of similar quality rather than dominating the extraction bonuses. We also keep
fulltext_bonus=0.0 here so that arbitrary long-form answers are not rewarded unless we
can extract a clearer final candidate from them.

###### NOTE We develop a scorer and do not use the verifier from chapter 3 here, because we assume we don't know the true answer to the question. The verifier is used only for evaluation purposes when we evaluate the model on an existing test set.

The prompt argument is a placeholder that does not serve a purpose in the function itself,
but it will make our lives easier (and the code simpler) when we develop the self-
refinement function with swappable scorer plugins later in the chapter.

Although brevity_bonus=500.0 may appear high, note that it is used as an
exponentially decaying term via 1.5 * math.exp(-len(answer) / brevity_bonus). The
plot below illustrates its effect on the score.

- Listing 5.4 Plotting the brevity penalty curve


import matplotlib.pyplot as plt

def plot_brevity_curve(brevity_bonus, max_len=2048):
lengths = torch.arange(1, max_len)
scores = 1.5 * torch.exp(-lengths / brevity_bonus)

plt.figure(figsize=(4, 3))
plt.plot(lengths, scores)
plt.xlabel("Text length (number of characters)")
plt.ylabel("Score contribution")
plt.tight_layout()
#plt.savefig("brevity_curve.pdf")
plt.show()

plot_brevity_curve(500)

The resulting plot is shown below:

![image 67](<input (1)_images/imageFile67.png>)

- Figure 5.6 A simple rule-based length penalty used by our scorer. Longer explanations receive a smaller score
contribution.


The score bonus approaches 1.5 for shorter answers, while answers longer than 1,000
characters receive a bonus of 0.2 or less.

As a rough intuition, a few hundred characters correspond to a short paragraph or a
concise worked solution, whereas 1,000 characters or more usually means a fairly long
multi-sentence explanation. So this term mildly favors compact answers, but it does not
force the model to collapse everything into a one-line response.

For simplicity, the brevity bonus is computed using the number of characters rather than
tokens, which avoids passing the tokenizer to the scoring function. Using token counts
would also be reasonable (and may even be preferable).

We now apply the heuristic scorer to the first (longer) response from the previous
section:

- print(round(heuristic_score(response_1), 3))


The resulting score is 2.088. Next, let's try the second (shorter) response:

- print(round(heuristic_score(response_2), 3))


The computed score is 2.517, which means the second (shorter) response would be the
preferred answer.

This section introduced the basic idea of scoring methods. Later in the chapter, we
develop another scorer and return to the heuristic scorer when using it as part of the self-
refinement method.

###### EXERCISE 5.1: USING THE HEURISTIC SCORER AS A TIE-BREAKER IN SELF-CONSISTENCY

Extend the self-consistency implementation (self_consistency_vote) in the
previous chapter so that it can handle ties among the candidate answers. For
instance, when two or more answers receive the same number of votes, apply the
heuristic scorer (heuristic_score) from this section to the tied candidates and
select the one with the highest score. Tip: You don't need to modify the
self_consistency_vote function itself to apply the tie-breaking, but you can apply
the tie-breaking to the self_consistency_vote function's returned results
dictionary.

###### EXERCISE 5.2: USING THE HEURISTIC SCORER IN A BEST-OF-N SETUP

Modify the self-consistency implementation so that the final answer is chosen using
the heuristic scorer rather than majority voting. Generate N candidate answers for
each problem (where N=2 or higher), score each candidate with the heuristic scorer,
and select the one with the highest score as the final prediction. (If we use a scorer
instead of majority vote, the method is called Best-of-N in the literature as opposed
to self-consistency.)

Apply this method to a small subset of MATH-500 and compare the results to both
plain Best-of-N and the self-consistency tie-breaker from the previous exercise.

Note that the scorer includes several parameters chosen by intuition, which is why we refer
to it as a heuristic score. To optimize the extraction and brevity reward settings, we could
plug this scorer into the self-consistency or Best-of-N methods from exercises 5.1 and 5.2
and evaluate which settings yield higher accuracy on a benchmark dataset such as MATH-
500.

As a practical rule of thumb, increase the extraction-related bonuses if the scorer too
often prefers answers whose final result is hard to parse. Increase the brevity pressure if
the scorer keeps favoring long rambling answers that do not improve the final result. On
the other hand, if the scorer starts preferring overly short but incomplete answers, reduce
the brevity penalty or increase the reward for clearly extractable final answers. Thinking
about these trade-offs is often more useful than focusing on the exact numeric values in
isolation.

##### 5.4 Understanding token probability scores

In this section, we take the first step toward building a scorer based on the model's own
confidence (step 5 in figure 5.7), where confidence means model-assigned probability. The
idea is that, at each position, the model distributes probability mass over the possible next
tokens. If the tokens in a proposed answer consistently receive high probability, this
suggests that the answer is more compatible with the model's own internal preferences.
Conversely, if the model assigns very low probability to many of the answer tokens, that is
a sign that the model itself does not strongly support that answer. Specifically, in this
section, we start by computing the token probability scores for a proposed answer and use
them to estimate how likely the model considers that answer to be (step 3 in figure 5.7).

This may seem like a detour at first, but these probability and log-probability concepts
become important again in the next chapter, where they reappear inside the training
objective rather than only as an inference-time scoring signal.

![image 68](<input (1)_images/imageFile68.png>)

- Figure 5.7 In this section, we move from a simple rule-based scorer to token-level probabilities. These
probabilities form the basis for the logprob scoring method that we will use later in the self-refinement
approach.


The token probability scores computed in this section are also known as next-token
probabilities, per-token probabilities, sequence likelihoods, or loosely token-level likelihoods
in the literature. They represent the probability the model assigns to each possible next
token, expressed as a normalized (softmax) distribution over the vocabulary. Although this
may sound complicated at first, it relies on the same mechanism used for text generation,
which we covered in the previous chapter.

###### NOTE The term logprobis a common shorthand in the AI literature for log-probability.

For instance, in the previous chapter, we saw that at each step the model assigns a so-
called logit score to each token in the vocabulary and then selects the next token based on
these scores. This process, which we discussed in section 4.4 of chapter 4, is illustrated
again in figure 5.8 for reference.

![image 69](<input (1)_images/imageFile69.png>)

- Figure 5.8 How an LLM selects the next token. The model converts the input text into token IDs, computes a
score for every vocabulary token, and chooses the token with the highest score. The plot on the right shows
the logit values for a subset of the vocabulary, with the token for Berlin having the highest score.


In the previous chapter, we looked up the vocabulary entry corresponding to the highest
score (vocabulary index 19846, corresponding to the token "Berlin" in figure 5.8), to get
the next generated token.

Here, we revisit the logits with a different goal in mind. Instead of generating the next
token as shown in figure 5.8, we want to use the scores to quantify the model's confidence
in a particular answer. In other words, we are not generating anything here based on the
scores but are just inspecting the scores.

For example, consider two candidate answers we want to score:

- 1. "The capital of Germany is Berlin"
- 2. "The capital of Germany is Bridge"


The goal is to quantify how confident the model is in each answer. It is important to keep in
mind that model confidence does not automatically imply correctness. A model can assign
high probability to an answer simply because it fits the patterns it has learned well, even if
the answer is factually wrong. In practice, when this happens, we still need other tools such

- as verifiers, external tools or retrieval, or comparison-based methods like self-consistency
to detect and correct confidently wrong answers.


In figure 5.8, the input text "The capital of Germany is" is fed to the model, and the
next token with the highest score ("Berlin") is selected. Here, instead of selecting a
token, we compare the scores assigned to two candidate next tokens, "Berlin" and
"Bridge", as shown in figure 5.9. ("Bridge" is used as an alternative because it appears
within the vocabulary range shown in figure 5.8.)

![image 70](<input (1)_images/imageFile70.png>)

- Figure 5.9 How we look up the logit scores for specific tokens. After passing the input text through the model,
we obtain a logit value for every vocabulary token. We then convert the candidate tokens we want to score
into token IDs and read off their corresponding logit values from the distribution.


Rather than working with the raw logits shown in figure 5.9, we convert them to
probabilities. Probabilities are easier to interpret, comparable across inputs, and form the
basis for the logprob scoring method used later in the chapter.

As discussed in chapter 4, applying torch.softmax converts logits into probability
values. Figure 5.10 shows the resulting probability distribution.

![image 71](<input (1)_images/imageFile71.png>)

- Figure 5.10 Next-token scoring. The input text is converted into token IDs and fed to the LLM, which outputs
logits for the next token. After applying a softmax, these logits become probabilities, where tokens like
"Berlin" receive high probability and unlikely alternatives such as "Bridge" receive values near zero.


The token probabilities shown in figure 5.10 are computed in the same way as in the
previous chapter, using torch.softmax, as part of the multinomial sampling procedure
described in section 4.4.3. The difference here is that we do not sample from the
distribution. Instead, we simply look up the probability assigned to specific tokens.

In figure 5.10, only the bar for "Berlin" is visible, with a probability of 0.1695. The
probabilities for other tokens in the plotted range are near zero. The bars shown do not
sum to 1 because the full vocabulary contains 151,000 tokens, and the figure displays only
a small slice of the distribution (token indices 19,800–19,900).

This indicates that the model assigns higher confidence to "Berlin" than to "Bridge" as
the next token given the input text "The capital of Germany is".

In practice we usually want to compare complete answers rather than individual tokens.
This extension from single-token to sequence-level scoring is illustrated in figure 5.11.

![image 72](<input (1)_images/imageFile72.png>)

- Figure 5.11 Computing token probability scores for a given sequence. For each position, we feed the
preceding text into the model and read off the softmax probability of the next token. Multiplying these
conditional probabilities yields the joint probability of the full sequence.


As shown in figure 5.11, we compute the probability of each token given its preceding
tokens. This is the same procedure as in figure 5.10, except that we repeat it for every
token in the sequence. During ordinary text generation, the model also does this one step

- at a time as it goes along and then selects the next token.
Here, we reuse the same mechanism in a different way where instead of sampling new


tokens, we take a completed candidate answer and score each of its tokens one by one
under the model. We then multiply these probabilities to obtain the probability of the full
sequence, also called the joint probability.

###### JOINT PROBABILITY OF A TOKEN SEQUENCE

For those comfortable with mathematical notation, the joint probability of a token
sequence can be written compactly as a product of conditional probabilities. For a
sequence 𝑥1, 𝑥2, ..., 𝑥𝑇 and model weights W, this is:

![image 73](<input (1)_images/imageFile73.png>)

Expanded, this becomes:

![image 74](<input (1)_images/imageFile74.png>)

This matches exactly what we compute in figure 5.11. At each position we feed the
context into the model, obtain the probability of the next token, and multiply these
scores across the sequence.

Ideally, the probability assigned to the sequence "The capital of Germany is Berlin"
should be much higher than that of the nonsensical answer "The capital of Germany is
Bridge". Because the joint probability is obtained by multiplying many small values, the
resulting probabilities for both sequences in figure 5.11 are close to zero. We address this
issue in the next section.

The two answers in figure 5.11 are nearly identical, differing only in the final token
("Berlin" versus "Bridge"), which we use here for simplicity and illustration. The same
method can also be applied to sequences that differ more substantially, such as the
answers generated earlier in this chapter for the MATH-500 prompt ("Half the value of
$3x-9$ is $x+37$. What is the value of $x$?").

###### HOW THIS SCORING DIFFERS FROM GREEDY AND TEMPERATURE PLUS TOP-P SAMPLING

When we compute token probability scores, we are not generating text. The full
sequence already exists, and we simply query the model for the probability of each
next token given the fixed context as input. So, nothing about these probabilities
influences later inputs. It is a retrospective scoring procedure, not a generation step.

Generation methods like the generate_text_stream_cache (greedy sampling)
and generate_text_top_p_stream_cache functions that we coded in chapter 4,
behave very differently. Greedy sampling always selects the most likely next token,
while temperature sampling and top-p sampling draw from a reshaped probability
distribution. These procedures modify the distribution at each step and then commit
to a single token, which becomes the input for the next position.

An important consequence is that none of these sampling strategies, including
greedy sampling, is guaranteed to produce the sequence with the highest overall
probability under the model. A token that looks suboptimal at one step can lead to a
future step where the following tokens are much more likely. On the other hand, a
locally optimal choice may steer the model towards low-probability sequences later.

And even if we did find the globally highest-probability sequence under the model,
that still would not guarantee that it is the correct answer or the most useful answer
for the user. It would only mean that it is the sequence the model itself prefers most
under its learned distribution.

Temperature and top-p sampling add even more variability by flattening or
truncating the distribution before drawing samples, which may further disconnect the
generated text from the globally most likely sequence.

By scoring an existing answer, we avoid all these complications. We do not run a
sampling algorithm and the model does not choose any tokens. We only evaluate
how probable the given sequence would have been if it had already been written.

After so much conceptual explanation, let's now see the token probability calculation in
action:

- Listing 5.5 Calculating next token probabilities


@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device, show=True):

token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

logits = model(token_ids.unsqueeze(0)).squeeze(0) #A
all_probas = torch.softmax(logits, dim=-1) #A

#B
t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)

next_ids = token_ids[1:] #C

next_token_probas = all_probas[t_idx, next_ids] #D

prod_next_token_probas = torch.prod(next_token_probas) #E

if show:
print("Next-token probabilities:", next_token_probas)
print("Joint probability:", prod_next_token_probas)

else:
return next_token_probas, prod_next_token_probas

- #A Get logits and probabilities similar to text generation functions
- #B Select positions we score (here: all)
- #C Since we have the text, we know the true next tokens
- #D Get probabilities for each next token
- #E Likelihood of the sequence is the product of the probability scores


As shown above, the calc_next_token_probas function, which computes the next-token
probabilities as shown in figure 5.11, is relatively short and appears very simple at first
glance. There's a lot happening in this function to carry out the computation, as illustrated
by a simpler input text example and a tiny five-word vocabulary in figure 5.12.

![image 75](<input (1)_images/imageFile75.png>)

- Figure 5.12 Extracting next-token probabilities. After converting the input text into token IDs, the model
computes logits that are transformed into probabilities with a softmax function. Using index tensors for
positions and true next tokens, we then get the model's computed probability for each next token.


The calc_next_token_probas function, illustrated with a simple example in figure 5.12,
computes next-token probabilities in a few steps. First, it computes the logits in the same
way as the text-generation functions introduced earlier (for example, the out =
model(token_ids) line in generate_text_basic from chapter 2).

The resulting logits (step 2 in figure 5.12) have shape [sequence_length, vocab_size]
and are then converted into normalized probability distributions using torch.softmax (step

- 3), so that the probabilities at each position sum to 1.


To extract the probability assigned to the actual next token at each position, we
construct an index tensor t_idx for the positions we want to score and another tensor,
next_ids, containing the corresponding target tokens. For example, if the input text is
"The capital of Germany is Berlin", then t_idx refers to the positions corresponding
to "The capital of Germany is", and next_ids contains the tokens shifted by one
position: "capital of Germany is Berlin".

These target tokens are simply the input tokens shifted by one position, since an LLM is
trained to predict the next token in a sequence. For example, given the input "The capital
of Germany is Berlin", the model is asked at position 2 (the token for "capital") to
predict position 3 (the token for "of"). Using all_probas[t_idx, next_ids] (steps 4 and

- 5) retrieves exactly these probability values for each position in the sequence.


Finally, the function uses torch.prod to compute the sequence likelihood as the product
of the per-token probabilities (step 6).

Let's finally see this function in action:

torch.set_printoptions(precision=4, sci_mode=True)
calc_next_token_probas(

model, tokenizer, device=device,
prompt="The capital of Germany is Berlin"

)

The resulting output is:

Next-token probabilities: tensor([6.1512e-05, 4.6484e-01,
1.6724e-02, 7.3828e-01, 1.6895e-01], dtype=torch.bfloat16)
Joint probability: tensor(5.9372e-08, dtype=torch.bfloat16)

Next, let's try another text:

calc_next_token_probas(
model, tokenizer, device=device,
prompt="The capital of Germany is Bridge"

)

This results in:

Next-token probabilities: tensor([6.1512e-05, 4.6484e-01, 1.6724e-02,
7.3828e-01, 2.9802e-07], dtype=torch.bfloat16)
Joint probability: tensor(1.0481e-13, dtype=torch.bfloat16)

Analyzing the results above, the first response (ending in "Berlin") gave us the joint
probability (sequence likelihood) of 5.9372e-08, which is larger than the joint probability of
the second response, 1.0481e-13. (In decimal form, 5.9372e-08 is 0.000000059372, and
1.0481e-13 is 0.00000000000010481.)

These results make sense since the "Berlin" answer receives a higher sequence
likelihood than the "Bridge" answer, as expected. Both values are extremely small because
multiplying many probabilities less than one quickly produces numbers that approach zero.
This makes raw likelihoods difficult to work with in practice, especially for longer sequences
where underflow becomes a problem.

###### NOTE These scores reflect the model's internal likelihood, not a calibrated probability of correctness. In other words, they tell us how strongly the model prefers one answer over another under its own distribution, not whether the answer is actually true, correct, or useful. A high score means the answer fits the model's learned patterns well, not that it is guaranteed to be right.

In the next section, we will introduce a modification to this calculation, log-probabilities,
which avoid these numerical issues and give us a more stable way to score entire
sequences.

###### PROBABILITIES VERSUS LIKELIHOODS

You may have noticed that this section used both terms, "probability" and
"likelihood." Is there a difference? In the field of statistics, probabilities describe how
likely an event is before we observe any data and must sum to one across all
possible outcomes. Likelihoods, in contrast, measure how well a specific model
explains observed data and are viewed as a function of the model's weights or
parameters, not the data. In short, probabilities predict data given a model, while
likelihoods evaluate models given data. (For a more detailed example, please see my
article on probabilities versus likelihoods: https://sebastianraschka.com/faq/docs/
probability-vs-likelihood.html)

In LLMs, the next-token probability is technically a probability, not a likelihood,
because it comes from a normalized distribution over all possible next tokens that
sums to one. In other words, the values in next_token_probas are probabilities, not
likelihoods. They come directly from the model's softmax over the vocabulary for
each position, which is a normalized probability distribution.

The product, which is computed as the joint probability
(torch.prod(next_token_probas)), is the model's probability assigned to the
sequence. This quantity is often referred to as the sequence likelihood (especially
when viewed as a function of the model parameters).

###### 5.5 From token probability scores to log-probabilities

The token probabilities computed in the previous section can be used as a scoring function
to rank different responses. As we saw, the probability scores can be very small, especially
when multiplied to obtain the joint probability or sequence likelihood.

In this section, to avoid numerical stability issues that often arise when working with
such small values, we apply a logarithmic scaling (also called log scaling) to these
probabilities, as shown in the overview in figure 5.13.

![image 76](<input (1)_images/imageFile76.png>)

- Figure 5.13 Overview of how we move from token probabilities to token log-probabilities, which provide a
numerically more stable basis for log-probability scoring used later in self-refinement.


Note that this section is simply about applying a scaling transformation to the probabilities
in the previous section, and the overall goal is still to compute the token probabilities, or, in
this case, log-probabilities, as illustrated in figure 5.14.

For instance, consider this simple example computing the probabilities for a selection of
logits values:

torch.set_printoptions(precision=4, sci_mode=False)
logits = torch.linspace(-2, 2, steps=7)
probas = torch.softmax(logits, dim=-1)
print(probas)

The probability values are:

tensor([0.0090, 0.0175, 0.0341, 0.0665, 0.1295, 0.2522, 0.4912])

We can turn them into log-probability scores via the torch.log function:

print(torch.log(probas))

The log-probability values are:

tensor([-4.7109, -4.0442, -3.3776, -2.7109, -2.0442, -1.3776, -0.7109])

Here, torch.log applies the mathematical, natural logarithm, where 𝑙𝑜𝑔(0.0090) = -4.7109
and vice versa 𝑒-4.7109 = 0.0090.

Instead of chaining torch.log(torch.softmax(...)), PyTorch also has an optimized
torch.log_softmax function that combines these two operations:

log_probas = torch.log_softmax(logits, dim=-1)
print(log_probas)

Similar to before, this returns:

tensor([-4.7109, -4.0442, -3.3776, -2.7109, -2.0442, -1.3776, -0.7109])

Note that the log-scaling only changes the magnitude of the values, not the ordering, so a
sequence with a higher likelihood will always have a higher log-likelihood as well. To see
this visually, let's plot the logits, probabilities, and log-probabilities side by side:

- Listing 5.6 Plotting logits, softmax probabilities, and log-softmax values


plt.figure(figsize=(9, 4))

- #A

- plt.subplot(1, 3, 1)
plt.bar(range(len(logits)), logits, color="C0", alpha=0.7)
plt.title("Logits")
plt.xlabel("Token index")
plt.ylabel("Value")
plt.grid(alpha=0.3)

#B

- plt.subplot(1, 3, 2)
plt.bar(range(len(probas)), probas, color="C1", alpha=0.7)
plt.title("torch.softmax(logits)")
plt.xlabel("Token index")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(alpha=0.3)

#C

- plt.subplot(1, 3, 3)
plt.bar(range(len(log_probas)), log_probas, color="C2", alpha=0.7)
plt.title("torch.log_softmax(logits)")
plt.xlabel("Token index")
plt.ylabel("Log-probability")
plt.grid(alpha=0.3)




plt.tight_layout()
plt.savefig("logits_softmax_log_softmax.pdf")
plt.show()

- #A Plotting logits
- #B Plotting softmax probabilities
- #C Plotting log-softmax values


The resulting plot is shown in figure 5.14.

![image 77](<input (1)_images/imageFile77.png>)

- Figure 5.14 Comparison of logits, softmax probabilities, and log-probabilities for a simple example. The log-
probabilities preserve the ordering of the probabilities.


As we can see in the resulting plot (figure 5.14), the log-probabilities have the same sorting
order as the logits and probability values. A larger logit corresponds to a larger probability
and a less negative log-probability. On the other hand, smaller logits produce smaller
probabilities and more negative log-probabilities.

In practical terms, higher probabilities are better because they indicate the model
assigns more confidence to a token. For log-probabilities, values closer to zero are better,
since zero corresponds to a probability of one, while very negative values correspond to
extremely small probabilities. This makes it easy to compare tokens: the least negative log-
probability is the most likely one, and the most negative log-probability is the least likely.

While figure 5.14 illustrates the relationship between logits, probabilities, and log-
probabilities for a simpler toy example, we can apply it to the probability values for the
"The capital of Germany is" prompt example from earlier, as shown in figure 5.15.

![image 78](<input (1)_images/imageFile78.png>)

- Figure 5.15 How logits are converted to probabilities and log-probabilities for next-token scoring. The correct
next token ("Berlin") receives a high logit, which becomes a high probability and a less negative log-
probability, while unlikely candidates like "Bridge" map to very small probabilities and large negative log-
probabilities.


We use probabilities instead of logits because probabilities are normalized and easier to
interpret as confidence values. Earlier we also noted that log-probabilities offer an
additional benefit, since they allow for numerically more stable calculations than working
with raw probabilities.

The numerical stability comes from the fact that even very small probability scores map
to log-probabilities in a reasonable numeric range. The other reason is that multiplying
many probabilities quickly drives the result toward zero, while taking the log turns this
multiplication into addition, which avoids underflow and is much more stable for long
sequences.

Figure 5.16 shows how the joint log-probability (or sequence log-likelihood) is computed
for the two example texts we used earlier.

![image 79](<input (1)_images/imageFile79.png>)

- Figure 5.16 How token-level log-probabilities accumulate to form sequence log-probabilities. Each row shows
the log-probability of the next token given the preceding text. Summing these values results in the joint log-
probability of the full sequence


Earlier, when we computed the regular joint probability score, we saw that both sequences
had values close to zero. Now, with the joint log-probability, we can see in figure 5.16 that
even changing just the final token (for example, "Berlin" versus "Bridge") results in a
noticeably different total log-probability (-16.6250 versus -29.8750).

###### WORKING WITH LOG-PROBABILITIES

When we compute the joint probability of a sequence, we multiply many numbers
between 0 and 1:

![image 80](<input (1)_images/imageFile80.png>)

In the expanded form, we can write it as:

![image 81](<input (1)_images/imageFile81.png>)

These products quickly become extremely small, which is inconvenient and can
cause numerical underflow. In plain terms, when numbers become too tiny, the
computer may round them down to zero, and then we lose useful information. To
avoid this, we often work in log space. Taking the logarithm of the joint probability
gives

![image 82](<input (1)_images/imageFile82.png>)

Using the fact that the logarithm of a product is the sum of the logarithms, this
expands to

![image 83](<input (1)_images/imageFile83.png>)

In other words, the log-probability of a sequence is just the sum of the log-
probabilities of its individual tokens. This is both numerically more stable and easier
to work with. It also matches the values returned by torch.log_softmax, which
provides the log-probabilities directly.

Summing log-probabilities is therefore the standard approach in machine learning
and AI.

Reusing the code from the previous section on calculating the token probabilities (listing
5.6), it is straightforward to implement the token log-probability function as it only requires
two small changes: changing torch.softmax to torch.log_softmax and changing
torch.prod to torch.sum.

###### WARNING Using mismatched combinations, such as torch.softmax with torch.sum or torch.log_softmax with torch.prod, is technically possible but mathematically incorrect.

- Listing 5.7 Calculating next token log-probabilities


@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device, show=True):

token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

logits = model(token_ids.unsqueeze(0)).squeeze(0)

- #A
all_logprobas = torch.log_softmax(logits, dim=-1)

t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
next_ids = token_ids[1:]
next_token_logprobas = all_logprobas[t_idx, next_ids]

- #B
sum_next_token_logprobas = torch.sum(next_token_logprobas)


if show:
print("Next-token log-probabilities:", next_token_logprobas)
print("Joint log-probability:", sum_next_token_logprobas)

else:
return next_token_logprobas, sum_next_token_logprobas

- #A We now use log_softmax
- #B We replace the product with a sum


Let's now try the updated function on the example prompts from earlier:

calc_next_token_logprobas(
model, tokenizer, device=device,
prompt="The capital of Germany is Berlin"

)

The result is:

Next-token log-probabilities: tensor([-9.6875, -0.7695, -4.0938,

-0.3008, -1.7812], dtype=torch.bfloat16)

- Joint log-probability: tensor(-16.6250, dtype=torch.bfloat16)


And now the second sequence:

calc_next_token_logprobas(
model, tokenizer, device=device,
prompt="The capital of Germany is Bridge"

)

This returns:

Next-token log-probabilities: tensor([ -9.6875, -0.7695, -4.0938,

-0.3008, -15.0000], dtype=torch.bfloat16)
Joint log-probability: tensor(-29.8750, dtype=torch.bfloat16)

As we can see, the difference between the "Berlin" and the "Bridge" sequence is now
much more pronounced (-16.6250 and -29.8750), with the former scoring much higher
(better).

##### 5.6 Scoring model confidence with log-probabilities

The previous two sections explained the concepts of token probabilities and log-
probabilities in great detail. In this section, we add some slight modifications to the log-
probability computation to develop a log-probability-based scorer function analogous to the
heuristic scorer we developed at the beginning of this chapter, as illustrated in figure 5.17.

![image 84](<input (1)_images/imageFile84.png>)

- Figure 5.17 This section implements a logprob scorer, based on token log-probabilities, which we will use in
the self-refinement method later in this chapter.


The logprob scoring method we develop is very similar to the procedure we covered in the
previous section. We make two main modifications.

First, we exclude the prompt from the score calculation and only calculate the score for
the answer tokens. Second, we average (instead of summing) over the token log-
probabilities so that it is fairer to compare two different sequences of different lengths. The
updated calculation with these two modifications is shown in figure 5.18.

![image 85](<input (1)_images/imageFile85.png>)

- Figure 5.18 The modified logprob scoring procedure. The prompt tokens are excluded from the calculation,
and only the log-probabilities of the answer tokens are collected. These values are then averaged to obtain a
length-normalized score, which allows us to compare answers of different lengths fairly.


For illustration purposes, we can implement the two modifications shown in figure 5.18
using the calc_next_token_logprobas function from the previous section. For example,
suppose we have the following example prompt and answer:

example_prompt = "What is the capital of Germany?"
example_answer = " The capital of Germany is Berlin."

next_token_logprobas, sum_next_token_logprobas = calc_next_token_logprobas(
model, tokenizer, device=device,
prompt=example_prompt+example_answer,
show=False

)

print("Next-token logprobas:", next_token_logprobas)
print("Joint log-probability:", sum_next_token_logprobas)

This prints the following output:

Next-token logprobas: tensor([-0.4512, -0.3418, -8.3125, -0.3906,

- -3.8125, -3.0469, -1.1719, 0.0000, -0.0155, 0.0000,
- -0.0078, -0.0752, -0.1582], dtype=torch.bfloat16)


- Joint log-probability: tensor(-17.7500, dtype=torch.bfloat16)


(Note that the tensor contains scores for the prompt token as well, which is why the
numbers appear different from figure 5.18, but this will become more clear when you read
on.)

We can then calculate the number of answer tokens via the following code, which yields
7:

print(len(tokenizer.encode(example_answer)))

Then, to calculate the average log-probability over the answer tokens, as shown in figure
5.18, we need to average over those 7 answer tokens:

last_7 = next_token_logprobas[-7:]
print(last_7)
print(torch.mean(last_7))
This prints:
tensor([-1.1719, 0.0000, -0.0155, 0.0000, -0.0078, -0.0752, -0.1582],

dtype=torch.bfloat16)
tensor(-0.2041, dtype=torch.bfloat16)

We can see that the resulting average -0.2041 is similar to what's shown in figure 5.18.

We can combine the calc_next_token_logprobas code with this calculation more
conveniently into a new function, avg_logprob_answer, which is shown below:

- Listing 5.8 Average log-probability scoring for answer tokens


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):

prompt_ids = tokenizer.encode(prompt) #A
answer_ids = tokenizer.encode(answer) #A
full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

logits = model(full_ids.unsqueeze(0)).squeeze(0) #B
logprobs = torch.log_softmax(logits, dim=-1) #B

start = len(prompt_ids) - 1 #C
end = full_ids.shape[0] - 1 #C

- #D
t_idx = torch.arange(start, end, device=device)
next_tokens = full_ids[start + 1 : end + 1]
next_token_logps = logprobs[t_idx, next_tokens]
- #E
return torch.mean(next_token_logps)


#A Encode prompt and answer tokens separately to get the prompt length later
#B Same as in calc_next_token_logprobas before
#C Index range for positions corresponding to answer tokens
#D Same as before, except for using start and end
#E Average over the answer token scores

The avg_logprob_answer function is overall similar to the calc_next_token_logprobas
function from the previous section, except that we only calculate the logprobs for the
answer tokens, instead of the whole sequence including the prompt, and average over the
calculated logprob values instead of summing them.

Let's apply this new function to the prompt and answer from earlier in this section:

- score_1 = avg_logprob_answer(
model, tokenizer,
prompt="What is the capital of Germany?",
answer=" The capital of Germany is Berlin.",
device=device


)

- print(score_1)


This returns -0.2041 similar to before, which indicates that we implemented the function
correctly. We can now also apply this function to the nonsensical "Bridge" answer:

- score_2 = avg_logprob_answer(
model, tokenizer,
prompt="What is the capital of Germany?",
answer=" The capital of Germany is Bridge.",
device=device


)

- print(score_2)


The resulting score here is -3.8906, much lower than the "Berlin" score, as expected.

With the average logprob scoring function in place, we could now also calculate the
scores for the MATH-500 prompt (prompt_cot) we defined at the beginning of this chapter
and the respective responses we stored as response_1 and response_2, which is left as an
exercise for the reader.

We spend a lot of time in this chapter to go through the concepts of next-token
probability and log-probability scoring. One reason for this is that we can use it as a scorer
in the upcoming section on self-refinement. A second reason is that the concept of log-
probabilities will also be relevant when we implement the reinforcement learning with
verifiable rewards training in the upcoming chapter.

At the same time, it is important not to over-interpret logprob scoring as a universal fix.
Because it measures what the model itself strongly prefers, it can still favor confident
mistakes over correct but less strongly preferred answers. So in practice, logprob scoring is
best viewed as one useful scoring signal among several, not as a guaranteed tie-breaker.

###### EXERCISE 5.3: USING THE LOGPROB SCORER AS A TIE-BREAKER IN SELF-CONSISTENCY

Using the logprob scorer instead of the heuristic scorer in exercise 5.1, extend the
self-consistency implementation (self_consistency_vote) in the previous chapter
so that it can handle ties among the candidate answers. Then run the two
implementations on (a subset of) the MATH-500 dataset to see which tie-breaking
method performs better.

###### EXERCISE 5.4: USING THE LOGPROB SCORER IN A BEST-OF-N SETUP

Extend the self-consistency implementation so that the final answer is chosen using
the logprob scorer (avg_logprob_answer) rather than the heuristic scorer or
majority voting (similar to exercise 5.2). Then run the different implementations on
(a subset of) the MATH-500 dataset to see which tie-breaking method performs
better.

Tips: You don't need to modify the self_consistency_vote function itself to
apply the tie-breaking, but you can apply the tie-breaking to the
self_consistency_vote function's returned results dictionary, which is used
inside the evaluate_math500_stream function from chapter 3.

##### 5.7 Self-refinement through iterative feedback

Having introduced several methods for scoring LLM answers, we now get to the core
inference-scaling technique of this chapter, self-refinement (figure 5.19).

![image 86](<input (1)_images/imageFile86.png>)

- Figure 5.19 The final step in our workflow, where the logprob scorer developed earlier is used inside the self-
refinement method.


This section introduces self-refinement and walks through the process manually for
illustration. The next section, as shown in figure 5.19, then automates the self-refinement
loop and adds support for the scoring methods developed earlier in the chapter (the
heuristic scorer and the average logprob scorer).

Self-refinement is a technique in which the LLM analyzes and refines its own answers. As
shown in figure 5.20, the LLM starts with an initial answer to the prompt as in regular LLM
usage. Then, it critiques the answer and refines it.

![image 87](<input (1)_images/imageFile87.png>)

- Figure 5.20 The self-refinement process. The LLM first produces an initial answer to the prompt, then receives
a critique prompt that asks it to analyze its own response and produce a short critique with a plan to refine
the answer. In the final step, the model is given a refine prompt that contains the original question, its draft
answer, and the critique, and it generates a revised answer that incorporates the suggested improvements.


The self-refinement procedure in figure 5.20 may look complicated at first, but it's
essentially just a sequential application of the text generation function on different prompts
and inputs. To make it more clear, let's walk through it step by step with a concrete code
example.

###### NOTE For this code example, we don't use chain-of-thought prompting for illustration purposes. In practice, we can combine self-refinement with chain-of-thought prompting if we work with a base model.

We start with the base prompt and answer (steps 1 and 2 in figure 5.20), which is based on
the code from section 5.2 at the beginning of this chapter, Loading a pre-trained model:

- Listing 5.9 Base prompt and answer


raw_prompt = (
"Half the value of $3x-9$ is $x+37$. "
"What is the value of $x$?"

)
prompt = render_prompt(raw_prompt)

torch.manual_seed(123)
initial_response = generate_text_stream_concat_flex(

model, tokenizer, prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.7,
top_p=0.9,

)

The LLM responds with " \boxed{18}", which is an incorrect answer (the correct answer is
83).

Next, we have the LLM critique the answer. To do this we write a critique prompt, which
includes the original question (raw_prompt) and answer (draft), as shown in steps 3 and 4
in figure 5.20. In code, this looks like as follows:

- Listing 5.10 Critique prompt and refinement plan


def make_critique_prompt(raw_prompt, draft):

return (
"You are a meticulous reviewer. Identify logical errors, missing "
"steps, or arithmetic mistakes. If the answer seems correct, "
"say so briefly. Then propose a concise plan to fix issues.\n\n"
f"Question:\n{raw_prompt}\n\n"
f"Draft answer:\n{draft}\n\n"
"Write a short critique and bullet-point fix plan "
"(under ~120 words).\n"
"Critique:"

)

critique_prompt = make_critique_prompt(raw_prompt, initial_response)
torch.manual_seed(123)
critique = generate_text_stream_concat_flex(

model, tokenizer, critique_prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.7,
top_p=0.9,

)

The critique prompt above lets the LLM write an elaborate critique, which even contains the
correct answer itself:

The question seems to have a logical error in its setup. The statement "Half the
value of $3x-9$ is $x+37$" is incorrect because half of $3x-9$ should be $(3x-
9)/2$, not $3x-9$. The equation should be $\frac{1}{2}(3x-9) = x + 37$.

Fix Plan:

- 1. Correct the equation to $\frac{1}{2}(3x-9) = x + 37$.
- 2. Multiply both sides by 2 to eliminate the fraction: $3x - 9 = 2(x + 37)$.
- 3. Distribute the 2 on the right side: $3x - 9 = 2x + 74$.
- 4. Subtract $2x$ from both sides: $x - 9 = 74$.
- 5. Add 9 to both sides: $x = 83$.


Note that the critique itself can contain factual errors. For instance, it says "The question
itself is incomplete", which is not true, yet it still proceeds with a correct fix plan. This
is important because, in self-refinement, a critique can still be useful even if parts of it are
wrong, as long as it pushes the model toward a better revision overall.

Finally, we use this refinement prompt to revise the original answer (steps 5 and 6 in
figure 5.20):

- Listing 5.11 Answer refinement


def make_refine_prompt(raw_prompt, draft, critique):

return (
"Revise the answer using the critique. Keep it concise and "
"end with a final boxed result: \\boxed{ANSWER}\n\n"
f"Question:\n{raw_prompt}\n\n"
f"Previous answer:\n{draft}\n\n"
f"Critique:\n{critique}\n\n"
"Revised answer:"

)

refine_prompt = make_refine_prompt(raw_prompt, initial_response, critique)
torch.manual_seed(123)
revised_answer = generate_text_stream_concat_flex(

model, tokenizer, refine_prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.7,
top_p=0.9,

)

While the base model is not the best instruction-follower, the refinement prompt has the
LLM generate the correct answer:

...
\boxed{83}
Final result: The value of $x$ is \boxed{83}.

The procedure in this section illustrates a single refinement loop. In practice, it is not
uncommon to repeat this loop for multiple iterations. In the next section, we will code a
function to automate this process.

##### 5.8 Coding the self-refinement loop

This final section packages the manual self-refinement steps from the previous section into
a convenient function that simplifies using self-refinement and allows the process to be
repeated for a fixed number of iterations.

In addition, the function supports plugging in the scoring methods developed earlier in
the chapter (heuristic scoring and logprob scoring) to compute a score for each answer
(figure 5.21), which can be used to decide whether a refined answer should be accepted.

![image 88](<input (1)_images/imageFile88.png>)

- Figure 5.21 The self-refinement loop with optional scoring. The model first produces an initial answer, then
critiques it, and generates a revised answer based on the critique. Both answers can be evaluated with the
scoring functions from this chapter (for example, logprob scoring), and the revised answer is only accepted if
its score improves on the previous one.


In figure 5.21, the revised answer receives a higher logprob score (-0.377) than the initial
answer (-1.258), which makes it reasonable to accept the revision. In practice self-
refinement can also produce worse answers, especially when run for multiple iterations. The
scorer therefore provides a way to decide whether a revised answer represents an
improvement.

###### NOTE Scoring does not always improve the results. Whether and which scorer to use in self- refinement depends on the LLM and needs to be figured out via experimentation on a benchmark dataset like MATH-500.

###### The complete code, which combines the steps from the previous section and adds an iterations parameter and a scoring function (score_fn) option, is shown in listing 5.12 below:

- Listing 5.12 Self-refinement with multiple iterations and scoring support


def self_refinement_loop(
model,
tokenizer,
raw_prompt,
device,
iterations=2,
max_response_tokens=2048,
max_critique_tokens=256,
score_fn=None,
prompt_renderer=render_prompt,
prompt_suffix="",
verbose=False,
temperature=0.7,
top_p=0.9,

):

steps = []

- #A
prompt = prompt_renderer(raw_prompt) + prompt_suffix
current_full = generate_text_stream_concat_flex(

model=model,
tokenizer=tokenizer,
prompt=prompt,
device=device,
max_new_tokens=max_response_tokens,
verbose=False,
generate_func=generate_text_top_p_stream_cache,
temperature=temperature,
top_p=top_p,

)

current_extracted = extract_final_candidate(
current_full, fallback="number_then_full"

)
if score_fn:

current_score = score_fn(answer=current_full, prompt=prompt)
else:

current_score = 0.0

- #B
for it in range(iterations):


draft_before_full = current_full

draft_before_extracted = current_extracted
score_before = current_score

#C
critique_prompt = make_critique_prompt(

raw_prompt, draft_before_full

)
critique_full = generate_text_stream_concat_flex(

model=model,
tokenizer=tokenizer,
prompt=critique_prompt,
device=device,
max_new_tokens=max_critique_tokens,
verbose=False,
generate_func=generate_text_top_p_stream_cache,
temperature=temperature,
top_p=top_p,

)

#D
refine_prompt = make_refine_prompt(

raw_prompt, draft_before_full, critique_full

)
revised_full = generate_text_stream_concat_flex(

model=model,
tokenizer=tokenizer,
prompt=refine_prompt,
device=device,
max_new_tokens=max_response_tokens,
verbose=False,
generate_func=generate_text_top_p_stream_cache,
temperature=temperature,
top_p=top_p,

)

revised_extracted = extract_final_candidate(
revised_full, fallback="number_then_full"

)
if score_fn:

revised_score = score_fn(

answer=revised_full, prompt=prompt
)

else:
revised_score = 0.0

- #E
step = {

"iteration": it + 1,
"draft_full": draft_before_full,
"draft_extracted": draft_before_extracted,
"critique": critique_full,
"revised_full": revised_full,
"revised_extracted": revised_extracted,
"score_before": score_before,
"score_after": revised_score,

}
steps.append(step)

if verbose:

print(
f"[Refinement {it+1}/{iterations}]"
f"\nCurrent: {draft_before_extracted}"
f"\nRevised: {revised_extracted}"
f"\nScore before: {score_before:.3f}"
f"\nScore after: {revised_score:.3f}"
f"\n{'=' * 25}\n"

)

- #F
if revised_score >= current_score:


current_full = revised_full
current_extracted = revised_extracted
current_score = revised_score

return {
"final_full": current_full,
"final_extracted": current_extracted,
"steps": steps,

}

- #A Initial response (draft)
- #B Run for one or more iterations
- #C Critique the response
- #D Refine the response
- #E Log the results
- #F Accept revised response if it's not worse


The self_refinement_loop function runs the self-refinement procedure from the previous
section for one or more iterations (for it in range(iterations)). At the end of each
iteration, it compares the revised score to the current score (revised_score >=
current_score) and keeps the revised answer only if the score is equal or higher.

Using a scoring function is optional. When score_fn=None (the default), the score is
always set to 0.0. Since 0.0 >= 0.0 evaluates to True, the most recent answer is always
accepted when running multiple refinement iterations.

Next, we run the self-refinement loop using the average logprob scorer,
avg_logprob_answer. The avg_logprob_answer function from listing 5.8 (section 5.6)
takes several arguments (model, tokenizer, prompt, answer, and device). As shown in
the previous listing, the scoring function is called with only two arguments:

# ...

if score_fn:
revised_score = score_fn(

answer=revised_full, prompt=prompt
)

# ...

To make avg_logprob_answer compatible with this score_fn call, we use the partial
function from Python’s built-in functools module to pre-specify the remaining arguments:

- Listing 5.13 Creating an average log-probability scorer


from functools import partial

avg_logprob_score = partial(
avg_logprob_answer,
model=model,
tokenizer=tokenizer,
device=device

)

Now, we can use the avg_logprob_score function in the self-refinement loop:

torch.manual_seed(1)

results_logprob = self_refinement_loop(
model=model,
tokenizer=tokenizer,
raw_prompt=raw_prompt,
device=device,
iterations=2,
max_response_tokens=2048,
max_critique_tokens=256,
score_fn=avg_logprob_score,
verbose=True,
temperature=0.7,
top_p=0.9,

)

The output is shown below:

- [Refinement 1/2]
Current: 10
Revised: 83
Score before: -0.855

- Score after: -0.226

=========================
[Refinement 2/2]
Current: 83
Revised: 83
Score before: -0.226

- Score after: -1.320




=========================

Looking at the results above, we see that the model corrected the initially incorrect answer
(10 to 83) in the first iteration, and the score improved from -0.855 to -0.226. In the
second iteration, the scores become worse, but that's okay since we already have the
correct answer.

We can also access the extracted number of the best-scoring answer via
results_logprob["final_extracted"], which in this case returns 83. If you are
interested in the full answer, use results_logprob["final_full"]. To read through the
critiques and longer answers, you can print the results_logprob dictionary, which
contains the detailed results for all iterations.

###### EXERCISE 5.5: USING THE HEURISTIC SCORE FOR SELF-REFINEMENT

Run the self_refinement_loop using the heuristic_score function, which we
defined in section 5.3

The previous example showed that self-refinement can help the model with generating the
correct answer. To get a better idea of how useful this self-refinement method really is, I
ran this method on the MATH-500 from chapter 3 and summarized the results in table 5.1.

Table 5.1 MATH-500 task accuracy for different self-refinement methods

| |Method|Scoring|Iterations|Model|Accuracy|Time|
|---|---|---|---|---|---|---|
|1|Baseline (chapter<br>3)|-|-|Base|15.2%|10.1<br>min|
|2|Self-refinement|None|1|Base|25.0%|84.8<br>min|
|3|Self-refinement|None|2|Base|22.0%|165.4<br>min|
|4|Self-refinement|Heuristic|1|Base|21.6%|84.7<br>min|
|5|Self-refinement|Heuristic|2|Base|20.8%|151.4<br>min|
|6|Self-refinement|Avg. logprob|1|Base|21.4%|85.3<br>min|
|7|Self-refinement|Avg. logprob|2|Base|22.0%|165.3<br>min|
| | | | | | | |
|8|Baseline (chapter<br>3)|-|-|Reasoning|48.2%|182.1<br>min|
|9|Self-refinement|None|1|Reasoning|56.6%|498.8<br>min|
|10|Self-refinement|Heuristic|1|Reasoning|57.8%|498.6<br>min|
|11|Self-refinement|Avg. logprob|1|Reasoning|48.4%|499.7<br>min|


The accuracy values shown in table 5.1 were computed on all 500 samples in the MATH-500
test set using a "cuda" GPU (DGX Spark).

As we can see in table 5.1, self-refinement improves the performance over the base
model (rows 1-7). The improvement is only very moderate, with the best accuracy being
achieved when no scoring is used in self-refinement. This means that both the heuristic
score and average logprob score can sometimes lead to incorrect answers being accepted
over the initial correct answer.

Note that when adding "Explain step by step." chain-of-thought for both the base
model and the self-refinement, the model failed to improve over the base model (results
not shown).

Looking at the reasoning model results (rows 8-11), we can see that the combination of
self-refinement and heuristic scoring improved the answer accuracy by almost 10%. (As
explained in chapter 3, the "reasoning" model can be used by changing
which_model="base" to which_model="reasoning" in the load_model_and_tokenizer in
section 5.2.)

In both cases, it seems that the average logprob scoring results in worse performance
than no scorer or the heuristic scorer. This is likely because the average logprob score is
more closely related to how natural or expected an answer looks under the model than to
whether the answer is actually correct. As a result, it can favor answers that are fluent,
familiar, or syntactically clean even when they are semantically wrong. Or, in other words,
the logprob criterion can unintentionally select confident mistakes, whereas the heuristic
score is more focused on the format and structure of the answer.

At this point, it seems we spent a lot of time discussing the average logprob scoring
concept. Logprob scoring is a fundamental concept when working with LLMs, and it will
come handy in the upcoming chapters when we implement the reinforcement learning
training procedure.

Comparing these results in table 5.1 with those from the previous chapter (table 4.1),
we also observe that self-refinement is less effective than self-consistency for this model on
this math task. Self-consistency remains widely used because it works well in practice,
despite its simplicity.

Another method we have not explored in this chapter is self-refinement with an external
model in an LLM-as-a-judge setup similar to what's described in appendix F in section F.5.
For instance, instead of using the heuristic score or average logprob, we use a second LLM
to compute a score and to write the critique.

In November 2025, with DeepSeekMath-V2, the DeepSeek team demonstrated that self-
refinement with a second LLM can be very successful, leading to gold-level performance in
several math competitions. Specifically, the DeepSeek team proposed a new method in
which they trained a second LLM to be a good critique model and further trained their base
model to become a better math problem solver in the context of self-refinement (in
contrast, in this chapter, we applied self-refinement without any additional training).

Speaking of inference and training, this chapter concludes the inference scaling without
additional training (figure 5.22). In the next chapter, we begin covering training techniques
that update the model weights to become better at reasoning tasks.

![image 89](<input (1)_images/imageFile89.png>)

- Figure 5.22 Overview of the book's progression from basic LLM usage to inference-time reasoning methods
and finally to training-based techniques. This chapter concludes the inference-scaling methods without
additional training, and the next chapters introduce approaches that update model weights to further improve
reasoning performance.


- 5.9 Summary


Self-refinement extends the inference-time scaling ideas from the
previous chapter by iteratively critiquing and improving a single answer
instead of relying on multiple independent samples as in self-consistency.

A simple rule-based scoring function ranks model outputs by rewarding
extractable final answers and shorter, more economical completions.

Next-token scoring quantifies model confidence by converting logits into
normalized probabilities and combining these into a sequence-level
likelihood.

Log-probabilities replace raw probabilities to avoid numerical underflow
and to turn products over many tokens into stable sums and averages.
These scoring functions are more generally useful beyond self-refinement,
for example, for breaking ties in self-consistency or implementing Best-
of-N selection strategies.

The self-refinement procedure consists of three stages: generating an
initial draft, producing a short critique and fix plan, and generating a
revised answer.

A reusable refinement function automates the self-refinement workflow
with multiple iterations (refinement rounds), and it can use a score-based
acceptance to keep only revisions that do not degrade a computed
answer score using one of the scoring functions we developed earlier.

# 6 Training reasoning models with reinforcement learning

This chapter covers

The difference between reinforcement learning with human feedback (RLHF) and
reinforcement learning with verifiable rewards (RLVR)

Training reasoning LLMs as a reinforcement learning problem with task-correctness
rewards

Sampling multiple responses per prompt to compute group-relative learning signals

Updating the LLM weights using group-based policy optimization for improved
reasoning

Reasoning performance and answer accuracy can be improved by both increasing the
inference compute budget and by specific model training methods. This chapter, as shown
in figure 6.1, focuses on reinforcement learning, which is the most commonly used training
method for reasoning models.

![image 90](<input (1)_images/imageFile90.png>)

- Figure 6.1 A mental model of the topics covered in this book. This chapter focuses on techniques that improve
reasoning with additional training (stage 4). Specifically, this chapter covers reinforcement learning.


The next section provides a general introduction to reinforcement learning in the context of
LLMs before discussing the two common reinforcement learning approaches used for LLMs.

###### 6.1 Introduction to reinforcement learning for LLMs

Inference-time scaling and training-time scaling are two distinct approaches for improving
the reasoning performance of large language models, as illustrated in figure 6.2. Inference-
time scaling increases accuracy by spending more computation per generated answer,
whereas training-time scaling improves accuracy by investing additional computation during
training. This chapter focuses on training-time scaling.

![image 91](<input (1)_images/imageFile91.png>)

- Figure 6.2 Conceptual comparison of inference-time scaling and training-time scaling. Increasing compute at
inference improves accuracy by spending more resources per answer generation, while increasing compute
during training improves accuracy by investing more resources upfront.


While figure 6.2 presents inference-time scaling and training-time scaling as separate
concepts, in practice, they can be combined. For example, after improving a model's
reasoning capabilities through reinforcement learning (RL), which is the core focus of this
chapter, inference-time scaling techniques such as those introduced in chapters 4 and 5 can
be applied to further boost performance.

RL is focused on how models learn from sequences of actions and their outcomes, with
classic examples such as agents trained to play Chess, Go, or video games.

While RL for LLMs builds on this literature, it differs in important ways and often
resembles familiar LLM training techniques. Since this book focuses on language models,
we do not cover RL in general but instead focus on how RL is applied in LLM training.

So what does RL mean in practice for LLMs? At a high level, RL is typically applied as a
post-training stage on top of a pre-trained language model, sometimes after instruction
fine-tuning. This setup is illustrated in figure 6.3.

![image 92](<input (1)_images/imageFile92.png>)

- Figure 6.3 Common training stages for LLMs. The ordering of the reasoning training and preference tuning
stages can vary, and some pipelines interleave reasoning and preference tuning rather than strictly
sequencing them.


There are two common RL stages for LLMs, reasoning training and preference tuning. As
shown in figure 6.3, RL is usually applied after instruction fine-tuning, and preference
tuning often follows reasoning training. As demonstrated by the DeepSeek team in their
DeepSeek-R1 work, though, reasoning-focused RL can also be applied directly to the pre-
trained base model, skipping both instruction fine-tuning and preference tuning.

The resulting model is generally weaker in terms of reasoning accuracy than one that
goes through the full training pipeline, but it still exhibits clear and robust reasoning
behavior. More importantly, applying reasoning training directly to the base model avoids
mixing the effects of multiple training stages, which makes it easier to attribute any
observed improvements specifically to the reasoning training itself.

Before getting further into the details of how RL is implemented for LLMs, it is useful to
briefly compare RL with pre-training at a conceptual level. During pre-training, an LLM is
trained to predict the next token in large amounts of text, and many of the model's core
abilities and emergent behaviors (for example, answering knowledge-based questions and
following simple instructions) already develop at this stage.

Note that pre-training is the main focus of my earlier book, Build a Large Language
Model (From Scratch), but familiarity with that material is not required to follow this
chapter or the rest of this book.

Applying RL on top of a pre-trained model is attractive because it lets us optimize whole
outputs, such as answer correctness or preferences, rather than individual tokens and next-
token prediction. In this sense, pre-training mainly builds knowledge, while RL shapes how
the model uses that knowledge, including its reasoning behavior.

As shown in Figure 6.4, the next two subsections give a high-level overview of how RL is
used in LLM training and set the context for the method used in this chapter.

![image 93](<input (1)_images/imageFile93.png>)

- Figure 6.4 Roadmap of this chapter. After a brief introduction to reinforcement learning (RL) for LLMs in this
section, we discuss the difference between two RL stages, RLHF and RLVR, in the next section. Then, we
focus on implementing RLVR using the GRPO algorithm in the remainder of this chapter.


- 6.1.1 The original reinforcement learning pipeline with human feedback (RLHF)


In 2022, the InstructGPT paper (https://arxiv.org/abs/2203.02155) introduced
reinforcement learning with human feedback (RLHF), a training approach that uses human
preference labels to modify model behavior.

In fact, RLHF was a key ingredient in transforming GPT-3 into the original model used in
ChatGPT, which was arguably what made LLMs broadly popular in 2022.

At a high level, RLHF is the most popular method to implement the preference tuning
stage discussed in the previous section. Before RLHF, LLMs were primarily trained through
pre-training and, in some cases, supervised fine-tuning, which are both based on next-
token prediction objectives.

RLHF moved beyond this token-level optimization and optimizes models on the whole
outputs (in this case, RLHF is optimizing for how humans rank and evaluate LLM outputs).

To make this more concrete, consider the following example. Suppose the prompt is: "I
am looking to buy a laptop for programming and everyday use. What should I consider?"

The model might generate two candidate responses:

- Response A: "You should consider the CPU, RAM, storage, GPU, screen
resolution, battery life, keyboard quality, port selection, thermal design,
and price."

- Response B: "For programming and everyday use, focus on a fast CPU, at
least 16 GB of RAM, and an SSD with at least 256 GB storage. A
comfortable keyboard and good battery life also matter. A GPU may only
matter if you plan to train small LLMs locally."


Both responses mention relevant points, but a human data annotator might prefer response
B because it is more concrete and specific to the use case stated in the prompt.

In RLHF, such pairwise preferences are collected across many prompts and used to train
the model to favor responses that humans consistently rank higher.

As shown in figure 6.5, RLHF has two main steps: (1) training a reward model (which is
itself an LLM), and (2) using that reward model to score the target LLM outputs and fine-
tune it.

![image 94](<input (1)_images/imageFile94.png>)

- Figure 6.5 Two-stage overview of reinforcement learning with human feedback (RLHF). First, a reward model
is trained on human-ranked responses to assign a preference score to each. Second, the LLM is updated using
these reward scores within an RL objective to encourage preferred responses and discourage less desirable
ones.


The first step in RLHF is training a reward model, as illustrated in the top subpanel of figure
6.5. Here, we have the LLM generate multiple responses per prompt (for example, using
the temperature scaling and top-p filtering approach explained in chapter 4), and ask
human annotators to rank the answers from best to worst based on their human
preference.

These human preferences are converted into training targets for the reward model using
a statistical preference model that maps pairwise comparisons into relative quality scores.
The reward model, which is usually a separate model initialized from a pretrained LLM, is
trained to output a single-number (scalar) reward score to each response that reflects its
perceived quality.

The motivation is that the reward model can automatically score new model outputs,
which eliminates the need for human annotation at every training step.

The second step in RLHF trains the LLM on the reward scores the reward model assigns
to each of the LLMs' answers (for example, a bad answer may receive a -2, and a good
answer may receive a +2).

We discuss RLHF here because it can be viewed as a precursor to reasoning-focused RL
methods such as reinforcement learning with verifiable rewards (RLVR).

While the source of the training signal for RLHF and RLVR differs, the underlying
structure of the pipeline (generating responses, scoring them, and updating the model via
RL) is closely related.

- 6.1.2 From human feedback to verifiable rewards (RLVR)


RLHF can be fairly involved because it requires training an additional reward model, which
is often a large LLM that is (also) expensive to train.

Reinforcement Learning with Verifiable Rewards (RLVR) simplifies this setup by replacing
the LLM reward model with verifiable rewards. Verifiable rewards are computed
deterministically and without human annotation. For example, the math verifier introduced
in chapter 3 is a verifiable reward generator for math problems.

For example, given a math problem (and a correct solution), a math verifier
automatically checks whether a generated solution's final answer matches the ground truth
and assigns a corresponding reward (1 for correct, and 0 for incorrect responses).

As a result, the two-step RLHF pipeline collapses into a single training loop in RLVR,
which resembles step 2 in RLHF: the model generates responses, the verifier assigns
rewards, and these rewards are used directly to update the model.

This simplified RLVR training procedure is outlined in figure 6.6.

![image 95](<input (1)_images/imageFile95.png>)

- Figure 6.6 Overview of reinforcement learning with verifiable rewards (RLVR). The LLM generates a response
that is evaluated by a deterministic verifier, for example the math verifier from chapter 3, which assigns a
correctness label used as a reward signal within an RL objective to update the model.


The popularity of RLVR can largely be traced to the success of DeepSeek-R1 in 2025
(https://arxiv.org/abs/2501.12948), which demonstrated that strong reasoning
performance can be achieved without relying on human preference data or a learned
reward model.

DeepSeek-R1 trained reasoning behavior using automatically verifiable rewards,
including correctness checks for math problems (similar to our verifier in chapter 3) and
code compilation and execution for code-related tasks.

Code execution is outside the scope of this book, because it would require training the
LLM in a secure execution environment. In practice, this means compiling and running
model-generated code inside an isolated sandbox so it cannot access the host system,
private files, or external services in unsafe ways, which makes the setup considerably more
complex than solving math problems.

Either way, the idea behind verifying math tasks (checking whether an answer is correct,
as a binary label) and code compilation (checking whether code compiles, also as a binary
label) is similar, and the RLVR training algorithm would be identical.

To summarize, RLVR offers several practical advantages over RLHF. It removes the need
to train and maintain a separate reward model, which is often comparable in size and cost
to the base LLM. RLVR requires access to reliable verification signals, which restricts it to
certain domains, for example, math and code.)

Verifiable rewards are also deterministic and reproducible, meaning they produce the
same result every time for the same output, which avoids the noise and inconsistency that
arise in human preference annotations.

Finally, RLVR scales naturally to large training sets, since, given a reference solution,
rewards can be computed automatically for any number of model-generated answers
without additional human labeling effort.

6.2 Reinforcement learning with verifiable rewards walkthrough
using GRPO

Now that we have introduced the big picture and seen how RL fits within the overall
development cycle of LLMs, we get to the concrete implementation. Specifically, we
implement RLVR to train a reasoning model, as illustrated in the chapter roadmap in figure

- 6.7. This method is similar to DeepSeek-R1-Zero, but we apply it at a much smaller scale,
on a smaller model with less data, since a comparable full training run would otherwise
require several hundred thousand dollars in GPU compute.


![image 96](<input (1)_images/imageFile96.png>)

- Figure 6.7 After introducing the two main reinforcement learning approaches for LLMs, RLHF and RLVR, the
remaining sections focus on implementing RLVR using the GRPO algorithm, from dataset loading to
implementing the full training loop.


Note that RLVR defines the overall training setup, namely, using automatically verifiable
rewards. In addition, we need a concrete policy optimization algorithm that can be used
within this RLVR framework to update the model weights and train the model. The term
"policy" is an RL-specific jargon and refers, in this specific context, to the LLM we want to
train.

Specifically, we will use the group relative policy optimization (GRPO) algorithm for
policy optimization.

In short, RLVR determines what learning signal is available, and GRPO determines how
that signal is used to update the model weights.

###### POLICY GRADIENT ALGORITHMS

A widely used policy gradient algorithm for LLMs is proximal policy optimization
(PPO), which was popularized in RLHF. In principle, the same algorithm could also be
applied in an RLVR setting.

When training the DeepSeek-R1 reasoning models, the DeepSeek team opted for
a simpler alternative, GRPO, which was originally introduced in their DeepSeekMath
paper (https://arxiv.org/abs/2402.03300). GRPO is more resource-friendly than PPO
because it does not require a separate value model to estimate a value function.
Instead, GRPO derives its learning signal from relative comparisons within a group of
sampled responses, which substantially reduces computational overhead.

Interested readers can find a more detailed side-by-side comparison of PPO and
GRPO in my article The State of Reinforcement Learning for LLM Reasoning
(https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-
training). In this chapter, we focus on implementing RLVR using GRPO, which is
similar to what DeepSeek-R1 and other popular reasoning models use.

In the following chapter, we build on this foundation and introduce several
practical extensions to GRPO that further improve training stability and the resulting
reasoning performance.

- 6.2.1 High-level GRPO intuition via a chef analogy


The technical details of the GRPO algorithm for RLVR can be a bit overwhelming at first,
since it has many moving parts and a lot of domain-specific RL terminology. So, before we
start with a concrete technical overview and implementation, it may be beneficial to
introduce the idea and mechanics behind GRPO with an example.

Figure 6.8 introduces the technical terms and approach of GRPO using a chef analogy
preparing a meal.

![image 97](<input (1)_images/imageFile97.png>)

- Figure 6.8 High-level overview of the GRPO algorithm for RLVR using a chef analogy. Multiple rollouts are
generated and scored, relative advantages are computed, and a policy gradient objective with a KL-based
regularization (loss) term is used to update the model parameters.


So, walking through the flowchart in figure 6.8, imagine we are a chef running a small food
delivery service:

- 1. Each day, we receive a single customer request (the prompt) asking us to
prepare a certain dish (for example, lasagna).
- 2. For that request, we cook our famous lasagna dish, but while doing so,
we try out several different recipe variations (the rollouts).
- 3. After preparing the recipes, we create multiple dishes (completions). Note
in the LLM setting, a rollout refers to the entire generation process for a
prompt, while the completion is the resulting output text. In practice, the
two terms are often used interchangeably.
- 4. Customers only give us feedback (the reward) after tasting the entire
dish, not while it is being prepared. This means we judge the final result
as a whole rather than individual cooking steps.
- 5. We compare how the dishes performed relative to each other based on
the customer feedback from stage 4. This helps us identify which recipe
variations received higher rewards and which received lower rewards for
this particular request.


- 6. For each completed dish, we also keep track of how typical it was of our
current cooking style, that is, how closely it followed our usual techniques
and ingredient choices (logprobs).
- 7. Using relative feedback from stage 5 and how characteristic each dish
was of our current cooking style from stage 6, we determine how to
tweak the cooking style (policy gradient loss). Here, we want to reinforce
choices that led to better dishes and reduce those that led to worse ones.
- 8. At the same time, we consult our original cookbook from the past.
- 9. We then apply a style-preservation penalty that encourages trying new
recipe variations, but prevents overly drastic changes to our cooking style
that could scare away existing customers.
- 10. The preference-based adjustment and the cooking style-preservation
penalty are combined into a single overall measure or decision (the
overall loss) about how the cooking style should change. The goal here is
to balance customer satisfaction with consistency.
- 11. Finally, we update the cooking style (gradient-based model weight
update) so that future dishes are more likely to satisfy customer
preferences while still remaining close to the original cookbook.


Even though this is a relatively intuitive analogy, we can see that there are lots of
components and moving parts in GRPO.

- 6.2.2 The high-level GRPO procedure


Following the chef analogy from the previous section, the flowchart in figure 6.9 provides a
technical overview of the GRPO algorithm for RLVR using concrete values and numbers that
we will compute when implementing GRPO step by step.

![image 98](<input (1)_images/imageFile98.png>)

- Figure 6.9 Step-by-step GRPO update for RLVR. (1) A prompt is sampled and multiple rollouts are generated.


(2) Each rollout is scored using a verifiable reward. (3) Group-relative advantages are computed from these
rewards. (4) The log-probability of each rollout under the current model is calculated. (5) Advantages and log-
probabilities are combined to form the policy gradient loss. (6) A KL regularization term against a reference
model is added, and the resulting total loss is used to update the model parameters.

The GRPO outline in figure 6.9 contains many technical components, and at first glance, it
may look overwhelming. We will start by implementing a simplified version of GRPO that
omits the KL loss term shown on the right side of the figure.

In chapter 7, we will add this missing KL loss term for completeness. While the KL loss is
part of the original GRPO formulation, it is not strictly essential. In fact, many LLM
developers omit it in practice, as doing so both simplifies implementation and can
sometimes improve modeling performance.

###### NOTE Readers experienced with GRPO or other policy gradient methods may wonder about the use of clipped policy ratios; these are covered in the upcoming chapter.

If this overview, shown in figure 6.9, feels complicated on first reading, don’t worry. We will
build up the algorithm step by step. For now, it is best to treat figure 6.9 as a high-level
roadmap that helps orient us as we work through the individual pieces.

##### 6.3 Loading a pre-trained model

Similar to previous chapters, we begin by loading the pre-trained model (see stage 5 in
figure 6.10), which we will then use to generate the rollouts (answers to a given prompt)
we need for GRPO.

![image 99](<input (1)_images/imageFile99.png>)

- Figure 6.10 In stage 5, we load the pre-trained model (this section) and dataset (next section) that we will use
for the model training.


Listing 6.1 is the same code we’ve used previously for loading the tokenizer and base
model.

- Listing 6.1 Loading tokenizer and base model


import torch

- from reasoning_from_scratch.ch02 import get_device
- from reasoning_from_scratch.ch03 import (
load_model_and_tokenizer


)

device = get_device()
device = torch.device("cpu") #A

model, tokenizer = load_model_and_tokenizer(
which_model="base",
device=device,
use_compile=False

)

#A Delete this line to run the code on a GPU (if supported by your machine)

Note that the code in listing 6.1 runs on the CPU by default to ensure you get results that
are more consistent with what's shown in this chapter. Once you have done a full pass
through this chapter, I recommend deleting the line device = torch.device("cpu") and
running the chapter code on a GPU.

Next, to ensure that the model is loaded correctly, let's use it together with the
temperature and top-p sampler code from chapter 4 on a simple math prompt:

- Listing 6.2 Generating text with temperature scaling and top-p sampling


- from reasoning_from_scratch.ch03 import render_prompt
- from reasoning_from_scratch.ch04 import (
generate_text_stream_concat_flex,
generate_text_top_p_stream_cache


)

raw_prompt = (
"Half the value of $3x-9$ is $x+37$. "
"What is the value of $x$?"

)
prompt = render_prompt(raw_prompt)

torch.manual_seed(0)
response = generate_text_stream_concat_flex(

model, tokenizer, prompt, device,
max_new_tokens=2048, verbose=True,
generate_func=generate_text_top_p_stream_cache,
temperature=0.9,
top_p=0.9

)
print(response)

The model prints " \boxed{58}", which is an incorrect answer (83 is correct), but that's
okay since the goal here was merely to make sure that the code runs without issue.

##### 6.4 Loading a MATH training subset

Next, we load the dataset that we will use when training the model with GRPO.

We use a non-overlapping training subset derived from the original MATH dataset, which
means that all MATH-500 evaluation problems are explicitly excluded from the training
data. This prevents data leakage and ensures a clean separation between training and
evaluation, as illustrated in figure 6.11.

NOTE For details on how this dataset was constructed, see https://github.com/rasbt/math_full_
minus_math500.

![image 100](<input (1)_images/imageFile100.png>)

- Figure 6.11 Structure and split of the MATH dataset. The full dataset contains about 12,500 problems that
are divided into a 500-problem test set (MATH-500), which we used for model evaluation in chapter 3. A non-
overlapping set of 12,000 problems is used for training in this chapter.


To load the 12,000 math problems from the MATH training set depicted in figure 6.11, we
define the following load_math_train function in listing 6.3, which is similar to the
load_math500_test function in chapter 3, except that we specify a different file path.

- Listing 6.3 Loading the MATH training set


import json
import requests
from pathlib import Path

def load_math_train(local_path="math_train.json", save_copy=True):
local_path = Path(local_path)

url = (

"https://raw.githubusercontent.com/rasbt/"
"math_full_minus_math500/refs/heads/main/"
"math_full_minus_math500.json"

)
backup_url = ( #A

"https://f001.backblazeb2.com/file/reasoning-from-scratch/"
"MATH/math_full_minus_math500.json"

)

if local_path.exists(): #B
with local_path.open("r", encoding="utf-8") as f:

data = json.load(f)
else:

try:

r = requests.get(url, timeout=30)
r.raise_for_status()

except requests.RequestException:
print("Using backup URL.")
r = requests.get(backup_url, timeout=30)
r.raise_for_status()

data = r.json()

if save_copy: #C
with local_path.open("w", encoding="utf-8") as f:
json.dump(data, f, indent=2)

return data

#A Alternative URL in case the GitHub-hosted file is unavailable
#B Load from local file if already downloaded
#C Optionally cache a local copy for future runs

The above code should print "Dataset size: 12000" to indicate that the dataset was
loaded correctly.

Next, we can also print one of the 12,000 entries to get an idea of the dataset structure.
For this, we use the pprint function from the pprint standard library, as it provides nicer
formatting for dictionary entries compared to the regular print function:

from pprint import pprint
pprint(math_train[4])

The resulting entry (index 4 corresponds to the fifth entry in the dataset) is shown below:

{

"answer": "6",
"level": "Level 3",
"problem": (

"Sam is hired for a 20-day period. On days that he "
"works, he earns $\\$60. For each day that he does "
"not work, $\\$30 is subtracted from his earnings. "
"At the end of the 20-day period, he received "
"$\\$660. How many days did he not work?"

),
"solution": (

"Call $x$ the number of days Sam works and $y$ the "
"number of days he does not... Thus, Sam did not "
"work for $\\boxed{6}$ days."

),
"type": "Algebra",
"unique_id": 4,

}

As we can see, each dataset example is stored as a dictionary with several fields. For
training, the relevant fields are "problem", which serves as the prompt, and "answer",
which is the target we verify the model's output against using the math verifier.

The dataset also includes a full worked solution in the "solution" field. While such step-
by-step solutions could in principle be used to evaluate intermediate reasoning steps, doing
so would unnecessarily constrain the model and risk overfitting to a specific solution and
style. Instead, we want to allow the model to explore the solution space more freely.

##### 6.5 Sampling rollouts

After setting up the pre-trained base model and loading the dataset we are now ready to
implement the GRPO stages, as illustrated in figure 6.12.

![image 101](<input (1)_images/imageFile101.png>)

- Figure 6.12 After outlining the RLVR method and GRPO algorithm, the following sections implement the
individual GRPO stages that we need to train the LLM via verifiable rewards on the MATH dataset.


Before we begin, figure 6.13 revisits the GRPO overview introduced earlier, but in a
simplified form that omits the KL loss term, short for Kullback-Leibler divergence loss,
which we will add in the next chapter. We can think of it as a penalty term that discourages
the updated model from drifting too far away from a reference model. For now, figure 6.13
serves as a simplified roadmap that we will refer back to as we step through the algorithm's
components.

![image 102](<input (1)_images/imageFile102.png>)

- Figure 6.13 Step-by-step GRPO update for RLVR (without KL loss term). We begin by prompting the LLM to
generate the different rollouts.


As shown in figure 6.13, we first use the LLM to generate multiple rollouts. Here, "rollout" is
a reinforcement learning term that simply refers to a complete answer generated by the
model for a given prompt.

We could reuse the generate_text_stream_concat_flex function from listing 6.2 to
generate the rollouts. When we defined that function earlier, though, we decorated it with
@inference_mode, which disables several PyTorch features for efficiency. Since we will later
perform a backward pass, this makes the function incompatible with our training setup.
(Here, "backward pass" refers to the step where PyTorch computes gradients from the loss
so that the optimizer can update the model weights.) Instead, we need to use the
@torch.no_grad decorator, which disables gradient tracking for the forward pass without
switching PyTorch into inference-only mode.

Let’s rewrite the function in a more compact form and call it sample_response.
Importantly, this function generates text identical to
generate_text_stream_concat_flex(...,
generate_func=generate_text_top_p_stream_cache). We can verify this by observing
that, with the same random_seed, temperature, and top_p settings, both functions
produce identical generated responses.

- Listing 6.4 Defining the rollout generation function


from reasoning_from_scratch.qwen3 import KVCache

- from reasoning_from_scratch.ch04 import top_p_filter


@torch.no_grad()
def sample_response(

model,
tokenizer,
prompt,
device,
max_new_tokens=512,
temperature=0.8,
top_p=0.9,

):

input_ids = torch.tensor(
tokenizer.encode(prompt),
device=device
)

cache = KVCache(n_layers=model.cfg["n_layers"]) #A
model.reset_kv_cache()
logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

generated = []
for _ in range(max_new_tokens):

if temperature and temperature != 1.0: #B
logits = logits / temperature

probas = torch.softmax(logits, dim=-1)
probas = top_p_filter(probas, top_p) #C
next_token = torch.multinomial(

probas.cpu(), num_samples=1
).to(device)

token_id = next_token.item()
generated.append(token_id)

if (

tokenizer.eos_token_id is not None
and token_id == tokenizer.eos_token_id

):

break

logits = model(next_token, cache=cache)[:, -1]

full_token_ids = torch.cat(
[input_ids,
torch.tensor(generated, device=device, dtype=input_ids.dtype),]

)
return full_token_ids, input_ids.numel(), tokenizer.decode(generated)

#A Cache past keys and values for efficient generation as introduced in chapter 2
#B Apply temperature scaling from chapter 4
#C Apply top-p filter from chapter 4

Note that the function now also returns the token IDs of the prompt plus answer tokens
and the number of tokens (input_ids.numel()) next to the answer text
(tokenizer.decode(generated)). Returning these allows us to simplify several
downstream functions later on.

Otherwise, there is nothing new here. The code is simply a leaner version of what we
have been developing previously; it combines the generate_text_basic_stream_cache
function from chapter 2 with temperature and top-p sampling from chapter 4 directly.

Next, we call the function on an example prompt similar to listing 6.2.

- Listing 6.5 Generating rollouts with temperature scaling and top-p sampling


torch.manual_seed(0)

raw_prompt = (
"Half the value of $3x-9$ is $x+37$. "
"What is the value of $x$?"

)
prompt = render_prompt(raw_prompt)

token_ids, prompt_len, answer_text = sample_response(
model=model,
tokenizer=tokenizer,
prompt=prompt,
device=device,
max_new_tokens=512,
temperature=0.9,
top_p=0.9,

)

print(answer_text)

Before, the generate_text_stream_concat_flex sampling function returned "
\boxed{58}". Now, the sample_response function explicitly includes the end-of-sequence
token in its response: " \boxed{58}<|endoftext|>". Including this <|endoftext|> token
is theoretically not necessary, but it prevents the model from unlearning to generate end-
of-sequence tokens in very long training runs.

Next, we could call sample_response multiple times to generate different rollouts, which
we will indeed do later when implementing the full training loop. For now, to keep the GRPO
walkthrough example simple and to make it easier to follow the step-by-step GRPO outline
from figure 6.13, we instead assume that the model produced the following four short
responses:

rollouts = [
r"\boxed{83}",
r"The correct answer is \boxed{83}",
r"The final answer is 83",
r"We get \boxed{38}",

]

##### 6.6 Calculating rewards

The second stage of the GRPO procedure involves computing rewards for each rollout, as
shown in figure 6.14.

![image 103](<input (1)_images/imageFile103.png>)

- Figure 6.14 The second stage in the GRPO pipeline computes the rewards for each rollout the LLM generated
in the previous section.


The rewards are computed using the math verifier from chapter 3 and are primarily based
on answer correctness. By setting fallback=None in the reward_rlvr function in listing

- 6.6, we also include an implicit format constraint. For instance, an answer only receives a
reward of 1.0 if it is both correct and expressed in the required \boxed{} format.


- Listing 6.6 Implementing the reward function

#A fallback=None requires \boxed{} format to return an extracted answer

from reasoning_from_scratch.ch03 import (

extract_final_candidate, grade_answer
)

def reward_rlvr(answer_text, ground_truth):
extracted = extract_final_candidate(
answer_text, fallback=None #A

)
if not extracted:

return 0.0
correct = grade_answer(extracted, ground_truth)
return float(correct)

- Listing 6.7 Applying the reward function to all rollouts


Since the model receives a non-zero reward only if it answers correctly and writes the final
answer in the \boxed{} format, we encourage it to learn to produce correct and properly
formatted answers.

Let's give the reward_rlvr function a try and apply it to the rollouts.

rollout_rewards = []

for answer in rollouts:
reward = reward_rlvr(answer_text=answer, ground_truth="83")
print(f"Answer: {answer!r}")
print(f"Reward: {reward}\n")
rollout_rewards.append(reward)

The resulting outputs are as follows:

Answer: '\\boxed{83}'
Reward: 1.0

Answer: 'The correct answer is \\boxed{83}'
Reward: 1.0

Answer: 'The final answer is 83'
Reward: 0.0

Answer: 'We get \\boxed{38}'
Reward: 0.0

As shown above, the reward function works as intended and only provides a reward of 1.0
if the answer contains the correct result (83) and uses the \boxed{} format.

Note that the DeepSeek-R1 team also tried to use process reward models to score
intermediate solution steps during training. These attempts were unsuccessful, and the
researchers concluded that it is better to train only on final-answer correctness rewards,
without intermediate rewards.

###### EXERCISE 6.1: ADDING FORMAT-AWARE REWARD SHAPING

Extend the reward_rlvr function from this chapter so that it assigns partial credit
based on output format. Specifically, modify the reward function so that it returns
1.0 if the model produces the correct answer in the required \boxed{} format, 0.5 if
the answer is correct but not boxed, and 0.0 otherwise.

##### 6.7 Preparing learning signals from rollouts via advantages

We now move on from rewards to the so-called advantages, as shown in figure 6.15. While
rewards tell us how well each individual rollout performed, advantages capture how a
rollout performed relative to other rollouts generated for the same prompt.

![image 104](<input (1)_images/imageFile104.png>)

- Figure 6.15 The third GRPO stage computes the advantage values from the answer (rollout) correctness
rewards.


The advantage values shown in figure 6.15 are computed by a simple formula:

![image 105](<input (1)_images/imageFile105.png>)

Here

𝑟𝑖 denotes the reward of the i-th rollout,

𝜇𝑟 is the mean reward across the group of rollouts,

σ𝑟 is the corresponding standard deviation,

and ϵ is a small constant added for numerical stability to avoid zero-
division errors.

In code, we can implement the advantage calculation as follows:

- Listing 6.8 Calculating advantages


rewards = torch.tensor(rollout_rewards, device=device)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
print(rewards)
print(advantages)

The rewards, which we computed in the previous section, are [1., 1., 0., 0.], and the
corresponding advantages are [0.8659, 0.8659, -0.8659, -0.8659].

These advantage values capture which responses performed better than average and
which performed worse than average within the group.

###### NOTE The "GR" (group relative) in GRPO refers to the fact that GRPO generates multiple answers (rollouts) per prompt, and compares them relative to each other to construct a learning signal.

You might still wonder what the point of converting rewards into advantages is. At a
practical level, the advantage values directly scale the gradients during the policy update
later.

If the advantage is positive, the gradient increases the likelihood of the actions that
produced that rollout. If it is negative, the gradient decreases their likelihood. Rollouts with
advantage close to zero contribute very little.

###### EXERCISE 6.2: ZERO-ADVANTAGE CASES

Modify the rollout rewards so that all rollouts receive the same reward (for example,
force all rewards to 0.0 or all to 1.0). Run the advantage computation and verify
that the resulting GRPO loss is zero. Why is this behavior desirable in group-relative
policy optimization?

##### 6.8 Scoring rollouts with sequence log-probabilities

We calculated the rewards and advantages from the rewards. Now, we take a separate
branch from the rollouts and compute their log-probabilities (logprobs), as shown in figure

- 6.16. Logprobs measure how likely the model considers each generated token under its
current parameters, as discussed in the previous chapter.


![image 106](<input (1)_images/imageFile106.png>)

- Figure 6.16 The fourth stage in the GRPO pipeline computes log-probabilities for each rollout, which is related
to the logprob scorer we developed in the previous chapter.


As shown in figure 6.16, these logprobs, together with the advantages, form the core
ingredients of the GRPO policy gradient loss, which we will implement later on.

Coming back to the logprobs, in the previous chapter we implemented an
avg_logprob_answer function that computes the per-token logprobs for the model's
answer tokens. These values, often referred to as token-level logprobs in the literature, are
obtained by evaluating the likelihood that the model assigns to each generated token under
its current parameters.

When averaged across the response, these values are often referred to as token-level
logprobs and are commonly used for scoring LLM outputs. Averaging is preferred in this
context because it provides length normalization, which makes the scores comparable
across responses of different lengths.

###### MATHEMATICAL DEFINITION OF TOKEN-LEVEL LOG-PROBABILITIES

Mathematically, the token-level logprobs we computed in the previous chapter can
be written as

![image 107](<input (1)_images/imageFile107.png>)

Here

𝑦1, 𝑦2, ..., 𝑦𝑇 denote the tokens in the generated response of length T

𝑦<𝑡 represents all previously generated tokens

x is the input prompt

W denotes the model's weight parameters.

This expression is mathematically identical to the one used in the previous chapter;
we simply switch from x to y to distinguish the generated output tokens from the
input prompt for clarity.

For reference, the avg_logprob_answer function from the previous chapter is copied below,
which we use to compute the logprobs for the previous prompt and answer_text for
illustration purposes.

- Listing 6.9 Computing token-level log-probabilities (similar to chapter 5)


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):

prompt_ids = tokenizer.encode(prompt)
answer_ids = tokenizer.encode(answer)
full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

logits = model(full_ids.unsqueeze(0)).squeeze(0)
logprobs = torch.log_softmax(logits, dim=-1)

start = len(prompt_ids) - 1
end = full_ids.shape[0] - 1

t_idx = torch.arange(start, end, device=device)
next_tokens = full_ids[start + 1 : end + 1]
next_token_logps = logprobs[t_idx, next_tokens]

return torch.mean(next_token_logps).item()

avg_logprob_val = avg_logprob_answer(
model, tokenizer,
prompt=prompt,
answer=answer_text,
device=device)

print(avg_logprob_val)

This returns -0.061279296875.

In GRPO we do not use averaged token-level logprobs. Instead, we work with sequence-
level logprobs.

The reason is that GRPO assigns a single reward and advantage to each rollout, which
applies to the entire generated response. Intuitively, logprob should reflect how likely the
model was to generate the whole sequence, not an average per token.

We compute sequence-level logprobs by summing the logprobs of all generated tokens.
(This sum corresponds to the log-likelihood of producing the full response under the current
model parameters.)

In contrast, averaging token-level logprobs normalizes by sequence length. While this
normalization is useful for scoring and comparing responses of different lengths, it would
unintentionally rescale the learning signal in policy optimization and cause longer and
shorter responses to contribute unevenly to the update. (While this is not the intention of
the original GRPO formulation, we will revisit token-level logprobs when improving the
algorithm in the next chapter.)

We can convert the token-level, length-normalized score into a sequence-level logprobs
by dropping the averaging step and replacing torch.mean(next_token_logps) with
torch.sum(next_token_logps) in the code in listing 6.9.

The same result can also be obtained by multiplying the averaged logprob by the
number of answer tokens, as long as the token count matches exactly:

sequence_logprob_val = avg_logprob_val * (
len(tokenizer.encode(answer_text))

)
print(sequence_logprob_val)

This now returns -16.239013671875.

These sequence-level logprobs scale linearly with the sequence length T, which means
that longer responses generally receive more negative logprob values. As a result, for two
equally good answers, the optimization implicitly favors the shorter one, since it is cheaper
in terms of likelihood. Summed logprobs therefore encourage the model to stop earlier
unless producing a longer response is justified by a higher reward.

So, as mentioned before, we can replace torch.mean with torch.sum in the function to
obtain sequence-level logprobs. Since the function was run in inference mode in the
previous chapter using the @torch.inference_mode() decorator, we need to redefine it
(without the decorator) anyway so that PyTorch can track gradients.

In addition, because sample_response from section 6.5 already returns the token_ids
and prompt_len, we can simplify the implementation by dropping the explicit tokenization
step and the construction of full_ids.

- Listing 6.10 Computing sequence-level log-probabilities


def sequence_logprob_draft(model, token_ids, prompt_len):
logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
logprobs = torch.log_softmax(logits, dim=-1)

start = prompt_len - 1 #A
end = token_ids.shape[0] - 1 #A

t_idx = torch.arange(start, end, device=token_ids.device)
next_tokens = token_ids[start + 1 : end + 1]
next_token_logps = logprobs[t_idx, next_tokens]

return torch.sum(next_token_logps) #B

print(sequence_logprob_draft(model, token_ids, prompt_len))

- #A Positions whose next-token probabilities we want to predict
- #B Sum log-probabilities over the answer tokens


This outputs:

tensor(-16.2998, grad_fn=<SumBackward0>)

First, the result is similar to the -16.2421875 we got earlier (the difference is due to
floating-point behavior and rounding).

PyTorch internally builds a computation graph that records each differentiable operation
applied to a tensor. The SumBackward0 entry shows that the summation of token log-
probabilities is part of this graph, which means gradients can propagate back through the
sequence-level log-probability to the model parameters.

This is exactly what we need for policy optimization later, when we implement the
training loop to update the model weights via a backward pass. The backward pass applies
the backpropagation algorithm, which is the standard method used in deep learning to
compute gradients and update neural network weights.

###### PYTORCH COMPUTATION GRAPHS, GRADIENTS, AND BACKPROPAGATION

As the model produces an output, PyTorch records each mathematical operation that
leads from the model parameters to that output. This record is called the
computation graph. When we later run the backward pass, PyTorch uses this graph
to perform backpropagation, that is, computing gradients that describe how changes
to the model parameters would affect the final result. If an operation is part of this
graph, PyTorch can compute gradients through it during training.

It is not required to understand PyTorch's computation graphs for this chapter, but
interested readers can read more about it in my tutorial at https://sebastianraschka.
com/teaching/pytorch-1h/#3-seeing-models-as-computation-graphs.

While this draft implementation is a fully working code implementation, we can simplify it a
bit and make it more efficient for GPUs. In particular, we can avoid explicitly constructing
index ranges and instead use torch.gather to directly select the logprob corresponding to
the generated tokens. Listing 6.11 shows a more compact, optimized version of the same
computation that produces identical results while being easier to read and faster to
execute.

- Listing 6.11 Optimized sequence-level log-probabilities code


def sequence_logprob(model, token_ids, prompt_len):
logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
logprobs = torch.log_softmax(logits, dim=-1)
selected = logprobs[:-1].gather(

1, token_ids[1:].unsqueeze(-1)
).squeeze(-1)
return torch.sum(selected[prompt_len - 1:])

print(sequence_logprob(model, token_ids, prompt_len))

Similar to the previous code, this returns tensor(-16.2998, grad_fn=<SumBackward0>).

###### TIP We could compute these logprobs directly in the sample_response function to improve efficiency by avoiding calling model(...) twice (in both the sample_response and sequence_logprob functions).

Finally, with a robust and efficient sequence-level logprob function in place, we can
calculate the respective logprobs of the four different rollouts.

- Listing 6.12 Computing sequence-level log-probabilities of all rollouts


rollouts = [
r"\boxed{83}",
r"The correct answer is \boxed{83}",
r"The final answer is 83",
r"We get \boxed{38}",

]

rollout_logps = []

for text in rollouts:
token_ids = tokenizer.encode(prompt + " " + text)
logprob = sequence_logprob(

model=model,
token_ids=torch.tensor(token_ids, device=device),
prompt_len=prompt_len,

)

print(f"Answer: {text}")
print(f"Logprob: {logprob.item():.4f}\n")

rollout_logps.append(logprob)

This prints the following:

Answer: \boxed{83}
Logprob: -7.9243

Answer: The correct answer is \boxed{83}
Logprob: -20.1546

Answer: The final answer is 83
Logprob: -16.6130

Answer: We get \boxed{38}
Logprob: -23.3677

As we can see, shorter and more concise answers receive higher (less negative) sequence-
level logprobs than longer or more verbose responses. The exception is the last answer,
which receives the lowest score. Note that this is the only answer that contains the
incorrect numerical answer value (38 instead of 83), so this also makes intuitive sense.

Overall, this illustrates how summed logprobs naturally favor concise outputs, which is
consistent with their role in GRPO when rewards and advantages are applied at the
sequence level.

##### 6.9 From advantages to policy updates via the GRPO loss

We now implement the fifth stage of the GRPO pipeline, where the previously computed
advantages and logprobabs are combined into a policy gradient loss. We implement the
subsequent weight update (stage six) later as part of the training loop, as shown in figure

- 6.17.


![image 108](<input (1)_images/imageFile108.png>)

- Figure 6.17 The fifth stage in the GRPO pipeline computes the policy gradient loss that we use to update the
model. Stage number 6, the model weight update, will be implemented as part of the training loop later.


First, we convert the list of the rollout logprobs from the previous section into a PyTorch
tensor. Then, we compute the policy gradient loss by multiplying each rollout's sequence-
level logprob by its corresponding advantage, taking the mean across rollouts, and applying
a negative sign. Regarding the negative sign, because PyTorch optimizers are defined to
minimize a loss, objectives that are naturally written as maximization problems must be
sign-flipped.

- Listing 6.13 Computing the policy gradient loss


logps = torch.stack(rollout_logps)
pg_loss =

- -(advantages.detach() * logps).mean()
print(logps)
print(pg_loss)

tensor([

- -7.9243, -20.1546, -16.6130, -23.3677],
grad_fn=<StackBackward0>)


This prints

for the logprobs, and the resulting policy gradient loss value is

tensor(-2.5764, grad_fn=<NegBackward0>)

Note that we use .detach() on the advantages because they are treated as fixed learning
signals during the policy update. This prevents gradients from flowing back through the
advantage computation. This ensures that only the model parameters influencing the
logprobs are updated.

For instance, in policy gradient methods, we want to maximize the advantage-weighted
logprob of the rollouts, since higher log-probabilities for high-advantage rollouts improve
the policy.

So, by multiplying this objective by −1, we convert the maximization problem into an
equivalent minimization problem. Note that minimizing the negative objective produces the
exact same parameter updates as maximizing the original one, but this way, it remains
compatible with PyTorch's optimizer implementations.

###### MAXIMIZATION VERSUS MINIMIZATION

To illustrate the sign flipping further with a concrete example, suppose the objective
we want to maximize is a simple scalar value 𝑓(𝑥) = 3.

Under the maximization view, a larger value is better, so 3 is preferable to 2.
PyTorch optimizers minimize, so they would try to reduce this value, which is the
opposite of what we want.

Now, suppose we flip the sign and define the loss as 𝐿(𝑥) = - 𝑓(𝑥) = -3.
Minimizing 𝐿(𝑥) is now is equivalent to maximizing 𝑓(𝑥). For instance, if 𝐿(𝑥)

decreases from −2 to −3, 𝑓(𝑥) increases from 2 to 3.

###### MATHEMATICAL DEFINITION OF THE POLICY GRADIENT LOSS

For readers who find it easier to follow the computations via mathematical notation,
the policy gradient loss used in GRPO can be written as

![image 109](<input (1)_images/imageFile109.png>)

Here,

N denotes the number of rollouts,

𝑦𝑡(𝑖), …, 𝑦𝑇𝑖(𝑖) are the tokens of the i-th generated response of length
𝑇𝑖,

𝑦<𝑡(𝑖) represents represents all previously generated tokens in that
response,

𝑥(𝑖) is the corresponding input prompt,

And 𝐴𝑖 is the advantage of the full rollout.

The inner sum computes the sequence-level log-probability of a rollout, while the
outer average computes the advantage-weighted log-probabilities across rollouts.

##### 6.10 Putting everything together in a single GRPO function

We have now completed the most challenging parts of this chapter and walked through
each GRPO stage in isolation.

Next, we combine the five GRPO stages shown in figure 6.18 into a single
compute_grpo_loss function, which we will later use as part of the full training loop.

(Note that here, the term stage refers to a conceptual component of the GRPO loss
computation, based on how we walked through the GRPO algorithm step-by-step, not a
training step in the outer optimization loop.)

![image 110](<input (1)_images/imageFile110.png>)

- Figure 6.18 The complete GRPO workflow where (1) multiple rollouts are generated for a prompt, (2) assigned
correctness rewards, (3) converted into group-relative advantages, and (4) combined with log probabilities to


(5) compute the policy gradient loss. The loss gradients (6) will be computed and used to update the model in
the next section.

The compute_grpo_loss function that combines all the stages from figure 6.18 that we
discussed previously is shown in listing 6.14 below.

- Listing 6.14 Combining all GRPO stages


def compute_grpo_loss(
model,
tokenizer,
example,
device,
num_rollouts=2,
max_new_tokens=256,
temperature=0.8,
top_p=0.9,

):

assert num_rollouts >= 2
roll_logps, roll_rewards, samples = [], [], []
prompt = render_prompt(example["problem"])

was_training = model.training
model.eval()

for _ in range(num_rollouts):

- #A
token_ids, prompt_len, text = sample_response(

model=model,
tokenizer=tokenizer,
prompt=prompt,
device=device,
max_new_tokens=max_new_tokens,
temperature=temperature,
top_p=top_p,

)

- #B
reward = reward_rlvr(text, example["answer"])
- #C
logp = sequence_logprob(model, token_ids, prompt_len)


roll_logps.append(logp)
roll_rewards.append(reward)
samples.append(

{

"text": text,
"reward": reward,
"gen_len": token_ids.numel() - prompt_len,

}
)

if was_training:
model.train()

- #D
rewards = torch.tensor(roll_rewards, device=device)
- #E
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
- #F
logps = torch.stack(roll_logps)
- #G
pg_loss = -(advantages.detach() * logps).mean()
loss = pg_loss # In the next chapter we add a KL term here


return {
"loss": loss.item(),
"pg_loss": pg_loss.item(),
"rewards": roll_rewards,
"advantages": advantages.detach().cpu().tolist(),
"samples": samples,
"loss_tensor": loss,

}

#A Stage 1: generate rollouts
#B Stage 2.1: compute rewards
#C Stage 4.1: compute logprobs
#D Stage 2.2: collect all rewards
#E Stage 3: compute advantages
#F Stage 4.2: collect all logprobs
#G Stage 5: compute policy gradient loss

The compute_grpo_loss function is relatively self-explanatory as it follows the exact stages
we implemented manually in the previous sections. Note, though, that after stage 1, the
code proceeds directly to stages 2 and 4, rather than stages 3 and 4. This is purely an
implementation choice that simplifies the code structure by avoiding the need for multiple
nested for-loops while preserving the same logical sequence of operations.

Also note that the pg_loss (policy gradient loss) is identical to the overall loss in our
examples. This is because we intentionally omit the KL loss term here and add it later in
chapter 7 for completeness. As discussed earlier, GRPO is often reported to perform better
on math problems when the KL loss term is omitted, which is why we adopt this simplified
objective at this stage.

Next, let's try the new compute_grpo_loss function on an example from the MATH
training dataset to ensure that it works. Here, we sample only a small number of rollouts
(num_rollouts=2) and generate a relatively small number of tokens (max_new_tokens=256)
for testing purposes. It may still take several seconds until you see the function return the
results.

- Listing 6.15 Computing GRPO stages on a MATH training example


torch.manual_seed(123)

stats = compute_grpo_loss(
model=model,
tokenizer=tokenizer,
example=math_train[4],
device=device,
num_rollouts=2,
max_new_tokens=256,
temperature=0.8,
top_p=0.9

)

pprint(stats)

The output is as follows:

{'advantages': [0.0, 0.0],
'loss': -0.0,
'loss_tensor': tensor(-0., grad_fn=<NegBackward0>),
'pg_loss': -0.0,
'rewards': [0.0, 0.0],
'samples': [{'gen_len': 4, 'reward': 0.0, 'text': ' 14<|endoftext|>'},

{'gen_len': 256,
'reward': 0.0,
'text': ' 4\n'

'To solve the problem, let's break it down step by step'
'...'}

As we can see from the 'rewards' and 'text' entries in the 'samples' field, the model
answers incorrectly. As discussed earlier in section 6.4, a correct answer must include
\boxed{6}. Since this condition is not met, the reward is zero, which in turn yields zero
advantages and a zero loss. In this case, if we were training the model, the gradient would
be zero and the model parameters would not be updated.

##### 6.11 Implementing the GRPO training loop

We now have all the necessary components in place to implement the full GRPO training
loop. In this final section of the chapter, we bring everything together to train the model via
reinforcement learning with verifiable rewards using the GRPO algorithm, as illustrated in
figure 6.19.

![image 111](<input (1)_images/imageFile111.png>)

- Figure 6.19 After implementing the individual GRPO stages, we now implement the surrounding training loop
to update the model weights.


The training loop consists of eight main stages, as shown in figure 6.20. Most of these
stages are standard components of a typical PyTorch training loop used for deep neural
networks, including LLMs. The only stage specific to reasoning models is stage 4, where we
compute the GRPO loss rather than a standard classification loss used in many other types
of deep neural networks and LLM pre-training.

![image 112](<input (1)_images/imageFile112.png>)

- Figure 6.20 Outline of the training loop. The overall structure follows a standard deep learning training loop.
The key difference lies in how the loss is computed: instead of a standard supervised objective, the loss is
obtained via the GRPO stages (stage 4).


Before discussing the eight stages one by one, it is helpful to first see how they are
implemented in code. listing 6.16 shows the full training loop corresponding to the eight
stages illustrated in figure 6.20.

- Listing 6.16 Implementing the RLVR training loop


import time

def train_rlvr_grpo(
model,
tokenizer,
math_data,
device,
steps=None,
num_rollouts=2,
max_new_tokens=256,
temperature=0.8,
top_p=0.9,
lr=1e-5,
checkpoint_every=50,
checkpoint_dir=".",
csv_log_path=None,

):

if steps is None:
steps = len(math_data)

#A
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.train()
current_step = 0

if csv_log_path is None:
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"

csv_log_path = Path(csv_log_path)

try:

#B
for step in range(steps):

#C
optimizer.zero_grad()

current_step = step + 1
example = math_data[step % len(math_data)]

- #D


stats = compute_grpo_loss(
model=model,
tokenizer=tokenizer,
example=example,
device=device,
num_rollouts=num_rollouts,
max_new_tokens=max_new_tokens,
temperature=temperature,
top_p=top_p,

)

- #E
stats["loss_tensor"].backward()
- #F
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
- #G
optimizer.step()
- #H
reward_avg = torch.tensor(stats["rewards"]).mean().item()
step_tokens = sum(

sample["gen_len"] for sample in stats["samples"]

)
avg_response_len = (

step_tokens / len(stats["samples"]) if stats["samples"] else 0.0

)
append_csv_metrics(

csv_log_path, current_step, steps,
stats["loss"], reward_avg, avg_response_len,

)
print(

f"[Step {current_step}/{steps}] "
f"loss={stats['loss']:.4f} "
f"reward_avg={reward_avg:.3f} "
f"avg_resp_len={avg_response_len:.1f}"

)

- #I
if checkpoint_every and current_step % checkpoint_every == 0:


ckpt_path = save_checkpoint(
model=model,
checkpoint_dir=checkpoint_dir,

step=current_step,

)
print(f"Saved checkpoint to {ckpt_path}")

#J
except KeyboardInterrupt:

ckpt_path = save_checkpoint(
model=model,
checkpoint_dir=checkpoint_dir,
step=max(1, current_step),
suffix="interrupt",

)
print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
return model

return model

def save_checkpoint(model, checkpoint_dir, step, suffix=""):
checkpoint_dir = Path(checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
suffix = f"-{suffix}" if suffix else ""
ckpt_path = (

checkpoint_dir /
f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"

)
torch.save(model.state_dict(), ckpt_path)
return ckpt_path

#K
def append_csv_metrics(

csv_log_path, step_idx, total_steps,
loss, reward_avg, avg_response_len,

):

if not csv_log_path.exists():

csv_log_path.write_text(
"step,total_steps,loss,reward_avg,avg_response_len\n",
encoding="utf-8",

)
with csv_log_path.open("a", encoding="utf-8") as f:

f.write(
f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
f"{avg_response_len:.6f}\n"

)

- #A Stage 1: initialize optimizer (model was already initialized outside the function)
- #B Stage 2: Iterate over training steps
- #C Stage 3: Reset loss gradient (best practice to do this at the beginning of each step)
- #D Stage 4: calculate GRPO loss
- #E Stage 5: Backward pass to calculate loss gradients
- #F Clip large gradients to improve training stability
- #G Stage 6: Update model weights using loss gradients
- #H Stage 7: Collect rewards, average response lengths, and losses
- #I Stage 8: Save model checkpoint
- #J Save a model checkpoint if we interrupt the training early
- #K Utility function to save the results to a CSV file


This function implements the full RLVR training loop using GRPO. As mentioned earlier, with
the exception of stage 4 (the GRPO loss computation), the structure mirrors a conventional
PyTorch training loop where we reset gradients, backpropagate a loss, optionally clip
gradients, update parameters via an optimizer step, log metrics, and periodically save
checkpoints.

###### TIP For a general introduction to training neural networks in PyTorch, see sections 3–8 of my PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs article (https://sebastianraschka.com/teaching/pytorch-1h/).

The key difference here, compared to other standard training loops, is how the loss is
constructed. Instead of a standard supervised objective (which is used for the majority of
neural networks, for example classifiers, but also for pre-training LLMs), the training signal
is computed via the GRPO algorithm as part of the RLVR procedure.

Note that, next to the logging steps that print results periodically to track progress, we
also save the model checkpoint (a snapshot of the model weights) periodically so we can
load, evaluate, and use it later, as discussed in the next section.

Let's now run the code and train the model. Earlier, we hardcoded the PyTorch device to
"cpu" so that all intermediate results more closely match those shown in the book, since
"mps" and "cuda" devices can introduce small floating-point differences. Training on a CPU
is very slow, though.

For convenience, the listing below already includes two lines that automatically select
and activate the appropriate device preceding the train_rlvr_grpo() function call. For
instance, it will automatically switch to "cuda" or "mps" simply by running the code in the
following listing.

- Listing 6.17 Training the model


device = get_device()
model.to(device)

torch.manual_seed(0)

train_rlvr_grpo(
model=model,
tokenizer=tokenizer,
math_data=math_train,
device=device,
steps=50,
num_rollouts=4,
max_new_tokens=512,
temperature=0.8,
top_p=0.9,
lr=1e-5,
checkpoint_every=5,
checkpoint_dir=".",

)

Let's briefly talk about the settings before we inspect the results. The temperature and
top_p settings were chosen in a common range that encourages some diversity in the
answers but still results in coherent text for this model. Optionally, you can change these
settings and see how they affect the results (reasonable and common ranges are 0.7-0.9).

The number of steps is relatively small at 50, even though the dataset contains 12,000
entries. This is purely due to achieving a reasonable runtime for this educational context,
and can be increased. Now, though, it is already sufficient for good results.

The learning rate (lr) can be tweaked, but it is in a reasonable range and works well.

The number of rollouts (num_rollouts=4) and allowed tokens (max_new_tokens=512)
are relatively small to reduce resource requirements. If you experience out-of-memory
errors, you can further reduce these values, for example, by setting num_rollouts=2 and
max_new_tokens=64 for testing purposes.

With the current checkpoint_every=5 setting, the model is saved every 5 steps, and
one checkpoint requires approximately 1.5 GB of disk space. In practice, in longer runs, I
recommend setting this number to 50 or 100. In this context it is purposefully small for
testing purposes. Also, note that an additional checkpoint should be created if the training
run is manually interrupted, for instance, by interrupting the Jupyter notebook execution
via the "Interrupt the kernel" button.

Let's now take a brief look at the run's output (where I interrupted the 50-step run at
step 7):

Using Apple Silicon GPU (MPS)

- [Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=88.0
- [Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=7.5
- [Step 3/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.5
- [Step 4/50] loss=0.0909 reward_avg=0.250 avg_resp_len=6.5
- [Step 5/50] loss=1.1001 reward_avg=0.500 avg_resp_len=300.5
Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00005.pth


KeyboardInterrupt. Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00006-
interrupt.pth

We can see that both the loss and reward values fluctuate, which is normal in RL training.

Note that we print the average reward, which indicates the proportion of sampled
responses that are correct. For example, with our num_rollouts=4 setting, a reward_avg of
0.5 means that 2 of the 4 rollouts received a positive reward. Similarly, values of 0.25 and
0.0 indicate one or zero correct responses, respectively.

The loss values themselves should not be over-interpreted. Steps with
reward_avg=0.000 produce a near-zero loss because all rollouts receive the same reward,
resulting in vanishing group-relative advantages and little to no gradient signal. Larger loss
magnitudes, such as at step 3 (see row 4 in the output above), simply reflect bigger
relative differences between rollouts and are typical for GRPO-style objectives, especially at
the beginning of training.

Ideally, we want to see two main trends over time:

- 1. The average reward should increase when averaged over many steps,
since the model learns to produce more accurate responses.
- 2. The reasoning accuracy should improve (we evaluate this in the next
section).


The supplementary materials contain a slightly more sophisticated script version that also
contains an option to evaluate the model periodically on subsets of the MATH-500 test
dataset to see if the model is improving on the target task: https://github.com/rasbt/
reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro

###### BATCHED TRAINING

Reinforcement learning can be relatively resource-intensive due to the fact that we
have to generate multiple (long) rollouts via the LLMs for each training step. For this
reason, the implementation does not support batching.

If you have access to multiple GPUs you can use the optional version of this code
with batch and multi-GPU support that can be found in the supplementary materials
at https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_
grpo_scripts_intro, which trains the model faster.

###### 6.12 Loading and evaluating saved model checkpoints

In this final section, as illustrated in figure 6.21, we discuss how to load the saved
checkpoints from the RLVR training run of the previous section and how to evaluate the
model on the MATH-500 dataset.

![image 113](<input (1)_images/imageFile113.png>)

- Figure 6.21 The final step of this chapter discusses how we can load the saved model checkpoints and
evaluate them.


The saved checkpoints can be loaded using the PyTorch state dict approach, as described in

- chapter 2, where model_path points to the corresponding .pth checkpoint file:


model.load_state_dict(torch.load("qwen3-0.6B-rlvr-grpo-step00050.pth"))

###### DOWNLOADING RLVR CHECKPOINTS

If you prefer not to run the GRPO training loop locally due to its runtime cost, you
can instead download pre-trained checkpoints that I have uploaded. These
checkpoints can be fetched either manually or directly from Python using the
provided helper function below, which downloads the selected checkpoint and makes
it available for evaluation or further experimentation:

from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints
download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050")

Note that this checkpoint was created with similar settings as listing 6.17 except that
num_rollouts=4 was increased to num_rollouts=8.

These checkpoints are fully compatible with the model evaluation utilities introduced in

- chapter 3, which allows us to evaluate RL-trained models using the same MATH-500
verification pipeline.


For convenience, you can reuse the evaluation scripts provided in the chapter 3 bonus
materials (https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-
verifier-scripts/evaluate_math500.py) to run evaluations on the MATH-500 dataset by
specifying the desired checkpoint path.

For example, to evaluate the qwen3-0.6B-rlvr-grpo-step00050.pth checkpoint file
you can run the aforementioned script as

uv run evaluate_math500.py \

--dataset_size 500 \
--which_model base \
--checkpoint_path "qwen3-0.6B-rlvr-grpo-step00050.pth"

(If you are not a uv user, replace uv run with python.)

Table 6.1 compares the original base and reasoning models (rows 1 and 2) to the base
model we trained via GRPO in this chapter (rows 3 and 4).

Table 6.1 MATH-500 task accuracy for different base and reasoning models

| |Method|Step|Max<br>tokens|Num<br>rollouts|Accuracy|Average<br>tokens|
|---|---|---|---|---|---|---|
|1|Base model<br>(chapter 3)|-|-|-|15.2%|78.85|
|2|Reasoning model<br>(chapter 3)|-|-|-|48.2%|1369.79|
|3|GRPO (this<br>chapter)|50|256|4|43.2%|560.22|
|4|GRPO (this<br>chapter)|50|512|4|45.6%|579.81|
|5|GRPO (this<br>chapter)|50|512|8|47.4%|586.11|


The accuracy column in table 6.1 refers to the accuracy on the full 500-sample MATH-500
dataset we used in chapter 3. Note that the "Max tokens" column corresponds to the
number of tokens that were allowed per rollout during training. If the number is 512, this
encourages the LLM to provide the final boxed answer within this 512 token limit because
otherwise it will not receive a reward during training.

The evaluation code was executed with a maximum token limit of 2048, allowing the LLM
to generate longer responses during evaluation. The "Average tokens" column averages the
response length during the evaluation on the MATH-500 dataset. What we can see is that,
compared to the reference reasoning model (row 2), our GRPO models generate shorter
responses on average, as expected due to the token-length restriction during training.

Note that the DeepSeek-R1 team observed that responses grow longer over the course
of training as the model begins to write intermediate (chain-of-thought) explanations. So,
to maximize accuracy, it makes sense not to restrict token length too aggressively during
training. Longer token lengths, especially when coupled with multiple rollouts, require more
computational resources, which is why we capped them at a lower limit in this chapter.

As shown in table 6.1, training the reasoning model via GRPO results in 47.4% accuracy
(row 5), which is close to the accuracy of the official Qwen3 0.6B reasoning model on
MATH-500.

The model's accuracy could be further improved by increasing the response length,
sampling more rollouts per prompt, and training for additional steps. The next chapter
introduces practical tips and techniques for monitoring and improving GRPO training
outcomes.

We trained the model for only 50 steps, which already appears sufficient to unlock
reasoning behavior in the base model. In my experiments, training for longer does not
necessarily improve performance and can even reduce accuracy, since the original GRPO
formulation can be unstable over longer runs.

It is also worth keeping in mind that this chapter implemented a simplified GRPO variant
without the KL regularization term. In the next chapter, we return to the full GRPO
objective, add the KL term back in, and discuss common improvements for making training
more stable over longer runs.

Interested readers can find and download additional checkpoints in the range from 50 to
9000 from https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main/
grpo_original_no_kl

- 6.13 Summary


Reinforcement learning (RL) can be used to train LLMs on human
preference labels and verifiable rewards.
RL is typically applied as post-training on top of a pre-trained base model,
and it can be inserted at different stages of an LLM pipeline, including
reasoning training and preference tuning.
RL with human feedback (RLHF) optimizes for human preferences via a
two-stage setup: train a reward model from ranked responses, then use
reward scores to update the LLM.
RL with verifiable rewards (RLVR) simplifies RLHF by replacing learned
reward models with deterministic, automatically computed verifiers (for
example, math answer checking)
We focussed on RLVR for math reasoning.
We used GRPO as the policy optimization algorithm that turns verifier
rewards into parameter updates; because GRPO directly optimizes the
model using sequence-level rewards without requiring a separate value
model, it is particularly convenient.
GRPO is a more resource-friendly alternative to other RL algorithms for
LLMs because it avoids training a separate value model and instead
derives learning signals from comparisons within a group of sampled
rollouts.
A "rollout" refers to a full model answer (completion) for a prompt;
rewards, advantages, and log-probabilities are computed from the rollout
in later steps.
Rewards are computed from a verifier that only grants a reward if the
final answer is both correct and extractable in a required format like
"\boxed{}".
Raw rewards are transformed into advantages by nor malizing each rollout
reward relative to the group mean and standard deviation.
GRPO also relies on sequence-level log-probabilities, which are computed
by summing token log-probabilities over the generated answer tokens.

Sequence log-probabilities, together with the advantages, form the core
policy-gradient objective in GRPO.

The full GRPO loss computation is combined into a single function that
performs rollout sampling, reward computation, advantage calculation,
log-prob computation, and policy-gradient loss calculation.

The surrounding training loop is a standard deep learning loop, with the
key difference being that the loss comes from GRPO rather than
conventional classification losses.

Training is resource-intensive because each step requires generating
multiple, potentially long rollouts, but even short GRPO runs can increase
MATH-500 accuracy from 15% to 47%.

# 7 Improving GRPO for reinforcement learning

This chapter covers

Interpreting training curves and evaluation metrics

Preventing the model from exploiting the reward signal

Extending task-correctness rewards with additional response-formatting rewards

Previously, we implemented the GRPO algorithm for reinforcement learning with verifiable
rewards (RLVR) end to end. Now, as shown in figure 7.1, we pick up from that baseline and
focus on what happens when we run longer training.

![image 114](<input (1)_images/imageFile114.png>)

- Figure 7.1 A mental model of the topics covered in this book. This chapter provides a deeper coverage of the
GRPO algorithm for reinforcement learning with verifiable rewards.


In particular, we will discuss which metrics are worth tracking (beyond reward and
accuracy), how to spot failure modes early, and why training can become unstable even
when the code is "correct." Also, as it turns out, basic GRPO can result in training
instability, this chapter also introduces practical GRPO extensions and fixes used in
reasoning-model training.

##### 7.1 Improving GRPO

After implementing GRPO (group relative policy optimization) in the previous chapter, we
now revisit and analyze the training run more thoroughly. Also, we revisit the KL loss term
that we omitted in the previous chapter and discuss a collection of practical tips and
algorithmic choices that become important in real training runs. These topics are
summarized in the chapter overview in figure 7.2.

![image 115](<input (1)_images/imageFile115.png>)

- Figure 7.2 A chapter overview showing the different topics being covered in this chapter.


The examples in this chapter are based on actual experiments, but the results should be
interpreted with care. To draw strong conclusions about the effect of individual settings,
each experiment would need to be repeated multiple times and the results averaged, since
randomness in sampling and optimization can lead to noticeable variation between runs.
That said, the examples shown here are sufficient to illustrate the main ideas and the
relevant trade-offs.

###### NOTE This chapter focuses on additional technical details and practical considerations when using GRPO. Readers who find the content in this chapter too technical and prefer to move on can do so, as the next chapter does not depend on it.

##### 7.2 Tracking GRPO performance metrics

In chapter 6, we ran a short GRPO training loop and briefly examined the results. For
example, the output of a short run was structured as follows:

- [Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=5.5
- [Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.8
- [Step 3/50] loss=0.3592 reward_avg=0.250 avg_resp_len=7.8
- [Step 4/50] loss=2.7401 reward_avg=0.250 avg_resp_len=56.5
- [Step 5/50] loss=3.3214 reward_avg=0.500 avg_resp_len=251.2
# ...


Let’s pick up from this point and discuss which metrics to track when training with GRPO,
and how they help interpret and debug training behavior.

- 7.2.1 Executing a GRPO training run


The code in chapter 6 is designed to strike a balance between implementing the complete
GRPO pipeline and keeping the implementation compact enough to remain readable. For
convenience, the supplementary materials include an equivalent script (https://github.
com/rasbt/reasoning-from-scratch/blob/main/ch06/02_rlvr_grpo_scripts_intro/rlvr_
grpo_original_no_kl.py), which contains the same code and can be run from a code
terminal (more on this later).

For each GRPO improvement introduced, we will use similar reference scripts from the
supplementary materials to avoid duplicating large code blocks that would otherwise
unnecessarily bloat the chapter and the surrounding discussion.

Using the helper function in listing 7.1, we can download the relevant scripts from the
supplementary materials and save them locally as needed throughout this chapter. This is
to avoid repeating lengthy code passages and to focus on the main changes compared to
the previous GRPO implementation.

- Listing 7.1 Helper function to download supplementary materials


from pathlib import Path
import requests

def download_from_github(rel_path, out=None):

github_raw_base = ( #A
"https://raw.githubusercontent.com/rasbt/"
"reasoning-from-scratch/refs/heads/main/"

)

rel_path = Path(rel_path)
#B
out = Path(out) if out is not None else Path(rel_path.name)

if out.exists(): #C
size_kb = out.stat().st_size / 1e3
print(f"{out}: {size_kb:.1f} KB (cached)")
return out

#D
r = requests.get(github_raw_base + rel_path.as_posix())
r.raise_for_status()

out.write_bytes(r.content)
size_kb = out.stat().st_size / 1e3
print(f"{out}: {size_kb:.1f} KB")

#A Base URL
#B Use URL file name as default output file name
#C Skip download if file already exists locally
#D Download file

Using the helper function, we can download the aforementioned script with the GRPO
training code from chapter 6 as follows:

download_from_github(

"ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py"
)

When executing the code above, you should see the following output:
rlvr_grpo_original_no_kl.py: 13.4 KB

If the downloaded file is much smaller than the size shown above, the download may not
have completed correctly. In that case, first double-check that the URLs don't have any
typos. If the problem persists, please see thetroubleshooting guide (https://github.
com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md) for additional
suggestions.

The resulting script can be run from a terminal as shown in figure 7.3.

![image 116](<input (1)_images/imageFile116.png>)

- Figure 7.3 Output from a GRPO training run using GRPO in a terminal with several training statistics, such as
the loss, average reward, tokens/sec throughput, and average response length.


It's also possible to run the training script in Jupyter notebooks in a code cell by prepending
an "!", i.e., !uv run rlvr_grpo_original_no_kl.py.

If you are not a uv user, replace uv run shown in figure 7.3 with python, that is, python
rlvr_grpo_original_no_kl.py:

python rlvr_grpo_original_no_kl.py \

--steps 500 \

--max_new_tokens 1024

TIP You can add --show_eta to the code execution command above to show a time estimate of
the total runtime of the script specific to your machine.

If you prefer not to run the training code yourself, which is reasonable given its
computational cost, the supplementary materials provide log files from this run. We can
download these as follows:

download_from_github(
"ch07/02_logs/rlvr_grpo_original_no_kl_metrics.txt"

)
download_from_github(

"ch07/02_logs/ch06_rlvr_grpo_original_no_kl_metrics.csv"
)

The first file, ending in .txt, is the plain output file that shows the output statistics similar
to figure 7.3. The second file, ending in .csv, is a comma-separated values file that
provides more structure to this information so that we can more easily extract and plot the
data for further analysis.

- 7.2.2 Inspecting the GRPO training run


To inspect the training run, we plot the results from the previous log files using Matplotlib.
For this, we define the following plotting function that we will use throughout this chapter:

- Listing 7.2 Plotting function to visualize training results from log files


import csv
import matplotlib.pyplot as plt

def moving_average(values, window_fraction=0.25):
#A
window_size = max(1, int(window_fraction * len(values)))
smoothed = []

for i in range(len(values)):
start_idx = max(0, i - window_size + 1)
window_mean = sum(values[start_idx : i + 1]) / (i - start_idx + 1)
smoothed.append(window_mean)

return smoothed

def plot_grpo_metrics(csv_path, columns, save_as=None):
data = {name: {"steps": [], "values": []} for name in columns}

with Path(csv_path).open(newline="") as f: #B
reader = csv.DictReader(f)
for row in reader:

if not row or not row.get("step"):
continue

step = int(row["step"]) #C

for name in columns:
value_str = row.get(name)
if value_str:

data[name]["steps"].append(step)
data[name]["values"].append(float(value_str))

#D
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6, 4))
axes = axes.ravel()

for i, name in enumerate(columns):
steps = data[name]["steps"]
values = data[name]["values"]

if not values: #E
fig.delaxes(axes[i])
continue

if name == "eval_acc": #F
axes[i].bar(steps, values, width=20)

else:
axes[i].plot(steps, values, alpha=0.4)
axes[i].plot(steps, moving_average(values))

axes[i].set_ylabel(name)

for j in (2, 3):
if axes[j] in fig.axes:
axes[j].set_xlabel("Step")

plt.tight_layout()
if save_as is not None:

plt.savefig(save_as)
plt.show()

- #A Smooth the noisy training signal to reveal longer-term trends during training
- #B Open and read CSV log file
- #C Use the training step as the shared x-axis across all metrics
- #D Create a fixed grid so loss, rewards, response length, etc. can be shown side by side
- #E Skip metrics that are not present
- #F Evaluation accuracy as barplot because we don't have data for each step


The .csv file we downloaded has multiple columns. Using the plot_grpo_metrics function
we defined in listing 7.2, we plot the loss, average reward, average response length, and
evaluation accuracy:

plot_grpo_metrics(
"rlvr_grpo_original_no_kl_metrics.csv",
columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]

)

The resulting plot is shown in figure 7.4.

![image 117](<input (1)_images/imageFile117.png>)

- Figure 7.4 The four metrics tracked during the GRPO training run (loss, average reward, average response
length, and evaluation accuracy). The orange centerline represents a moving average over the last 25% of
values, which helps reveal overall trends in the otherwise noisy training signals. The evaluation accuracy is
shown as a bar plot since it is computed only every 50 steps rather than at each step.


A few general observations stand out in the plots shown in figure 7.4. The average
response length should initially increase, ideally together with an improvement in accuracy,
which is largely what we see here, although there is a noticeable decline later in the run
just before step 400. Compared to LLM pre-training, which is outside the scope of this book
and covered in more detail in Build A Large Language Model (From Scratch), the loss value
itself is less informative and mainly serves as a sanity check. Overall, the loss should
remain relatively stable. Some fluctuations are expected, but the larger spikes that appear
halfway through the run are a bit concerning.

The average reward should also increase over time. In principle, an average reward of
1.00 means that all sampled responses are correct, which is desirable, but it also means
that the training signal has disappeared. At that point, further training is unlikely to be
useful, and stopping early can save us time and resources.

Finally, performance on the external target task, here measured by MATH-500 accuracy,
should improve. In this run, accuracy increases at first but then begins to decline, which
points to problems and instabilities in the training process.

In summary, the training run shows fast gains early on and is followed by diminishing
returns after approximately fifty steps. One likely reason is that the underlying algorithm is
not particularly stable over longer runs. Another possible explanation could be that later
training examples are more difficult, but this would not account for decline in evaluation
accuracy, which is computed from the same 500 MATH-500 test samples every fifty steps.

Note that the main goal of this section is to introduce, analyze, and discuss various
training metrics. This section focused on some basic ones, and in the next section, we will
extend this list and look at additional metrics that are commonly tracked during GRPO
training.

###### CALCULATING THE EVALUATION ACCURACY

Evaluation accuracy (eval_acc), here measured on the MATH-500 benchmark, is not
tracked by default during training. It can be computed periodically by adding the
setting --eval_on_checkpoint 500 when running the training script. This is not
recommended as this significantly slows down training. Alternatively, it can be
calculated separately after training using the evaluate_math500.py script introduced
in chapter 3:

download_from_github(

"ch03/02_math500-verifier-scripts/evaluate_math500.py"
)

The evaluation can then be run as follows on a given checkpoint:

uv run evaluate_math500.py \

--dataset_size 500 \

--checkpoint_path \

"checkpoints/rlvr_grpo_original_no_kl/\
qwen3-0.6B-rlvr-grpo-step00050.pth"

In the run discussed above, evaluation accuracy is already included in the log file. (If
you are not a uv user, replace uv run with python.)

##### 7.3 Tracking more advanced GRPO performance metrics

Beyond the small set of basic metrics for monitoring GRPO training runs, there are several
additional ones that can be useful for understanding training dynamics, as illustrated in
figure 7.5.

![image 118](<input (1)_images/imageFile118.png>)

###### Figure 7.5 After analyzing basic GRPO training metrics, we now add more advanced metrics to analyze the training run.

Two examples of the more advanced metrics mentioned in figure 7.5 are the rollout
advantages already computed in the compute_grpo_loss function from chapter 6, and
the entropy of the generated sequences.

- 7.3.1 Advantage tracking


As part of the GRPO algorithm, we compute so-called advantages, as discussed in chapter

###### 6, and shown in figure 7.6. Beyond their role in the loss computation, these values are also useful for analyzing and understanding training dynamics.

![image 119](<input (1)_images/imageFile119.png>)

- Figure 7.6 GRPO overview figure from chapter 6. The advantages are shown in step 3.


In particular, we compute two summary statistics derived from the advantages shown in
figure 7.6, namely, their average value (the sample mean) and their standard deviation:

- Listing 7.3 Computing advantage statistics


import torch

def compute_advantage_stats(rewards_list):

- #A
rewards = torch.tensor(rewards_list)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
- #B
adv_avg = advantages.mean().item()
adv_std = advantages.std().item()


return advantages, adv_avg, adv_std

#A The rewards and advantages below are what we already compute in GRPO
#B Now, we also compute the average (mean) and standard deviation (std)

As a simple illustration, consider a small sample of four reward values, similar to those
shown in the previous figure 7.6.

adv, adv_avg, adv_std = compute_advantage_stats([1., 1., 0., 0.])
print(f"Advantages: {adv}")
print(f"Advantage mean = {adv_avg:.4f}, std = {adv_std:.4f}")

The output is:

Advantages: tensor([ 0.8659, 0.8659, -0.8659, -0.8659])
Advantage mean = 0.0000, std = 0.9998

Because of how advantages are computed in GRPO, their mean is always zero. In practice,
this makes the mean mostly a sanity check that the implementation behaves as expected.

The standard deviation is more informative. Values close to 1 indicate a well-scaled
gradient signal and are usually associated with stable updates. Very small values point to a
vanishing learning signal, which often happens when rewards collapse. Very large values
indicate overly spiky updates that can destabilize training.

An extreme case occurs when all rewards are identical. When all advantages are
identical, which is indicated by a standard deviation that is zero, the policy gradient
becomes zero and no model weight update occurs. In other words, the model fails to learn
to improve in these cases. For example, consider the following scenario:

adv, adv_avg, adv_std = compute_advantage_stats([0., 0., 0., 0.])
print(f"Advantages: {adv}")
print(f"Advantage mean = {adv_avg:.4f}, std = {adv_std:.4f}")

which prints the following outputs:

Advantages: tensor([0., 0., 0., 0.])
Advantage mean = 0.0000, std = 0.0000

And the outputs are similar if we replace the all-zero rewards in

- compute_advantage_stats([0., 0., 0., 0.]) with all-one rewards
- compute_advantage_stats([1., 1., 1., 1.]).
In practice, it is best to consider the advantage statistics together with the average


reward. As mentioned earlier, an average reward of 1.0 is actually a desirable outcome,
even though it means that the training signal has disappeared because the model is
answering everything correctly. At this point, it usually makes sense to stop training or to
switch to more challenging examples.

- 7.3.2 Entropy tracking


Before we track and analyze the advantage statistics introduced in the previous section, we
introduce another metric to track, entropy.

Entropy measures how uncertain the model is when generating the next token. High
entropy means the probabilities are spread across many possible tokens, which encourages
exploration. Low entropy means most of the probability is concentrated on a single token,
which makes the model increasingly deterministic. It also potentially signals training
collapse, where the model stops exploring and keeps producing the same outputs.

Before computing entropy, it is useful to briefly revisit how we calculated log-probability
values (logprobs) in chapter 5, as summarized in figure 7.7.

![image 120](<input (1)_images/imageFile120.png>)

- Figure 7.7 Log-probability (logprob) computation of a single token ("this") in the LLM's generated answer.
The LLM returns the logits of the token, which are then converted to softmax probability values via
torch.softmax() or logprob values via torch.log_softmax().


In figure 7.7, instead of using real logits created by prompting the base model, we use
example logits assuming a vocabulary size of 7 (otherwise, with the original 151-thousand-
token vocabulary, it would be impossible to visualize this concept in a plot):

logits = torch.tensor([

0.6667, -2.0000, 1.3333, -0.0000, -0.6667, 2.0000, -1.3333
])

The following code implements the calculation shown in the previous figure 7.7:

logprobs = torch.log_softmax(logits, dim=-1)
print("All token logprobs:", logprobs)

selected_token = torch.argmax(logprobs)
selected = logprobs[selected_token]
print("Selected token ID:", selected_token)
print("Selected token logprob:", selected)

The outputs are:

All token logprobs: tensor([-2.0442, -4.7109, -1.3776,

-2.7109, -3.3776, -0.7109, -4.0442])
Selected token ID: tensor(5)
Selected token logprob: tensor(-0.7109)

In short, the torch.log_softmax() function computes all log-probabilities (logprobs), the
torch.argmax() returns the index (token ID) of the largest logprob (here: 5), and the
logprobs[selected_token] returns the logprob value of that token ID (-0.7109).

Entropy is closely related to the log-probabilities. We compute it by multiplying each
probability by its log-probability and then summing these products, as illustrated in figure
7.8.

![image 121](<input (1)_images/imageFile121.png>)

- Figure 7.8 The entropy term is calculated by multiplying the token probabilities with the token logprobs.


- Figure 7.8 shows that the entropy is computed by multiplying probability and logprob
values. In principle, we could also track simpler quantities than entropy during training,
such as the sum of probabilities or logprobs, but entropy is a widely used and easy-to-
interpret measure of uncertainty in a probability distribution.


In the previous figure, we see an entropy of 1.37. As a rough rule of thumb:

Very low entropy (≈ 0-0.5) means that one token dominates the
distribution. The model is highly confident and behaves almost
deterministically.

Moderate entropy (≈ 1-2) means the probabilities are shared across a
reasonably small set of tokens, which is typical during stable training.

High entropy (≫ 2, approaching log(vocabulary size); here: log(7) =
1.9459) means the probabilities are spread across many tokens. In this
case, the model is highly uncertain and behaves close to random.

We can calculate the entropy as follows in code:

- Listing 7.4 Calculating entropy


probs = torch.softmax(logits, dim=-1)
logprobs = torch.log_softmax(logits, dim=-1)
entropy = torch.sum(-(probs * logprobs))

print("Probs:", probs)
print("Entropy:", entropy)

The resulting outputs are:

Probs: tensor([0.1295, 0.0090, 0.2522, 0.0665, 0.0341, 0.4912, 0.0175])
Entropy: tensor(1.3700)

As mentioned earlier, entropy quantifies how spread out the model's probability distribution
is over the vocabulary. In this specific example, the entropy of 1.37 indicates a moderate
level of uncertainty. One token (with probability 0.4912) clearly dominates, but other
tokens still have meaningful probabilities, too.

Note that in the previous code listing, we compute the probs and logprobs separately
using torch.softmax() and torch.log_softmax(), respectively. The
torch.log_softmax() function combines two separate function calls:
torch.log(torch.softmax()). And the torch.exp() function is the inverse of
torch.log(). This means, if we only had the logprobs, we could also calculate the probs
by applying torch.exp(logprobs), as follows:

print("Probs:", torch.exp(logprobs))

This outputs the same probs values as before:

Probs: tensor([0.1295, 0.0090, 0.2522, 0.0665, 0.0341, 0.4912, 0.0175])

Building on this idea, we can extend the sequence_logprob function from the previous
chapter to a sequence_logprob_and_entropy function that returns both the logprobs and
the entropy.

- Listing 7.5 Calculating sequence logprob and average entropy


def sequence_logprob_and_entropy(model, token_ids, prompt_len):

- #A
logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
logprobs = torch.log_softmax(logits, dim=-1)

targets = token_ids[1:]
selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)

- #B
selected_answer_logprobs = selected[prompt_len - 1:]
logp_all_steps = torch.sum(selected_answer_logprobs)
- #C
all_answer_logprobs = logprobs[:-1][prompt_len - 1:]
if all_answer_logprobs.numel() == 0: #D


entropy_all_steps = logp_all_steps.new_tensor(0.0)
else:

- #E
all_answer_probs = torch.exp(all_answer_logprobs)
- #F
plogp = all_answer_probs * all_answer_logprobs
- #G
step_entropy = -torch.sum(plogp, dim=-1)
- #H
entropy_all_steps = torch.mean(step_entropy)


return logp_all_steps, entropy_all_steps

- #A Code below is identical to the sequence_logprob code in chapter 5
- #B Logprob of the generated answer tokens (sum over answer steps)
- #C Below is the new code that calculates the entropy
- #D A safeguard that is triggered if the model immediately returns EOS token
- #E Convert logprob to prob
- #F Calculate elementwise p * log p
- #G Calculate entropy for single tokens (generation steps) by summing over all plogp values in the vocabulary
- #H Average over all answer steps to calculate average entropy for a given LLM answer


To compute and track the entropy during training, we can use this
sequence_logprob_and_entropy function inside compute_grpo_loss in place of the
sequence_logprob function.

Note that the previous figure illustrated the entropy computation for a single token in
the rollout (step_entropy), whereas sequence_logprob_and_entropy returns the average
entropy over all answer tokens, that is, entropy_all_steps =
torch.mean(step_entropy). For example, for the answer "this is the LLM response",
the step entropy refers to the entropy at a single token position (for instance at "this"),
while the average entropy is the mean taken across all answer tokens in "this is the LLM
response".

With this modified sequence_logprob_and_entropy function, we can now use entropy
as a diagnostic for the model's generation behavior during training. By tracking the average
entropy of the generated answer tokens, we can monitor whether the model behaves
randomly (high entropy), remains exploratory (moderate entropy), or becomes overly
confident and deterministic (low entropy).

In particular, we expect entropy to gradually decrease as the model becomes more
confident. A sudden collapse to very low entropy can be a warning sign for unstable
training.

- 7.3.3 Plotting additional GRPO metrics


Let's build on the earlier computation of advantage statistics and entropy by analyzing
these quantities for a concrete training run. To do this, we could update
compute_grpo_loss to use sequence_logprob_and_entropy, and then print and log the
entropy in train_rlvr_grpo, which calls compute_grpo_loss internally. To avoid
duplicating a long code listing here, the full modified version is provided in the
supplementary materials, and we can download it as follows:

download_from_github(

"ch07/03_rlvr_grpo_scripts_advanced/7_3_plus_tracking.py"
)

The code can be run in the same way as the training script we used earlier:

- uv run 7_3_plus_tracking.py \


--steps 500 \

--max_new_tokens 1024

Since training takes a long time, we can download the resulting CSV log file and plot the
new advantage statistics and entropy as follows:

download_from_github(
"ch07/02_logs/7_3_plus_tracking_metrics.csv"

)
plot_grpo_metrics(

"7_3_plus_tracking_metrics.csv",
columns=["reward_avg", "adv_avg", "adv_std", "entropy_avg"]

)

Note that we already looked at the average reward earlier, but we include it again here for
comparison purposes. The resulting plots are shown in figure 7.9. Here, moderate entropy
should still be understood as exploratory behavior, just less random than the later high-
entropy regime.

![image 122](<input (1)_images/imageFile122.png>)

- Figure 7.9 Visualizing advantage statistics and entropy tracked during the GRPO training run (next to the
average reward, which we tracked previously).


Let's begin by analyzing the advantage shown in figure 7.9. As expected with GRPO-style
normalization, the average advantage stays at zero throughout training. Since advantages
are computed relative to the group mean, they sum to zero by design. As mentioned
earlier, in practice, this metric mainly serves as a sanity check. If the advantage averages
were to drift away from zero, that would point to a bug or a normalization issue.

The standard deviation of the advantages is more informative. Early in training, we see
relatively high variance, which indicates that the model is producing rollouts with a wide
range of quality. Over time, the advantage standard deviation gradually decreases and
stabilizes, which means that the rollouts become more similar in quality. The key point is
that as long as the advantage standard deviation remains nonzero and reasonably stable
(meaning the value doesn’t vary much), there is still a usable learning signal and training
happening.

Next, let's look at the entropy. Early in training, the entropy is relatively low and fairly
flat, which indicates that the model behaves in a largely deterministic way. In practice,
increasing the sampling temperature would lead to more diverse rollouts, although it does
not change the underlying entropy of the model itself.

Later in training, after roughly step 200, the entropy increases quite noticeably. This
means that the next-token probabilities are more spread out and the model behaves more
randomly. Very low entropy can also be a sign of collapse, where the model repeatedly
produces the same or very similar outputs. In this run, considering the entropy together
with the increasing average reward and the non-vanishing advantage standard deviation
suggests still somewhat healthy exploration rather than collapse. This is also consistent
with the earlier MATH-500 evaluation accuracy that stays in the 30%-40% range, which is
not great but not near zero.

The main takeaway here is that each metric tells a slightly different part of the story,
and they are most useful when considered together and in context.

##### 7.4 Stabilizing sequence-level GRPO using clipped policy ratios

So far, we have mainly focused on analyzing the GRPO training results. Next, we start
making additions to the GRPO algorithm itself. The version we have used so far is a
simplified form of GRPO, and in this section we introduce a clipped policy ratio in the GRPO
loss, as illustrated in the overview in figure 7.10.

![image 123](<input (1)_images/imageFile123.png>)

- Figure 7.10 After plotting basic and advanced GRPO training metrics, we now modify the GRPO algorithm and
add clipped policy ratios.


The clipped policy ratios mentioned in figure 7.10 help limit overly large model weight
updates and make training more stable, especially over longer runs. Ideally, we want to see
that the model performance doesn't noticeably decline as we have seen previously.

- 7.4.1 Computing clipped policy ratios


The clipped policy ratio measures how much the current policy, that is, the LLM being
trained, has changed relative to an earlier version of itself. Concretely, it compares
sequence logprobs computed before an update step with those computed after the update.
You can think of it as asking: "If the LLM previously assigned a certain likelihood to this
answer, how much more or less likely does it consider the same answer after we adjusted
its weights?"

In the GRPO pipeline shown in figure 7.11, this corresponds to comparing the logprobs
from step 4, which are computed using the old weight parameters, with the logprobs
produced by the updated model (the model is updated via step 6).

![image 124](<input (1)_images/imageFile124.png>)

- Figure 7.11 GRPO overview figure from chapter 6. We now use the sequence logprobs from step 4 to compute
policy ratios and clipped policy ratios.


- Figure 7.11 shows the logprobs that are computed via the model that is being trained. The
model is then updated via step 6. The policy ratios are computed from logprobs of the
model in two different states: before and after a weight update. This will become clearer
once we implement this concept in code. But first, to recap, in chapter 6 we computed the
policy gradient loss as follows:


- Listing 7.6 Compute policy gradient


# ... #A

- #B
rewards = torch.tensor([1., 1., 0., 0.])
- #C
logprobs = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677])


advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

pg_loss = -(advantages.detach() * logprobs).mean()
print("Policy gradient loss:", pg_loss)

- #A Code to compute rollouts omitted for brevity
- #B Compute rewards
- #C Compute sequence logprobs


The resulting policy gradient value, also as shown in figure 7.11, is -2.5764.
Next, we compute the policy ratio and clipped policy ratio, as shown in figure 7.12.

![image 125](<input (1)_images/imageFile125.png>)

- Figure 7.12 Calculating the policy ratio (ratio) and clipped policy ratio (clipped ratio) added to the GRPO from
"new" and "old" logprobs.


The policy ratio and clipped policy ratio are computed by comparing log-probabilities from a
previous version of the policy ("old" logprobs) with those from the current policy ("new"
logprobs). The mathematical derivation is outside the scope of this chapter, but interested
readers can find more details in the Proximal Policy Optimization Algorithms paper
(https://arxiv.org/pdf/1707.06347).

For the following code-based illustration of the calculation shown in figure 7.12, we reuse
the logprobs from the example above as old_logps, assume that a model update has
taken place, and then compute the corresponding new_logps using the updated model. In
practice, both would be computed using the same prompt to ensure a fair, apples-to-apples
comparison between the old and current policy:

- Listing 7.7 Compute policy ratios and clipped policy ratios:


new_logps = logprobs

#A
old_logps = torch.tensor([

- -10.9243, # -7.9243
- -20.3546, # -20.1546
- -14.6130, # -16.6130
- -23.3677, # -23.3677


])

log_ratio = new_logps - old_logps
ratio = torch.exp(log_ratio)
clip_eps = 10.0
clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

print("Ratio: ", ratio)
print("Clipped ratio:", clipped_ratio)

#A The values of the new_logps are shown side by side in the comments

The resulting unclipped and clipped policy ratios are shown below:

Ratio: tensor([20.0855, 1.2214, 0.1353, 1.0000])
Clipped ratio: tensor([11.0000, 1.2214, 0.1353, 1.0000])

The ratio tells us how different the old and new logprobs are. If they are identical, the ratio
is 1.0.

The clipped_ratio, which we use to compute a clipped version of the policy gradient
loss, limits how far the new policy is allowed to move away from the old one in a single
update. Concretely, if the new model suddenly assigns a much higher or much lower
probability to a rollout compared to the old model, the raw ratio can become very large or
very small. Without clipping, this would scale the advantage term substantially, which can
lead to a very large gradient step that potentially destabilizes the training.

Using the clip_eps parameter in the previous code example, in practice, it is common
to clamp the ratio to the range 1 ± clip_eps. For example, DeepSeek-R1 used clip_eps =
10, which corresponds to very weak clipping, while other RL training setups (for example,
reinforcement learning with human feedback using the PPO algorithm) often use much
smaller values, such as 0.1, which results in aggressive clipping and thus substantially
smaller per-step policy changes.

As we can see, based on this very generous clip_eps value, only the first value is
clipped from 20.0855 down to 11.0000.

###### NOTE The “eps” in clip_eps is short for epsilon (ε), a Greek letter commonly used in mathematics to denote a small positive quantity.

Next, we apply the ratio and clipped_ratio in the loss computation. Previously, we
multiplied the advantages directly by the logprobs. Now, we instead scale the advantages
by the policy ratios and use the clipped objective to limit how much each rollout can
influence the update:

- Listing 7.8 Compute clipped policy gradient loss


adv = advantages.detach() #A

unclipped = ratio * adv
clipped = clipped_ratio * adv

obj = torch.where(
adv >= 0, #B
torch.minimum(unclipped, clipped), #C
torch.maximum(unclipped, clipped), #D

)

clipped_pg_loss = -torch.mean(obj)
policy_ratio = torch.mean(ratio)

print("Clipped policy gradient loss:", clipped_pg_loss)
print("Policy ratio:", policy_ratio)

- #A Treat advantages as fixed learning signals (no backprop through rewards)
- #B Choose the more conservative update depending on the advantage signal
- #C Cap large positive updates
- #D Cap large negative updates


The resulting clipped policy gradient loss is -2.3998, which is slightly lower than the regular
policy gradient loss we computed previously in this section (-2.5764). This small difference
makes sense, because in our example, only one of the policy ratios was effectively clipped
via the generous clip_eps=10.0 ratio. In general, this clipping can prevent overly
aggressive updates that could destabilize training. Large, aggressive updates can move the
policy too far in a single step, which can cause large shifts in token probabilities and then
lead to reward crashes and entropy collapses, or other related problems.

In addition, we also compute the average policy ratio, policy_ratio =
torch.mean(ratio), which we can track and report in the training run. Here, the result is a
relatively large 5.6106. The large policy ratio in this example means that the updated policy
(model) assigns a much higher probability to the sampled tokens than the previous policy.

- 7.4.2 Training with clipped policy ratios


We can apply the discussed clipped policy ratio and policy loss computations from listings
7.7 and 7.8 to see if they help stabilize the training.

The modified code can be downloaded from the supplementary materials as follows:

download_from_github(

"ch07/03_rlvr_grpo_scripts_advanced/7_4_plus_clip_ratio.py"
)

And we can run it similar to before:

- uv run 7_4_plus_clip_ratio.py \


--steps 500 \

--max_new_tokens 1024

As discussed in the previous section, the policy ratio compares logprobs computed under
the current policy with those from a previous version of the policy.

In practice, this means that we first need a batch of sampled responses whose
old_logps are measured under a fixed reference policy. We can then compare those same
responses against the current model, which changes during optimization, to form the
clipped policy-ratio update. DeepSeek-R1 does this at large scale by generating a large
rollout pool (8,192 responses) once, holding those responses fixed, and then walking
through them in 16 minibatches. At a high level, the process is as follows:

- 1. Make a copy of the current model as the reference policy
- 2. Sample a big rollout pool using the reference policy
- 3. Split those fixed rollouts into minibatches
- 4. For each minibatch:
- 5. Compute old_logps under the reference model
- 6. Compute new_logps under the current model (which changes after each
minibatch update)


- 7. Update the current model
- 8. Update the reference model


Due to resource constraints, we cannot generate as many rollouts and minibatches as
DeepSeek-R1 does, so we use a smaller approximation. Instead of generating one huge
fixed rollout pool and slicing it into minibatches, we generate a small batch of rollouts,
update the model on that batch, then repeat the process with a fresh batch. So, the
procedure in the 7_4_plus_clip_ratio.py script is as follows:

- 1. Make a copy of the current model as the reference policy
- 2. Sample 8 rollouts using the reference policy
- 3. Then, we:
- 4. Compute old_logps under the reference model
- 5. Compute new_logps under the current model
- 6. Update the current model
- 7. Repeat steps 2 and 3 (by default one more time) with different rollouts,
instead of using minibatches
- 8. Update the reference model


In short, DeepSeek-R1 generates a huge rollout pool once and then reuses it across
minibatches before refreshing the reference model. In our code, we mimic the same idea
more cheaply by generating a small batch of rollouts multiple times instead.

A log file of the 7_4_plus_clip_ratio.py script training run can be downloaded and
visualized as follows:

download_from_github(
"ch07/02_logs/7_4_plus_clip_ratio_metrics.csv"

)
plot_grpo_metrics(

"7_4_plus_clip_ratio_metrics.csv",
columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]

)

- Figure 7.13 shows the resulting plot.


![image 126](<input (1)_images/imageFile126.png>)

Figure 7.13 Selected metrics from a GRPO training run using clipped policy ratios.

Compared to the previous training runs, the training with clipped policy ratios in figure 7.13
is indeed more stable as there is no visible drop in the average reward or evaluation
accuracy around the 400-step mark. Optionally, we can also plot and inspect the other
metrics, such as "policy_ratio", "adv_avg", "adv_std", "entropy_avg", which is left
as an exercise for the reader.

###### 7.5 Controlling how much the model changes with a KL term

Previously, we skipped the KL loss term of the GRPO algorithm to keep the presentation
manageable. To complete the GRPO algorithm, we now add the KL loss term as shown in
figure 7.14.

###### NOTE Readers familiar with reinforcement learning may recognize that what we implemented up to this point is essentially REINFORCEwith group-normalized advantages.

![image 127](<input (1)_images/imageFile127.png>)

Figure 7.14 Implementing a KL loss term, which is a part of the original GRPO algorithm.

KL is short for Kullback–Leibler divergence and measures how much the current policy, that
is, the LLM being trained, deviates from a reference policy, typically the original model at
the start of training.

The KL term acts as a constraint (technically called a regularizer) that discourages overly
large updates and keeps the model close to its original behavior to prevent drastic or
unstable changes. This is closely related to the clipped policy ratios we introduced earlier.
Using clipped policy ratios, we limited how large individual update steps can be. The KL
term is another mechanism to control how much the model can change. While the clipped
policy ratios are more focused on limiting the changes between each step, the KL term is
often used to control the changes more long term over the training trajectory.

- 7.5.1 Implementing the KL loss term


Similar to when we computed the clipped policy ratio, calculating the KL term involves two
sets of logprobs we compare, as shown in figure 7.15.

![image 128](<input (1)_images/imageFile128.png>)

Figure 7.15 Overview of the GRPO algorithm with the KL loss term calculation added to the right.

- Figure 7.15 shows the KL loss term computation as part of the GRPO algorithm. The KL loss
term calculations involve a comparison between logprobs and reference logprobs. It is
computed by summing the differences between these corresponding logprobs, which A
small KL loss term value indicates that the current model and the reference model output
relatively similar logprobs, meaning the current model has not changed much compared to
the reference model. This is generally desirable for stability, as it prevents large, abrupt
shifts in behavior. If the KL term remains too small for too long, it may also indicate that
the learning here is limited.


The resulting KL loss term is then added to the policy gradient loss to compute the total
loss, so weight updates that strongly increase the difference (divergence) from the
reference model (even if they result in a high reward) are penalized during training.

In the clipped policy ratio section, we replaced the reference model in each iteration. For
the KL loss term, the reference model, which is used to compute the reference logprobs, is
the original model at the beginning of the training and is either kept fixed or replaced
relatively rarely (in the case of DeepSeek-R1, every 400 steps).

The following code illustrated how the compute_grpo_loss function can be modified to
include this KL loss term. To avoid code duplication, the code in listing 7.9 does not show
the fully modified compute_grpo_loss_with_kl, which focuses only on the changes we
have to make to the compute_grpo_loss function.

The new additions are marked with comments or are located inside the if kl_coeff
blocks.

- Listing 7.9 Adding a KL term to the GRPO loss computation


import copy

kl_coeff = 0.0 #A

if kl_coeff: #B
ref_model = copy.deepcopy(model).to(device)
ref_model.eval()
for p in ref_model.parameters():

p.requires_grad = False
else:

ref_model = None

def compute_grpo_loss_with_kl(
model,
ref_model, #C
# ...
kl_coeff=0.02, #D

):

roll_logps, roll_ref_logps, roll_rewards, samples = [], [], [], []
# ...

for _ in range(num_rollouts):
token_ids, prompt_len, text = sample_response(
# ...
)

if kl_coeff: #E
with torch.no_grad():
ref_logp = sequence_logprob(

ref_model, token_ids, prompt_len
)

else:
ref_logp = None

reward = reward_rlvr(text, example["answer"])

roll_rewards.append(reward)
roll_logps.append(logp)
if kl_coeff:

roll_ref_logps.append(ref_logp)

# ...

rewards = torch.tensor(roll_rewards, device=device)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

logps = torch.stack(roll_logps)
if kl_coeff:

ref_logps = torch.stack(roll_ref_logps).detach()

pg_loss = -(advantages.detach() * logps).mean()

if kl_coeff: #F

kl_loss = kl_coeff * torch.mean(logps - ref_logps)
else:

kl_loss = torch.tensor(0.0, device=logps.device)

loss = pg_loss + kl_loss #G

#A kl_coeff = 0.0 deactivates the KL term and the code is similar to before
#B Make copy of the original model and disable gradients so it isn't updated
#C New: Pass reference model
#D New: Specify KL strength
#E Compute logprobs with reference model
#F Compute KL loss term
#G New: Add KL loss term to policy gradient loss

Note that adding the KL loss term increases resource usage, since we now need to keep an
additional copy of the original model. The strength of this penalty is controlled by
kl_coeff. Larger kl_coeff values add a stronger constraint on how far the current policy
is allowed to change from the reference model.

The setting kl_coeff = 0.0 at the top of listing 7.9 is only added so that if we were to
run this abbreviated code snippet in a code environment, it wouldn't crash. In practice,
kl_coeff is typically a small value between 0.001 and 0.05.

- 7.5.2 Training with a KL loss term


The following script from the supplementary materials implements the KL loss term code
modification we discussed previously:

download_from_github(

"ch07/03_rlvr_grpo_scripts_advanced/7_5_plus_kl.py"
)

Similar to before, we can run it as follows:

- uv run 7_5_plus_kl.py \


--steps 500 \

--max_new_tokens 1024

Let's take a look at the log files from a training run that was executed with the settings
shown above:

download_from_github(
"ch07/02_logs/"
"7_5_plus_kl_metrics.csv"

)
plot_grpo_metrics(

"7_5_plus_kl_metrics.csv",
columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]

)

The resulting plots are shown in figure 7.16.

![image 129](<input (1)_images/imageFile129.png>)

- Figure 7.16 Selected metrics from a GRPO training run after adding a KL loss term. Here, loss (in the upper
left) refers to the total GRPO loss, that is, the sum of the policy-gradient loss and the KL loss term.


We can see that the first fifty steps improve the model performance noticeably from slightly
above 15% accuracy to approximately 40% accuracy. We then see a total collapse where
the loss values explode and the average reward goes towards 0. This is accompanied by a
crash in the evaluation accuracy that also goes towards 0%.

On manual inspection of generated responses, the model eventually started producing
nonsensical text and random tokens, such as "framework.raises Self profess(import".
There are a couple of likely reasons for this behavior.

First, the KL term is computed using summed logprobs over the answer tokens, that is,
kl_loss = kl_coeff * mean(new_logps - ref_logps). As a result, longer sequences
naturally lead to larger KL values. This can unintentionally encourage very long outputs,
which we also observe in the growing response length, and can destabilize training.

Second, once rewards collapse to 0 around step 200, the KL term becomes the only
remaining source of gradient signal. When all rewards are zero, the advantages are zero as
well, so the policy gradient loss no longer contributes any gradient. The KL term is still
active and starts to dominate the updates. This means the KL loss term can push the model
toward higher entropy and near-uniform token distributions, which explains both the
increasingly random outputs and the drop to 0.0% evaluation accuracy. If you want, you
can plot the other evaluation metrics, for instance, "policy_ratio", "adv_avg",
"adv_std", "entropy_avg", and confirm that the entropy becomes very large.

While the KL term is a standard part of the GRPO algorithm, which is why we introduce it
here, several recent works report that models can train better without it. Examples include
Dr. GRPO, Olmo 3, and DeepSeek-V3.2, all of which observe improved stability or
performance when the KL term is reduced or omitted (see appendix A for more detailed
references).

###### IMPROVING TRAINING STABILITY WITH A KL TERM

Alternatively, instead of removing the KL term, there are several ways to improve
training stability. First, we can reduce kl_coeff from 0.02 to a smaller value (for
example, 0.001) to weaken the KL penalty, although in my experiments this was not
sufficient.

Since we saw that response length keeps growing until it hits the maximum,
another option is to length-normalize the sequence logprobs by dividing by the
number of generated tokens, which approximates average per-token logprobs. This
could reduce the incentive to increase response length to accumulate larger absolute
logprob differences.

A third option is to use a reweighted KL term based on importance sampling, as
proposed by DeepSeek-V3.2. Instead of

kl_loss = kl_coeff * torch.mean(logps - ref_logps)
we use
kl_loss = (

kl_coeff * torch.mean(torch.exp(new_logps - old_logps)

* (new_logps - ref_logps)
)

Here, the importance weight torch.exp(new_logps - old_logps) corrects for
rollouts being generated by the old policy, and it downweights samples that the
current policy would rarely produce and upweights those it would.

Finally, we could add a small format reward to prevent advantages from collapsing
to zero (we will revisit format rewards in the next section).

That said, for math data, a common recommendation is to omit the KL term
altogether (e.g., Dr. GRPO, Olmo 3, and DeepSeek-V3.2), which also simplifies the
code and reduces resource requirements since we don't have to keep an additional
reference model in memory.

##### 7.6 Adding an explicit format reward

So far, we only used one type of verifiable reward, an answer-correctness reward, when
training the model. In practice, this is sufficient to train reasoning models with RLVR. It is
also common to use one or more auxiliary rewards, for example, a format reward that
checks whether the generated answer follows certain formatting guidelines (figure 7.17).

![image 130](<input (1)_images/imageFile130.png>)

- Figure 7.17 Implementing a format reward that encourages the model to generate <think>...</think>
tokens


Specifically, we will focus on a format reward that encourages the model to use so-called
<think>...</think> tokens, as shown in figure 7.17.

- 7.6.1 Using <think> tokens


Some reasoning models, such as DeepSeek-R1 and Qwen3, use special <think> and
</think> tokens. These <think> tokens are ordinary text tokens that mark the beginning
and end of the model's intermediate reasoning. In practice, we can prompt the model to
write its step-by-step reasoning inside <think> ... </think> and then provide the final
answer outside these tags. This entire approach is optional, but it makes the output
structure explicit and helps separate reasoning from the final result.

To start illustrating this concept, we load the tokenizer of the base model we used
earlier.

- Listing 7.10 Loading the base model tokenizer


- from reasoning_from_scratch.ch02 import get_device
- from reasoning_from_scratch.ch03 import load_model_and_tokenizer


device = get_device()
model, tokenizer_base = load_model_and_tokenizer(

which_model="base",
device=device,
use_compile=False

)

Next, let's use the tokenizer to encode <think>:

print(tokenizer_base.encode("<think>"))

The output is [13708, 766, 29], which means that the tokenizer breaks up the <think>
token into several subword tokens, which indicates that the <think> token is not part of
the tokenizer's vocabulary.

Nowadays, when developing tokenizers and creating the embedding layers of LLMs, the
developers often leave some unused placeholder token IDs that can be used for specific
purposes when fine-tuning a model. In this case, the Qwen3 base model tokenizer has
some empty placeholder slots that we can use to add the <think>:

tokenizer_base._tok.add_special_tokens(

["<tool_response>", "</tool_response>", "<think>", "</think>"]
)

Note that we also added the <tool_response> and </tool_response> tokens above to
match it up with the tokenizer of the Qwen3 reasoning variant, which we will discuss
shortly. But first, let's check if the modified base tokenizer now supports the newly added
<think> and </think> tokens:

print(tokenizer_base.encode("<think>"))
print(tokenizer_base.encode("</think>"))

The outputs are single tokens [151667] and [151668], which indicates that the <think>
and </think> tokens are now part of the vocabulary and are no longer broken up into
subword tokens.

These new tokens would also be supported by the base model, which supports token IDs
from 0 to 151935 as it has a vocabulary size of 151936, which we can confirm by printing
the size of the embedding layer:

print("Vocabulary size:", model.tok_emb.weight.shape[0])

Alternatively, instead of modifying the base tokenizer, the tokenizer of the reasoning model
variant already supports these <think> tokens natively, which means that we don't have to
modify the tokenizer manually. For instance, similar to what we have done in chapter 2, we
can load the tokenizer separately:

- Listing 7.11 Loading the reasoning model tokenizer


from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from reasoning_from_scratch.qwen3 import download_qwen3_small

download_qwen3_small(

kind="reasoning", tokenizer_only=True, out_dir="qwen3"
)

tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

Now, let's use the reasoning tokenizer (tokenizer) instead of the tokenizer_base to
encode the token IDs similar to what we have done earlier:

print(tokenizer.encode("<think>"))
print(tokenizer.encode("</think>"))

Similar to when we used tokenizer_base, this returns [151667] and [151668].

Now that we have a tokenizer that supports these <think> </think> tokens, we can
then develop a reward function that checks whether the LLM uses these tokens and
provides a non-zero reward if it does, as illustrated in figure 7.18.

![image 131](<input (1)_images/imageFile131.png>)

- Figure 7.18 Using the previous correctness reward (left) and a correctness plus format reward (right).


Similar to figure 7.18, we develop a function that returns 1.0 if both <think> and </think>
occur in the output and in the right order; otherwise, it returns 0.0:

- Listing 7.12 Implementing a format reward function


def reward_format(
token_ids,
prompt_len,
start_think_id=151667,
end_think_id=151668,

):

try:

gen = token_ids[prompt_len:].tolist()
return float(

gen.index(start_think_id) < gen.index(end_think_id)
)

except ValueError:
return 0.0

Let's give this a try on a simple sample text:

prompt = "Calculate ..."
rollout = "Let's ... <think> ... </think> ..."
token_ids = tokenizer.encode(prompt + rollout)
reward_format(

token_ids=torch.tensor(token_ids),
prompt_len=len(tokenizer.encode(prompt))

)

Running the following code returns 1.0 as expected, since the answer (rollout) contains
both <think> and </think>.

###### EXERCISE 7.1: TESTING THE <THINK> FORMAT REWARD

Run the example and confirm it returns 1.0, since both <think> and </think>
appear in the correct order.

Now modify the rollout text by:

Introducing a typo in one of the <think> </think> tags

Reversing the tag order

Removing one tag

Verify that the reward becomes 0.0 in each case.

Using this new reward_format function, we can use the format reward as an additional
reward signal to train the model. For instance, we could convert the compute_grpo_loss
from chapter 6 into a compute_grpo_loss_plus_format_reward function as follows (the
changes are highlighted in the comments):

- Listing 7.13 Updating the GRPO loss function with a format reward

- #A A setting parameter that lets us regulate the magnitude of the format reward relative to the existing correctness
reward
- #B New format reward lines that modify the previous reward to now consist of a correctness and a format reward


def compute_grpo_loss_plus_format_reward(
# ...
format_reward_weight=1.0, #A

):

# ...

logp = sequence_logprob(model, token_ids, prompt_len)
rlvr_reward = reward_rlvr(text, example["answer"])

#B
format_reward = reward_format(token_ids, prompt_len) # NEW
reward = rlvr_reward + format_reward_weight * format_reward # NEW

# ...

- Listing 7.14 Updated prompt rendering function for <think> tokens


In addition, we can modify the render_prompt to direct the model to emit these <think>
and </think> tokens explicitly:

def render_prompt_with_think_tokens(prompt):

template = (
"You are a helpful math assistant.\n"
"When solving the problem, first write your reasoning inside <think> and

</think> tags.\n"
"Then write the final result on a new line in the exact format:\n"
"\\boxed{ANSWER}\n\n"
f"Question:\n{prompt}\n\nAnswer:"

)
return template

In theory, the modifications in the previous code listings 7.11 to 7.14 should be sufficient to
train a reasoning model with an additional format reward. The next section will explain how
this works in practice and discuss additional caveats.

- 7.6.2 Training a model to emit <think> tokens


Similar to previous sections, the modifications needed to train a reasoning model including
a formatting reward are implemented in a script that we can download from the
supplementary materials:

download_from_github(

"ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py"
)

There are a few caveats about this script to keep in mind. For instance, if we train the base
model with the modified prompt to emit <think> and </think> tokens, as shown in listing

- 7.14, the results would be quite poor because the base model has never seen these tokens
before.


Instead, the proper way to introduce these new tokens would be to pre-train or
instruction fine-tune the model on data that includes <think> and </think> tokens first
(instruction fine-tuning is covered in the next chapter).

So, for this chapter, where we focus on reinforcement learning, we train the existing
"reasoning" variant of the model, which is already familiar with <think> tokens. (This is
the same "reasoning" variant we introduced in chapter 3).

Similar to before, we can run the script as follows:

- uv run 7_6_plus_format_reward.py \


--steps 500 \

--max_new_tokens 1024

Note that this script is already configured to use the "reasoning" model variant instead of
the "base" model.

Also, similar to before, we can download a log file of this run for further analysis, as
running the experiment can be resource-intensive and expensive:

download_from_github(
"ch07/02_logs/7_6_plus_format_reward_metrics.csv"

)
plot_grpo_metrics(

"7_6_plus_format_reward_metrics.csv",
columns=["loss", "reward_avg", "avg_response_len", "eval_acc"]

)

The resulting plot is shown in figure 7.19.

![image 132](<input (1)_images/imageFile132.png>)

- Figure 7.19 Basic metrics from a GRPO training run with a format reward.


As we can see in figure 7.19, the model improves at first but then starts to get worse in
terms of evaluation accuracy while the average reward stays approximately constant. The
decline in evaluation accuracy appears to correlate with the shorter response lengths the
model produces after that point.

To investigate this further, let's also have a look at some additional metrics:

plot_grpo_metrics(
"7_6_plus_format_reward_metrics.csv",
columns=["reward_avg", "format_reward_avg", "adv_std", "entropy_avg"],

)

The results are shown in figure 7.20.

![image 133](<input (1)_images/imageFile133.png>)

- Figure 7.20 Additional metrics from a GRPO training run with a format reward.


As figure 7.20 shows, one possible explanation for the declining model performance is that
the model receives too much reward from simply following the <think> tag format, as
shown in the format_reward_avg plot in figure 7.20. In other words, the model does not
focus enough on answer correctness (see reward_avg).

One way to potentially address this is to reduce the strength of the format reward, for
example by lowering the default format_reward_weight from 1.0 to 0.1. Another option is
to make the format reward conditional, so that it is only applied when the answer is also
correct. In the current setup, the format reward is granted regardless of correctness.

###### NOTE ON LOSS AND KL_LOSS

This experiment used 7_6_plus_format_reward.py with the default settings --
kl_coeff 0.0 and --inner_epochs 1. Because --kl_coeff is set to 0.0, the KL
penalty is disabled, so the logged kl_loss remains 0 throughout the run.

If you want to enable KL regularization with the same settings used in Section
7.5, run:

uv run 7_6_plus_format_reward.py \

--steps 500 \

--max_new_tokens 1024 \

--kl_coeff 0.001 \

--inner_epochs 2

The logged loss (figure 7.19) is also 0 with the default 7.6 settings. This happens
because the run uses only one inner update (--inner_epochs 1) and compares the
model against a copy of itself from the same step. As a result, the policy ratio starts
at 1, and since the advantages are normalized to have mean 0, the scalar policy loss
evaluates to 0.

So in this case, loss = 0 is a computation artifact that is a consequence of the
default settings. I.e., it mainly reflects that the run uses a single inner epoch and no
KL term, which makes the logged scalar loss uninformative.

The more useful quantities to monitor here are metrics such as reward_avg,
format_reward_avg, adv_std, and entropy_avg.

###### EXERCISE 7.2: MAKING THE FORMAT REWARD CONDITIONAL

Modify the format reward calculation

reward = rlvr_reward + format_reward_weight * format_reward

Note on `loss` and `kl_loss`so that the format reward is only given if the
correctness reward (rlvr_reward) is non-zero. Repeat the training run with the
conditional format reward. Does conditioning the format reward change training
behavior and final accuracy?

Note: Since running these experiments can be resource-intensive, reference
results are provided in appendix B.

- 7.6.3 More GRPO modifications, tips, and tricks


Previously, we implemented a version of GRPO without KL loss term and clipped policy
ratios, and format reward for introductory purposes. Then, we added these additional
concepts to get a solid foundation and understanding of the original GRPO algorithm, which
underlies most of the modern reasoning training frameworks. Figure 7.21 briefly shows this
next step which is additional modifications, tips, and tricks.

![image 134](<input (1)_images/imageFile134.png>)

- Figure 7.21 This last section in this chapter outlines some additional GRPO modifications that emerged in
recent months.


As far as RLVR with GRPO goes, we learned that there are many knobs to tune. Besides
changing the basic settings such as the learning rate, maximum response length, prompt
format, and so on, there is a near infinite number of settings and modifications to try,
where exploring a single modification or a small handful of modifications thoroughly can be
a whole month- or year-long research project for a small research team.

It is expected that algorithms keep changing and evolving. It is important, though, to
understand the fundamentals and general idea behind it to also understand and appreciate
these improvements.

In the months following the success of DeepSeek-R1, many researchers proposed
changes to the original GRPO algorithm that improved both training stability and final
performance. To provide a taste of several suggested improvements, below is a selection
from DAPO, Dr. GRPO, DeepSeek-V3.2, and others (you can find a full list of references and
links in appendix A):

- 1. Zero gradient signal filtering (DAPO)
- 2. Active sampling (DAPO)
- 3. Switch from sequence- to token-level loss (DAPO)
- 4. No KL loss (DAPO and Dr. GRPO)
- 5. Clip higher (DAPO)
- 6. Truncated importance sampling (VERL)
- 7. No standard deviation normalization (Dr. GRPO)
- 8. KL tuning with domain-specific KL strengths; zero for math (DeepSeek-
V3.2)
- 9. Reweighted KL (DeepSeek-V3.2)
- 10. Off-policy sequence masking (DeepSeek-V3.2)
- 11. Keep sampling mask for top-p / top-k (DeepSeek-V3.2)
- 12. Keep original GRPO advantage normalization (DeepSeek-V3.2)
- 13. Per-reward group-wise normalization before aggregation (GDPO)
- 14. Sequence-level importance sampling and clipping (GSPO)
- 15. Clip importance-sampling weights rather than token updates (CISPO)


Given the already substantial length here, detailed explanations, code examples, and
training run results concerning those modifications are outside the scope of this chapter. If
you’re interested, you can find these in the supplementary materials at https://github.
com/rasbt/reasoning-from-scratch/tree/main/ch07/03_rlvr_grpo_scripts_advanced.

- 7.7 Summary


Training reasoning models with GRPO can become unstable over longer
runs, even when the implementation is correct and rewards initially
improve.

Interpreting GRPO training requires tracking multiple metrics jointly
(average rewards, response length, evaluation accuracy, advantage
statistics, and entropy)

Basic metrics such as loss mainly serve as sanity checks
in GRPO and should not be over-interpreted in isolation.
Advantage statistics provide useful diagnostics: the mean
should remain near zero by design, while the standard
deviation reflects the strength and stability of the
learning signal.

Entropy measures how uncertain the model is during
generation. Very low entropy can signal collapse, and
very large entropy can indicate unstable updates and
randomness in the model responses.

Clipped policy ratios limit how much the policy can change between
updates and can substantially improve training stability over longer runs.

Adding a KL divergence term constrains long-term drift from a reference
model but can destabilize training when rewards collapse.

For math reasoning tasks, several recent systems report better stability
and performance by omitting the KL term altogether.

Auxiliary format rewards can improve the response structure, such as
encouraging the use of <think> and </think> tokens.

Beyond the original GRPO algorithm, many recent extensions modify
advantage normalization, importance sampling, clipping strategies, and
KL handling to improve stability and efficiency.

# 8 Distilling reasoning models for efficient reasoning

This chapter covers

Hard and soft distillation for reasoning models

Creating and preparing a teacher-generated reasoning dataset

Training and evaluating a distilled student model using cross-entropy loss

Reasoning performance can be improved not only through inference-time scaling and
reinforcement learning, but also through distillation. In distillation, a smaller student model
is trained on reasoning traces and answers generated by a larger teacher model. As shown
in the overview in figure 8.1, this chapter focuses on this training-time technique.

![image 135](<input (1)_images/imageFile135.png>)

- Figure 8.1 A mental model of the topics covered in this book. This chapter focuses on distillation, where a
smaller student model is trained on reasoning traces generated by a larger teacher model.


First, we’ll take a look at a general introduction to model distillation before discussing the
individual steps shown in figure 8.1 in more detail.

##### 8.1 Introduction to model distillation for reasoning tasks

Model distillation means training a smaller LLM, the student, on outputs produced by a
larger LLM, the teacher. For reasoning models, these outputs usually include not only the
final answer but also the intermediate reasoning trace that leads to it.

Distillation is especially relevant because the strongest reasoning models are often too
large and expensive to work with directly. For example, DeepSeek-R1 has 671-billion-
parameters. Systems at that scale are expensive to develop, expensive to deploy, and far
outside what most practitioners can run on local hardware.

This chapter is meant to show the same general production pattern at a smaller scale.
The team behind DeepSeek-R1 created smaller model variants by distilling the 671-billion-
parameter teacher model. We follow the same basic idea here, but at a scale that is
practical for this book.

Throughout this book, we deliberately worked with small models rather than training a
very large LLM from scratch. But the workflow in this chapter is still the same. We use a
stronger teacher to generate reasoning traces, and then train a smaller student to
reproduce them. In our setup, this distillation stage also takes only a fraction of the time.
Here, the distillation training run takes about 3 hours on a DGX Spark using about 15 GB of
RAM, whereas even a few GRPO rounds in chapter 6 took around 12 hours and about 70 GB
of RAM on the same hardware.

Distillation can also be more effective than training a small model with reinforcement
learning with verifiable rewards (RLVR) from scratch. For example, the DeepSeek-R1 paper
reported that its smaller distilled variants outperformed comparable models trained with
reinforcement learning alone. In that setup, the largest DeepSeek-R1 model, with 671
billion parameters, acted as the teacher and generated the supervision used to train smaller
student models.

There are two main types of distillation: hard distillation and soft distillation. In hard
distillation, the student is trained on text generated by the teacher, so the teacher's tokens
are treated as the targets. In soft distillation, the student is trained to match the teacher's
probability distribution over the vocabulary by minimizing the KL divergence, a measure of
how different two probability distributions are. These two approaches are illustrated in
figure 8.2.

![image 136](<input (1)_images/imageFile136.png>)

- Figure 8.2 Hard distillation trains the student on teacher-generated tokens, soft distillation trains the student
on the teacher's full output distribution.


As illustrated in figure 8.2, one option is pure hard distillation. Here, we use only the
teacher-generated text as the training target.

For those familiar with the typical LLM training pipeline, which is covered in more detail
in my other book, Build A Large Language Model (From Scratch), hard distillation is just
supervised fine-tuning on synthetic data.

This is also the setup used for the smaller DeepSeek-R1 distilled models. A large teacher
model generates reasoning traces and answers, and the student model is fine-tuned to
reproduce them.

According to the DeepSeek-R1 paper, for small models, this distillation approach can
result in higher accuracy than training with reinforcement learning.

The main practical advantage of hard distillation is that we only need access to the
teacher's generated text, not its logits.

The second option is pure soft distillation. Instead of matching only the teacher's chosen
tokens, the student is trained to match the teacher's full output distribution of each token
over the whole vocabulary. This gives the student richer information about which
alternative tokens the teacher considered plausible, but it requires access to teacher logits
or log-probabilities at training time.

A third option combines hard and soft distillation. This is the classic knowledge-
distillation setup popularized earlier in computer vision, for example in the paper Distilling
the Knowledge in a Neural Network. In this case, we train on the teacher's actual output
tokens while also encouraging the student to match the teacher's full distribution.

In practice, hard distillation is much more common for LLMs. One reason is that full
teacher logits are usually inaccessible. Proprietary systems such as ChatGPT or Claude may
expose generated text, but they generally do not expose the full vocabulary distribution
needed for classical soft distillation.

###### CAUTION Reusing generated text for distillation may be restricted by provider-specific usage policies, including the OpenAI and Anthropic Terms of Service, so these should be reviewed carefully in practice before using such outputs for training.

Even when logits are available, soft distillation is more cumbersome. The student and
teacher usually require the same tokenizers so that their vocabulary distributions line up,
which makes this approach easier within the same model family. It is also much more
expensive to store and use full token distributions for long reasoning traces. By
comparison, storing plain text outputs is cheap and simple.

Here, we focus on hard distillation in the style of DeepSeek-R1 because it is the more
practical setup for most readers. The chapter steps are summarized in figure 8.3.

![image 137](<input (1)_images/imageFile137.png>)

- Figure 8.3 The chapter has four main steps: (1) distillation introduction and overview; (2) dataset preparation
(steps 2a-2c); (3) training via distillation (steps 3a-3c); and (4) evaluation.


With this overview in place, we can now move from the high-level ideas behind distillation
to the practical steps required to implement it. In the next section, we begin by preparing a
dataset for distillation, which serves as the foundation for training the student model.

##### 8.2 Generating a dataset for reasoning distillation

The first practical step is to create a dataset for training the student model. For us, the
student is again the Qwen3 0.6B base model that we used throughout the earlier chapters.

To build the dataset, we use the 12,000 math problems from the MATH split that do not
overlap with the MATH-500 evaluation set. These are the same 12,000 problems that we
used in chapters 6 and 7 for RLVR. Instead of sampling multiple student responses and
computing rewards, we now feed these problems to an existing reasoning model,
DeepSeek-R1 (the teacher model), and collect its responses as training targets. This setup
is illustrated in figure 8.4.

![image 138](<input (1)_images/imageFile138.png>)

- Figure 8.4 Distillation setup used in this chapter. We use the 12,000 non-overlapping MATH training problems
to obtain synthetic solutions from DeepSeek-R1 and later evaluate the distilled Qwen3 student on the
separate MATH-500 test set.


In the RLVR setup in chapters 6 and 7, we trained the Qwen3 base model to produce the
correct solution and then used a verifier to compare the model's final answer against the
reference answer. The verifier produced the reward signal.

In distillation, the supervision is more direct. Instead of comparing the student's answer
against the reference solution with a reward function, we compare the student's generated
tokens against the teacher's generated tokens, as shown in figure 8.4. In other words, the
teacher's response becomes the target sequence. This comparison with RLVR is illustrated
in more detail in figure 8.5.

![image 139](<input (1)_images/imageFile139.png>)

- Figure 8.5 In RLVR, the generated answer is compared against the ground-truth reference solution (top
subpanel), whereas in distillation the student answer is compared against the teacher-generated solution
(bottom subpanel).


A practical advantage of distillation is that we can generate the teacher dataset ahead of
time before training the student.

Because generating teacher answers for all 12,000 math problems can be time- and
resource-intensive, I created this dataset ahead of time using the 671-billion-parameter
DeepSeek-R1 model hosted via OpenRouter.

The full data generation cost was approximately $50 in API usage. Next, we simply load
this pre-generated dataset, so you do not need to generate it yourself to follow along. If
you are curious about the data-generation process, the code and usage instructions are
available in the supplementary materials at https://github.com/rasbt/reasoning-from-
scratch/tree/main/ch08/02_generate_distillation_data.

###### 8.3 Loading the MATH training dataset for distillation

We now load the distilled MATH dataset generated by DeepSeek-R1 in the previous step.
Each example contains a math problem together with a reasoning trace and final answer
that we can use as the target for supervised fine-tuning. This step is highlighted in figure
8.6.

![image 140](<input (1)_images/imageFile140.png>)

- Figure 8.6 Chapter overview with the current section highlighted. Here, we load the DeepSeek-R1-generated
dataset from a JSON file before preparing it for training.


I made the dataset available via the Hugging Face Hub rather than GitHub because the
JSON file is approximately 107 MB, which exceeds GitHub's 100 MB file size limit:
https://huggingface.co/datasets/rasbt/math_distill. The following helper function downloads
the selected partition if it is not already cached locally and returns it as a Python object.

- Listing 8.1 Loading the distilled MATH training split


import json
import requests
from pathlib import Path

def load_distill_data(
local_path=None,
partition="deepseek-r1-math-train",
save_copy=True,

):

if local_path is None:

local_path = f"{partition}.json"
local_path = Path(local_path)

url = (
"https://huggingface.co/datasets/rasbt/math_distill"
"/resolve/main/data/"
f"{partition}.json"

)
backup_url = (

"https://f001.backblazeb2.com/file/reasoning-from-scratch/"
f"MATH/{partition}.json"

)

if local_path.exists(): #A
with local_path.open("r", encoding="utf-8") as f:
data = json.load(f)

size_kb = local_path.stat().st_size / 1e3
print(f"{local_path}: {size_kb:.1f} KB (cached)")
return data

assert partition in (
"deepseek-r1-math-train",
"deepseek-r1-math500",
"qwen3-235b-a22b-math-train",
"qwen3-235b-a22b-math500",

)

try: #B
r = requests.get(url, timeout=30)

r.raise_for_status()

except requests.RequestException:
print("Using backup URL.")
r = requests.get(backup_url, timeout=30)
r.raise_for_status()

data = r.json()

if save_copy: #C
with local_path.open("w", encoding="utf-8") as f:
json.dump(data, f, indent=2)

size_kb = local_path.stat().st_size / 1e3
print(f"{local_path}: {size_kb:.1f} KB")

return data

- #A Reuse a cached copy if the dataset was already downloaded
- #B Try downloading from Hugging Face first
- #C Save a local copy so later runs can skip the download


The output is:

deepseek-r1-math-train.json: 107538.0 KB
Dataset size: 12000

Next, let's inspect one of the training examples to understand the dataset structure and see
exactly what fields the teacher-generated data contains. For this, we pick one of the
training examples (the fifth one) for illustration purposes:

from pprint import pprint
pprint(math_train[4])

The printed output is as follows:

{'gtruth_answer': '6',
'message_content': 'Sam worked \\( x \\) days and did...'
'message_thinking': "Okay, let's see. Sam was hired for 20 days...'
'problem': 'Sam is hired for a 20-day period...'

}

Each dataset entry contains the math problem itself (problem), the ground-truth answer
(gtruth_answer), the teacher's reasoning trace in message_thinking, and the final answer
in message_content.

For distillation, the two most important fields are the reasoning trace and the final
answer, because together they form the target text that the student should learn to
reproduce.

The format_distilled_answer function in listing 8.2 combines these two fields into a
single training target by placing the reasoning trace inside <think>...</think> tags and
then appending the final answer.

- Listing 8.2 Formatting teacher responses for distillation


def format_distilled_answer(entry):
content = str(entry["message_content"]).strip()
if not content:

raise ValueError("Missing non-empty 'message_content' field.")

thinking = str(entry["message_thinking"]).strip()
return f"<think>{thinking}</think>\n\n{content}"

print(format_distilled_answer(math_train[4]))

The printed output is:

<think>Okay, let's see. Sam was hired for 20 days. Each day he works, he earns
$60...So answer is 6 days not worked.</think>
Sam worked \( x \) days and did not work \( y \) days. We know:...
Sam did not work \(\boxed{6}\) days.

As discussed in the previous chapter, the <think></think> tokens are optional. They are
not required for distillation itself, though they can be useful for clearly separating the
reasoning trace from the final answer.

This separation becomes helpful when implementing user interfaces that hide the
verbose reasoning trace from end users. Some systems, including products such as
ChatGPT, may display only the final answer while hiding portions of the internal reasoning.
Teaching the model to use explicit <think> tags makes these traces easier to parse and
handle.

Since the dataset also contains ground-truth labels, we can measure how accurate the
teacher model was on this set. For convenience, we use the evaluate_json.py script from
the supplementary materials of chapter 3, which compares generated answers against the
reference answers using the verifier implemented in that chapter. This gives us a quick
estimate of DeepSeek-R1's performance on the distillation dataset.

from reasoning_from_scratch.ch07 import download_from_github
_ = download_from_github(

"ch03/02_math500-verifier-scripts/evaluate_json.py"

)
After downloading the script via the preceding Python code, we can run it in a
code terminal as follows:
uv run evaluate_json.py \
--json_path "deepseek-r1-math-train.json" \

--gtruth_answer gtruth_answer \

--generated_text message_content

(If you are not a uv user, replace uv run with python.)
The output is:

Accuracy: 90.6% (10871/12000)

While the model is not perfect, 90.6% is a relatively high accuracy. Furthermore, on the
MATH-500 test set from chapter 3, it achieved 91.2% accuracy, which is much higher than
the Qwen3 0.6B base model we are going to train (15.2% accuracy on MATH-500) or the
official Qwen3 0.6B reasoning reference model (50.8% on MATH-500).

##### 8.4 Building training examples

Next, we convert the raw dataset entries into training examples that can be consumed by
the model. At a high level, this means formatting the prompts and answers consistently,
tokenizing them, and storing the resulting token IDs together with the prompt length. The
overall preprocessing stage is highlighted in figure 8.7.

![image 141](<input (1)_images/imageFile141.png>)

- Figure 8.7 Bringing the loaded dataset into a format suitable for model training by understanding the
tokenizer, tokenizing the examples, and filtering and splitting the dataset.


Most of the work here is straightforward preprocessing. In the RLVR chapters, we
performed similar preparation on the fly because each training example was sampled once
and then discarded. Distillation is different, and we usually loop over the same examples for
multiple training epochs. It is therefore more efficient to format and tokenize the dataset
once, store the processed examples, and reuse them during training. This is also one
reason distillation is often easier to iterate on than RLVR once the teacher data has been
collected.

###### WHAT ARE TRAINING EPOCHS?

Training epoch, or epoch for short, is a classical machine learning and deep learning
term. An epoch is one complete pass through the full training dataset. For example,
if we train for three epochs, the model sees each training example three times,
usually in a different order each time. Multiple epochs help the model gradually
improve by revisiting the same data more than once.

- 8.4.1 Loading and understanding the tokenizer


We begin with the tokenizer. Here, we use the Qwen3 reasoning tokenizer because it
supports the <think>...</think> tokens introduced in chapter 7.

- Listing 8.3 Loading the Qwen3 reasoning tokenizer


from reasoning_from_scratch.qwen3 import (
download_qwen3_small,
Qwen3Tokenizer,

)

def load_reasoning_tokenizer(local_dir="qwen3"):
download_qwen3_small(

kind="reasoning", tokenizer_only=True, out_dir=local_dir
)

tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(

tokenizer_file_path=tokenizer_path,
apply_chat_template=True,
add_generation_prompt=True,
add_thinking=True,

)

return tokenizer

tokenizer = load_reasoning_tokenizer()

We set apply_chat_template=True and add_generation_prompt=True so that the
tokenizer applies the same style of prompt formatting used by Qwen3's chat and reasoning
models. The example below shows the additional wrapper tokens that are inserted
automatically.

prompt = "Sam is hired for a 20-day period..."
prompt_ids = tokenizer.encode(prompt)
decoded_prompt = tokenizer.decode(prompt_ids)
print(decoded_prompt)

The formatted text is as follows:

<|im_start|>user
Sam is hired for a 20-day period...<|im_end|>
<|im_start|>assistant

In particular, <|im_start|>user marks the start of the user prompt, <|im_end|> marks the
end of the prompt, and <|im_start|>assistant marks the start of the model response.
This chat-style wrapping is optional, but it is a common convention for instruction and chat
fine-tuning.

For the target answer, we disable this wrapping by setting chat_wrapped=False.
Otherwise, both the prompt and the answer would introduce their own assistant-start
tokens, which is not what we want when concatenating them into a single training
sequence:

answer = (
"<think>Okay, let me try to solve "
"this problem...</think> \\boxed{4}"

)
answer_ids = tokenizer.encode(answer, chat_wrapped=False)
decoded_answer = tokenizer.decode(answer_ids)
print(decoded_answer)

As shown, the chat_wrapped=False setting suppressed the chat template for the answer
tokens:

<think>Okay, let me try to solve this problem...</think> \boxed{4}

For training, we need the full sequence of token IDs: the wrapped prompt, followed by the
teacher answer, followed by an end-of-sequence token. This is the sequence the model will
see during next-token prediction:

token_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
decoded_token_ids = tokenizer.decode(token_ids)
print(decoded_token_ids)

The formatted string looks like as follows:

<|im_start|>user
Sam is hired for a 20-day period...<|im_end|>
<|im_start|>assistant
<think>Okay, let me try to solve this problem...</think>\boxed{4}<|im_end|>

It is worth emphasizing that the reasoning tokenizer is mainly convenient because it
already includes the <think></think> tokens. But in principle, we could also use the base
tokenizer. Likewise, the chat template is not strictly required. We keep it here because it
matches the formatting used by Qwen3's own reasoning models and helps keep the training
and evaluation setup consistent.

If you later evaluate the model with the scripts from earlier chapters, remember to use -
-which_model "reasoning" setting so that the evaluation uses the same tokenizer variant.

- 8.4.2 Formatting and tokenizing the dataset


After loading and understanding the tokenizer, we can now apply the formatting and
tokenization steps to the whole dataset. This transition is highlighted in figure 8.8.

![image 142](<input (1)_images/imageFile142.png>)

- Figure 8.8 With the tokenizer step complete, we now move on to apply the formatting and tokenization steps
to the whole dataset.


With the tokenizer in place, we can now apply the formatting and tokenization steps
consistently across the whole dataset via a build_examples function. The full formatting
and tokenization pipeline for a single training sample is illustrated in figure 8.9.

![image 143](<input (1)_images/imageFile143.png>)

- Figure 8.9 Example of the tokenization pipeline for one training sample. The math problem is rendered into
the chat prompt format, the teacher reasoning trace and final answer are combined via
format_distilled_answer, and both parts are concatenated into one token sequence.


As shown in figure 8.9, the build_examples function follows three steps. First, it renders
and tokenizes the prompt. Second, it formats and tokenizes the teacher answer. Third, it
concatenates both parts and records the prompt length so that we can later compute the
loss only on the answer tokens. Listing 8.4 shows how to implement that in code.

- Listing 8.4 Building and inspecting tokenized distillation examples


from reasoning_from_scratch.ch03 import render_prompt

def build_examples(data, tokenizer):
examples = []
skipped = 0

for entry in data:
try:

- #A
prompt = render_prompt(entry["problem"])
prompt_ids = tokenizer.encode(prompt)
- #B
target_answer = format_distilled_answer(entry)
answer_ids = tokenizer.encode(

target_answer, chat_wrapped=False
)

- #C
token_ids = (

prompt_ids + answer_ids + [tokenizer.eos_token_id]
)

if len(token_ids) < 2:
skipped += 1
continue

- #D
examples.append({


"token_ids": token_ids,
"prompt_len": len(prompt_ids),

})

except (KeyError, TypeError, ValueError):
#E
skipped += 1

return examples, skipped

examples, skipped = build_examples(math_train, tokenizer)

print("Number of examples:", len(examples))
print("Number of skipped examples:", skipped)

- #A Step 1: Render the problem in the chat format
- #B Step 2: Tokenize the teacher reasoning trace and final answer
- #C Step 3: Combine prompt and answer for training
- #D Store prompt length so we can ignore prompt tokens in the loss later
- #E Skip misformatted examples


The resulting numbers, after running the code in listing 8.4, are:

Number of examples: 12000
Number of skipped examples: 0

Next, let's decode one of the training examples to inspect it further:

print(tokenizer.decode(examples[4]["token_ids"]))

The output is:

<|im_start|>user
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
Sam is hired for a 20-day period...

Answer:<|im_end|>
<|im_start|>assistant
<think>Okay, let's see. Sam was hired for 20 days.... So answer is 6 days not
worked.</think>

...Sam did not work \(\boxed{6}\) days.<|im_end|>

Looking at the output above, we can confirm that it checks all the formatting requirements.
For instance, it uses the chat template correctly, and the answer's reasoning trace is
correctly enclosed in <think></think> tags. Finally, the answer ends with an end-of-
sequence token, <|im_end|>.

- 8.4.3 Filtering and splitting the dataset


Once the examples are tokenized, we still need to filter long sequences and split the
dataset into training and validation subsets. This step is highlighted in figure 8.10.

![image 144](<input (1)_images/imageFile144.png>)

- Figure 8.10 After tokenization, we filter out long sequences and split the remaining examples into training
and validation subsets.


After tokenization, it is useful to inspect the sequence lengths. This tells us how long the
examples are on average, which samples are extreme outliers, and how aggressive our
filtering needs to be. We then remove examples above the chosen maximum length, shuffle
the remaining data with a fixed random seed, and split off a small validation set.

Let's begin with analyzing the sequence lengths via listing 8.5.

- Listing 8.5 Computing lengths and filtering long examples


def compute_length(examples, answer_only=False):
lengths = []
for ex in examples:

total = len(ex["token_ids"])
length = total - ex["prompt_len"] if answer_only else total
lengths.append(length)

avg_len = round(sum(lengths) / len(lengths))

shortest_len = min(lengths)
longest_len = max(lengths)
shortest_idx = lengths.index(shortest_len)
longest_idx = lengths.index(longest_len)

print(f"Average: {avg_len} tokens")
print(f"Shortest: {shortest_len} tokens (index {shortest_idx})")
print(f"Longest: {longest_len} tokens (index {longest_idx})")

compute_length(examples)

The resulting output is:

Average: 2946 tokens
Shortest: 236 tokens (index 10846)
Longest: 42005 tokens (index 2529)

As we can see, the average response length is at 2,946 tokens, which is typical for
reasoning models. There are outliers , though. For instance, the longest answer is 42,005
tokens, which is very excessive. The index positions (index 10846 and index 2529) denote
the positions of the shortest and longest examples in the dataset, respectively, in case we
want to inspect them.

To keep the computational costs reasonable for this distillation example, we filter the
dataset to include only dataset entries of up to 2,048 tokens, using the code in listing 8.6.
In practice, controlling the sequence length is one of the main steps that makes distillation
feasible on smaller hardware.

- Listing 8.6 Filtering long examples


def filter_examples_by_max_len(examples, max_len=2048):

filtered_examples = [
s for s in examples
if len(s["token_ids"]) <= max_len

]

print("Original:", len(examples))
print("Filtered:", len(filtered_examples))
print("Removed:", len(examples) - len(filtered_examples))

return filtered_examples

filtered_examples = filter_examples_by_max_len(examples, max_len=2048)

After running the filtering code in listing 8.6, 5305 training examples were removed:

Original: 12000
Filtered: 6695
Removed: 5305

Let's compute the dataset lengths on this new subset:

compute_length(filtered_examples)

As we can see, the average token length is now down to 1180 tokens, and none of the
formatted training examples exceed the 2048 tokens.

Average: 1180 tokens
Shortest: 236 tokens (index 5971)
Longest: 2048 tokens (index 5587)

Lastly, we split the dataset into training and validation examples, where the latter are used
for quick evaluations throughout the training run later.

- Listing 8.7 Partitioning into training and validation sets


import random

rng = random.Random(123)
rng.shuffle(filtered_examples)

train_examples = filtered_examples[25:]
val_examples = filtered_examples[:25]

print("Number of train examples:", len(train_examples))
print("Number of validation examples:", len(val_examples))

The resulting numbers of training and validation example are as follows:

Number of train examples: 6670
Number of validation examples: 25

Note that we keep the validation set purposefully small so that we don't unnecessarily slow
down the training loop. In addition, after the training is completed, we will also use the 500
samples of the MATH-500 set to evaluate the performance of the model.

###### EXERCISE 8.1: TRAINING AND VALIDATION SET LENGTHS

Apply the compute_length function to the new train_examples and val_examples
partitions to check whether they are balanced based on the sample lengths.

##### 8.5 Loading a pre-trained model

With the dataset preparation complete, we can turn to the actual distillation training. We
begin by loading the pre-trained Qwen3 base model, as shown in figure 8.11.

![image 145](<input (1)_images/imageFile145.png>)

- Figure 8.11 With the dataset preparation complete, we begin the distillation training by loading the pre-
trained Qwen3 base model.


We start from the pre-trained Qwen3 base model, not from an RL-trained model, because
distillation itself is the training stage we want to study here. This keeps the setup clean and
makes it easier to attribute any improvement to the distilled reasoning traces. It also
mirrors a common practical setup where a general base model is adapted later using
teacher-generated data.

- Listing 8.8 Loading the Qwen3 base model for distillation


import torch

- from reasoning_from_scratch.ch02 import get_device
- from reasoning_from_scratch.ch03 import (
load_model_and_tokenizer,


)

device = get_device()

model, _ = load_model_and_tokenizer(
which_model="base",
device=device,
use_compile=False,

)

Note that we can ignore the loaded base tokenizer in listing 8.8, since we will be using the
tokenizer with <think></think> token support we loaded previously in section 8.4.1.

##### 8.6 Computing the training and validation losses

Next, we implement the cross-entropy loss that serves as the training signal during
distillation when we implement the training loop later. We will also reuse the same
computation on the validation set to monitor progress during training.

Readers with a deep-learning background may already know cross-entropy from
classification tasks. The idea is the same here, except that the target class we want to
predict at each position is the next token in the teacher-generated sequence.

We can connect this loss directly to the log-probability computations from chapters 5 and
6. As discussed there, log-probability measures how much probability the model assigns to
the correct next token. Higher log-probability means the model is more confident in the
correct token, and lower log-probability means the opposite. Re-using the "The capital of
Germany is Berlin" example from previous chapters, this is recapped in figure 8.12.

![image 146](<input (1)_images/imageFile146.png>)

- Figure 8.12 Illustration of token and sequence log-probabilities. The log-probabilities of the correct next
tokens are summed to obtain the sequence log-probability, which is the basis for the cross-entropy loss used
later in this chapter.


Cross-entropy is simply the negative average of these token log-probabilities shown in
figure 8.12. For instance, if -16.6250 is the log-probability, the negative log-probability is
16.6250, the negative average log-probability is 16.6250/5 = 3.325. Note that the 5 is
there because the sequence has 5 target predictions whose log-probabilities are being
averaged: "The", "capital", "of", "Germany", "is", and "Berlin".

The cross-entropy is also 3.325. (Similar to log-probability, the closer the value is to 0,
the better, since the model is more confident in the correct target token.)

The sequence_logprob function from chapter 6 performs exactly this computation
shown in figure 8.12, which makes it a useful starting point for understanding how the
distillation loss works. In this case, we use the training example at index position 5730,
because it's the shortest example in train_examples, which we can determine via
compute_length(train_examples), and thus computes a bit more quickly than the other
examples:

token_ids = train_examples[5730]["token_ids"]
prompt_len = train_examples[5730]["prompt_len"]

Instead of reporting the summed log-probabilities returned by sequence_logprob, we
average them over the number of answer tokens. The reason is that summed log-
probabilities grow with sequence length, so they are not directly comparable across
examples with shorter or longer answers. By dividing by the number of answer tokens, we
obtain a per-token quantity, which matches the form used by cross-entropy loss. We
compute the number of answer tokens by subtracting prompt_len from the total sequence
length:

- Listing 8.9 Computing average negative log-probabilities


- from reasoning_from_scratch.ch06 import sequence_logprob


tok = torch.tensor(token_ids, dtype=torch.long, device=device)

with torch.no_grad():
seq_logprob = sequence_logprob(model, tok, prompt_len)
num_answer_tokens = tok.numel() - prompt_len
avg_logprob = -seq_logprob / num_answer_tokens

print(f"Average logprob: {avg_logprob:.2f}")

The resulting negative average log-probability is 1.68.

Now, we can compute the same quantity using PyTorch's cross_entropy function.
Although the function is usually introduced for classification, it is a natural choice here. For
instance, the model logits provide the predicted class distribution over the vocabulary, and
the target sequence provides the correct class label at each position.

As in the previous average-logprob calculation, we compute the loss only over the
answer tokens and ignore the prompt tokens (the prompt is the context provided as input,
and it is not something we want the model to be penalized for reproducing). This is
illustrated in figure 8.13.

![image 147](<input (1)_images/imageFile147.png>)

- Figure 8.13 Input for the cross-entropy loss over the answer tokens. The model receives the prompt and
answer shifted by one token as input, and the answer-token logits are compared against the reference answer
tokens.


During distillation, we want the student to learn the teacher's reasoning trace and final
answer conditioned on the prompt, so, as shown in figure 8.13, we discard the logits and
targets that correspond only to the prompt portion of the sequence when computing the
cross-entropy in listing 8.10:

- Listing 8.10 Computing cross-entropy loss directly


- #A
input_ids = tok[:-1].unsqueeze(0)
target_ids = tok[1:]
logits = model(input_ids).squeeze(0)
- #B
first_answer_logit_idx = max(prompt_len - 1, 0)
answer_logits = logits[first_answer_logit_idx:]
answer_targets = target_ids[first_answer_logit_idx:]
- #C
with torch.no_grad():


ce_mean_direct = torch.nn.functional.cross_entropy(

answer_logits, answer_targets
)

print(f"Cross-entropy: {ce_mean_direct:.2f}")

- #A Shift the sequence by one token so each input predicts the next token target
- #B Drop the prompt positions so the loss only covers the teacher answer
- #C Compute cross-entropy loss


The resulting cross-entropy loss is 1.68, which is similar to listing 8.9, when we used the
sequence_logprob function. In other words, cross_entropy is implementing the same
core calculation in a more optimized way. For training, we therefore use the built-in
cross_entropy function rather than our custom log-probability function.

The compute_example_loss helper in listing 8.11 below wraps this logic into a
convenient function that calculates the answer-only loss for a single example. "Answer-only
loss" means the training loss is computed only on the teacher's answer tokens, not on the
prompt tokens.

We focus on the answer-only loss because the prompt is already given. And the model's
job in distillation is not to learn to reconstruct the input instruction. Its job is to produce the
target answer conditioned on that instruction. So we use the prompt as context, but we do
not penalize the model for prompt-token predictions.

Now, let's put it all together into a function that applies the whole logic, from target
preparation to cross entropy computation, on a given training example in listing 8.11:

- Listing 8.11 Defining the loss for one distillation example


def compute_example_loss(model, example, device):
token_ids = example["token_ids"]
prompt_len = example["prompt_len"]

- #A
input_ids = torch.tensor(

token_ids[:-1], dtype=torch.long, device=device
).unsqueeze(0)
target_ids = torch.tensor(

token_ids[1:], dtype=torch.long, device=device
)

logits = model(input_ids).squeeze(0)

- #B
answer_start = max(prompt_len - 1, 0)
answer_logits = logits[answer_start:]
answer_targets = target_ids[answer_start:]
- #C
loss = torch.nn.functional.cross_entropy(


answer_logits, answer_targets

)
return loss

- #D
with torch.no_grad():


loss = compute_example_loss(

model, train_examples[5730], device
)

print(f"Loss: {loss:.2f}")

- #A Create input-target pairs that are shifted by one token
- #B Ignore prompt tokens so the loss is computed on the distilled answer only
- #C Compute cross-entropy loss
- #D Use to verify that the helper returns the same loss as before


The resulting loss is 1.68 again, which indicates that the function in listing 8.11 works as
intended.

###### BATCHING

It is also possible to process multiple examples in parallel by batching them together.
We omit batching here to keep the implementation compact and the resource
requirements lower. Appendix E discusses batching and throughput-oriented
execution in more detail for the loss computation and training in general.

Next, we define a small wrapper that iteratively computes the average loss across multiple
examples. This will be useful both for quick sanity checks and for tracking the validation
loss during training.

- Listing 8.12 Evaluating loss across multiple examples


@torch.no_grad()
def evaluate_examples(model, examples, device):

was_training = model.training

- #A
model.eval()
total_loss = 0.0
num_examples = 0
- #B
for example in examples:

loss = compute_example_loss(model, example, device)
total_loss += loss.item()
num_examples += 1

- #C
if was_training:

model.train()

- #D
return total_loss / num_examples


- #E
train_loss = evaluate_examples(model, train_examples[:3], device)
print(f"Train loss (3 examples): {train_loss:.2f}")


val_loss = evaluate_examples(model, val_examples[:3], device)
print(f"Validation loss (3 examples): {val_loss:.2f}")

- #A Temporarily switch to evaluation mode while scoring the examples
- #B Sum the loss over all examples
- #C Restore training mode so this helper is safe to call during training
- #D Average the loss over all examples
- #E Estimate the current training loss on a small subset


The output is:

Train loss (3 examples): 0.98
Validation loss (3 examples): 1.02

We will reuse this evaluation function during training. Ideally, both quantities should
decrease over time, which indicates that the student model is becoming better at matching
the teacher-generated target sequences.

In practice, the training loss, which is the loss computed on the training examples used
for optimization, is often noisier because it is measured on the examples currently being
optimized and can fluctuate depending on the sample order and recent parameter updates.

The validation loss, which is measured on a separate held-out validation set rather than
on the examples currently being used for optimization, is computed without updating the
model weights. As a result, it usually provides a cleaner signal of whether the student is
improving in a way that generalizes beyond the training set. For this reason, the validation
loss is often the more reliable metric to watch.

##### 8.7 Implementing the training loop for distillation

With the dataset preparation and loss computation in place, we can now implement the
training loop for distillation. This stage is highlighted in figure 8.14.

![image 148](<input (1)_images/imageFile148.png>)

- Figure 8.14 With dataset preparation and loss computation complete, we now turn to the training loop for
distillation.


The training loop is very similar to the one from chapter 6. The main difference is that we
now revisit the same training set multiple times across epochs and optimize the cross-
entropy distillation loss instead of the GRPO objective (GRPO loss) used in RLVR. The
detailed steps are shown in figure 8.15.

![image 149](<input (1)_images/imageFile149.png>)

- Figure 8.15 Distillation training loop. In each epoch, the training examples are shuffled, the student model
computes a cross-entropy loss for each example, gradients are backpropagated, and the model weights are
updated. The validation loss is reported in certain intervals to track progress.


The train_distillation function in listing 8.13 implements the loop shown in figure 8.15.
It shuffles the training examples at the start of each epoch, computes the loss for each
example, applies an optimizer step, optionally clips large gradients, and periodically
evaluates on the validation set. The metrics are also written to a CSV file so that we can
inspect the learning curves later.

- Listing 8.13 Implementing the distillation training loop


import time

def train_distillation(
model,
train_examples,
val_examples,
device,
epochs=2,
lr=5e-6,
grad_clip_norm=None,
seed=123,
log_every=50,
checkpoint_dir="checkpoints",
csv_log_path=None,

):

- #A
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.train()

total_steps = epochs * len(train_examples)
global_step = 0
rng = random.Random(seed)

if csv_log_path is None:
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_log_path = f"train_distill_metrics_{timestamp}.csv"

csv_log_path = Path(csv_log_path)

- #B
for epoch in range(1, epochs + 1):
- #C
epoch_examples = list(train_examples)
rng.shuffle(epoch_examples)
- #D
for example in epoch_examples:
global_step += 1
- #E
optimizer.zero_grad()


- #F
loss = compute_example_loss(model, example, device)
- #G
loss.backward()
- #H
if grad_clip_norm is not None:

torch.nn.utils.clip_grad_norm_(

model.parameters(), grad_clip_norm
)

- #I
optimizer.step()
- #J
if log_every and global_step % log_every == 0:


val_loss = evaluate_examples(
model=model,
examples=val_examples,
device=device,

)
model.train()
print(

f"[Epoch {epoch}/{epochs} "
f"Step {global_step}/{total_steps}] "
f"train_loss={loss.item():.4f} "
f"val_loss={val_loss:.4f}"

)
append_csv_metrics(

csv_log_path=csv_log_path,
epoch_idx=epoch,
total_steps=global_step,
train_loss=loss.item(),
val_loss=val_loss,

)

#K
ckpt_path = save_checkpoint(

model=model,
checkpoint_dir=checkpoint_dir,
step=global_step,
suffix=f"epoch{epoch}",

)

print(f"Saved checkpoint to {ckpt_path}")
return model

def save_checkpoint(model, checkpoint_dir, step, suffix=""):
checkpoint_dir = Path(checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
suffix = f"-{suffix}" if suffix else ""
ckpt_path = (

checkpoint_dir /
f"qwen3-0.6B-distill-step{step:05d}{suffix}.pth"

)
torch.save(model.state_dict(), ckpt_path)
return ckpt_path

def append_csv_metrics(
csv_log_path,
epoch_idx,
total_steps,
train_loss,
val_loss,

):

if not csv_log_path.exists():

csv_log_path.write_text(
"epoch,total_steps,train_loss,val_loss\n",
encoding="utf-8",

)
with csv_log_path.open("a", encoding="utf-8") as f:

f.write(
f"{epoch_idx},{total_steps},{train_loss:.6f},"
f"{val_loss:.6f}\n"

)

- #A Step 1: initialize optimizer (model is already loaded)
- #B Step 2: iterate over training epochs
- #C Step 3: shuffle the training examples at the start of the epoch
- #D Step 4: iterate over training examples in epoch
- #E Stage 5: reset loss gradient
- #F Step 6: compute the cross-entropy loss for the current example
- #G Step 7: backpropagate gradients
- #H Optionally clip large gradients to improve training stability
- #I Step 8: update the model weights
- #J Step 9: periodically evaluate the current model on the validation set
- #K Step 10: record the metrics and save a checkpoint for this epoch


With a maximum sequence length of 2048, the full training run requires about 15 GB of
GPU memory. If this is too high for your hardware, you can lower the resource
requirements by filtering out longer sequences earlier in the notebook, for example by
changing max_len from 2048 to 1024 or 512 in the filter_examples_by_max_len step in
listing 8.6 (section 8.4.3).

Let's execute a short training run:

- Listing 8.14 Training the model


torch.manual_seed(0) #A

train_distillation(
model,
train_examples=train_examples[:10], #B
val_examples=val_examples[:10], #B
device=device,
epochs=2,
lr=5e-6,
grad_clip_norm=1.0, #C
seed=123,
log_every=5,
csv_log_path="train_distill_metrics.csv",

)

- #A Seed PyTorch so the short demo is reproducible
- #B Train on a tiny subset so this notebook run stays lightweight
- #C Same as in chapter 6


Let's briefly talk about the settings before we inspect the results. We keep the training run
intentionally short. For instance, we use only the first 10 training examples and 10
validation examples, and we train for just 2 epochs. This is purely to keep the notebook run
lightweight and fast enough for experimentation. For a real distillation run, we would of
course train on many more examples, ideally the whole training set.

The learning rate (lr=5e-6) is in a reasonable range for fine-tuning a pre-trained model
and works well in practice for this setup and worked well in practice in my experiments, as
reflected in the loss curves discussed next.

The gradient clipping setting (grad_clip_norm=1.0) is the same as in chapter 6 and
helps prevent unstable updates when a particular example produces unusually large
gradients.

The log_every=5 setting means that validation loss is measured every 5 training steps.
Since this demo uses only a handful of examples, this produces frequent progress updates
so that we can quickly verify that the training loop behaves as expected. In a larger run, we
would usually increase this interval to reduce evaluation overhead.

Finally, the csv_log_path="train_distill_metrics.csv" argument stores the training
and validation losses in a CSV file so that we can inspect and plot them later.

The main goal of this run is not to achieve the best possible reasoning performance, but
to confirm that the distillation pipeline works end to end. Once that is established, we can
move on to larger runs and inspect the resulting learning curves and checkpoints in more
detail.

Let's now take a brief look at the run's output:

- [Epoch 1/2 Step 5/20] train_loss=0.9648 val_loss=0.9082
[Epoch 1/2 Step 10/20] train_loss=0.9844 val_loss=0.8871
Saved checkpoint to checkpoints/qwen3-0.6B-distill-step00010-epoch1.pth
[Epoch 2/2 Step 15/20] train_loss=0.8008 val_loss=0.8707
- [Epoch 2/2 Step 20/20] train_loss=0.7148 val_loss=0.8586
Saved checkpoint to checkpoints/qwen3-0.6B-distill-step00020-epoch2.pth


Even though this is only a tiny demonstration run, the output already shows the expected
overall behavior. Both the training loss and the validation loss decrease over time, which
indicates that the student model is becoming better at matching the teacher-generated
target sequences. The validation loss is especially useful here because it is computed on
held-out examples and therefore provides a cleaner signal that the improvement is not
limited to the training samples alone.

We can also see that checkpoints are saved at the end of each epoch. This is useful in
practice because it allows us to resume training later or evaluate intermediate versions of
the distilled model using the MATH-500 test set.

##### 8.8 Evaluating the distilled model

After implementing the training loop, the final stage is evaluation of the distilled model, as
shown in figure 8.16.

![image 150](<input (1)_images/imageFile150.png>)

- Figure 8.16 After implementing the training loop, we evaluate the distilled model on the MATH-500 test set.


Instead of running the full distillation process inside this notebook, we can also download a
convenience script from the supplementary materials. As in the previous chapter on RLVR,
it is often convenient to keep the notebook focused on the core ideas and move the longer-
running training job into a standalone script.

- from reasoning_from_scratch.ch07 import download_from_github
download_from_github(


"ch08/04_train_with_distillation/distill.py"
)

After downloading the script via the preceding code, we can run it as follows in a code
terminal (if you are not a uv user, replace uv run by python):

uv run distill.py \

--data_path deepseek-r1-math-train.json \

--validation_size 25 \

--epochs 3 \

--lr 1e-5 \

--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0

Using these settings, the full training run takes about 3 hours and 5 minutes on a DGX
Spark and uses roughly 15.02 GB of GPU memory. (This is relatively modest compared with
the earlier RLVR runs because most of the expensive work has already been moved into the
one-time teacher data generation step.) If you do not want to run the training yourself, we
can download the resulting metrics file and inspect it directly.

download_from_github(

"ch08/03_logs/deepseek-r1-2048_distill_metrics.csv"
)

The following listing 8.15 implements a utility function to visualize the training metrics
stored in the CSV file:

- Listing 8.15 Plotting distillation losses from a CSV log


import csv
import matplotlib.pyplot as plt

def plot_distill_metrics(csv_path="train_distill_metrics.csv"):
total_steps, train_losses, val_losses, epoch_bounds = [], [], [], {}

- #A
with open(csv_path, newline="", encoding="utf-8") as f:

for row in csv.DictReader(f):
step = int(row["total_steps"])
epoch = int(row["epoch"])
total_steps.append(step)
train_losses.append(float(row["train_loss"]))
val_losses.append(float(row["val_loss"]))
epoch_bounds.setdefault(epoch, [step, step])[1] = step

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(total_steps, train_losses, label="train_loss", alpha=0.3)
ax.plot(total_steps, val_losses, label="val_loss")
ax.set_xlabel("Total Steps")
ax.set_ylabel("Loss")
ax.legend()

- #B
epoch_axis = ax.secondary_xaxis("bottom")
epoch_axis.spines["bottom"].set_position(("outward", 45))
epochs = sorted(epoch_bounds)
epoch_axis.set_xticks(


[

(epoch_bounds[epoch][0] + epoch_bounds[epoch][1]) / 2
for epoch in epochs

]

)
epoch_axis.set_xticklabels(epochs)
epoch_axis.set_xlabel("Epoch")

plt.tight_layout()
plt.show()

plot_distill_metrics("deepseek-r1-2048_distill_metrics.csv")

#A Load and plot the logged losses
#B Add a second x-axis so the epoch numbers are visible below the step axis

The resulting plot is shown in figure 8.17.

![image 151](<input (1)_images/imageFile151.png>)

- Figure 8.17 The training and validation loss of a 3-epoch distillation training run on DeepSeek-R1 reasoning
traces.


The training-loss curve and validation-loss curve shown in figure 8.17 should be interpreted
slightly differently. The training loss is computed for each training example during training
at the current step, whereas the validation loss is computed on a small fixed held-out set.
The validation curve is therefore less noisy and the more informative signal here. We could
also compute the training loss on a subset of examples, similar to the validation loss, but
this would slow down the training; the training loss is much less important than the
validation loss for estimating the training progress.

As we would hope, the validation loss drops sharply at first and then begins to flatten
out, indicating that the model is learning from the distillation data but that additional
training yields diminishing returns. We could experiment with a larger learning rate or a
different schedule to train more aggressively, but overall the curve looks healthy.

Just as in chapter 7, we can evaluate saved checkpoints with the verifier-based utilities
from chapter 3. The following command downloads the evaluation script from the
supplementary materials:

download_from_github(

"ch03/02_math500-verifier-scripts/evaluate_math500.py"
)

Next, we evaluate the distilled checkpoint on MATH-500 using the reasoning tokenizer in a
code terminal, as follows:

uv run evaluate_math500.py \

--dataset_size 500 \

--which_model reasoning \

--max_new_tokens 4096 \

--checkpoint_path \
"run_11/checkpoints/distill/qwen3-0.6B-distill-step06682-epoch1.pth"

For the later checkpoints from the same DeepSeek-R1 run, replace ...step06682-
epoch1.pth with ...step13364-epoch2.pth and ...step20046-epoch3.pth, respectively.

The evaluation results for the DeepSeek-R1 run used in this chapter are summarized in
table 8.1 below. For reference, I also include a second run trained on Qwen3 235B-A22B
teacher outputs (a 235-billion parameter Qwen3 model).

Table 8.1 MATH-500 task accuracy for different model checkpoints

| |Method|Epoch|Final val loss|MATH-500<br>Acc.|
|---|---|---|---|---|
|1|Base Qwen3 0.6B (chapter<br>3)|-|-|15.2%|
|2|Reasoning Qwen3 0.6B<br>(chapter 3)|-|-|48.2%|
|3|DeepSeek-R1|1|0.5436|30.6%|
|4|DeepSeek-R1|2|0.5349|32.4%|
|5|DeepSeek-R1|3|0.5343|33.6%|
|6|Qwen3 235B-A22B|1|0.4043|45.0%|
|7|Qwen3 235B-A22B|2|0.3963|43.8%|
|8|Qwen3 235B-A22B|3|0.3948|44.2%|


Based on the results shown in table 8.1, for the DeepSeek-R1 run, we see that the MATH-
500 accuracy improves from 30.6% after the first epoch to 33.6% after the third epoch,
while the validation loss decreases from 0.5436 to 0.5343. This matches the previous
learning-curve discussion based on figure 8.17, where the student clearly learns from the
teacher-generated reasoning traces, but the gains begin to taper off after the initial
improvement.

The Qwen3 235B-A22B run performs noticeably better in this setup. One likely reason is
that the teacher and student come from the same model family, which means that the
tokenizer, prompting conventions, and overall response style are more closely aligned. This
can make the teacher targets easier for the smaller Qwen3 student to imitate.

Considering the MATH-500 accuracy of 45% (row 6), our distillation recipe reaches
almost the same performance as the reasoning reference model (48.2%, row 2), which is
itself a distilled model but has been trained on a much larger dataset generated by Qwen3
235B-A22B. This is the main tradeoff behind distillation.

Note that in general, we do not expect the smaller student to match the teacher exactly.
For instance, Qwen3 235B-A22B has a MATH-500 accuracy of 92.4%, and DeepSeek-R1 has
a MATH-500 accuracy of 91.2%. But we can still recover a useful portion of the teacher's
reasoning behavior in a much cheaper model. Also, note that our distilled model accuracy
could be higher if we used a larger model (e.g., a 4- or 30-billion-parameter version of
Qwen3 instead of Qwen3 0.6B), but this would increase the computational cost during
training.

Let's step back and connect the individual pieces we implemented into one complete
workflow. Figure 8.18 summarizes the full distillation pipeline, starting with the introductory
setup, followed by dataset generation and preprocessing, then the distillation training loop
itself, and finally the evaluation of the distilled model on MATH-500.

![image 152](<input (1)_images/imageFile152.png>)

- Figure 8.18 The evaluation of the distilled model completes the technical content of this chapter.


This workflow shown in figure 8.18 completes the core technical recipe for distilling a
smaller reasoning model from a stronger teacher. In practice, each stage offers room for
variation, such as changing how the teacher data is generated and modifying the training
settings.

###### EXERCISE 8.2: DISTILLING WITHOUT <THINK> TOKENS

Repeat the distillation experiment without --use_think_tokens and compare the
results against the version trained with reasoning traces wrapped in <think>...
</think>. Inspect the validation loss and, if possible, evaluate the saved checkpoint
on MATH-500. Then compare the results with those listed in table 8.1. How much do
the explicit reasoning tags matter in this setup?

##### 8.9 Future directions for reasoning models

Before closing the chapter, let’s discuss where reasoning models are headed next.

For the foreseeable future, the overall broad pattern remains the same. For instance, the
general strategy is to develop a stronger reasoning teacher, collect high-quality reasoning
traces, and distill them into smaller student models.

One obvious direction is continued refinement of the training recipe popularized by the
DeepSeek-R1 paper, which includes both RLVR (chapters 6 and 7) for the flagship models
and distillation for the smaller models geared towards computational efficiency.

A second direction is the optimization at inference time. In practice, a reasoning model
should not always produce equally long answers. Some tasks benefit from short, direct
responses, whereas others benefit from more detailed multi-step reasoning. This creates
room for more automatic and flexible inference scaling at the application layer, where the
surrounding system decides when to ask for a short answer, when to allocate more
reasoning budget, and when to stop early. For example, OpenAI implemented such a
system with the launch of GPT-5 in 2025, where they added an "auto" mode to steer the
reasoning effort and reasoning trace generation length.

A third direction for improvement is the reward generation for RLVR. Much of the current
work still relies heavily on rewards based on final answers, especially in math and code. But
process rewards that check intermediate reasoning steps, not just the final result, may
provide a richer training signal and help models learn more reliable reasoning strategies.
For example, the DeepSeek-Math-V2 paper (https://arxiv.org/abs/2511.22570) recently
demonstrated that judging the whole answer during training can meaningfully improve
reasoning performance.

A fourth direction is the growing role of reasoning models as the engine inside larger
agent applications such as OpenAI Codex, Claude Code, and OpenClaw. In these settings,
the model must not only solve a simple math or coding problem, but also plan, call tools,
recover from failures, and coordinate longer workflows. This naturally pushes reward design
beyond math and code correctness. We may want rewards for successful tool use,
information retrieval, policy compliance, and more. In turn, this leads to multi-reward
training, where several objectives are optimized together instead of relying on a single
correctness score.

Distillation will likely remain important in this setting because it offers a practical way to
transfer these richer behaviors from larger and more expensive teacher systems into
smaller models that are easier to deploy or even run locally while being more cost-effective
for users.

##### 8.10 Conclusions

This completes the main technical material of the book. The remaining sections are brief
pointers on what to try next, how to keep up with a fast-moving field, and where to find
additional material.

- 8.10.1 What's next

A practical next step is to start combining the methods from this book instead of treating
them as isolated techniques. For example, you could distill a smaller model from a strong
teacher, continue training it with RLVR, and then apply inference-time scaling methods such
as self-consistency or self-refinement at deployment time. Running these kinds of
comparisons is often the fastest way to develop intuition for which method helps most in a
given setting.

The appendices are also a good place to continue. They cover additional topics such as
LLM architecture details, batched execution for higher throughput, and alternative
evaluation approaches.

Lastly, the supplementary code repository (https://github.com/rasbt/reasoning-from-
scratch) includes bonus material and standalone scripts that are better suited for longer
runs and larger experiments than a notebook.

- 8.10.2 Staying up to date in a fast-moving field


I hope this book gave you a clearer picture of how modern reasoning models work in
practice!

Reasoning-model research is moving quickly, and specific algorithms, datasets, and best
practices will continue to change. The core ideas in this book tend to remain relevant and
useful. This includes careful model evaluation, distinguishing between inference-time and
training-time methods, and a solid understanding of how the training losses and reward
signals are computed.

When new methods appear, it helps to map them back to these fundamentals. Many new
techniques are best understood as variations or combinations of the building blocks we
implemented here, even when the surrounding training recipe becomes more complex.

In practice, I recommend several resources. First, you may skim recent machine
learning and AI papers on arXiv (https://arxiv.org/list/cs.LG/recent), to spot new training
and evaluation ideas early. However, be aware that the volume of new papers is now so
large that keeping up comprehensively is nearly impossible, so it is usually better to treat
arXiv as a way to scan for themes and promising papers than to try to read everything.

Second, read technical summaries and reports for newly released models, since they
often contain the most useful details about prompting conventions, benchmarks, and
limitations. These are usually shared directly by the developer on the Hugging Face model
hub, announcement blog articles, and social media.

Third, keep an eye on practitioner discussions on the social media platform X and in
communities such as r/LocalLLaMA (https://www.reddit.com/r/LocalLLaMA/), where
implementation details, replications, and failure cases often show up before they make it
into polished papers.

AI-assistant or "deep research" tools can also be useful for monitoring these sources and
summarizing new developments, but they work best as filters rather than substitutes for
reading the original papers and model cards.

I also regularly write about AI and LLM topics on my blog at https://magazine.
sebastianraschka.com.

- 8.11 Summary


Distillation trains a smaller student LLM on outputs produced by a larger
teacher LLM.

Hard distillation is usually more practical than soft distillation because
teacher logits are often unavailable and teacher text outputs are much
cheaper to store and reuse.

We used DeepSeek-R1 as the teacher model and Qwen3 0.6B as the
student model

The distillation dataset was built from the 12,000 MATH training problems
that do not overlap with MATH-500.

Each training sample combines a rendered prompt with the teacher
reasoning trace and final answer, optionally separated via <think>...
</think> tags.

For efficiency, we tokenize the dataset once, filter it by sequence length,
and reuse the processed examples across multiple epochs.

The training objective is answer-only cross-entropy, which is equivalent to
the negative average log-probability of the correct next tokens.

The distillation training loop is a standard supervised learning loop, which
includes shuffling the training examples each epoch, computing the loss,
backpropagating, updating the model weights, and tracking validation
loss.

The validation loss is the main signal to watch during training, and saved
checkpoints can later be evaluated on MATH-500 with the verifier from
chapter 3.

The distillation approach in this chapter improves the Qwen3 0.6B base
model from 15.2% accuracy on MATH-500 to 45.0% accuracy.

## Appendix A. References and further reading

##### A.1 Chapter 1: Understanding reasoning models

- A.1.1 References
The announcement article of OpenAI's o1 model, which is regarded as the first LLM-


based reasoning model:

Introducing OpenAI o1-preview, https://openai.com/index/introducing-
openai-o1-preview/

DeepSeek-R1 is the first open-source reasoning model that was accompanied by a
comprehensive technical report, which was the first to show that reasoning emerges from
reinforcement learning with verifiable rewards (a topic covered in more detail in chapter 5):

DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning, https://arxiv.org/abs/2501.12948

OpenAI CEO’s comment on the reasoning ("chain-of-thought") capabilities of future models:

"[...] We will next ship GPT-4.5, the model we called Orion internally, as
our last non-chain-of-thought model. [...]", https://x.com/sama/
status/1889755723078443244

A research paper by AI researchers at Apple finding that reasoning models are
sophisticated (but very capable) pattern matchers:

The Illusion of Thinking: Understanding the Strengths and Limitations of
Reasoning Models via the Lens of Problem Complexity,
https://machinelearning.apple.com/research/illusion-of-thinking

An in-depth book and guide on implementing and training large language models step-by-
step:

Build a Large Language Model (From Scratch),http://mng.bz/orYv

- A.1.2 Further Reading


An introduction to how DeepSeek-R1 works, providing insights into the foundations of
reasoning in LLMs:

Understanding Reasoning LLMs, https://magazine.sebastianraschka.
com/p/understanding-reasoning-llms

###### A.2 Chapter 2: Generating text with a pre-trained LLM

- A.2.1 References
Official installation page for the uv Python package and project manager:

Installing uv, https://docs.astral.sh/uv/getting-started/installation/

User-friendly and popular cloud compute platforms with GPU support:

Lightning AI, https://lightning.ai/

Google Colab, https://colab.research.google.com/

Qwen3 resources with additional benchmark performance and comparison to other models:

Blog post, https://qwenlm.github.io/blog/qwen3/

Technical report, https://arxiv.org/abs/2505.09388

Readers curious about KV cache sizes for different sequence lengths can find a handy
calculator app here:

KV cache size calculator, https://lmcache.ai/kv_cache_calculator.html

- A.2.2 Further Reading


A PyTorch tutorial for readers who are new to PyTorch or would like a refresher:

PyTorch in One Hour: From Tensors to Training Neural Networks on
Multiple GPUs tutorial, https://sebastianraschka.com/teaching/pytorch-1h

Additional resources on tokenization:

Build a Large Language Model (from Scratch) chapter 2, https://mng.
bz/M96o

Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch,
https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

For readers interested in a more in-depth PyTorch coverage (optional), I can recommend
the following two books:

Deep Learning with PyTorch, https://www.manning.com/books/deep-
learning-with-pytorch-second-edition

Machine Learning with PyTorch and Scikit-Learn, https://www.
amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/
dp/1801819319/

###### A.3 Chapter 3: Evaluating reasoning models

- A.3.1 References
The MATH-500 dataset originated from the MATH dataset (with 12,500 problems across

algebra, geometry, probability, number theory, and more) that was introduced in the
following paper:

Measuring Mathematical Problem Solving With the MATH Dataset,
https://arxiv.org/abs/2103.03874

The MATH-500 split (created from the original MATH dataset) was proposed in the following
paper:

Let's Verify Step by Step, https://arxiv.org/abs/2305.20050

- A.3.2 Further Reading


Readers who are interested in learning more about SymPy, Python library for math and
symbolic computation (not required for this book), can learn about it in this official tutorial:

SymPy introductory tutorial, https://docs.sympy.org/latest/tutorials/intro-
tutorial/index.html

An example of a system (here, a fine-tuned LLM) to also evaluate intermediate reasoning
steps:

Evaluating Mathematical Reasoning Beyond Accuracy, https://arxiv.
org/pdf/2404.05692

A large-scale dataset containing 800,000 step-level correctness labels for model-generated
solutions to problems from the MATH dataset:

Let's Verify Step by Step, https://arxiv.org/abs/2305.20050

An article describing the rising cost of LLM evaluation, finding that evaluating reasoning
models such as o1 on (seven) popular benchmarks costs approximately $1500:

The rise of AI "reasoning" models is making benchmarking more
expensive, https://techcrunch.com/2025/04/10/the-rise-of-ai-reasoning-
models-is-making-benchmarking-more-expensive/

A comprehensive 2025 survey on LLM benchmarks:

A Survey on Large Language Model Benchmarks, https://arxiv.
org/abs/2508.15361

Instead of only relying on deterministic and symbolic verifiers, a recent research project
highlighted that small reasoning models themselves can be used successfully as verifiers for
other reasoning models:

xVerify: Efficient Answer Verifier for Reasoning Model Evaluations,
https://arxiv.org/abs/2504.10481

###### A.4 Chapter 4: Improving reasoning with inference-time scaling

- A.4.1 References
The following paper that formally described chain-of-thought prompting. Note that the

paper suggested "Let's think step by step" as a prompt modification. However, in my
experiments, I found that "Explain step by step" performs better when using the Qwen3
base model, which is why we use the latter in chapter 4.

Large Language Models are Zero-Shot Reasoners, https://arxiv.
org/abs/2205.11916

A description of self-consistency sampling with additional comparison studies:

Self-Consistency Improves Chain-of-Thought Reasoning in Language
Models, https://arxiv.org/abs/2203.11171

- A.4.2 Further Reading


An overview and discussion of additional inference scaling methods:

The State of LLM Reasoning Model Inference, https://magazine.
sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling

##### A.5 Chapter 5: Inference-time scaling via self-refinement

- A.5.1 References
Google keeps the methods behind their proprietary Gemini 3 model a secret, but based

on a recent announcement, we can speculate that it uses inference scaling techniques
similar to self-consistency or Best-of-N: "We’re pushing the boundaries of intelligence even
further with Gemini 3 Deep Think. This mode meaningfully improves reasoning capabilities
by exploring many hypotheses simultaneously to solve problems."

Public announcement by Google DeepMind, which develops Gemini,
https://x.com/GoogleDeepMind/status/1996658401233842624?s=20

The DeepSeekMath-V2 paper showed that self-consistency scaling can noticeably improve
the answer accuracy, and combining self-consistency with their version of self-refinement
(Best@32 in figure 2), the model achieved gold-level performance in several math
competitions:

DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning,
https://arxiv.org/abs/2511.22570v1

Instead of using a majority vote in self-consistency, we can use a scoring function to rank
the different answers and select the best one. This approach is also known as Best-of-N.
However, if applicable, majority voting often tends to give better results:

Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-
time-scaling Methods, https://arxiv.org/abs/2504.14047

- A.5.2 Further Reading
A short article explaining the difference between probabilities and likelihood:


What is the difference between likelihood and probability?,
https://sebastianraschka.com/faq/docs/probability-vs-likelihood.html

###### A.6 Chapter 6: Training reasoning models with reinforcement learning

- A.6.1 References
The InstructGPT paper demonstrated the effectiveness of RLHF and was instrumental in


popularizing RLHF as a standard alignment and fine-tuning approach for LLMs:

Training language models to follow instructions with human feedback,
https://arxiv.org/abs/2203.02155

The DeepSeekMath paper that introduced the GRPO algorithm

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
Language Models, https://arxiv.org/abs/2402.03300

The DeepSeek-R1 paper showed that strong reasoning behavior can emerge in LLMs
through reinforcement learning alone (via RLVR with GRPO). This was most clearly shown in
the R1-Zero variant. However, combining this approach with a multi-stage training pipeline
yields an even better reasoning model:

DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning, https://arxiv.org/abs/2501.12948

- A.6.2 Further Reading


A comprehensive walkthrough of the DeepSeek-R1 training pipeline involving RLVR:

Understanding Reasoning LLMs: Methods and Strategies for Building and
Refining Reasoning Models, https://magazine.sebastianraschka.
com/p/understanding-reasoning-llms

A comparison of GRPO and PPO for reinforcement learning in the context of LLMs:

The State of Reinforcement Learning for LLM Reasoning: Understanding
GRPO and New Insights from Reasoning Model Papers, https://magazine.
sebastianraschka.com/p/the-state-of-llm-reasoning-model-training

###### A.7 Chapter 7: Improving GRPO for reinforcement learning

- A.7.1 References
The original PPO paper that introduced clipped policy ratios, which we also use here to


stabilize GRPO:

Proximal Policy Optimization Algorithms, https://arxiv.org/abs/1707.
06347

Additional papers that recommend improvements to the GRPO algorithm:

DAPO: An Open-Source LLM Reinforcement Learning System at Scale,
https://arxiv.org/abs/2503.14476

Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO),
https://arxiv.org/abs/2503.20783

Your Efficient RL Framework Secretly Brings You Off-Policy RL Training
(VERL), https://fengyao.notion.site/off-policy-rl

DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models,
https://arxiv.org/abs/2512.02556

GDPO: Group reward-Decoupled Normalization Policy Optimization for
Multi-reward RL Optimization, https://arxiv.org/abs/2601.05242

Group Sequence Policy Optimization (GSPO), https://arxiv.org/abs/2507.
18071

MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning
Attention (CISPO), https://arxiv.org/abs/2506.13585

- A.7.2 Further Reading


A comparison between PPO (the original algorithm used for RLHF) and GRPO:

The State of Reinforcement Learning for LLM Reasoning,
https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-
model-training

A good technical deep dive that discusses different GRPO improvements:

GRPO++: Tricks for Making RL Actually Work, https://cameronrwolfe.
substack.com/p/grpo-tricks

###### A.8 Chapter 8: Distilling Reasoning Models for Efficient Reasoning

- A.8.1 References
The original knowledge-distillation paper that popularized the combination of hard and

soft distillation objectives:

Distilling the Knowledge in a Neural Network, https://arxiv.org/abs/1503.
02531

The DeepSeek-R1 paper that described the reasoning-distillation recipe, which motivated
this chapter, where a large teacher model generates reasoning traces that are then used to
train smaller student models:

DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning, https://arxiv.org/abs/2501.12948

A paper on distilling large language models that reported strong results for carefully
designed soft-distillation objectives:

MiniLLM: Knowledge Distillation of Large Language Models, https://arxiv.
org/abs/2306.08543

- A.8.2 Further Reading


More details on supervised fine-tuning (the technique in hard distillation) and masking
when working with batches of training examples

Build A Large Language Model (From Scratch) chapter 7, https://www.
manning.com/books/build-a-large-language-model-from-scratch

A practical walkthrough of reasoning-model training pipelines, including distillation and
RLVR:

Understanding Reasoning LLMs: Methods and Strategies for Building and
Refining Reasoning Models, https://magazine.sebastianraschka.
com/p/understanding-reasoning-llms

###### A.9 Appendix F: Common approaches to LLM evaluation

- A.9.1 References
The paper that introduced the popular multiple-choice MMLU dataset:

Measuring Massive Multitask Language Understanding, https://arxiv.
org/abs/2009.03300

A detailed description of the Elo rating system:

Elo rating system, https://en.wikipedia.org/wiki/Elo_rating_system

The Chatbot Arena paper describing the original methodology behind the popular LLM
leaderboard:

Chatbot Arena: An Open Platform for Evaluating LLMs by Human
Preference, https://arxiv.org/abs/2403.04132

- A.9.2 Further Reading
A paper discussing the problems with leaderboards such as LM Arena:


The Leaderboard Illusion, http://arxiv.org/abs/2504.20879

An article by the author describing gpt-oss in more detail:

From GPT-2 to gpt-oss: Analyzing the Architectural Advances,
https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-
analyzing-the

A survey of different LLM judge approaches:

A Survey on LLM-as-a-Judge, https://arxiv.org/abs/2411.15594

Example of a small LLM fine-tuned to act as a judge:

###### PHUDGE: Phi-3 as Scalable Judge, https://arxiv.org/abs/2405.08029

## Appendix B. Exercise solutions

The complete code examples for the exercise solutions can be found in the supplementary
GitHub repository at https://github.com/rasbt/reasoning-from-scratch.

###### B.1 Chapter 2

- EXERCISE 2.1: ENCODING UNKNOWN WORDS
We can use a prompt similar to "Hello, Ardwarklethyrx. Haus und Garten.", which


contains a made-up word ("Ardwarklethyrx") and three words in a non-English language
(German):

"Haus und Garten":
prompt = "Hello, Ardwarklethyrx. Haus und Garten."
input_token_ids_list = tokenizer.encode(prompt)
for i in input_token_ids_list:

print(f"{[i]} --> {tokenizer.decode([i])}")

The output is:

[9707] --> Hello
[11] --> ,
[1644] --> Ar
[29406] --> dw
[838] --> ark
[273] --> le
[339] --> th
[10920] --> yr
[87] --> x
[13] --> .
[47375] --> Haus
[2030] --> und
[93912] --> Garten
[13] --> .

As we can see, unknown words are broken into smaller pieces of subwords or even single
tokens; this allows the tokenizer and LLM to handle any input.

German words are not broken down into characters or even subwords here, suggesting
that the tokenizer has seen German texts during training. This also suggests that the LLM
was likely trained on German texts, too, and should be able to handle at least certain non-
English languages well.

###### EXERCISE 2.2: RERUN CODE ON NON-CPU DEVICES

We can simply delete the line device = torch.device("cpu") in section 2.5, and then
rerun the rest of the code in chapter 2 as is. Reference numbers for the hardware I tried
the code on are provided in table 2.1 at the end of chapter 2.

##### B.2 Chapter 3

- EXERCISE 3.1: ADDING MORE TEST CASES
There is an endless number of different test cases we may add. Below is a selection of


some interesting ones:

from reasoning_from_scratch.ch03 import (

run_demos_table
)

more_tests = [

- ("check_17", "[1, 2]", "(1, 2)", True), #A
- ("check_18", "1e-3", "0.001", True), #B
- ("check_19", "(-3)^2", "9", True), #C
- ("check_20", "−1", "-1", True), #D

]
run_demos_table(more_tests)

Test | Expect | Got | Status

- check_17 | True | True | PASS
- check_18 | True | True | PASS
- check_19 | True | True | PASS
- check_20 | True | False | FAIL


text = text.replace("−", "-")

text = text.replace("\u2212", "-")

- extra_tests_1 = [


- ("check_21", "Text around answer 3.", "3", True)


- #A Different bracket types
- #B Scientific notation
- #C Algebraic simplification with caret exponent
- #D Unicode minus (U+2212) vs ASCII hyphen-minus


The output is:

As we can see, the tests pass in all cases except for check_20, which swaps the regular
sign with a Unicode version of a minus sign that looks indistinguishable to the human eye
(depending on which font or editor we use). We could fix this test case by adding one of the
following lines anywhere to the normalize_text function:

or

Another interesting test is the following one:

]

We can run it via the following code:

- run_demos_table(extra_tests_1)

Test | Expect | Got | Status
check_21 | True | False | FAIL

- Passed 0/1

from reasoning_from_scratch.ch03 import (
extract_final_candidate

)
extra_tests_2 = [

("check_21",
extract_final_candidate("Text around answer 3."),
"3", True)

]

run_demos_table(extra_tests_2)
Test | Expect | Got | Status
check_21 | True | True | PASS

- Passed 1/1




However, it fails the test:

While it may seem that our code cannot handle such text-containing cases, this is actually a
poorly designed test. In practice, the run_demos_table function is intended specifically to
test the grade_answer function; nothing more, nothing less.

The grade_answer function would never receive the entire answer in this text form,
since the answer would have been extracted from the text before being passed to it. For
instance, if we want to test text answers, we need to call the test as follows:

As we can see based on the output, it now passes the test:

###### EXERCISE 3.2: CALCULATING THE AVERAGE RESPONSE LENGTH

There are two options to calculate the average response length. The first option is to modify
the evaluate_math500_stream function (listing 3.13 in chapter 3) by adding the following
lines:

# ...
# below `num_correct = 0`
total_len = 0

# ...
# inside for i, row in enumerate(math_data, start=1):
# anywhere below `gen_text = ...`
total_len += len(tokenizer.encode(gen_text))

# ...
# anywhere at the bottom before the return statement
avg_len = total_len / num_examples
print(f"Average length: {avg_len:.2f} tokens")

Alternatively, the second option is to calculate the response lengths from the .jsonl files
that were created when we ran the evaluate_math500_stream function in the main
chapter. This way, we avoid having to rerun the evaluation.

First, we load the .jsonl file as follows:

import json
from pathlib import Path

WHICH_MODEL = "base"
dev_name = "mps"

local_path = Path(f"math500-{dev_name}.jsonl") #A
if not local_path.exists():

raise FileNotFoundError(

f"{local_path} not found. Run ch03_main.ipynb to create it."
)

results = []
with open(local_path, "r") as f:

for line in f:
if line.strip():
results.append(json.loads(line))

print("Number of entries:", len(results))

#A You may need to adjust this path

Let's print the dictionary keys to get a better idea of how the results dataset is structured:

print(results[0].keys())

This prints:

dict_keys(['index', 'problem', 'gtruth_answer', 'generated_text',
'extracted', 'correct'])

Note that each entry has multiple keys, however, we are only interested in the
"generated_text" key, which contains the model's full answer. Next, we need to load the
tokenizer so that we can tokenize the answer text before we can calculate the number of
tokens. This is similar to the code we used in listing 3.1 in chapter 3:

from reasoning_from_scratch.qwen3 import (
download_qwen3_small,
Qwen3Tokenizer

)

if WHICH_MODEL == "base":

download_qwen3_small(
kind="base", tokenizer_only=True, out_dir="qwen3"

)
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

elif WHICH_MODEL == "reasoning":

download_qwen3_small(
kind="reasoning", tokenizer_only=True, out_dir="qwen3"

)
tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(

tokenizer_file_path=tokenizer_path,
apply_chat_template=True,
add_generation_prompt=True,
add_thinking=True,

)

Then, we can calculate the average length as follows, which is similar to how we could have
modified the evaluate_math500_stream function:

total_len = 0

for item in results:
num_tokens = len(tokenizer.encode(item["generated_text"]))
total_len += num_tokens

avg_len = total_len / len(results)
print(f"Average length: {avg_len:.2f} tokens")

The resulting average length is as follows:

Average length: 98.00 tokens

- Table B.1 lists the average lengths for the different models and subsets.


- Table B.1 Average number of tokens on MATH-500


|Model|Device|Average<br>length|MATH-500<br>size|
|---|---|---|---|
|Base|CPU|97.30|10|
|Base|CUDA|96.74|500|
|Reasoning|CPU|891.80|10|
|Reasoning|CUDA|1361.21|500|


- As we can see based on the results in table B.1, and as expected, the reasoning model
generates much longer responses (in this case, approximately 10-times longer).


###### EXERCISE 3.3: EXTENDING OR CHANGING THE EVALUATION DATASET

To evaluate the model on a larger dataset, we can simply change the math_data[:10] to a
different slice or larger number (up to 500) in the following function call:

num_correct, num_examples, acc = evaluate_math500_stream(
model, tokenizer, device,
math_data=math_data[:10],
max_new_tokens=2048,
verbose=False

)

- Table B.2 below shows the accuracy values for different dataset sizes. (Since the MATH-500
test set is already shuffled, no additional shuffling was applied.)


- Table B.2 Accuracies for different MATH-500 dataset sizes


|Model|Device|Accuracy|MATH-500<br>size|
|---|---|---|---|
|Base|CUDA|30.0%|10|
|Base|CUDA|34.0%|50|
|Base|CUDA|27.0%|100|
|Base|CUDA|15.3%|500|
|Reasoning|CUDA|90.0%|10|
|Reasoning|CUDA|58.0%|50|
|Reasoning|CUDA|56.0%|100|
|Reasoning|CUDA|48.2%|500|


- As we can see based on the results in table B.2, the first 10 examples are not very
representative of the MATH-500 performance evaluated on the whole 500 examples.


In addition, we can create an entirely new dataset in a similar style to MATH-500. For
example, a dataset in MATH-500 style is included in this repository; we can use it in the
main chapter by changing the filename from math500_test.json to
math_new50_exercise.json (this dataset is included in this book's GitHub repository at
https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03/01_main-chapter-code).

The performance of the models is as follows:

base: 36.0% (18/50)

reasoning: 80.0% (40/50)

Accuracy is similar for the base model and higher for the reasoning model compared to the
50-example subset of the MATH-500 test set (table B.2). This indicates that, despite the
possibility of overlap with Qwen3’s training data, the model generalizes well to new math
questions and does not show signs of extensive overfitting to the original MATH-500 data.

###### EXERCISE 3.4: EXPERIMENTING WITH DIFFERENT PROMPT TEMPLATES

We could use the alternative prompt similar to the one suggested in the chapter, which
modifies the prompt to use the word "problem" instead of "question":

def render_prompt(prompt):

template = (
"You are a helpful math assistant.\n"
"Solve the problem and write the final "
"result on a new line as:\n"
"\\boxed{ANSWER}\n\n"
f"Problem:\n{prompt}\n\nAnswer:"

)
return template

Using this prompt improves the performance of the base model, on the 500 examples, from
15.3% to 31.2%. Also, it improves the performance of the reasoning model from 48.2% to
50.0%

From these observations, we may conclude that the base model is much more sensitive
to the prompt format (likely due to memorizing some prompt-formatted MATH-500
examples from the training set) than the reasoning model; the latter seems largely
unaffected.

##### B.3 Chapter 4

- EXERCISE 4.1: USE CHAIN-OF-THOUGHT PROMPTING ON MATH-500
The modification only requires adding a prompt suffix such as "\n\nExplain step by


step." after applying the prompt template. There is only a very small portion of code that
needs to be updated in the MATH-500 evaluation function from chapter 3, as shown below:

def evaluate_math500_stream(...):
# ...

for i, row in enumerate(math_data, start=1):
prompt = render_prompt(row["problem"])
prompt += "\n\nExplain step by step." # NEW
gen_text = generate_text_stream_concat(

model, tokenizer, prompt, device,
max_new_tokens=max_new_tokens,
verbose=verbose,

)

# ...

The improvements are shown in row 3 in table 4.1, which can be found in section 4.6 in
chapter 4.

###### EXERCISE 4.2: USE TEMPERATURE SCALING AND TOP-P FILTERING ON MATH-500

Here, we replace the generate_text_stream_concat function with
generate_text_stream_concat_flex and pass in generate_text_top_p_stream_cache as
its generation function. The updated MATH-500 evaluation function from chapter 3 is shown
below, and the changes are marked with comments labeled # NEW.

def evaluate_math500_stream(
model,
tokenizer,
device,
math_data,
out_path=None,
max_new_tokens=512,
verbose=False,
temperature=1.0, # NEW
top_p=1.0, # NEW

):

# ...

with open(out_path, "w", encoding="utf-8") as f:

for i, row in enumerate(math_data, start=1):
prompt = render_prompt(row["problem"])
gen_text = generate_text_stream_concat_flex( # NEW

model, tokenizer, prompt, device,
max_new_tokens=max_new_tokens,
verbose=verbose,
generate_func=generate_text_top_p_stream_cache, # NEW
temperature=temperature, # NEW
top_p=top_p # NEW

)
# ...

The difference between this modified function and the baseline from chapter 3 can be seen
in rows 1 and 4 in table 4.1, which can be found in section 4.6 in chapter 4.

###### EXERCISE 4.3: USE SELF-CONSISTENCY SAMPLING ON MATH-500

Starting from the evaluate_math500_stream function in chapter 3, the first modification is
to replace the line gen_text = generate_text_stream_concat(...) with a call to
results = self_consistency_vote(...) from chapter 4. The second modification adds a
simple tie-breaking rule that selects the first occurrence of the most frequent answer. For
instance, if the sampled results are 1, 3, 5, 3, 5, the function would return 3 because it is
the earliest member of the most frequent group.

Since the most frequent answers are stored in results["majority_winners"], one
straightforward way to break ties is to take the first element of this list, that is,
results["majority_winners"][0].

Those changes are illustrated in the code excerpts below:

def evaluate_math500_stream(
model,
tokenizer,
device,
math_data,
out_path=None,
max_new_tokens=2048,
verbose=False,
prompt_suffix="", # NEW
temperature=1.0, # NEW
top_p=1.0, # NEW
seed=None, # NEW
num_samples=10, # NEW

):

if out_path is None:
dev_name = str(device).replace(":", "-")
out_path = Path(f"math500-{dev_name}.jsonl")

num_examples = len(math_data)
num_correct = 0
start_time = time.time()

with open(out_path, "w", encoding="utf-8") as f:
for i, row in enumerate(math_data, start=1):

prompt = render_prompt(row["problem"])

##############################################################
# NEW
prompt += prompt_suffix
results = self_consistency_vote(

model=model,
tokenizer=tokenizer,
prompt=prompt,
device=device,
num_samples=num_samples,
temperature=temperature,
top_p=top_p,
max_new_tokens=max_new_tokens,
show_progress=False,
show_long_answer=False,
seed=seed,

)

# resolve ties
if results["final_answer"] is None:

extracted = results["majority_winners"][0]
else:

extracted = results["final_answer"]

# extracted = extract_final_candidate(
# gen_text
# )

# Optionally, get long answer
if extracted is not None:

for idx, s in enumerate(results["short_answers"]):

if s == extracted:
long_answer = results["full_answers"][idx]
break

gen_text = long_answer
##############################################################

is_correct = grade_answer(
extracted, row["answer"]

)
num_correct += int(is_correct)

# ...

The performance improvements when using self-consistency sampling are summarized and
discussed in table 4.1 in chapter 4 (rows 5-7 and rows 9-12), which can be found in section
4.6 of chapter 4.

###### EXERCISE 4.4: EARLY STOPPING IN SELF-CONSISTENCY SAMPLING

The early stopping check can be implemented by adding a few lines of code that check
whether the given answer is already counted multiple times, or, more specifically, if the
given answer count is greater than num_samples / 2:

if early_stop and counts[short] > num_samples / 2:
majority_winners = [short]
final_answer = short
break

The excerpt of the modified self_consistency_vote function below illustrates more
specifically where to insert this code:

def self_consistency_vote(
# ...
early_stop=True, # NEW

):

# ...

if show_progress:
print(f"[Sample {i+1}/{num_samples}] → {short!r}")

#########################################################
# NEW
# Early stop if one answer already meets >= 50% majority
if early_stop and counts[short] > num_samples / 2:

majority_winners = [short]
final_answer = short
break

#########################################################

if final_answer is None:
mc = counts.most_common()
if mc:

top_freq = mc[0][1]
majority_winners = [s for s, f in mc if f == top_freq]
final_answer = mc[0][0] if len(majority_winners) == 1 else None

return {
"full_answers": full_answers,
"short_answers": short_answers,
"counts": dict(counts),
"groups": groups,
"majority_winners": majority_winners,
"final_answer": final_answer,

}

##### B.4 Chapter 5

- EXERCISE 5.1: USING THE HEURISTIC SCORER AS A TIE-BREAKER IN SELF-
CONSISTENCY


There are many ways to implement this. Perhaps the easiest approach is to handle it
outside the self-consistency function and work directly with the returned dictionary, similar
to what we did in exercise 4.4 when we implemented the tie-breaking logic directly inside
the evaluate_math500_stream function. The relevant lines are shown below:

# ...
from reasoning_from_scratch.ch05 import heuristic_score

def evaluate_math500_stream(
#...
# ...

results = self_consistency_vote(...)
# Majority vote winner available
if results["final_answer"] is not None:

extracted = results["final_answer"]

### NEW: Break tie with heuristic_score
else:

best = None
best_score = float("-inf")
for cand in results["majority_winners"]:

scores = [
heuristic_score(results["full_answers"][idx],
prompt=prompt)
for idx in results["groups"][cand]

]
score = max(scores)
if score > best_score:

best_score = score
best = cand

extracted = best
# ...

# ...
return num_correct, num_examples, acc

The results are shown in table B.3.

- Table B.3 MATH-500 self-consistency score with different tie-breaking


| |Method|Model|Accuracy|Time|
|---|---|---|---|---|
|1|Baseline with chain-of-thought<br>prompting|Base|33.4%|129.2<br>min|
|2|Self-consistency (n=3)|Base|43.2%|328.2<br>min|
|3|Self-consistency (n=3) + heuristic|Base|43.4%|326.5<br>min|
|4|Self-consistency (n=3) + avg. logprob|Base|44.8%|327.7<br>min|


The accuracy values and runtimes shown in the table were computed on all 500 samples in
the MATH-500 test set using a "cuda" GPU (DGX Spark).

Row 1 in table B.3 is the baseline from chapter 4 without self-consistency. Row 2 doesn't
use a scorer for tie-breaking, so if there is a tie among the answers, it chooses the answer
with the first appearance. Using a heuristic scorer (row 3) as tie-breaker results in a slight
improvement. And the best (but also minimal) improvement is achieved with the logprob
scorer as tie-breaker (row 4).

###### EXERCISE 5.2: USING THE HEURISTIC SCORER IN A BEST-OF-N SETUP

Best-of-N is similar to self-consistency in that we generate multiple answers. However,
instead of selecting the final answer via a majority vote, we score all generated answers
using a scoring function, such as heuristic_score, and return the highest-scoring one.
There are several ways to implement this behavior, but the simplest approach is to use the
existing self-consistency function from chapter 4 as a template and swap in
heuristic_score, as shown below:

# ...
from reasoning_from_scratch.ch05 import (

heuristic_score
)

def self_consistency_vote( #...):
full_answers, short_answers = [], []
counts = Counter()
groups = {}
majority_winners, final_answer = [], None
best_score, best_idx = float("-inf"), None

for i in range(num_samples):
if seed is not None:

torch.manual_seed(seed + i + 1)

answer = generate_text_stream_concat_flex(
model=model,
tokenizer=tokenizer,
prompt=prompt,
device=device,
max_new_tokens=max_new_tokens,
verbose=show_long_answer,
generate_func=generate_text_top_p_stream_cache,
temperature=temperature,
top_p=top_p,

)

short = extract_final_candidate(answer, fallback="number_then_full")
full_answers.append(answer)
short_answers.append(short)
counts[short] += 1

if short in groups:

groups[short].append(i)
else:

groups[short] = [i]

score = heuristic_score(answer, prompt=prompt)

if score > best_score:
best_score, best_idx = score, i

# ...

- Table B.4 MATH-500 Best-of-N scores with heuristic and average logprob scores


| |Method|Model|Accuracy|Time|
|---|---|---|---|---|
|1|Baseline with chain-of-thought<br>prompting|Base|33.4%|129.2<br>min|
|2|Best-of-N (n=3) + heuristic|Base|40.6%|327.7<br>min|
|3|Best-of-N (n=3) + avg. logprob|Base|43.2%|330.2<br>min|


The accuracy values and runtimes shown in the table were computed on all 500 samples in
the MATH-500 test set using a "cuda" GPU (DGX Spark).

- EXERCISE 5.3: USING THE LOGPROB SCORER AS A TIE-BREAKER IN SELF-
CONSISTENCY


The code is similar to exercise 5.1, except that we swap heuristic_score with
avg_logprob_answer, as shown below:

# ...
# from reasoning_from_scratch.ch05 import heuristic_score
from reasoning_from_scratch.ch05 import avg_logprob_answer

def evaluate_math500_stream(# ...)
# ...

# score = heuristic_score(
# candidate_full, prompt=prompt
# )
score = avg_logprob_answer(

model=model,
tokenizer=tokenizer,
prompt=prompt,
answer=candidate_full,
device=device,

)
# ...

The results were already included in the previous table B.3 (exercise 5.1) in row 4.

###### EXERCISE 5.4: USING THE LOGPROB SCORER IN A BEST-OF-N SETUP

To implement Best-of-N with a logprob scorer, we can use the code from exercise 5.2 and
swap the heuristic_score with avg_logprob_answer, as shown below:

from reasoning_from_scratch.ch05 import (
avg_logprob_answer

)
# ...

score = avg_logprob_answer(
model=model,
tokenizer=tokenizer,
prompt=prompt,
answer=answer,
device=device

)
if score > best_score:

best_score, best_idx = score, i
# ...

The resulting MATH-500 score is shown in table B.4 above (exercise 5.2).

###### EXERCISE 5.5: USING THE HEURISTIC SCORE FOR SELF-REFINEMENT

Using the heuristic_score is actually even simpler than using the logprob score; all we
need to do is change the following code:

from functools import partial

avg_logprob_score = partial(
avg_logprob_answer,
model=model,
tokenizer=tokenizer,
device=device

)
torch.manual_seed(0)

results_logprob = self_refinement_loop(
model=model,
tokenizer=tokenizer,
raw_prompt=raw_prompt,
device=device,
iterations=2,
max_response_tokens=2048,
max_critique_tokens=256,
score_fn=avg_logprob_score,
verbose=True,
temperature=0.7,
top_p=0.9,

)

The updated code is:
torch.manual_seed(0)
results_logprob = self_refinement_loop(

model=model,
tokenizer=tokenizer,
raw_prompt=raw_prompt,
device=device,
iterations=2,
max_response_tokens=2048,
max_critique_tokens=256,
score_fn=heuristic_score, # NEW
verbose=True,
temperature=0.7,
top_p=0.9,

)

The improvements over the baseline in chapter 3 and self-consistency from chapter 4 are
shown in table 5.1 (rows 4, 5, and 10) in the main chapter.

##### B.5 Chapter 6

EXERCISE 6.1: ADDING FORMAT-AWARE REWARD SHAPING

We can assign a partial reward (score 0.5) if no "\boxed{}" answer is found as follows,
using the fallback="number_then_full" fallback we coded in chapter 3:

from reasoning_from_scratch.ch03 import (

extract_final_candidate, grade_answer
)

def reward_rlvr(answer_text, ground_truth):

- # 1) Try to extract a boxed answer
boxed = extract_final_candidate(

answer_text, fallback=None

)
if boxed:

correct = grade_answer(boxed, ground_truth)
return 1.0 if correct else 0.0

- # 2) If no boxed answer is found, look for number
unboxed = extract_final_candidate(


answer_text, fallback="number_then_full"

)
if unboxed:

correct = grade_answer(unboxed, ground_truth)
return 0.5 if correct else 0.0

return 0.0

When plugged into the chapter 6 code and trained under the same settings, the partial-
reward variant achieves lower accuracy (37.8%) than the standard GRPO setup (47.4%),
despite using a similar number of tokens on average, as shown in table B.5.

- Table B.5 MATH-500 accuracies for strict and partial rewards


| |Method|Step|Max<br>tokens|Num<br>rollouts|Accuracy|Average<br>tokens|
|---|---|---|---|---|---|---|
|1|GRPO (chapter 6)|50|512|8|47.4%|586.11|
|2|GRPO partial<br>rewards (exercise<br>6.2)|50|512|8|37.8%|550.33|


###### EXERCISE 6.2: ZERO-ADVANTAGE CASES

If the rewards are all equal (for instance, they are all 0 or all 1), the advantages will all be
0, because subtracting the mean removes the shared reward value and leaves only zeros,
which we can demonstrate below:

import torch

rollout_rewards = [0., 0., 0., 0.]
rewards = torch.tensor(rollout_rewards)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
print(advantages)

This returns tensor([0., 0., 0., 0.]).

Similarly, if we change the rollout rewards to rollout_rewards = [0., 0., 0., 0.],
we get the same all-zero tensor, tensor([0., 0., 0., 0.]).

In short, if all rewards in a group are identical, for example all rewards 0 or all rewards
are 1, then 𝑟𝑖 -𝜇𝑖 = 0 for all i rollouts. As a result, the policy gradient is zero and the model
parameters are not updated for that prompt.

This behavior is intentional. If all rollouts are equally bad or equally good, there is no
relative signal to tell the model which behavior to reinforce or suppress. Intuitively, if the
model answers all the questions correctly, there is no need to update it. Vice versa, if the
model answers all questions incorrectly, we don't want to update the model to reinforce this
behavior.

##### B.6 Chapter 7

- EXERCISE 7.1: TESTING THE <THINK> FORMAT REWARD
The following code checks that the format reward is zero if the think tokens are used


incorrectly:

from pathlib import Path
import torch
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from reasoning_from_scratch.qwen3 import download_qwen3_small
from reasoning_from_scratch.ch07 import reward_format

download_qwen3_small(
kind="reasoning", tokenizer_only=True, out_dir="qwen3"

)
tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

prompt = "Calculate ..."

def check_case(name, rollout):
token_ids = tokenizer.encode(prompt + rollout)
prompt_len = len(tokenizer.encode(prompt))
reward = reward_format(

token_ids=torch.tensor(token_ids),
prompt_len=prompt_len,

)
print(f"{name}: {reward}")

- # 1) Correct case
check_case(

"Correct order",
"Let's ... <think> ... </think> ..."

)

- # 2) Typo in tag
check_case(

"Typo in <think>",
"Let's ... <thnik> ... </think> ..."

)

- # 3) Reversed order
check_case(

"Reversed order",
"Let's ... </think> ... <think> ..."

)

- # 4) Missing one tag
check_case(


"Missing </think>",

"Let's ... <think> ..."
)

The output is as follows, indicating that the function requires correct <think>...</think>
tag use to award a reward of 1.0

Correct order: 1.0
Typo in <think>: 0.0
Reversed order: 0.0
Missing </think>: 0.0

###### EXERCISE 7.2: MAKING THE FORMAT REWARD CONDITIONAL

The implementation of the conditional reward is very simple; in the main chapter, we
discussed implementing the overall reward as follows:

reward = rlvr_reward + format_reward_weight * format_reward

So, one way to disable the reward if the correctness reward (rlvr_reward) is 0.0 is

if conditional_reward:
format_reward *= rlvr_reward
reward = rlvr_reward + format_reward_weight * format_reward

To use it in practice, you can run the 7_6_plus_format_reward.py script, which we used in
section 7.6 in chapter 7 with the --conditional_reward flag enabled.

We can download the log file of this run (using similar settings as in section 7.6) and plot
it as follows:

from reasoning_from_scratch.ch07 import download_from_github
from reasoning_from_scratch.ch07 import plot_grpo_metrics
download_from_github(

"ch07/02_logs/7_6_plus_format_reward_conditional_metrics.csv"

)
plot_grpo_metrics(

"7_6_plus_format_reward_conditional_metrics.csv",
columns=["loss", "reward_avg", "avg_response_len", "eval_acc"],

)

![image 153](<input (1)_images/imageFile153.png>)

- Figure B.1 Basic metrics from a GRPO training run with a conditional format reward.


The plots in figure 7.1 show that the evaluation accuracy and reward average take a big hit,
but seem to recover.

Overall, despite this performance crash, this looks more stable than before, and the
trend indicates that the performance would improve further if we trained longer.

plot_grpo_metrics(
"7_6_plus_format_reward_conditional_metrics.csv",
columns=["reward_avg", "format_reward_avg", "adv_std", "entropy_avg"],

)

![image 154](<input (1)_images/imageFile154.png>)

- Figure B.2 Additional metrics from a GRPO training run with a conditional format reward.


In figure 7.2, we see the average format reward mimicking the average reward graph
almost perfectly, which is a good sanity check that the conditional logic is working. Also, the
average format reward shows how much of the total reward is coming from the format term
on the subset of correct answers.

As we can see though, since the average format reward graph echoes the average
reward one, it's mainly a bonus (and it looks like it's always awarded if the model is
correct; this makes sense, because the trained reasoning model already knows how to use
<think>...</think> tags correctly (and we can see that it doesn't unlearn this ability).

The entropy increase is still a bit troubling, though, and could hint towards training
instabilities that could potentially be addressed by other means (like tighter clipping with
smaller clip_eps).

##### B.7 Chapter 8

- EXERCISE 8.1: TRAINING AND VALIDATION SET LENGTHS
To calculate the training and validation answer length statistics, you can add the


following commands at the end of section 8.4.3, following the partitioning:

compute_length(train_examples)
# Prints
# Average: 1180 tokens
# Shortest: 236 tokens (index 5730)
# Longest: 2048 tokens (index 1319)

and

To calculate the training and validation answer length statistics, you can add the
following commands at the end of section 8.4.3, following the partioning:

compute_length(train_examples)
# Prints
# Average: 1180 tokens
# Shortest: 236 tokens (index 5730)
# Longest: 2048 tokens (index 1319)

As we can see, the average token length (1180 versus 1106) is fairly similar, and the
datasets should be relatively balanced.

As a bonus, we can also plot histograms to visualize the distributions:

import matplotlib.pyplot as plt

train_lengths = [len(ex["token_ids"]) for ex in train_examples]
val_lengths = [len(ex["token_ids"]) for ex in val_examples]

# Normalize counts because the validation split is much smaller
bins = range(0, max(train_lengths + val_lengths) + 64, 64)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(train_lengths, bins=bins, density=True, alpha=0.6, label="Train")
ax.hist(val_lengths, bins=bins, density=True, alpha=0.6, label="Validation")
ax.set_xlabel("Token length")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()

The resulting plot is shown in figure B.3.

![image 155](<input (1)_images/imageFile155.png>)

- Figure B.3 Distribution of training and validation set lengths.


There are much fewer validation samples, which is why the validation histogram seems a
bit jagged, but as we can see, it has a good distribution coverage.

- EXERCISE 8.2: DISTILLING WITHOUT <THINK> TOKENS
To replicate the runs without <think></think> tokens in the script execution command:


uv run distill.py \

--data_path deepseek-r1-math-train.json \

--validation_size 25 \

--epochs 3 \

--lr 1e-5 \

--max_seq_len 2048 \

--grad_clip 1.0

Then, for the evaluation, we use the base instead of reasoning model:

uv run evaluate_math500.py \

--dataset_size 500 \
--which_model base \
--max_new_tokens 4096 \

--checkpoint_path \
run_11/checkpoints/distill/qwen3-0.6B-distill-step05746-epoch1.pth

The results are shown in table B.6.

- Table B.6 MATH-500 task accuracy with and without think tokens


| |Method|Epoch|Final val loss|MATH-500<br>Acc.|
|---|---|---|---|---|
|1|Base Qwen3 0.6B (chapter<br>3)|-|-|15.2%|
|2|Reasoning Qwen3 0.6B<br>(chapter 3)|-|-|48.2%|
|3|DeepSeek-R1|1|0.5436<br>(0.5404)|31.8%<br>(30.6%)|
|4|DeepSeek-R1|2|0.5349<br>(0.5339)|31.8%<br>(32.4%)|
|5|DeepSeek-R1|3|0.5343<br>(0.5306)|30.2%<br>(33.6%)|
|6|Qwen3 235B-A22B|1|0.4043<br>(0.3130)|44.8%<br>(45.0%)|
|7|Qwen3 235B-A22B|2|0.3963<br>(0.3087)|39.4%<br>(43.8%)|
|8|Qwen3 235B-A22B|3|0.3948<br>(0.3078)|39.8%<br>(44.2%)|


In table B.6, the new results (without think tokens) are shown first, with corresponding
think-token results (from the main chapter) in parentheses.

Interestingly, the Qwen3 model has a lower validation loss when <think></think>
tokens are omitted, but this doesn't translate into better modeling performance.

As we can see, the omission of <think></think> makes the results slightly worse in
almost all cases.

## Appendix C. Qwen3 LLM source code

While this is a from scratch book, as mentioned in the main chapters, the from scratch part
refers to the reasoning techniques, not the LLM itself. Implementing an LLM entirely from
scratch would require a separate book, which is the topic of my Build A Large Language
Model (From Scratch) book (http://mng.bz/orYv).

However, for readers interested in seeing the Qwen3 implementation we use in this Build
A Reasoning Model (From Scratch) book, this appendix lists the source code for the
Qwen3Model model that I implemented in and that we import from the book's
reasoning_from_scratch Python package:

from reasoning_from_scratch.qwen3 import Qwen3Model, Qwen3Tokenizer

As shown in figure C.1, the Qwen3 architecture is very similar to GPT-2, which is covered in
my Build A Large Language Model (From Scratch) book. While familiarity with GPT-2 is not
required for this book, this appendix mentions comparisons to GPT-2 for those who are
familiar with it. In fact, I wrote the Qwen3 implementation by porting the GPT-2 model
from my other book piece by piece into the Qwen3 architecture, such that it follows similar
style conventions to improve readability.

![image 156](<input (1)_images/imageFile156.png>)

- Figure C.1 Architectural comparison between Qwen3 and GPT-2. Both models process text through
embedding layers and stacked transformer blocks, but they differ in certain design choices.


As shown in figure C.1, both Qwen3 (released in 2025) and GPT-2 (released in 2019) are
very similar overall in that they are both based on the decoder submodule of the original
transformer architecture. However, some of the design choices have evolved since 2019.
Note that most of these design choices found in Qwen3 are not unique to Qwen3 but are
found in many other contemporary LLMs, which I discussed in my The Big LLM Architecture
Comparison (https://magazine.sebastianraschka.com/p/the-big-llm-architecture-
comparison) article.

For readers new to LLMs who want to understand how these architectures are
implemented, I recommend starting with GPT-2. Its design is simpler to implement, which
makes it an easier entry point before exploring more modern variations.

Since this book does not focus on architecture implementations, the remainder of this
appendix will cover only a brief overview of Qwen3's code.

##### C.1 Root mean square layer normalization (RMSNorm)

In contrast to GPT-2, which used standard LayerNorm, the newer Qwen3 architecture
replaces it with root mean square layer normalization (RMSNorm). This is a trend that has
become increasingly common in recent model architectures.

RMSNorm fulfills the same core function as LayerNorm: normalizing layer activations to
stabilize and improve training. However, it simplifies the computation by removing the
mean-centering step, as shown in figure C.2. This means that activations will still be
normalized, but they are not centered at 0.

![image 157](<input (1)_images/imageFile157.png>)

- Figure C.2 Comparison of LayerNorm (used in GPT-2) and RMSNorm (used in Qwen3). LayerNorm (left)
normalizes activations so that their average value (mean) is exactly zero and their spread (variance) is exactly
one. RMSNorm (right) instead scales activations based on their root mean square, which does not enforce
zero mean or unit variance, but still keeps the mean and variance within a reasonable range for stable
training.


- As we can see in figure C.2, both LayerNorm and RMSNorm scale the layer outputs to be in
a reasonable range.


LayerNorm subtracts the mean and divides by the standard deviation such that the layer
outputs have a zero mean and unit variance (variance of one and standard deviation of
one), which results in favorable properties, in terms of gradient values, for stable training.

RMSNorm divides the inputs by the root mean square. This scales activations to a
comparable magnitude without enforcing zero mean or unit variance. In this particular
example shown in figure C.2, the mean is 0.77 and the variance is 0.41.

Both LayerNorm and RMSNorm stabilize activation scales and improve optimization;
however, RMSNorm is often preferred in large-scale LLMs because it is computationally
cheaper. Unlike LayerNorm, RMSNorm does not use a bias (shift) term by default, which
reduces the number of trainable parameters. Moreover, RMSNorm reduces the expensive
mean and variance computations to a single root-mean-square operation. This reduces the
number of cross-feature reductions from two to one, which lowers communication overhead
on GPUs and slightly improves training efficiency.

Listing C.1 shows what RMSNorm looks like in code.

- Listing C.1 RMSNorm .


import torch.nn as nn

class RMSNorm(nn.Module):

def __init__(
self,
emb_dim,
eps=1e-6,
bias=False,
qwen3_compatible=True,

):

super().__init__()
self.eps = eps
self.qwen3_compatible = qwen3_compatible
self.scale = nn.Parameter(torch.ones(emb_dim))
self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

def forward(self, x):
input_dtype = x.dtype

if self.qwen3_compatible:
x = x.to(torch.float32)

variance = x.pow(2).mean(dim=-1, keepdim=True)
norm_x = x * torch.rsqrt(variance + self.eps)
norm_x = norm_x * self.scale

if self.shift is not None:
norm_x = norm_x + self.shift

return norm_x.to(input_dtype)

Note that, for brevity, this appendix does not provide detailed code walkthroughs for each
LLM component. Instead, in section C.6, we will integrate all components into the
Qwen3Model class, load the pre-trained weights into it, and then use this model to generate
text in section C.9.

##### C.2 Feed forward module

The feed forward module (a small multi-layer perceptron) is replaced with a gated linear
unit (GLU) variant, introduced in a 2020 paper (https://arxiv.org/abs/2002.05202). In this
design, the standard two fully connected layers are replaced by three, as shown in figure
C.3.

![image 158](<input (1)_images/imageFile158.png>)

- Figure C.3 In GPT-2 (top), the feed forward module consists of two fully connected (linear) layers separated by
a non-linear activation function. In Qwen3 (bottom), this module is a gated linear unit (GLU) variant, which
adds a third linear layer (linear layer 3) and multiplies the output of this linear layer 3 elementwise with the
activated output of linear layer 1.


Qwen3's feed forward module (figure C.3) can be implemented as shown in listing C.2.

- Listing C.2 Qwen3 feed forward module .


class FeedForward(nn.Module):

def __init__(self, cfg):
super().__init__()
self.fc1 = nn.Linear(

cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"],
bias=False

)
self.fc2 = nn.Linear(

cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"],
bias=False

)
self.fc3 = nn.Linear(

cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"],
bias=False

)

def forward(self, x):
x_fc1 = self.fc1(x)
x_fc2 = self.fc2(x)
x = nn.functional.silu(x_fc1) * x_fc2 #A
return self.fc3(x)

#A The non-linear activation function here is a SiLU function, which will be discussed later

- At first glance, it might seem that the GLU feed forward variant used in Qwen3 should
outperform the standard feed forward variant in GPT-2, simply because it adds an extra
linear layer (three instead of two) and therefore appears to have more parameters.


However, this intuition is misleading. In practice, the fc1 and fc2 layers in the GLU
variant are each half the width of the fc1 layer in a standard feed forward module. In
practice, the GLU variant has fewer parameters.

To illustrate this with a concrete example, suppose the input dimension to the "Linear
layer 1" in figure C.3 is 1024. This corresponds to cfg["emb_dim"] in listing C.2. The
output dimension of fc1 is 3,072 (cfg["hidden_dim"]). Note that these are the actual
numbers used in the Qwen3 0.6B variant. In this case, we have the following parameter
counts for the GLU variant in listing C.2:

fc1: 1024 × 3,072 = 3,145,728
fc2: 1024 × 3,072 = 3,145,728
fc3: 1024 × 3,072 = 3,145,728
Total: 3 × 3,145,728 = 9,437,184 parameters

If we assume that fc1 in this GLU variant has half the width as would be typically chosen
for an fc1 in a standard feed forward module, the parameter counts of the standard feed
forward module would be as follows:

fc1: 1024 × 2×3,072 = 6,291,456
fc2: 1024 × 2×3,072 = 6,291,456
Total: 2 × 6,291,456 = 12,582,912 parameters

While GLU variants usually have fewer parameters than regular feed forward modules, they
perform better. The improvement comes from the additional multiplicative interaction
introduced by the gating mechanism, activation(x_fc1) * x_fc2, which increases the
model's expressivity. This is similar to how deeper, slimmer networks can outperform
shallower, wider ones, given proper training.

Before we proceed to the next section, there is one more thing to address. Note that the
feed forward module shown in figure C.3 contains an element labeled as "Activation
function, " whereas we used a nn.functional.silu activation as a concrete example in
listing C.2.

Historically, activation functions were a hot topic of debate until the deep learning
community largely converged on the rectified linear unit (ReLU) more than a decade ago.
ReLU is simple and computationally cheap, but it has a sharp kink at zero. This motivated
researchers to explore smoother functions such as the Gaussian error linear unit (GELU)
and the sigmoid linear unit (SiLU), as shown in figure C.4.

![image 159](<input (1)_images/imageFile159.png>)

- Figure C.4 Different activation functions that can be used in a feed forward module (neural network). GELU
and SiLU (Swish) offer smooth alternatives to ReLU, which has a sharp kink at input zero.


GELU involves the Gaussian cumulative distribution function (CDF). Computing this CDF is
slow because it uses piecewise logic and exponentials, which makes it hard to write fused,
optimized GPU kernels (although a tanh approximation exists that uses cheaper operations
and runs faster with near-identical results).

In short, while GELU produces smooth activation curves, it is overall computationally
more expensive than simpler functions.

Newer models have largely replaced GELU with the SiLU (also known as Swish) function,
which smoothly suppresses large negative inputs toward ~0 and is approximately linear for
large positive inputs, as shown in figure C.4.

SiLU has a similar smoothness, but it is slightly cheaper to compute than GELU and
offers comparable modeling performance. In practice, SiLU is now used in most
architectures, while GELU remains in use in only some models, such as Google's Gemma
open-weight LLM. In the implementation of the feed forward module in listing C.2, this SiLU
function is called via nn.functional.silu. The feed forward module in listing C.2 is also
often called SwiGLU, an abbreviation that is derived from the terms Swish and GLU.

##### C.3 Rotary position embeddings (RoPE)

In transformer-based LLMs, positional encoding is necessary because of the attention
mechanism. By default, attention treats the input tokens as if they have no order. In the
original GPT architecture, absolute positional embeddings addressed this by adding a
learned embedding vector for each position in the sequence, which is then added to the
token embeddings.

RoPE (short for rotary position embeddings) introduced a different approach: instead of
adding position information as separate embeddings, it encodes position information by
rotating the query and key vectors in the attention mechanism (section C.4) in a way that
depends on each token's position. RoPE is an elegant idea, but also a long topic in itself.
Interested readers can find more information in the original RoPE paper at https://arxiv.
org/abs/2104.09864. (While first introduced in 2021, RoPE became widely adopted with the
release of the original Llama model in 2023 and has since become a staple in modern LLMs,
so it is not unique to Qwen3.)

RoPE can be implemented in two mathematically equivalent ways: the interleaved form,
which pairs adjacent dimensions for rotation, or in a two-halves form, which splits the
dimension into cosine and sine halves for convenience. Listing C.3 implements the two-
halves variant, which can be easier to read.

- Listing C.3 RoPE functions .


import torch

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096,

dtype=torch.float32):
assert head_dim % 2 == 0, "Embedding dimension must be even"
inv_freq = 1.0 / (theta_base ** (

torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
/ head_dim

))
positions = torch.arange(context_length, dtype=dtype)
angles = positions[:, None] * inv_freq[None, :]
angles = torch.cat([angles, angles], dim=1)

cos = torch.cos(angles)
sin = torch.sin(angles)

return cos, sin

def apply_rope(x, cos, sin, offset=0):
batch_size, num_heads, seq_len, head_dim = x.shape #A
assert head_dim % 2 == 0, "Head dimension must be even"

x1 = x[..., : head_dim // 2] # First half #B
x2 = x[..., head_dim // 2:] # Second half #B

cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
# Shape after: (1, 1, seq_len, head_dim)

rotated = torch.cat((-x2, x1), dim=-1)
x_rotated = (x * cos) + (rotated * sin)

return x_rotated.to(dtype=x.dtype) #C

#A The shape is (batch_size, num_heads, seq_len, head_dim)
#B Split x into first half and second half
#C It's ok to use lower-precision after applying cos and sin rotation

The RoPE code in listing C.3 will be used in the grouped query attention mechanism in
section C.4.

###### ROPE IMPLEMENTATION VARIANTS

Readers familiar with the original RoPE paper (https://arxiv.org/abs/2104.09864),
you may be wondering about the particular implementation I have chosen.

There are two common styles to implement RoPE, which are mathematically
equivalent. The implementations mainly differ in how the rotation matrix pairs
dimensions. I chose the split-halves style as it is a bit easier to read and implement.

1) Split-halves style (this book, Hugging Face Transformers):
[ x0 x1 x2 x3 x4 x5 x6 x7 ]

│ │ │ │ │ │ │ │
▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
cos cos cos cos sin sin sin sin

Rotation matrix:

[ cosθ -sinθ 0 0 ... ]
[ sinθ cosθ 0 0 ... ]
[ 0 0 cosθ -sinθ ... ]
[ 0 0 sinθ cosθ ... ]

Here, the embedding dimensions are split into two halves and then each one is
rotated in blocks.

2) Interleaved (even/odd) style (original paper, Llama repo):

[ x0 x1 x2 x3 x4 x5 x6 x7 ]

│ │ │ │ │ │ │ │
▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
cos sin cos sin cos sin cos sin

Rotation matrix:

[ cosθ -sinθ 0 0 ... ]
[ sinθ cosθ 0 0 ... ]
[ 0 0 cosθ -sinθ ... ]
[ 0 0 sinθ cosθ ... ]

Here, embedding dims are interleaved as even/odd cosine/sine pairs.

Both layouts encode the same relative positions. The only difference is how
dimensions are paired.

##### C.4 Grouped query attention (GQA)

Grouped query attention (GQA) has become the standard, more compute- and parameter-
efficient alternative to the original multi-head attention (MHA) mechanism.

Unlike MHA, where each head also has its own set of keys and values, to reduce memory
usage, GQA groups multiple heads to share the same key and value projections, as shown
in figure C.5.

![image 160](<input (1)_images/imageFile160.png>)

- Figure C.5 A comparison between MHA and GQA. Here, the group size is 2, where a key and value pair is
shared among 2 queries.


So, the core idea behind GQA, shown in figure C.5, is to reduce the number of key and
value heads by sharing them across multiple query heads. This (1) lowers the model's
parameter count and (2) reduces the memory bandwidth usage for key and value tensors
during inference since fewer keys and values need to be stored and retrieved from the KV
cache (section C.7).

While GQA is primarily a computational efficiency workaround for MHA, ablation studies
(as presented in the original GQA paper, https://arxiv.org/abs/2305.13245) show that it
performs comparably to standard MHA in terms of LLM modeling performance.

Listing C.4 implements the GQA mechanism with KV cache support.

- Listing C.4 Grouped query attention .


class GroupedQueryAttention(nn.Module):
def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None,

qk_norm=False, dtype=None):
super().__init__()
assert num_heads % num_kv_groups == 0

self.num_heads = num_heads
self.num_kv_groups = num_kv_groups
self.group_size = num_heads // num_kv_groups

if head_dim is None:

assert d_in % num_heads == 0
head_dim = d_in // num_heads

self.head_dim = head_dim
self.d_out = num_heads * head_dim

self.W_query = nn.Linear(
d_in, self.d_out, bias=False, dtype=dtype

)
self.W_key = nn.Linear(

d_in, num_kv_groups * head_dim, bias=False,dtype=dtype

)
self.W_value = nn.Linear(

d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
)

self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

if qk_norm:

self.q_norm = RMSNorm(head_dim, eps=1e-6)
self.k_norm = RMSNorm(head_dim, eps=1e-6)

else:
self.q_norm = self.k_norm = None

def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
b, num_tokens, _ = x.shape

queries = self.W_query(x) #A
keys = self.W_key(x) #B
values = self.W_value(x) #B

queries = queries.view(b, num_tokens, self.num_heads,
self.head_dim).transpose(1, 2)

keys_new = keys.view(b, num_tokens, self.num_kv_groups,
self.head_dim).transpose(1, 2)
values_new = values.view(b, num_tokens, self.num_kv_groups,
self.head_dim).transpose(1, 2)

if self.q_norm:

queries = self.q_norm(queries)
if self.k_norm:

keys_new = self.k_norm(keys_new)

queries = apply_rope(queries, cos, sin, offset=start_pos)
keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

if cache is not None:
prev_k, prev_v = cache
keys = torch.cat([prev_k, keys_new], dim=2)
values = torch.cat([prev_v, values_new], dim=2)

else:

start_pos = 0 #C
keys, values = keys_new, values_new

next_cache = (keys, values)

keys = keys.repeat_interleave( #D

self.group_size, dim=1 #D
) #D
values = values.repeat_interleave( #D

self.group_size, dim=1 #D
) #D

attn_scores = queries @ keys.transpose(2, 3)
attn_scores = attn_scores.masked_fill(mask, -torch.inf)
attn_weights = torch.softmax(

attn_scores / self.head_dim**0.5, dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
return self.out_proj(context), next_cache

#A The shape is (b, num_tokens, num_heads * head_dim)
#B The shapes are (b, num_tokens, num_kv_groups * head_dim)
#C Reset RoPE
#D Expand K and V to match number of heads

You may have noticed that the GQA mechanism in listing C.4 also includes a qk_norm
parameter. This is not part of the standard GQA design. When qk_norm=True, an additional
Query/Key-RMSNorm-based normalization, called QKNorm, is applied to both the queries
and keys, which is a technique used in Qwen3. As discussed earlier in the RMSNorm section
(section C.1), QKNorm helps improve training stability.

##### C.5 Transformer block

The transformer block is the central component of an LLM, which combines all the individual
elements covered in this appendix so far. As shown in figure C.6, it is repeated multiple
times; in the 0.6-billion-parameter version of Qwen3, it is repeated 28 times.

![image 161](<input (1)_images/imageFile161.png>)

- Figure C.6 The Structure of the transformer block in Qwen3. Each block includes RMSNorm, RoPE, masked
grouped-query attention, and a feed-forward module, and is repeated 28 times in the 0.6B-parameter model.


- Listing C.5 implements the transformer block shown in figure C.6.


- Listing C.5 Transformer block .


class TransformerBlock(nn.Module):

def __init__(self, cfg):
super().__init__()
self.att = GroupedQueryAttention(

d_in=cfg["emb_dim"],
num_heads=cfg["n_heads"],
head_dim=cfg["head_dim"],
num_kv_groups=cfg["n_kv_groups"],
qk_norm=cfg["qk_norm"],
dtype=cfg["dtype"]

)
self.ff = FeedForward(cfg)
self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
shortcut = x
x = self.norm1(x)
x, next_cache = self.att(

x, mask, cos, sin, start_pos=start_pos,cache=cache
) #A
x = x + shortcut

shortcut = x
x = self.norm2(x)
x = self.ff(x)
x = x + shortcut

return x, next_cache

#A The shape is (batch_size, num_tokens, emb_size)

As we can see, in listing C.5, the transformer block simply connects various elements we
implemented in previous sections.

##### C.6 Main model code

In this section, we will define the Qwen3Model class that we imported and used in chapter 2.

To implement the Qwen3Model class, the code in listing C.6 follows the architecture
previously shown in figure C.6, where the transformer block sits at the heart of the LLM.

- Listing C.6 Main Qwen3Model code .


class Qwen3Model(nn.Module):
def __init__(self, cfg):

super().__init__()

# Main model parameters
self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"],

dtype=cfg["dtype"])

self.trf_blocks = nn.ModuleList(
[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]

)
self.final_norm = RMSNorm(cfg["emb_dim"])
self.out_head = nn.Linear(

cfg["emb_dim"], cfg["vocab_size"],
bias=False, dtype=cfg["dtype"]

)

# Reusable utilities
if cfg["head_dim"] is None:

head_dim = cfg["emb_dim"] // cfg["n_heads"]
else:

head_dim = cfg["head_dim"]

cos, sin = compute_rope_params(
head_dim=head_dim,
theta_base=cfg["rope_base"],
context_length=cfg["context_length"]

)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)
self.cfg = cfg
self.current_pos = 0 # Track current position in KV cache

def forward(self, in_idx, cache=None):
tok_embeds = self.tok_emb(in_idx)
x = tok_embeds

num_tokens = x.shape[1]
if cache is not None:

pos_start = self.current_pos
pos_end = pos_start + num_tokens
self.current_pos = pos_end
mask = torch.triu(

torch.ones(
pos_end, pos_end, device=x.device, dtype=torch.bool

),
diagonal=1

)[pos_start:pos_end, :pos_end]

else:
pos_start = 0 # Not strictly necessary but helps torch.compile
mask = torch.triu(

torch.ones(num_tokens, num_tokens, device=x.device,

dtype=torch.bool),
diagonal=1

)

mask = mask[None, None, :, :] #A

for i, block in enumerate(self.trf_blocks):
blk_cache = cache.get(i) if cache else None
x, new_blk_cache = block(x, mask, self.cos, self.sin,

start_pos=pos_start,
cache=blk_cache)

if cache is not None:
cache.update(i, new_blk_cache)

x = self.final_norm(x)
logits = self.out_head(x.to(self.cfg["dtype"]))
return logits

def reset_kv_cache(self):
self.current_pos = 0

#A Prefill: Shape (1, 1, T, T) to broadcast across batch and heads. Cached: Shape (1, 1, T, K+T) where T=new tokens,
K=cached keys.

Since we already have all the main ingredients, the Qwen3Model class in listing C.6 only
adds a few more components around the transformer block, namely the embedding and
output layers (including one more RMSNorm layer). However, the code may appear
somewhat complicated, which is due to the KV cache option.

As discussed in chapter 2, the KV cache can speed up the text generation process, but it
is a topic outside the scope of this book. Interested readers can find more information
about KV caching in my Understanding and Coding the KV Cache in LLMs from Scratch
article at https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms.

Note that the Qwen3Model class, as implemented in listing C.6, supports various model
sizes (see appendix D for more information). In chapter 2, we use the 0.6-billion-parameter
model as it is the least resource-intensive model in the Qwen3 model family. The specific
configuration of this model is visualized in figure C.7.

![image 162](<input (1)_images/imageFile162.png>)

- Figure C.7 Architecture of the Qwen3 0.6B model. The model consists of a token embedding layer followed by
28 transformer blocks, each containing RMSNorm, RoPE, QKNorm, masked grouped-query attention with 16
heads, and a feed-forward module with an intermediate size of 3,072.


To use the 0.6B model shown in figure C.7 via the Qwen3Model class, we can define the
following configuration in listing C.7 that we provide as input (cfg=QWEN_CONFIG_06_B)
upon instantiating a new Qwen3Model instance.

- Listing C.7 Qwen3 0.6B configuration


QWEN_CONFIG_06_B = {
"vocab_size": 151_936, # Vocabulary size
"context_length": 40_960, # Length originally used during training
"emb_dim": 1024, # Embedding dimension
"n_heads": 16, # Number of attention heads
"n_layers": 28, # Number of layers
"hidden_dim": 3072, # Size of intermediate dim in FeedForward
"head_dim": 128, # Size of the heads in GQA
"qk_norm": True, # Whether to normalize queries & keys in GQA
"n_kv_groups": 8, # Key-Value groups for GQA
"rope_base": 1_000_000.0, # The base in RoPE's "theta"
"dtype": torch.bfloat16, # Lower-precision dtype to reduce memory

}

We will use the QWEN_CONFIG_06_B configuration from listing C.7 to instantiate the Qwen3
0.6B model later in section C.9.

###### MORE PRECISE CONTEXT LENGTH INFORMATION

For Qwen3 0.6B, the default maximum supported context length is 40,960 tokens.
However, the model itself was trained with only 32,768 tokens. The 40,960-token
allocation reserves 32,768 tokens for model outputs (the generated text) and 8,192
tokens for typical prompts (the user's question or instruction).

To put this in perspective, 40 thousand tokens correspond to roughly one half of
the first Harry Potter book.

Since this length is sufficient for most reasoning tasks, the developers note that
context-extension methods (such as YaRN) are not recommended unless the average
context regularly exceeds 32 thousand tokens, as enabling them in shorter contexts
can slightly degrade performance.

##### C.7 KV cache

The KV-cache-related heavy-lifting is mostly done in the Qwen3Model (listing C.6) and
GroupedQueryAttention (listing C.4) code. The KVCache, shown in listing C.8, stores the
key-value pairs themselves during text generation, which results in the speedup we
experienced when enabling KV caching in chapter 2.

- Listing C.8 KV Cache


class KVCache:
def __init__(self, n_layers):
self.cache = [None] * n_layers

def get(self, layer_idx):
return self.cache[layer_idx]

def update(self, layer_idx, value):
self.cache[layer_idx] = value

def get_all(self):
return self.cache

def reset(self):
for i in range(len(self.cache)):
self.cache[i] = None

The KVCache class in listing C.8 is used inside the generate_text_basic_stream_cache
function that we implemented in chapter 2.

##### C.8 Tokenizer

The tokenizer code is somewhat complicated, as it supports a variety of special tokens, in
addition to the base model and the so-called "Thinking" model variant of Qwen3, which is a
reasoning model. The full reimplementation of the tokenizer is shown in listing C.9.

- Listing C.9 Tokenizer


import re
from tokenizers import Tokenizer

class Qwen3Tokenizer:
_SPECIALS = [
"<|endoftext|>",
"<|im_start|>", "<|im_end|>",
"<|object_ref_start|>", "<|object_ref_end|>",
"<|box_start|>", "<|box_end|>",
"<|quad_start|>", "<|quad_end|>",
"<|vision_start|>", "<|vision_end|>",
"<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",

]
_SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

def __init__(self,
tokenizer_file_path="tokenizer-base.json",
apply_chat_template=False,
add_generation_prompt=False,
add_thinking=False):

self.apply_chat_template = apply_chat_template
self.add_generation_prompt = add_generation_prompt
self.add_thinking = add_thinking

tok_path = Path(tokenizer_file_path)
if not tok_path.is_file():

raise FileNotFoundError(

f"Tokenizer file '{tok_path}' not found. "
)

self._tok = Tokenizer.from_file(str(tok_path))
self._special_to_id = {t: self._tok.token_to_id(t)

for t in self._SPECIALS}

self.pad_token = "<|endoftext|>"
self.pad_token_id = self._special_to_id.get(self.pad_token)

f = tok_path.name.lower() #A
if "base" in f and "reasoning" not in f: #A

self.eos_token = "<|endoftext|>" #A
else: #A

self.eos_token = "<|im_end|>" #A
self.eos_token_id = self._special_to_id.get(self.eos_token)

def encode(self, prompt, chat_wrapped=None):
if chat_wrapped is None:
chat_wrapped = self.apply_chat_template

stripped = prompt.strip()
if stripped in self._special_to_id and "\n" not in stripped:

return [self._special_to_id[stripped]]

if chat_wrapped:
prompt = self._wrap_chat(prompt)

ids = []
for part in filter(None, self._SPLIT_RE.split(prompt)):

if part in self._special_to_id:

ids.append(self._special_to_id[part])
else:

ids.extend(self._tok.encode(part).ids)
return ids

def decode(self, token_ids):
return self._tok.decode(token_ids, skip_special_tokens=False)

def _wrap_chat(self, user_msg):
s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
if self.add_generation_prompt:

s += "<|im_start|>assistant"
if self.add_thinking:

s += "\n" #B
else:

s += "\n<think>\n\n</think>\n\n"
return s

- #A Match HF behavior: chat model: <|im_end|>, base model: <|endoftext|>
- #B insert no <think> tag, just a new line


Note that my Qwen3Tokenizer reimplementation in listing C.9 may appear somewhat
complicated, as it aims to replicate the behavior of the official tokenizer released by the
Qwen3 team in the Hugging Face Transformers library.

At first glance, it appears to have a few quirks. For example, when add_thinking=True,
no "\n<think>\n\n</think>\n\n" tokens are inserted (where \n is anewline character),
and when add_thinking=False, these tokens are added. This is intentional because the
non-base Qwen3 0.6B model is a hybrid that supports both reasoning ("thinking") and
standard modes.

##### C.9 Using the model

Let's now instantiate and use the model to confirm that the code works by reusing the text
generation approach from chapter 2.

First, we instantiate the model using the pre-trained model weights:

from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import download_qwen3_small

# device = get_device() #A
device = torch.device("cpu")

download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")

tokenizer_file_path = Path("qwen3") / "tokenizer-base.json"
model_file = Path("qwen3") / "qwen3-0.6B-base.pth"

tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_file_path)
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file))

model.to(device)

#A Optional: Uncomment to use automatic device picker

The output shows the structure of the instantiated model, which should match the values
we used in the configuration file in listing C.7:

✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
Qwen3Model(

(tok_emb): Embedding(151936, 1024)
(trf_blocks): ModuleList(

(0-27): 28 x TransformerBlock(

(att): GroupedQueryAttention(
(W_query): Linear(in_features=1024, out_features=2048, bias=False)
(W_key): Linear(in_features=1024, out_features=1024, bias=False)
(W_value): Linear(in_features=1024, out_features=1024, bias=False)
(out_proj): Linear(in_features=2048, out_features=1024, bias=False)
(q_norm): RMSNorm()
(k_norm): RMSNorm()

)
(ff): FeedForward(

(fc1): Linear(in_features=1024, out_features=3072, bias=False)
(fc2): Linear(in_features=1024, out_features=3072, bias=False)
(fc3): Linear(in_features=3072, out_features=1024, bias=False)

)
(norm1): RMSNorm()
(norm2): RMSNorm()

)

)
(final_norm): RMSNorm()
(out_head): Linear(in_features=1024, out_features=151936, bias=False)

)

Next, we re-use the text generation functions from chapter 2 to generate text:

import time

from reasoning_from_scratch.ch02 import (
generate_text_basic_stream_cache,
generate_stats

)

prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(

tokenizer.encode(prompt),
device=device
).unsqueeze(0)

max_new_tokens = 200

start_time = time.time()
generated_ids = []

for token in generate_text_basic_stream_cache(
model=model,
token_ids=input_token_ids_tensor,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id

):

token_id = token.squeeze(0).tolist()
print(

tokenizer.decode(token_id),
end="",
flush=True

)

next_token_id = token.squeeze(0)
generated_ids.append(next_token_id) # Collect generated tokens

end_time = time.time()

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

Since we used the same prompt as in chapter 2, the generated text matches the generated
text from chapter 2 exactly:

Time: 1.46 sec
28 tokens/sec

Large language models are artificial intelligence systems that can

understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.

While the main chapters use the 0.6-billion-parameter variant of Qwen3 to lower the
resource requirements for this book, interested readers can find more information on how
to use the larger models in appendix D.

## Appendix D. Using larger LLMs

The main chapters use the 0.6-billion-parameter (0.6B) Qwen3 base model because it is
the smallest model in the Qwen3 family and therefore the easiest to run on consumer
hardware.

However, the same Qwen3Model implementation from appendix C is not limited to the
0.6B checkpoint. We can also use it to load larger Qwen3 checkpoints with the same from-
scratch PyTorch code. In practice, this means that once we understand how to work with
the 0.6B model, moving to a larger model mainly involves three changes:

- 1. selecting the matching configuration dictionary;
- 2. downloading the larger checkpoint from Hugging Face;
- 3. loading the appropriate tokenizer for the base or reasoning variant.


This appendix illustrates this process using the Qwen3 4B model as a concrete example,
because it is large enough to be meaningfully stronger than the 0.6B model while still being
easier to handle than the larger 8B, 14B, and 32B variants.

##### D.1 Larger dense Qwen3 configurations

The repository includes configuration dictionaries for several larger Qwen3 models in the
reasoning_from_scratch.appendix_c Python library, which are listed in table D.1. (You
can also view the source code in the supplementary materials at https://github.com/rasbt/
reasoning-from-scratch/blob/main/reasoning_from_scratch/appendix_c.py).

Table D.1 Qwen3 configurations (larger than 0.6B)

|Model size|Configuration Python<br>dictionary|
|---|---|
|1.7B|QWEN3_CONFIG_1_7B|
|4B|QWEN3_CONFIG_4B|
|8B|QWEN3_CONFIG_8B|
|14B|QWEN3_CONFIG_14B|
|32B|QWEN3_CONFIG_32B|


Note that the models listed in table D.1 are the dense Qwen3 variants, similar to the
Qwen3 0.6B model used in the main chapters and discussed in appendix C. Here, "dense"
means these are not the sparse Mixture-of-Experts (MoE) variants of Qwen3. MoE models
are not supported by this book's code, but a from-scratch implementation is available here:
https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3.

Figure D.1 summarizes the differences between the models in table D.1 more visually.

![image 163](<input (1)_images/imageFile163.png>)

Figure D.1 Comparison of different Qwen3 variants, from 0.6-billion to 32-billion parameters.

As can be seen in figure D.1, all the different Qwen3 size variants use the same overall
architecture pattern as the 0.6B model from appendix C. Concretely, they still use token
embeddings, grouped-query attention, rotary position embeddings, feed-forward layers,
and RMS normalization. What changes are the model dimensions, such as the embedding
size, the number of layers, the number of attention heads, and the feed-forward hidden
dimension.

Table D.2 lists the memory requirement estimates when loading the models into RAM
(and GPU RAM). Here, as a rough lower bound, storing weights in bfloat16 precision (which
is the common default for LLMs) requires about 2 bytes per parameter.

Table D.2 Qwen3 models sizes

|Model size|Rough weight memory in<br>bfloat16|
|---|---|
|1.7B|about 3.4 GB|
|4B|about 8 GB|
|8B|about 16 GB|
|14B|about 28 GB|
|32B|about 64 GB|


The numbers in table D.2 are only approximate lower bounds for the weights themselves.
In practice, the real runtime memory usage is higher because we also need memory for
activations, temporary buffers, the KV cache, and so on. In addition, model loading can
temporarily increase memory usage because tensors may exist in more than one place
while they are being copied into the Qwen3Model instance. Please see the supplementary
materials of my Build A Large Language Model (From Scratch) book for more details on
this: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/08_memory_efficient_
weight_loading

###### D.2 Downloading larger checkpoints overview

Unlike the 0.6B checkpoints used in the main chapters, which I converted into a PyTorch
checkpoint to minimize external code dependencies, larger official Qwen3 models are
typically distributed as safetensors files, sometimes split across multiple shards.

The reasoning-from-scratch Python library provides a helper function,
download_from_huggingface_from_snapshots, to handle this. This function, which we will
use in the next section, downloads the Hugging Face snapshot into a local directory and
then loads either a single model.safetensors file or multiple shards referenced via a
model.safetensors.index.json file.

Because this helper relies on additional packages that were not required in the earlier

chapters, you may need to install them first:
uv add huggingface_hub safetensors
or
pip install huggingface_hub safetensors

Once these packages are installed, loading a larger checkpoint follows the same broad
pattern as before. For instance, we download the files, instantiate the model, load the
weights, prepare the tokenizer, and then run the text generation, as we will see in the next
section.

##### D.3 Loading a larger base model

We will now walk through a complete example using the official Qwen3 4B base model. The
first step, shown in listing D.1, is to download the model snapshot from the Hugging Face
Model hub (https://huggingface.co/Qwen/Qwen3-4B-Base):

- Listing D.1 Download model weights


from pathlib import Path
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.appendix_c import (

download_from_huggingface_from_snapshots,
)

device = get_device()
local_dir = Path("qwen3-4b-base")

weights = download_from_huggingface_from_snapshots(
repo_id="Qwen/Qwen3-4B-Base",
local_dir=local_dir,

)

The local_dir path specifies where the snapshot should be stored on disk. In this
example, we keep the 4B base model in a dedicated qwen3-4b-base folder so that it stays
separate from the smaller qwen3 directory used throughout the main chapters.

Next, we use the code in listing D.2 to instantiate the 4B model and load the Hugging
Face weights into the from-scratch Qwen3Model implementation:

- Listing D.2 Load weights into Qwen3Model

from reasoning_from_scratch.qwen3 import (
Qwen3Model,
load_hf_weights_into_qwen,

)
from reasoning_from_scratch.appendix_c import QWEN3_CONFIG_4B

model = Qwen3Model(QWEN3_CONFIG_4B)
load_hf_weights_into_qwen(

model,
param_config={

"n_layers": QWEN3_CONFIG_4B["n_layers"],
"hidden_dim": QWEN3_CONFIG_4B["hidden_dim"],

},
params=weights,

)
model.to(device)
model.eval()

- Listing D.3 Load base tokenizer


The QWEN3_CONFIG_4B dictionary defines the architecture dimensions for the 4B model. The
load_hf_weights_into_qwen helper then maps the Hugging Face parameter names into
the parameter names used by our own implementation. After that, model.to(device)
moves the model to the selected device, and model.eval() switches the model into
inference mode.

After the model is loaded, we prepare the tokenizer via the code in listing D.3:

from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
import shutil

tokenizer_src = local_dir / "tokenizer.json"
tokenizer_path = local_dir / "tokenizer-base.json"

if not tokenizer_path.exists():
shutil.copyfile(tokenizer_src, tokenizer_path)

tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

The Hugging Face snapshot stores the tokenizer under the generic name
"tokenizer.json". In listing D.3, we copy it to "tokenizer-base.json" so that it is easy
to distinguish from the reasoning tokenizer that we will use in the next section.

We can now use the model for generation, exactly as we did with the smaller 0.6B model
in chapter 2, using the code in listing D.4:

- Listing D.4 Generate text


import torch
from reasoning_from_scratch.ch02 import (
generate_text_basic_stream_cache,
)

prompt = "Explain large language models in two sentences."
input_ids = torch.tensor(

tokenizer.encode(prompt),
device=device,

).unsqueeze(0)

for token in generate_text_basic_stream_cache(
model=model,
token_ids=input_ids,
max_new_tokens=64,
eos_token_id=tokenizer.eos_token_id,

):

print(tokenizer.decode(token.squeeze(0).tolist()), end="", flush=True)

Similar to the main book content in chapter 2, we encode the input prompt, add a batch
dimension via unsqueeze(0), and then stream the generated tokens one by one.

The output is shown below:

Large language models are artificial intelligence systems that use deep learning
techniques to understand and generate human-like text. They are trained on vast
amounts of data and can perform a wide range of natural language processing
tasks,
such as translation, summarization, and question answering.

This is one of the nice properties of reusing the same Qwen3Model abstraction across model
sizes. Once the checkpoint and tokenizer are set up correctly, the inference code looks
essentially the same.

##### D.4 Loading a larger reasoning variant

The same idea also works for larger reasoning-style Qwen3 models. The architecture for a
given model size stays the same. Compared to the previous section, only the checkpoint
and tokenizer settings change.

For example, to load the 4B reasoning variant instead of the 4B base variant, we would:
switch the repository ID from Qwen/Qwen3-4B-Base to Qwen/Qwen3-4B;

copy or rename the tokenizer.json file to tokenizer-reasoning.json;

Initialize the tokenizer as follows, using the code in listing D.5:

- Listing D.5 Load reasoning tokenizer


tokenizer = Qwen3Tokenizer(
tokenizer_file_path=tokenizer_path,
apply_chat_template=True,
add_generation_prompt=True,
add_thinking=True,

)

These three tokenizer settings match the chat-style prompt formatting used by the
reasoning models, as discussed in chapter 2 and 8. In other words, we still use the same
4B architecture configuration (QWEN3_CONFIG_4B), but we pair it with the reasoning
checkpoint and the reasoning-style tokenizer behavior.

The rest of the model-loading and model-usage code stays the same. We still download
the snapshot, instantiate Qwen3Model(QWEN3_CONFIG_4B), load the weights via
load_hf_weights_into_qwen, move the model to the target device, and then generate text
using the same streaming function from chapter 2.

##### D.5 Practical recommendations

The main point of this appendix is that the same from-scratch model code from appendix C
can be reused across larger official Qwen3 checkpoints, provided that we select the
matching configuration dictionary and tokenizer setup.

This means moving to a larger model is mostly a matter of loading a different checkpoint
and configuration rather than learning an entirely new concept.

If you want to experiment beyond the 4B example shown here, the 1.7B and 8B variants
are natural next steps. The 1.7B model is a modest step up from 0.6B, in case the 4B is too
resource-intensive for your hardware. The 8B model is substantially larger and may give
stronger outputs if your hardware can handle it.

## Appendix E. Batching and throughput-oriented execution

In the code we implemented throughout the main chapters, we usually process one prompt
or example at a time. This keeps the code compact and easier to understand, and it also
helps keep the resource requirements somewhat more manageable.

In practice, the code in this book is already expensive to run, so adding batching support
everywhere would often increase complexity without any real benefits in many cases,
depending on your available hardware.

That being said, if you have access to relatively modern GPUs with enough memory,
batched execution, illustrated in figure E.1, is very useful in some settings. For example, if
we want to evaluate many problems, sample several responses per prompt, or train on
multiple examples at once, then batching can help increase the throughput and lower the
overall time.

This appendix explains the broad idea behind batching and shows how to use the
batched implementations in the supplementary materials across different chapters.

![image 164](<input (1)_images/imageFile164.png>)

Figure E.1 An overview of single-example versus batched generation. In batched generation, several prompts
are packed into one batch, processed in parallel, and then decoded together to improve throughput.

###### E.1 Why batching helps

When we talk about computational performance, it helps to separate between the following
overall goals:

- 1. latency: how quickly we get the answer for one prompt;
- 2. throughput: how many prompts we can process in a fixed amount of
time.


Single-example generation is often best for understanding the conceptual implementation
of a concept, minimizing latency, and for debugging.

Batching can be seen as an extension of single-example generation that primarily
targets throughput. For example, if we want to evaluate hundreds of problems on MATH-
500, generate several self-consistency samples, or train on a larger distillation dataset,
batching can reduce total runtime substantially on suitable hardware.

However, batching is not automatically faster on every device. Small models on CPUs or
less optimized GPUs may benefit only a little, or not at all, because we have to introduce
additional code complexity, including padding and other overheads, that can offset the
gains from the parallelism that we gain through batching.

So batching is best understood as a throughput optimization, not as a universal
speedup.

##### E.2 Running batched generation

The main technical obstacle in batching is that prompts usually have different lengths. One
math problem may tokenize to 40 tokens while another may tokenize to 120 tokens, but
PyTorch tensors still have to be rectangular. So if we want to run several prompts together,
we need a way to pad shorter rows and keep track of which positions are real tokens and
which are just padding tokens.

Padding tokens are placeholder tokens that are added to a prompt to make it a certain
length, for example to, extend a three-token prompt "3 + 3" to a five-token prompt, we
may add two additional padding tokens to the right or the left: "<pad><pad> 3 + 3". (In
practice, it doesn't matter which token we use as padding token, and it's common to use an
end-of-sequence token like <eos> or <|endoftext|>.)

Conceptually, padding and keeping track of padded positions makes batched generation
more complicated than single-prompt generation. In the main chapters, we used the
Qwen3Model class from reasoning_from_scratch.qwen3, which is the plain single-example
implementation explained in appendix C.

For batched generation, the supplementary materials therefore include a separate
Qwen3Model class in reasoning_from_scratch.qwen3_batched, which can be used as a
drop-in replacement but adds the bookkeeping needed for padding-aware execution that is
required in batching.

To make this concrete, it helps to start with a single-sequence example that looks very
similar to the code we already used in the main chapters. Here, in listing E.1, we apply it to
two very small prompts, but we still run them one after the other:

- Listing E.1 Single-example generation


import torch

- from reasoning_from_scratch.ch02 import (
get_device,
generate_text_basic_stream_cache,

)

- from reasoning_from_scratch.ch03 import (
load_model_and_tokenizer,
render_prompt,


)

device = get_device()
model, tokenizer = load_model_and_tokenizer(

which_model="base",
device=device,
use_compile=False,

)

for problem in ["2+2?", "3+3=6?"]: #A
prompt = render_prompt(problem)
input_ids = torch.tensor(

tokenizer.encode(prompt),
dtype=torch.long,
device=device,

).unsqueeze(0)

for token in generate_text_basic_stream_cache(
model=model,
token_ids=input_ids,
max_new_tokens=32,
eos_token_id=tokenizer.eos_token_id,

):

next_token_id = token.squeeze(0)
print(tokenizer.decode(next_token_id.tolist()), end="", flush=True)

print() #B

#A The two prompts we sequentially process
#B Force a new line after each answer

The resulting output is:

\boxed{4}
\boxed{6}

The code in listing E.1 is still ordinary single-example generation. We render the prompt,
tokenize it, add a batch dimension of size 1, and then stream the generated tokens as they
arrive. This is conceptually simple, which is why the main chapters prefer this style.

Listing E.2 now shows the corresponding batched version. The overall workflow is similar,
but now we switch to reasoning_from_scratch.qwen3_batched, tokenize both prompts up
front, left-pad them to the same length, and generate them together:

- Listing E.2 Batched generation


from reasoning_from_scratch.qwen3_batched import (
generate_text_basic_batched_cache,
load_model_and_tokenizer,

)

model, tokenizer = load_model_and_tokenizer(
which_model="base",
device=device,
use_compile=False,

)

problems = ["2+2?", "3+3=6?"]
prompts = [render_prompt(problem) for problem in problems]
tokenized = [tokenizer.encode(p) for p in prompts]
pad_id = tokenizer.pad_token_id
max_len = max(len(t) for t in tokenized)

left_padded = [ #A
[pad_id] * (max_len - len(t)) + t
for t in tokenized

]
input_ids = torch.tensor(left_padded, dtype=torch.long, device=device)

generated = generate_text_basic_batched_cache(
model=model,
token_ids=input_ids,
max_new_tokens=32,
eos_token_id=tokenizer.eos_token_id,
pad_id=pad_id, #B

)

for row in generated:
eos_pos = (row == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
if len(eos_pos) > 0:

row = row[:eos_pos[0]]
print(tokenizer.decode(row.tolist()))

#A Left-pad shorter sequences so all prompts in the batch have the same length
#B Needed so generation can distinguish real tokens from padding

The results are the same as before, but this time they are produced in parallel:

\boxed{4}
\boxed{6}

In this example, we use the non-streaming batched helper, so we wait until the whole batch
is finished before decoding and printing the results. That keeps the usage example simple
and makes it easier to see the role of padding before we dig into the internal masking logic.

There is also a more optimized text generation function variant,
generate_text_basic_batched_cache_stop, which removes finished rows from the active
compute batch instead of carrying them along until the longest row finishes. In contrast,
the generate_text_basic_batched_cache we use in listing E.2 keeps every row in the
active batch for every decode step.

In generate_text_basic_batched_cache, once a row has emitted EOS, the
implementation forces that row to keep producing EOS so that the batch shape stays
aligned, even though the row is already finished from the perspective of the final decoded
output. The difference between these two functions is illustrated in figure E.2.

![image 165](<input (1)_images/imageFile165.png>)

- Figure E.2 Sequential generation (1) versus regular batched generation (2) and early-stop batched generation


(3). The regular batched helper (2) keeps finished rows in the batch as placeholder EOS rows, whereas the
_stop variant (3) removes finished rows from the active compute batch as soon as they emit EOS.

As a side note, Qwen3 uses <|endoftext|> as the EOS token for the base model, but
figure E.2 labels it simply as <eos> for visual compactness.

The next section goes into a bit more detail regarding how the padding is handled
internally.

###### E.3 Padding and attention masks

In single-example mode, if we tokenize a short prompt such as "2+2?", we can pass it to
the model as a simple tensor of shape (1, 4):

input_ids = torch.tensor([[17, 10, 17, 30]])

Internally, the model builds a standard causal attention mask internally so that each
position can only attend to itself and earlier tokens. If you are unfamiliar with causal
attention masks and self-attention, I have an article that provides more background
information: https://magazine.sebastianraschka.com/p/understanding-and-coding-self-
attention.

The causal attention mask for the "2+2?" prompt is illustrated in figure E.3.

![image 166](<input (1)_images/imageFile166.png>)

- Figure E.3 The causal mask used in single-example generation. Tokens can attend only to themselves and
earlier positions, which is the standard autoregressive setup.


In the causal mask shown in figure E.3, 1 means "masked out" and 0 means "allowed".

In this context, as illustrated in figure E.3, the first token cannot look ahead to later
positions, the second token can only look at the first two positions, and so on, which is the
standard autoregressive masking pattern.

Batching changes the situation because different prompts usually have different lengths.
Suppose we process "2+2?" together with the slightly longer prompt, "3+3=6?".

Since PyTorch tensors must be rectangular, the shorter row has to be padded to match
the longer one, which is illustrated in figure E.4 using left padding.

![image 167](<input (1)_images/imageFile167.png>)

- Figure E.4 Left padding in batched generation. The shorter row is padded up to the maximum sequence length
in the batch, and the padding-aware mask ensures that these padded positions do not affect attention.


Note that figure E.4 shows that we keep an additional attn_mask internally. This mask is
used to keep track of the padded positions, where True means padded and False means
not padded.

We use this additional attn_mask to identify the tokens in the causal mask that
correspond to the pad token IDs.

Masking padded keys and zeroing padded queries are important steps to make batching
behave similarly to the single-example execution.

In the remainder of this appendix, we will see how we can use scripts from the
supplementary materials that implement the optional batched variants for the different
chapters.

###### E.4 Chapter 3: batched MATH-500 evaluation

The supplementary materials include a script for the single-example evaluation method
from chapter 3 that we can download and run directly using the download function we
introduced in chapter 7:

- Listing E.3 Download single-example evaluation script

from reasoning_from_scratch.ch07 import download_from_github

download_from_github(
"ch03/02_math500-verifier-scripts/evaluate_math500.py"

)
download_from_github(

"ch03/01_main-chapter-code/math500_test.json",
out="math500_test.json",

)

uv run evaluate_math500.py \

--dataset_size 500 \

--which_model "reasoning"

- Listing E.4 Download evaluation script with batch support


After downloading the script and dataset via listing E.3, we can run the single-example
evaluation from the terminal as shown below (if you are not using uv, simply replace uv
run with python):

The supplementary material also includes a batched version that applies the batching
method we discussed above. The download is almost identical, except that we replace
evaluate_math500.py with evaluate_math500_batched.py, as shown in listing E.4:

download_from_github(

"ch03/02_math500-verifier-scripts/evaluate_math500_batched.py"
)

The usage of the batched script is also very similar to the non-batched version, except that
we now provide an additional --batch_size argument to specify how many prompts and
answers the model should process in parallel:

uv run evaluate_math500_batched.py \

--dataset_size 500 \

--which_model "reasoning" \

--batch_size 64

The ideal batch size depends entirely on what your hardware can handle, so in practice it is
best to start with a smaller value and then scale upward until you hit the throughput and
memory limit that makes sense for your setup.

We will return to a comparison of the single-example and batched generation
performances at the end of the chapter.

##### E.5 Chapter 4: batched self-consistency sampling

The optional self_consistency_math500_batched.py script for chapter 4 does not mix
different prompts into one padded tensor. Instead, it repeats the same prompt
num_samples times and samples several continuations in parallel for self-consistency
voting, since parallelizing over the different number of samples for each prompt is a natural
angle here that we can take advantage of.

Because every row starts from the same prompt length, this script can use the regular
Qwen3Model from reasoning_from_scratch.qwen3 instead of
reasoning_from_scratch.qwen3_batched. In other words, the batch dimension here is
used for multiple sampled rollouts of the same prompt, not for padding together unrelated
prompts of different lengths.

We can download the script as shown in listing E.5:

- Listing E.5 Download self-consistently sampling with batch support


download_from_github(
"ch04/02_math500-inference-scaling-scripts/"
"self_consistency_math500_batched.py"

)

To download the non-batched version instead, simply remove the "_batched" part from the
file name in listing E.5.

We can run the batched script as follows, and the syntax for the non-batched script is
otherwise the same:

uv run self_consistency_math500_batched.py \

--which_model base \

--temperature 0.9 \

--top_p 0.9 \

--num_samples 3 \

--dataset_size 500 \

--prompt_suffix "\n\nExplain step by step."

##### E.6 Chapter 6: batched GRPO rollouts

Self-refinement in chapter 5 is fundamentally sequential and therefore does not benefit
much from simple batching. One could in principle run self-refinement loops for multiple
inputs in parallel, but that is non-trivial to implement and is therefore not part of the
supplementary material. Instead, the next batched example appears in the chapter 6 RLVR
code.

In chapter 6, we again use the same prompt for multiple rollouts, so no padding is
required. As in section E.5, the code can therefore stay with the regular Qwen3Model class
from reasoning_from_scratch.qwen3.

The relevant scripts can be fetched as follows:

- Listing E.6 Download RLVR script with batch support


download_from_github(
"ch06/02_rlvr_grpo_scripts_intro/"
"rlvr_grpo_original_no_kl_batched.py"

)

Similar to before, to download the single-generation script, simply drop "_batched" from
the file name in listing E.6.

A typical batched run looks like this:
uv run rlvr_grpo_original_no_kl_batched.py \

--num_rollouts 8 \

--batch_size 4 \

--max_new_tokens 512

Here, --batch_size controls how many rollouts are generated in parallel within one
training step. This improves throughput, but it also increases memory pressure, so in
practice you may need to reduce --num_rollouts or --max_new_tokens.

##### E.7 Chapter 8: batched distillation

Chapter 8 returns to the padding-aware style from chapter 3 because the distillation
examples have different prompt and answer lengths. In other words, we are again batching
unrelated examples of different lengths into a single rectangular tensor, which means that
masking and padding matter again.

We can download the batched training script and fetch a sample training dataset as
follows:

- Listing E.7 Download distillation script with batch support


from reasoning_from_scratch.ch08 import load_distill_data

download_from_github(
"ch08/04_train_with_distillation/distill_batched.py"

)
_ = load_distill_data(

partition="deepseek-r1-math-train",
local_path="deepseek-r1-math-train.json",

)

For the non-batched version, remove the "_batched" part from the file name in listing E.7.
A representative batched run looks like this:

uv run distill_batched.py \

--data_path deepseek-r1-math-train.json \

--dataset_size 500 \

--validation_size 10 \

--epochs 2 \

--use_think_tokens \

--batch_size 32

Here, --batch_size lets us process multiple distillation examples per optimization step, but
because training also has to store activations, the resource demands can become much
higher than in the chapter 3 evaluator. So chapter 8 gives us the training analogue of the
padding-aware batching strategy introduced earlier in this appendix.

##### E.8 Single-sequence versus batch generation

The supplementary scripts in chapters 3 and 8 batch together different-length examples, so
they need padding-aware attention masks and more careful bookkeeping. Chapters 4 and 6
batch repeated copies of the same prompt or rollout setup, so they can stay with the
regular model implementation and avoid the more complicated padding logic.

Either way, batched generation can result in higher throughput and shorter runtimes, at
the cost of increased RAM usage. Table 8.1 compares single to batched generation using
the settings shown in the previous sections.

Table E.1 RAM usage and runtime comparison

| |Script|Batch<br>size|RAM|H100<br>Total<br>time|DGX<br>Spark<br>Total<br>tim|
|---|---|---|---|---|---|
|1|evaluate_math500.py|-|1.8<br>GB|90<br>min|174<br>min|
|2|evaluate_math500_batched.py|64|23.39<br>GB|16<br>min|108<br>min|
| | | | | | |
|3|self_consistency_math500.py|-|1.79<br>GB|252<br>min|340<br>min|
|4|self_consistency_math500_batched.py|3|2.45<br>GB|129<br>min|243<br>min|
| | | | | | |
|5|rlvr_grpo_original_no_kl.py|-|43.35<br>GB|68<br>min|63<br>min|
|6|rlvr_grpo_original_no_kl_batched.py|4|44.91<br>GB|19<br>min|23<br>min|
| | | | | | |
|7|distill.py|-|8.29<br>GB|10<br>min|32<br>min|
|8|distill_batched.py|4|8.34<br>GB|9 min|28<br>min|


As shown in table E.1, the batched variants are always faster than the single-sequence
variants. In most cases, the difference is very noticeable. One exception is distillation (row
7 and 8), where batching only yields a modest advantage.

This could be because the GPU is already well utilized in single-batch mode, and the
extra overhead in the batched generation due to the additional mask generation is not
worth it, given the small model size of the Qwen3 6-billion-parameter model.

## Appendix F. Common approaches to model evaluation

##### F.1 Understanding the main evaluation methods for LLMs

There are four common ways of evaluating trained LLMs in practice: multiple choice,
verifiers, leaderboards, and LLM judges, as shown in figure F.1. Research papers, marketing
materials, technical reports, and model cards (a term for LLM-specific technical reports)
often include results from two or more of these categories.

![image 168](<input (1)_images/imageFile168.png>)

Figure F.1 A mental model of the topics covered in this book with a focus on the two broad evaluation
categories, benchmark-based evaluation and judgment-based evaluation, covered in this appendix.

Furthermore, as shown in figure F.1, the four categories introduced here fall into two
groups: benchmark-based evaluation and judgment-based evaluation.

Other measures, such as training loss, perplexity, and rewards, are typically used
internally during model development. (They are covered in the model training chapters.)

The following subsections provide brief overviews of each method.

###### F.2 Evaluating answer-choice accuracy

We begin with a benchmark‑based method: multiple‑choice question answering.

Historically, one of the most widely used evaluation methods is multiple-choice
benchmarks such as MMLU (short for Massive Multitask Language Understanding,
https://huggingface.co/datasets/cais/mmlu). An example task from the MMLU dataset is
shown in figure F.2.

![image 169](<input (1)_images/imageFile169.png>)

Figure F.2 Evaluating an LLM on MMLU by comparing its multiple-choice prediction with the correct answer
from the dataset.

- Figure F.2 shows just a single example from the MMLU dataset. The complete MMLU dataset
consists of 57 subjects (from high school math to biology) with about 16 thousand multiple-
choice questions in total, and performance is measured in terms of accuracy (the fraction of
correctly answered questions), for example 87.5% if 14,000 out of 16,000 questions are
answered correctly.


Multiple-choice benchmarks, such as MMLU, test an LLM's knowledge recall in a
straightforward, quantifiable way similar to standardized tests, many school exams, or
theoretical driving tests.

Note that figure F.2 shows a simplified version of multiple-choice evaluation, where the
model's predicted answer letter is compared directly to the correct one. Two other popular
methods exist that involve log-probability scoring (log-probabilities are discussed in chapter

- 4 in more detail).
The following subsections illustrate how the MMLU scoring shown in figure F.2 can be


implemented in code. End-to-end MMLU scripts, including the different scoring variants, will
be provided as bonus materials in this book's code repository.

- F.2.1 Loading the model


First, before we can evaluate it on MMLU, we have to load the pre-trained model. The
following code is identical to listing 3.1 in chapter 3.

- Listing F.1 Loading a pre-trained model


from pathlib import Path
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import (

download_qwen3_small, Qwen3Tokenizer,
Qwen3Model, QWEN_CONFIG_06_B

)

device = get_device()
torch.set_float32_matmul_precision("high") #A

# device = "cpu" #B

WHICH_MODEL = "base" #C

if WHICH_MODEL == "base":
download_qwen3_small(

kind="base", tokenizer_only=False, out_dir="qwen3"

)
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

elif WHICH_MODEL == "reasoning":
download_qwen3_small(
kind="reasoning", tokenizer_only=False, out_dir="qwen3"

)
tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
model_path = Path("qwen3") / "qwen3-0.6B-reasoning.pth"
tokenizer = Qwen3Tokenizer(

tokenizer_file_path=tokenizer_path,
apply_chat_template=True,
add_generation_prompt=True,
add_thinking=True,

)

else:
raise ValueError(f"Invalid choice: WHICH_MODEL={WHICH_MODEL}")

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_path))
model.to(device)

USE_COMPILE = False #D
if USE_COMPILE:

torch._dynamo.config.allow_unspec_int_on_nn_module = True
model = torch.compile(model)

- #A Lower precision from "highest" to enable Tensor Cores if applicable
- #B Uncomment this line if you have compatibility issues with your device
- #C Uses the base model, similar to chapter 2, by default
- #D Optionally set to true to enable model compilation


- F.2.2 Checking the generated answer letter


In this section, we implement the simplest and perhaps most intuitive MMLU scoring
method, which relies on checking whether a generated multiple-choice answer letter
matches the correct answer. This is similar to what was illustrated earlier in figure F.2.

For this, we will work with an example from the MMLU dataset:

example = {

"question": (
"How many ways are there to put 4 distinguishable"
" balls into 2 indistinguishable boxes?"

),
"choices": ["7", "11", "16", "8"],
"answer": "D",

}

Next, we define a function to format the LLM prompts:

- Listing F.2 Formatting the LLM prompt


def format_prompt(example):
return (
f"{example['question']}\n"

- f"A. {example['choices'][0]}\n"
- f"B. {example['choices'][1]}\n"
- f"C. {example['choices'][2]}\n"
- f"D. {example['choices'][3]}\n"
"Answer: " #A


)

#A trailing space encourages a single-letter next token

Let's execute the function on the MMLU example to get an idea of what the formatted LLM
input looks like:

prompt = format_prompt(example)
print(prompt)

The output is:

How many ways are there to put 4 distinguishable balls into 2
indistinguishable boxes?

- A. 7
- B. 11
- C. 16
- D. 8
Answer:


The model prompt, as shown above, provides the model with a list of the different answer
choices and ends with an "Answer: " text that encourages the model to generate the
correct answer.

While it is not strictly necessary, it can sometimes also be helpful to provide additional
questions along with the correct answers as input, so that the model can observe how it is
expected to solve the task. (For example, cases where 5 examples are provided are also
known as 5-shot MMLU.) However, for current generations of LLMs, where even the base
models are quite capable, this is not required.

###### LOADING DIFFERENT MMLU SAMPLES

You can load examples from the MMLU dataset directly via the datasets library
(which can be installed via pip install datasets or uv add datasets):

from datasets import load_dataset
configs = get_dataset_config_names("cais/mmlu")
dataset = load_dataset("cais/mmlu", "high_school_mathematics")

example = dataset["test"][0] #A
print(example)

#A Inspect the first example from the test set

Above, we used the "high_school_mathematics" subset; to get a list of the other
subsets, use the following code:

from datasets import get_dataset_config_names
subsets = get_dataset_config_names("cais/mmlu")
print(subsets)

Next, we tokenize the prompt and wrap it in a PyTorch tensor object as input to the LLM
(similar to what we did in chapter 2):

prompt_ids = tokenizer.encode(prompt)
prompt_fmt = torch.tensor(prompt_ids, device=device).unsqueeze(0)

Then, we define the main scoring function in listing F.3, which generates a few tokens
(here, 8 tokens by default) and extracts the first instance of letter A/B/C/D that the model
prints.

- Listing F.3 Extracting the generated letter


from reasoning_from_scratch.ch02 import (

generate_text_basic_stream_cache
)

def predict_choice(

model, tokenizer, prompt_fmt, max_new_tokens=8
):

pred = None
for t in generate_text_basic_stream_cache(

model=model,
token_ids=prompt_fmt,
max_new_tokens=max_new_tokens,
eos_token_id=tokenizer.eos_token_id,

):

answer = tokenizer.decode(t.squeeze(0).tolist())
for letter in answer:

letter = letter.upper()
if letter in "ABCD": #A

pred = letter
break

if pred:

break
return pred

#A stop as soon as a letter appears

We can then check the generated letter using the function from listing F.3 as follows:

pred1 = predict_choice(model, tokenizer, prompt_fmt)
print(

f"Generated letter: {pred1}\n"
f"Correct? {pred1 == example['answer']}"

)

The result is:

Generated letter: C
Correct? False

As we can see, the generated answer is incorrect (False) in this case.

###### MULTIPLE-CHOICE ANSWER FORMATS

Note that this section implemented a simplified version of multiple-choice evaluation
for illustration purposes, where the model's predicted answer letter is compared
directly to the correct one. In practice, more widely used variations exist, such as
log-probability scoring, where we measure how likely the model considers each
candidate answer rather than just checking the final letter choice. (We discuss
probability-based scoring in chapter 4.) For reasoning models, evaluation can also
involve assessing the likelihood of generating the correct answer when it is provided
as input.

Regardless of the variant, the evaluation still amounts to checking whether the
model selects from the predefined answer options. Examples of these variations will
be included in the code repository as optional bonus material.

A limitation of multiple‑choice benchmarks like MMLU is that they only measure an LLM's
ability to select from predefined options and thus is not very useful for evaluating reasoning
capabilities besides checking if and how much knowledge the model has forgotten
compared to the base model. It does not capture free-form writing ability or real-world
utility. Still, it remains a simple and useful diagnostic: a high MMLU score doesn't
necessarily mean the model is strong in practical use, but a low score can highlight
potential knowledge gaps.

##### F.3 Using verifiers to check answers

Related to multiple-choice question answering discussed in the previous section,
verification-based approaches quantify the LLMs capabilities via an accuracy metric.
However, in contrast to multiple-choice benchmarks, verification methods allow LLMs to
provide a free-form answer. We then extract the relevant answer portion and use a so-
called verifier to compare the answer portion to the correct answer provided in the dataset,
as illustrated in figure F.3.

![image 170](<input (1)_images/imageFile170.png>)

- Figure F.3 Evaluating an LLM with a verification-based method in free-form question answering. The model
generates a free-form answer (which may include multiple steps) and a final boxed answer, which is extracted
and compared against the correct answer from the dataset.


When we compare the extracted answer with the provided answer, as shown in figure F.3,
we can employ external tools, such as code interpreters or calculator software.

The downside is that this method can only be applied to domains that can be easily (and
ideally deterministically) verified, such as math and code. Also, this approach can introduce
additional complexity and dependencies, and it may shift part of the evaluation burden
from the model itself to the external tool.

However, because it allows us to generate an unlimited number of math problem
variations programmatically and benefits from step-by-step reasoning, it has become a
cornerstone of reasoning model evaluation and development.

An extensive example of this method is provided in chapter 3, which is why we skip a
code demonstration here.

##### F.4 Comparing models using preferences and leaderboards

So far, we have covered two methods that offer easily quantifiable metrics such as model
accuracy. However, none of the aforementioned methods evaluate LLMs in a more holistic
way, including judging the style of the responses. In this section, as illustrated in figure F.4,
we discuss a judgment-based method, namely, LLM leaderboards.

![image 171](<input (1)_images/imageFile171.png>)

- Figure F.4 A mental model of the topics covered in this book with a focus on the judgment- and benchmark-
based evaluation methods covered in this appendix. Having already covered benchmark-based approaches
(multiple choice, verifiers) in the previous section, we now introduce judgment-based approaches to measure
LLM performance, with this subsection focusing on leaderboards.


The leaderboard method mentioned in figure F.4 is a judgment-based approach where
models are ranked not by accuracy values or other fixed benchmark scores but by user (or
other LLM) preferences on their outputs.

A popular leaderboard is LM Arena (formerly Chatbot Arena, https://lmarena.ai/), where
users compare responses from two user-selected or anonymous models and vote for the
one they prefer, as shown in figure F.5.

![image 172](<input (1)_images/imageFile172.png>)

- Figure F.5 Example of a judgment-based leaderboard interface (LM Arena). Two LLMs are given the same
prompt, their responses are shown side by side, and users vote for the preferred answer.


These preference votes, which are collected as shown in figure F.5, are then aggregated
across all users into a leaderboard that ranks different models by user preference. In the
remainder of this section, we will implement a simple example of a leaderboard.

To create a concrete example, consider users prompting different LLMs in a setup similar
to figure F.5. The list below represents pairwise votes where the first model is the winner:

votes = [
("GPT-5", "Claude-3"),
("GPT-5", "Llama-4"),
("Claude-3", "Llama-3"),
("Llama-4", "Llama-3"),
("Claude-3", "Llama-3"),
("GPT-5", "Llama-3"),

]

In the list above, each tuple in the votes list represents a pairwise preference between two
models, written as (winner, loser). So, ("GPT-5", "Claude-3") means that a user
preferred GPT-5 over a Claude-3 model answer.

In the remainder of this section, we will turn the votes list into a leaderboard. For this,
we will use the popular Elo rating system, which was originally developed for ranking chess
players. Before we look at the concrete code implementation, in short, it works as follows.
Each model starts with a baseline score. Then, after each comparison and the preference
vote, the model’s rating is updated. Specifically, if a user prefers a current model over a
highly ranked model, the current model will get a relatively large ranking update and rank
higher in the leaderboard. Vice versa, if the current model wins against a lowly ranked
model, it increases the rating only a little. (And if the current model loses, it is updated in a
similar fashion, but with ranking points getting subtracted instead of added.)

The code to turn these pairwise rankings into a leaderboard is shown in listing F.4.

- Listing F.4 Constructing a leaderboard


def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):

ratings = { #A
model: initial_rating
for pair in vote_pairs
for model in pair

}

for winner, loser in vote_pairs: #B
rating_winner, rating_loser = ratings[winner], ratings[loser]

expected_winner = 1.0 / ( #C

1.0 + 10 ** ((ratings[loser] - ratings[winner]) / 400.0)
)

ratings[winner] = ( #D
ratings[winner] + k_factor * (1 - expected_winner)

)
ratings[loser] = (

ratings[loser] + k_factor * (0 - (1 - expected_winner))
)

return ratings

#A Initialize all models with the same base rating
#B Update ratings after each match
#C Expected score for the current winner given the ratings
#D k_factor determines sensitivity of rating updates

The elo_ratings function in listing F.4 takes the votes as input and turns it into a
leaderboard, as follows:

ratings = elo_ratings(votes, k_factor=32, initial_rating=1000)
for model in sorted(ratings, key=ratings.get, reverse=True):

print(f"{model:8s} : {ratings[model]:.1f}")

This results in the following leaderboard ranking, where the higher the score, the better:

GPT-5 : 1043.7
Claude-3 : 1015.2
Llama-4 : 1000.7
Llama-3 : 940.4

So, how does this work? For each pair, we compute the expected score of the winner using
the following formula:

expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))

This value expected_winner is the model's predicted chance to win in a no-draw setting
based on the current ratings. It determines how large the rating update is.

First, each model starts at initial_rating = 1000. If the two ratings (winner and
loser) are equal, we have expected_winner = 0.5, which indicates an even match. In this
case, the updates are:

rating_winner + k_factor * (1 - 0.5) = rating_winner + 16
rating_loser + k_factor * (0 - (1 - 0.5)) = rating_loser - 16

Now, if a heavy favorite (a model with a high rating) wins, we have expected_winner ≈ 1.
The favorite gains only a small amount and the loser loses only a little:

rating_winner + 32 * (1 - 0.99) = rating_winner + 0.32
rating_loser + 32 * (0 - (1 - 0.99)) = rating_loser - 0.32

However, if an underdog (a model with a low rating) wins, we have expected_winner ≈ 0,
and the winner gets almost the full k_factor points while the loser loses about the same
magnitude:

rating_winner + 32 * (1 - 0.01) = rating_winner + 31.68
rating_loser + 32 * (0 - (1 - 0.01)) = rating_loser - 31.68

###### ORDER MATTERS

The Elo approach updates ratings after each match (model comparisons), so later
results build on ratings that have already been updated. This means the same set of
outcomes, when presented in a different order, can end with slightly different final
scores. This effect is usually mild, but it can happen especially when an upset
happens early versus late.

To reduce this order effect, we can shuffle the votes pairs and run the
elo_ratings function multiple times and average the ratings.

Leaderboard approaches such as the one described above provide a more dynamic view of
model quality than static benchmark scores. However, the results can be influenced by user
demographics, prompt selection, and voting biases. Benchmarks and leaderboards can also
be gamed, and users may select responses based on style rather than correctness. Finally,
compared to automated benchmark harnesses, leaderboards do not provide instant
feedback on newly developed variants, which makes them harder to use during active
model development.

###### OTHER RANKING METHODS

The LM Arena originally used the Elo method described in this section but recently
transitioned to a statistical approach based on the Bradley–Terry model. The main
advantage of the Bradley–Terry model is that, being statistically grounded, it allows
the construction of confidence intervals to express uncertainty in the rankings. Also,
in contrast to the Elo ratings, the Bradley–Terry model estimates all ratings jointly
using a statistical fit over the entire dataset, which makes it immune to order effects.

To keep the reported scores in a familiar range, the Bradley–Terry model is fitted
to produce values comparable to Elo. Even though the leaderboard no longer
officially uses Elo ratings, the term "Elo" remains widely used by LLM researchers
and practitioners when comparing models. A code example showing the Elo rating is
included in this book's bonus materials at https://github.com/rasbt/reasoning-from-
scratch/tree/main/chF/03_leaderboards.

##### F.5 Judging responses with other LLMs

In the early days, LLMs were evaluated using statistical and heuristics-based methods,
including a measure called BLEU, which is a crude measure of how well generated text
matches reference text. The problem with such metrics is that they require exact word
matches and don't account for synonyms, word changes, and so on.

One solution to this problem, if we want to judge the written answer text as a whole, is
to use relative rankings and leaderboard-based approaches as discussed in the previous
section. However, a downside of leaderboards is the subjective nature of the preference-
based comparisons as it involves human feedback (as well as the challenges that are
associated with collecting this feedback).

A related method is to use another LLM with a pre-defined grading rubric (i.e., an
evaluation guide) to compare an LLM's response to a reference response and judge the
response quality based on a pre-defined rubric, as illustrated in figure F.6.

![image 173](<input (1)_images/imageFile173.png>)

- Figure F.6 Example of an LLM-judge evaluation. The model to be evaluated generates an answer, which is then
scored by a separate judge LLM according to a rubric and a provided reference answer.


In practice, the judge-based approach shown in figure F.6 works well when the judge LLM is
strong. Common setups use leading proprietary LLMs via API, though specialized judge
models also exist (see appendix A for references). One of the reasons why judges work so
well is also that evaluating an answer is often easier than generating one.

To implement a judge-based model evaluation as shown in figure F.6 programmatically in
Python, we could either load one of the Qwen3 models (appendix D) and prompt it with a
grading rubric and the model answer we want to evaluate.

Alternatively, we can use other LLMs through an API, for example the ChatGPT or Ollama
API. In the remainder of the section, we will implement the judge-based evaluation shown
in figure F.6 using the Ollama API in Python.

Specifically, we will use the 20-billion parameter gpt-oss open-weight model by OpenAI

- as it offers a good balance between capabilities and efficiency. For more information about
gpt-oss, please see my From GPT-2 to gpt-oss: Analyzing the Architectural Advances article
- at https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the.


- F.5.1 Implementing a LLM-as-a-judge approach in Ollama


Ollama (https://ollama.com) is an efficient open-source application for running LLMs on a
laptop. It serves as a wrapper around the open-source llama.cpp library (https://github.
com/ggerganov/llama.cpp), which implements LLMs in pure C/C++ to maximize efficiency.
However, note that Ollama is only a tool for generating text using LLMs (inference) and
does not support training or fine-tuning LLMs.

To execute the following code, please install Ollama by visiting https://ollama.com and
follow the provided instructions for your operating system:

For macOS and Windows users: Open the downloaded Ollama application.
If prompted to install command-line usage, select "yes."

For Linux users: Use the installation command available on the Ollama
website.

Before implementing the model evaluation code, let's first download the gpt-oss model and
verify that Ollama is functioning correctly by using it from the command line terminal.

Execute the following command on the command line (not in a Python session) to try out
the 20 billion parameter gpt-oss model:

ollama run gpt-oss:20b

The first time you execute this command, the 20 billion parameter gpt-oss model, which
takes up 14 GB of storage space, will be automatically downloaded. The output looks as
follows:

$ ollama run gpt-oss:20b
pulling manifest
pulling b112e727c6f1: 100% ▕██████████████████████▏ 13 GB
pulling fa6710a93d78: 100% ▕██████████████████████▏ 7.2 KB
pulling f60356777647: 100% ▕██████████████████████▏ 11 KB
pulling d8ba2f9a17b3: 100% ▕██████████████████████▏ 18 B
pulling 55c108d8e936: 100% ▕██████████████████████▏ 489 B
verifying sha256 digest
writing manifest
removing unused layers
success

###### ALTERNATIVE OLLAMA MODELS

Note that the gpt-oss:20b in the ollama run gpt-oss:20b command refers to the
20 billion parameter gpt-oss model. Using Ollama with the gpt-oss:20b model
requires approximately 13 GB of RAM. If your machine does not have sufficient RAM,
you can try using a smaller model, such as the 4 billion parameter qwen3:4b model
via ollama run qwen3:4b, which only requires around 4 GB of RAM.

For more powerful computers, you can also use the larger 120-billion parameter
gpt-oss model by replacing gpt-oss:20b with gpt-oss:120b. However, keep in mind
that this model requires significantly more computational resources.

Once the model download is complete, we are presented with a command-line interface
that allows us to interact with the model. For example, try asking the model, "What is
1+2?":

>>> What is 1+2?
Thinking...
User asks: "What is 1+2?" This is simple: answer 3. Provide explanation? Possibly
ask for simple
arithmetic. Provide answer: 3.

...done thinking.

1 + 2 = **3**

You can end this ollama run gpt-oss:20b session using the input /bye.

In the remainder of this section, we will use the ollama API. This approach requires that
Ollama is running in the background. There are three different options to achieve this:

- 1. Run the ollama serve command in the terminal (recommended). This runs the Ollama

backend as a server, usually on http://localhost:11434. Note that it doesn’t load a
model until it's called through the API (later in this section).

- 2. Run the ollama run gpt-oss:20b command similar to earlier, but keep it open and

don't exit the session via /bye. As discussed earlier, this opens a minimal convenience
wrapper around a local Ollama server. Behind the scenes, it uses the same server API as
ollama serve.

- 3. Ollama desktop app. Opening the desktop app runs the same backend automatically


and provides a graphical interface on top of it as shown in the earlier figure F.6.

###### OLLAMA SERVER IP ADDRESS

Ollama runs locally on our machine by starting a local server-like process. When
running ollama serve in the terminal, as described above, you may encounter an
error message saying Error: listen tcp 127.0.0.1:11434: bind: address
already in use.

If that's the case, try use the command OLLAMA_HOST=127.0.0.1:11435 ollama
serve (and if this address is also in use, try to increment the numbers by one until
you find an address not in use.)

The following code verifies that the Ollama session is running properly before we use
Ollama to evaluate the test set responses generated in the previous section:

- Listing F.5 Checking Ollama is running


import psutil

def check_if_running(process_name):
running = False
for proc in psutil.process_iter(["name"]):

if process_name in proc.info["name"]:
running = True
break

return running

ollama_running = check_if_running("ollama")

if not ollama_running:
raise RuntimeError(

"Ollama not running. Launch ollama before proceeding."
)

print("Ollama running:", check_if_running("ollama"))

Ensure that the output from executing the previous code displays Ollama running: True.
If it shows False, please verify that the ollama serve command or the Ollama application
is actively running.

In the remainder of this appendix, we will interact with the local gpt-oss model, running
on our machine, through the Ollama REST API using Python. The following query_model
function demonstrates how to use the API:

- Listing F.6 Querying a local Ollama model


import json
import requests

def query_model(
prompt,
model="gpt-oss:20b",
url="http://localhost:11434/api/chat" #A

):

data = { #B
"model": model,
"messages": [

{"role": "user", "content": prompt}

],
"options": { #C

"seed": 123,
"temperature": 0,
"num_ctx": 2048

}
}

#D
with requests.post(url, json=data, stream=True, timeout=30) as r:

r.raise_for_status() #E
response_data = ""
for line in r.iter_lines(decode_unicode=True): #F

if not line:

continue
response_json = json.loads(line) #G
if "message" in response_json: #H

response_data += response_json["message"]["content"]

return response_data

return response_data

- #A If you used OLLAMA_HOST=127.0.0.1:11435 ollama serve, update the address
- #B Create the data payload as a dictionary
- #C Settings required for deterministic responses
- #D Send the POST request with a JSON payload and open a streaming response
- #E Raise an error if the server response indicates failure
- #F Iterate over each streamed line from the response
- #G Parse each line into JSON format
- #H Extract and accumulate the message content from the response


Here's an example of how to use the query_model function from listing F.6 that we just
implemented:

ollama_model = "gpt-oss:20b"
result = query_model("What is 1+2?", ollama_model)
print(result)

The resulting response is "3". (It differs from what we'd get if we ran Ollama run or the
Ollama application due to different default settings.)

Using the query_model function, we can evaluate the responses generated by our model
with a prompt that includes a grading rubric asking the gpt-oss model to rate our target
model's responses on a scale from 1 to 5 based on a correct answer as a reference.

The prompt we use for this is shown in listing F7:

- Listing F.7 Setting up the prompt template including grading rubric


def rubric_prompt(instruction, reference_answer, model_answer):
rubric = (

"You are a fair judge assistant. You will be given an instruction, "
"a reference answer, and a candidate answer to evaluate, according "
"to the following rubric:\n\n"

- "1: The response fails to address the instruction, providing "
"irrelevant, incorrect, or excessively verbose content.\n"
- "2: The response partially addresses the instruction but contains "
"major errors, omissions, or irrelevant details.\n"
- "3: The response addresses the instruction to some degree but is "
"incomplete, partially correct, or unclear in places.\n"
- "4: The response mostly adheres to the instruction, with only "
"minor errors, omissions, or lack of clarity.\n"
- "5: The response fully adheres to the instruction, providing a "
"clear, accurate, and relevant answer in a concise and efficient "
"manner.\n\n"
"Now here is the instruction, the reference answer, and the "
"response.\n"


)

prompt = (
f"{rubric}\n"
f"Instruction:\n{instruction}\n\n"
f"Reference Answer:\n{reference_answer}\n\n"
f"Answer:\n{model_answer}\n\n"
f"Evaluation: "

)
return prompt

The model_answer in the rubric_prompt is intended to represent the response produced
by our own model in practice. For illustration purposes, we hardcode a plausible model
answer here rather than generating it dynamically. (However, feel free to use the Qwen3
model we loaded in section F.2.1 to generate a real model_answer).

Next, let's generate the rendered prompt for the Ollama model:

rendered_prompt = rubric_prompt(

instruction=(
"If all birds can fly, and a penguin is a bird, "
"can a penguin fly?"

),
reference_answer=(

"Yes, according to the premise that all birds can fly, "
"a penguin can fly."

),
model_answer=(

"Yes – under those premises a penguin would be able to fly."
)

)
print(rendered_prompt)

The output is as follows:

You are a fair judge assistant. You will be given an instruction, a
reference answer, and a candidate answer to evaluate, according to the
following rubric:

- 1: The response fails to address the instruction, providing irrelevant,
incorrect, or excessively verbose content.
- 2: The response partially addresses the instruction but contains major
errors, omissions, or irrelevant details.
- 3: The response addresses the instruction to some degree but is
incomplete, partially correct, or unclear in places.
- 4: The response mostly adheres to the instruction, with only minor
errors, omissions, or lack of clarity.
- 5: The response fully adheres to the instruction, providing a clear,
accurate, and relevant answer in a concise and efficient manner.


Now here is the instruction, the reference answer, and the response.

Instruction:
If all birds can fly, and a penguin is a bird, can a penguin fly?

Reference Answer:
Yes, according to the premise that all birds can fly, a penguin can
fly.

Answer:
Yes – under those premises a penguin would be able to fly.

Evaluation:

Ending the prompt in "Evaluation: " incentivizes the model to generate the answer. Let's
see how the gpt-oss:20b model judges the response:

result = query_model(rendered_prompt, ollama_model)
print(result)

The response is as follows:

**Score: 5**

The candidate answer directly addresses the question, correctly applies the given
premises, and concisely states that a penguin would be able to fly. It is
accurate, relevant, and clear.

As we can see, the answer receives the highest score, which is reasonable, as it is indeed
correct. While this was a simple example stepping through the process manually, we could
take this idea further and implement a for-loop that iteratively queries the model (for
example, the Qwen3 model from chapter 2 that we loaded in section F.2.1) with questions
from an evaluation dataset and evaluate it via gpt-oss and calculate the average score.
Then, doing this for two models (for example, the Qwen3 base and reasoning model), we
can compare the models relative to each other.

###### SCORING INTERMEDIATE REASONING STEPS WITH PROCESS REWARD MODELS

Related to symbolic verifiers and LLM judges, there is a class of learned models
called process reward models (PRMs). Like judges, PRMs can evaluate reasoning
traces beyond just the final answer, but unlike general judges, they focus specifically
on the intermediate steps of reasoning. And unlike verifiers, which check correctness
symbolically and usually only at the outcome level, PRMs provide step-by-step
reward signals during training in reinforcement learning. We can categorize PRMs as
"step-level judges," which are predominantly developed for training, not pure
evaluation. (In practice, PRMs are difficult to train reliably at scale. For example,
DeepSeek R1 did not adopt PRMs and instead combined verifiers for the reasoning
training.)

Judge-based evaluations offer advantages over preference-based leaderboards, including
scalability and consistency, as they do not rely on large pools of human voters. (Technically,
it is possible to outsource the preference-based rating behind leaderboards to LLM judges
as well). However, LLM judges also share similar weaknesses with human voters: results
can be biased by model preferences, prompt design, and answer style. Also, there is a
strong dependency on the choice of judge model and rubric, and they lack the
reproducibility of fixed benchmarks.

## Appendix G. Building a chat interface

The supplementary code repository for this book includes a small browser-based chat
interface built with the open-source Chainlit Python package (https://github.com/Chainlit/
chainlit) to interact with the various LLMs and reasoning models in this book via a ChatGPT-
like interface, as shown in the screenshot in figure G.1.

![image 174](<input (1)_images/imageFile174.png>)

Figure G.1 Chainlit interface for the Qwen3 model, which mimics the ChatGPT interface.

This interface shown in figure G.1 can be useful when we want a more convenient way to
interact with the models than typing prompts into a notebook cell or terminal.

Building upon the Chainlit library, the implementation is relatively straightforward.
Instead of printing tokens to the terminal, we wrap the same model loading and generation
functions from the earlier chapters inside a small web application, which stays local on our
computer and runs in our browser.

In practice, this means that we can reuse the from-scratch Qwen3Model implementation
and the chapter 2 streaming helper while letting Chainlit handle the browser UI and
message events.

The supplementary materials contain two variants in chG/01_main-chapter-code
(https://github.com/rasbt/reasoning-from-scratch/tree/main/chG/01_main-chapter-code):

qwen3_chat_interface.py, which is a single-turn interface;

qwen3_chat_interface_multiturn.py, which stores conversation history
and supports multi-turn interactions.

In this appendix, we will walk through both versions. We will also discuss how to install
Chainlit, how to download the scripts via the download_from_github helper from chapter 7.

##### G.1 Installing Chainlit

Before running the chat interface, we first need to install Chainlit. The simplest option is:

pip install chainlit

If you are using uv, the equivalent command is:

uv add chainlit

If you are working directly inside the reasoning-from-scratch repository, Chainlit is also
listed as an optional dependency in the project's pyproject.toml. In that case, a
convenient alternative is:

uv sync --extra extra

or:

pip install -e ".[extra]"

##### G.2 Running the code as a script

Chainlit runs ordinary Python scripts. This is important because Chainlit imports the script
as a module and looks for special functions decorated with @chainlit.on_chat_start and
@chainlit.on_message as we will see later.

This means that if we want to reuse the code shown in this appendix, we should place it
into a .py file such as:

qwen3_chat_interface.py

or:

app.py

and then launch it from the terminal via:

chainlit run app.py

Or, if you are using uv, this command becomes:

uv run chainlit run app.py

In other words, while we can inspect the code in a notebook, the actual Chainlit application
is expected to live in a standalone Python script.

##### G.3 Downloading the scripts

If you cloned the repository, the files are already present in chG/01_main-chapter-code
and you can skip this step. Otherwise, as discussed in chapter 7, we can use
download_from_github(...) to fetch the two scripts:

- Listing G.1 Download Chainlit scripts


from reasoning_from_scratch.ch07 import download_from_github

download_from_github(
"chG/01_main-chapter-code/qwen3_chat_interface.py"

)
download_from_github(

"chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py"
)

##### G.4 The regular single-turn script

The regular single-turn version in qwen3_chat_interface.py is intentionally compact. It
reuses the earlier chapter code almost directly and wraps it in the minimum amount of
Chainlit-specific logic.

Here, "single-turn" means that we can still submit multiple queries, but each query is
independent of each other and doesn't know about the previous query.

The code, which is relatively short, is shown in its entirety in listing G2.

- Listing G.2 Chainlit script for single-turn queries


import os
from pathlib import Path

import torch
import chainlit

from reasoning_from_scratch.ch02 import (
get_device,

)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import (

load_model_and_tokenizer,
load_tokenizer_only

)
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

#A
WHICH_MODEL = "reasoning" #B
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False

DEVICE = get_device()

def load_app_model_and_tokenizer():
if CHECKPOINT_PATH is None:

return load_model_and_tokenizer(
which_model=WHICH_MODEL,
device=DEVICE,
use_compile=COMPILE,
local_dir=LOCAL_DIR,

)

checkpoint_path = Path(CHECKPOINT_PATH) #C
if not checkpoint_path.exists():
raise FileNotFoundError(

f"Checkpoint file not found: {checkpoint_path}"
)

tokenizer = load_tokenizer_only(
which_model=WHICH_MODEL, local_dir=LOCAL_DIR

)
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(

torch.load(checkpoint_path, map_location="cpu")

)
model.to(DEVICE)

if COMPILE:
torch._dynamo.config.allow_unspec_int_on_nn_module = True
model = torch.compile(model)

return model, tokenizer

MODEL, TOKENIZER = load_app_model_and_tokenizer()

EOS_TOKEN_IDS = (
TOKENIZER.encode("<|im_end|>")[0],
TOKENIZER.encode("<|endoftext|>")[0]

)

- #D
@chainlit.on_chat_start
async def on_start():

chainlit.user_session.set("history", [])
chainlit.user_session.get("history").append(

{"role": "system", "content": "You are a helpful assistant."}
)

- #E
@chainlit.on_message
async def main(message: chainlit.Message):
- #F
input_ids = TOKENIZER.encode(message.content)
input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
- #G
out_msg = chainlit.Message(content="")
await out_msg.send()
- #H


for tok in generate_text_basic_stream_cache(
model=MODEL,
token_ids=input_ids_tensor,
max_new_tokens=MAX_NEW_TOKENS,

):

token_id = tok.squeeze(0).item()
if token_id in EOS_TOKEN_IDS:

break
piece = TOKENIZER.decode([token_id])
await out_msg.stream_token(piece)

#I
await out_msg.update()

- #A Configuration section
- #B Change to "base" for base model
- #C Optionally load trained checkpoint from chapter 7 or 8
- #D Initialize per-session state and seed it with a default system prompt
- #E Handle one user message and stream the model response into the Chainlit UI
- #F 1) Encode input
- #G 2) Start an outgoing message we can stream into
- #H 3) Stream generation
- #I 4) Finalize the streamed message


The CHECKPOINT_PATH environment variable in listing G.2 optional. If it is set, the script
loads that custom .pth checkpoint instead of the default official weights. The
load_app_model_and_tokenizer() helper handles these two cases. If no custom
checkpoint is provided, it delegates to load_model_and_tokenizer(...) from chapter 3. If
a custom checkpoint path is provided, it loads only the tokenizer via
load_tokenizer_only(...), constructs a fresh Qwen3Model(QWEN_CONFIG_06_B), and then
loads in the weight with the custom state dictionary.

This is useful because it means the single-turn Chainlit interface can also be used to
inspect chapter 6, 7, or 8 checkpoints interactively, provided that the tokenizer setting
remains aligned with the checkpoint. The next section will explain how exactly the custom
checkpoint can be provided as input from the command line terminal.

The @chainlit.on_chat_start function initializes a history object and inserts a default
system message, which is a common convention:

{"role": "system", "content": "You are a helpful assistant."}

However, the single-turn script does not actually use that stored history when generating
the response. Inside main(...), the prompt is built only from the current
message.content:

input_ids = TOKENIZER.encode(message.content)

So even though the browser UI looks like a chat application, the regular version behaves
more like a prompt box that happens to show earlier messages visually without the model
actually remembering earlier conversations. In other words, the model itself only sees the
current user message and treats each turn independently.

The rest of the generation logic is very similar to what we already used in chapter 2.

##### G.5 Running the single-turn script

If we are inside the supplementary code repository, the most convenient command to
launch the user interface is usually:

uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface.py

If we saved the code under another filename such as app.py, we would simply replace the
path accordingly:

uv run chainlit run app.py

After launching the server, Chainlit usually opens a browser tab automatically. If not, we
can copy the local address shown in the terminal into the browser manually. Typically this is
http://localhost:8000.

Figure G.2 shows an example output in the Chainlit interface.

![image 175](<input (1)_images/imageFile175.png>)

Figure G.2 Example response in the Chainlit interface.

By default, the script initializes the Qwen3 reasoning variant we briefly introduced in
chapter 3, which provided the response shown in figure G.2. To run the same script with a
custom checkpoint, we pass the path via the CHECKPOINT_PATH environment variable. For
example:

CHECKPOINT_PATH=path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface.py

When doing so, WHICH_MODEL must remain aligned with the tokenizer expected by the
checkpoint. The chapter 8 distillation checkpoints use the reasoning tokenizer, so in that
case we should keep WHICH_MODEL = "reasoning". For chapters 6 checkpoints, for
example, we change it to WHICH_MODEL = "base".

Editing the configuration directly in the script may seem a bit cumbersome. The reason
for this, over using command line arguments is that Chainlit itself launches and manages
the application entry point via chainlit run ..., and as a result, the script does not
behave like a normal standalone Python program where we can freely define and parse
custom command-line arguments with the Python argparse library, for example. For simple
options such as model choice, generation length, or checkpoint paths, it is therefore more
convenient to keep the settings in the script or pass them via environment variables.

###### DOWNLOADING MODEL CHECKPOINTS

If you want to explore some of the model checkpoints from chapters 6, 7, or 8 but
don't have the compute capacity to train them yourself, you can find information on
how to download these in the supplementary materials at https://github.com/rasbt/
reasoning-from-scratch/tree/main/ch07/04_download_trainining_checkpoints and
https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/05_download_
training_checkpoints.

###### G.6 The multi-turn interface

The second script, qwen3_chat_interface_multiturn.py, extends the basic Chainlit
interface so that the model can condition on earlier turns in the same conversation. The
following subsections explain what multi-turn means in practice, how the script implements
it, and when it is appropriate to use.

- G.6.1 What multi-turn means

In a single-turn setup, the model only sees the current input. For illustration purposes,
suppose the interaction is:

In the regular single-turn script, when the model receives the second message, "What did
I just ask?", it might respond with "I don't know what you are asking". The reason
is that the earlier question and answer are visible in the browser, but they are not included
in the model input. So the model does not truly have conversation memory.

In a multi-turn setup (qwen3_chat_interface_multiturn.py), we explicitly include (or,
more specifically, prepend) the earlier messages in the prompt similar to what ChatGPT and
other chat assistants do. So, in this case, the model might answer the question "What did
I just ask?" with "You asked me what 1+1 is".

- G.6.2 The multi-turn variant


User: What is 1 + 1?
Assistant: 2
User: What did I just ask?

The multi-turn script begins in the same way as the regular version, but the main difference
is that it adds two helper functions and then actually uses the stored session history during
generation.

The first helper, shown in listing G.2, turns the stored message history into the Qwen
chat format we require here.

- Listing G.3 Building a prompt from conversation history


def build_prompt_from_history(history, add_assistant_header=True):
parts = []
for m in history:

role = m["role"]
content = m["content"]
parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

if add_assistant_header:

parts.append("<|im_start|>assistant\n")
return "".join(parts)

The build_prompt_from_history function in listing G.3 serializes the message list into the
chat-template style that the Qwen reasoning model expects. Conceptually, the resulting
prompt looks like this:

<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 1 + 1?<|im_end|>
<|im_start|>assistant
2<|im_end|>
<|im_start|>user
What did I just ask?<|im_end|>
<|im_start|>assistant

The model then continues generating from that final assistant header.

The second helper trims the messages, shown in listing G.3, is needed to trim the
messages if they exceed the context length, since multi-turn conversations can grow very
long.

- Listing G.4 Trimming the prompt to fit the context window


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
assert max_new_tokens < context_len
keep_len = max(1, context_len - max_new_tokens)

#A If the prompt is too long, left-truncate to keep_len
if input_ids_tensor.shape[1] > keep_len: #A

input_ids_tensor = input_ids_tensor[:, -keep_len:]

return input_ids_tensor

#A If the prompt is too long, left-truncate to

This trim_input_tensor helper function ensures that enough room remains in the context
window for the new tokens. If the conversation becomes too long, the oldest tokens are
discarded first via left truncation. (Other, more sophisticated options also exist in practice,
such as summarizing the previous context; however, for the sake of simplicity, this is not
implemented here.)

- G.6.3 How the multi-turn script uses history


The actual multi-turn behavior appears inside the message handler. At the beginning of
each turn, the script retrieves the current session history and appends the new user
message:

history = chainlit.user_session.get("history")
history.append({"role": "user", "content": message.content})

It then builds the full prompt from that history, tokenizes it, optionally trims it, and streams
the answer in the same way as before.

At the end of generation, it stores the model reply back into the same history list:

history.append({"role": "assistant", "content": out_msg.content})
chainlit.user_session.set("history", history)

This is the core difference relative to the single-turn script. The multi-turn version does not
merely display a chat transcript in the browser, that is, it actually feeds the earlier turns
back into the model so that future responses can condition on them.

The launch command is analogous to the single-turn case:

uv run chainlit run chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py

Operationally, the script behaves just like the regular Chainlit interface, as shown in figure
G.3. The difference is only in how the prompt is constructed internally.

![image 176](<input (1)_images/imageFile176.png>)

Figure G.3 Example response in the Chainlit interface with multiple prompts and responses.

The second response in figure G.3 indicates that the model remembers the first question
through the provided context.

- G.6.4 Recommendations


There is an important practical limitation worth emphasizing. The multi-turn Chainlit script
is intended for the official Qwen3 reasoning variant. It is not a good fit for the chapter 8
distillation checkpoints, for example.

The reason is that the distillation checkpoints were trained on prompt-response style
supervision rather than on persistent multi-turn chat histories. In other words, they were
not trained to continue a conversation over several turns in the same way as the official
reasoning checkpoint.

As a consequence, the multi-turn script should not be expected to work reliably with the
distillation checkpoints. The practical recommendation is:

use the single-turn Chainlit script for quick interactive tests, including
custom chapter 6, 7, and 8 checkpoints;

use the multi-turn Chainlit script with the official reasoning checkpoint
when you want conversation memory.

© Manning Publications Co. To comment go to liveBook

