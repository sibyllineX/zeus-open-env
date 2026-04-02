# Agent Behavior Evaluation Environment 

## 1. Overview

The environment simulates a setting where an AI agent must answer queries using provided context and optional tools, while being evaluated across multiple behavioral dimensions.

Unlike traditional QA systems, this environment focuses not just on what the agent answers, but how it behaves during the process.

### Core Objective

To evaluate and quantify:

* Correctness (Is the answer factually correct?)
* Hallucination (Is the answer grounded in the provided context?)
* Context Drift (Is the response relevant to the query?)
* Tool Misuse / Efficiency (Are actions necessary and efficient?)

---

## 2. Fit to Hackathon Requirements

* Simulates a real-world problem: evaluating reliability of AI agents
* Supports multi-step interaction (actions, reasoning, tool use)
* Enables deterministic grading via structured evaluation rules
* Provides meaningful reward signals across the trajectory
* Naturally supports easy → medium → hard task scaling

---

## 3. Detection Dimensions 

The system works for both short and long answers. Short answers are treated as a single unit, while long answers are evaluated sentence-by-sentence.

---

### 3.1 Correctness 

Goal: Determine if the agent’s answer matches the ground truth.

**Steps:**

1. If the answer is long, split it into sentences

2. Normalize each sentence:

3. Extract key entities / tokens from:

   * ground truth answer
   * each sentence

4. For each sentence:

   * check if it matches or aligns with ground truth facts

5. Compute:

   * correctness ratio = correct sentences / total sentences

**Decision Rule:**

* High ratio → Correct
* Moderate ratio → Partially correct
* Low ratio → Incorrect

**Role in Training:**
Correctness provides the primary learning signal that pushes the agent toward producing accurate answers. It ensures the agent converges toward factual correctness while interacting with the environment.

---

### 3.2 Hallucination 

Goal: Determine if the answer is supported by the provided context.

**Steps:**

1. Split answer into sentences (if long)

2. For each sentence:

   * extract tokens/entities
   * compare with context

3. Identify:

   * supported sentences (present in context)
   * unsupported sentences (new information)

4. Compute:

   * grounded ratio = supported sentences / total sentences

**Decision Rule:**

* High grounded ratio → Grounded
* Mixed → Partial hallucination
* Low grounded ratio → Hallucination

**Role in Training:**
Hallucination detection teaches the agent to rely on provided context rather than generating unsupported information. This improves faithfulness and makes the agent more reliable in real-world scenarios where grounding is critical.

---

### 3.3 Context Drift 

Goal: Determine if the agent is answering the query.

**Steps:**

1. Split answer into sentences (if long)

2. Extract key terms/entities from:

   * query
   * each sentence

3. For each sentence:

   * measure overlap with query terms
   * check if it refers to the same subject

4. Compute:

   * relevance ratio = relevant sentences / total sentences

**Decision Rule:**

* High relevance → Relevant
* Mixed → Partial drift
* Low relevance → Context drift

**Role in Training:**
Context drift detection ensures the agent stays focused on the task. It discourages irrelevant or off-topic responses and helps the agent learn to align its reasoning with the query.

---

### 3.4 Tool Misuse / Efficiency 

Goal: Evaluate how efficiently the agent behaves.

**Steps:**

1. Track:

   * number of steps
   * sequence of actions

2. For each action, check:

   * necessity (was it required?)
   * redundancy (was it repeated?)
   * relevance (aligned with query?)

3. Evaluate:

   * whether fewer steps could achieve the same result

**Decision Rule:**

* Efficient minimal steps → Good behavior
* Extra unnecessary steps → Inefficient
* Repeated or irrelevant actions → Misuse

**Role in Training:**
Tool misuse detection shapes the agent’s action policy. By penalizing unnecessary or redundant actions, the environment encourages efficient decision-making, proper tool selection, and optimal sequencing of steps. This helps the agent learn not just to solve tasks, but to solve them efficiently.

---

## 4. How These Signals Differ 

Each signal is computed against a different reference:

* Correctness compares the answer with the ground truth and checks factual accuracy
* Hallucination compares the answer with the context and checks whether information is supported
* Context Drift compares the answer with the query and checks relevance
* Tool Misuse compares actions with history and task and checks behavior efficiency

This separation ensures that:

* an answer can be correct but hallucinated
* an answer can be incorrect but still relevant
* an answer can be relevant but inefficiently produced

---

## 5. Reward / Grading Strategy

we can uses a partial reward system that captures both final outcome and intermediate behavior.


### Reward Function
(This just as example, we should come up partial reward Strategy for single and mulit step)
$$TotalReward = R_{correct} - (P_{drift} \times n) - (P_{hallucination} \times m) - (P_{efficiency} \times steps)$$

Where:

* R_correct = reward for correctness
* P_drift = penalty per drift occurrence
* P_hallucination = penalty per hallucination
* P_efficiency = penalty per step
* n, m = counts of drift and hallucination events

### Reward Components
(just as example)
* +1.0 → Correct final answer
* +0.2 → Successfully citing a relevant part of context
* -0.3 → Each hallucination occurrence
* -0.1 → Context drift or inefficient steps

### Design Principles

* Encourage correct and grounded answers
* Penalize unsupported reasoning
* Penalize irrelevant responses
* Encourage efficiency (fewer steps)
* Provide partial credit for intermediate progress

---

## 6. OpenEnv Functions Responsibility

### reset()

* Selects a task (easy / medium / hard)
* Loads:

  * query
  * context
  * ground truth
* Initializes:

  * action history
  * step counter

---

### step(action)

* Receives agent action
* Updates state
* Runs evaluation pipeline:

  * context drift detection
  * correctness check
  * hallucination detection
  * tool usage / efficiency evaluation
* Computes reward using defined strategy
* Returns updated observation and evaluation signals

---

### state()

* Returns full internal state:

  * task definition
  * ground truth
  * action history
  * evaluation metrics

Used for:

* debugging
* transparency
* reproducibility

---

## 7. Evaluation Pipeline Summary

Answer
↓
Split into sentences (if long)
↓
For each sentence:

* check relevance (drift)
* check correctness
* check grounding
  ↓
  Aggregate scores
  ↓
  Compute reward

---

## 8. How This Differs from QA and RAG Evaluation Systems

Traditional QA and RAG evaluation frameworks focus on evaluating the final output of a model. They treat evaluation as a static process where a query, context, and answer are given, and the system measures how good the final answer is.

This environment takes a fundamentally different approach.

First, it evaluates the Reasoning Trace, not just the Answer String. That means it looks at the sequence of steps, actions, and decisions the agent takes while solving the task, rather than only the final output.

Second, it operates as a stateful environment, where each action affects future evaluation and reward. This allows tracking of behavior such as inefficiency, repeated mistakes, or unnecessary actions over time.

Third, it goes beyond static benchmarks. While traditional RAG evaluation frameworks are primarily used to test models, this environment is designed to support training and improving agents by providing structured reward signals that encourage better reasoning, grounding, and efficiency.

Finally, instead of binary or single-score evaluation, this system provides a multi-dimensional assessment that captures correctness, grounding, relevance, and behavioral efficiency across the entire interaction.

---
