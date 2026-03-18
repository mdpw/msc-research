# CHAPTER 12: STUDENT REFLECTIONS

This chapter is a personal reflection on my experience throughout this project — what I learned, the problems I hit, and what I would do differently if I were starting again.

---

## 12.1 Personal Growth and Skills Development

This was the most technically demanding project I have undertaken during my studies. Going in, I had no practical experience with on-device machine learning deployment, Android development with Jetpack Compose, or real-time WebSocket communication. By the end, I had built working implementations of all three — not from following tutorials, but from solving real problems under real constraints.

The biggest area of growth was learning to work across multiple technology stacks at the same time. On any given day I might be debugging a TFLite model conversion in Python, figuring out why the Kotlin tokeniser was producing wrong token sequences, and investigating a WebSocket disconnection on the Android client — all for the same system. The ability to context-switch between different languages and ecosystems and still make progress was something I developed gradually over the course of the project. By Iteration 4, tasks that had taken me days of research in Iteration 1 were taking hours.

I also gained a much more realistic understanding of the gap between research papers and working implementations. Reading that MobileBERT achieves 62ms inference latency on mobile devices sounds straightforward. Actually getting the model to load, convert correctly, tokenise inputs the same way Python does, and produce correct outputs on a budget Android tablet is a completely different challenge — and a more valuable learning experience than any benchmark result.

---

## 12.2 Problems Encountered and How They Were Resolved

### 12.2.1 The Vosk Model Memory Problem

The first significant problem appeared early in Iteration 1, before I had written much code at all. I loaded the large Vosk model onto the test Android tablet and watched the app crash with an out-of-memory error. The load time alone was over 15 seconds — completely unusable for a guest-facing system.

I had planned to use a larger English model because I assumed bigger meant better for speech recognition. What I had not accounted for was that a budget Android tablet with limited RAM simply cannot hold a model that size alongside the Android OS and other running processes. Switching to `vosk-model-small-en-in-0.4` (~36MB, Indian English) not only solved the memory problem but turned out to be a better choice for the target use case — the Indian English acoustic model handles South Asian accents much better than a US English model would have.

The lesson here was immediate and concrete: always test on your actual target hardware as early as possible. If I had left this until later in the project, it could have forced a major redesign with very little time to spare.

### 12.2.2 The TFLite Conversion and Label Ordering Bug

Converting the trained MobileBERT model from PyTorch to TFLite was supposed to take two or three days. It ended up taking nearly a week, mostly because of one bug that was very difficult to diagnose.

The initial conversion worked — the TFLite model loaded successfully and produced output. But the predictions were wrong. A request for towels would come back classified as something completely unrelated. I spent a long time checking whether the issue was in the model itself, the tokeniser, or the inference code, before finally tracing it to a mismatch in how PyTorch and TensorFlow ordered the classification labels in the model output. The label map I was using during inference was indexed differently from the one used during training.

The fix itself was simple once I found it — aligning the label ordering between the training output and the TFLite inference. But finding it required carefully verifying the model's raw logit outputs against known test inputs in both frameworks. I should have done this verification step first, before writing any inference code. The lesson I took from this is that model conversion is not a black-box export operation — every stage needs to be verified against the previous one.

### 12.2.3 The Tokeniser Compromise

This was the problem I found most frustrating because I never fully solved it — I worked around it.

BERT models use a WordPiece tokeniser, which splits words into sub-word units (for example, "housekeeping" becomes ["house", "##keeping"]). The HuggingFace tokeniser that handles this in Python is not available on Android. I spent a significant amount of time trying to re-implement the WordPiece algorithm in Kotlin from scratch, which involves iteratively matching the longest possible sub-word tokens from a 30,522-entry vocabulary, managing special tokens ([CLS], [SEP], [UNK]), and padding to a fixed sequence length of 32.

My Kotlin implementation produced wrong token sequences, which caused the model to output nonsensical predictions. Debugging this required comparing token-by-token outputs between my Kotlin code and the Python tokeniser, which eventually revealed several edge cases I had missed in the subword matching logic.

In the end, I made a pragmatic decision: rather than continue trying to perfectly replicate the WordPiece algorithm, I implemented a simplified word-level tokeniser that splits text by whitespace and looks up whole words in a vocabulary file. This is not as accurate as full WordPiece tokenisation, particularly for less common words that need sub-word decomposition, and it means the actual on-device accuracy is lower than the 99.06% measured in the Python evaluation. I documented this honestly as a limitation (Section 10.3.8). In a production system, properly solving this — either by implementing a correct Kotlin WordPiece tokeniser or using TFLite's built-in tokenization support — would be a priority.

### 12.2.4 The WebSocket Screen-Off Disconnection

During Iteration 3, the staff dashboard was receiving real-time request updates perfectly — until I noticed that the Android device was silently dropping its WebSocket connection whenever the screen turned off. This was not obvious at first because the app did not report any error; it simply stopped receiving notifications.

The fix was to implement an exponential backoff reconnection strategy in the WebSocket client: on disconnect, wait 2 seconds and retry, then 4 seconds, then 8 seconds, up to a maximum of 30 seconds. This resolved the problem, and the reconnection behaviour was explicitly verified during integration testing.

The lesson was that mobile networking is less reliable than desktop networking in ways that are not always obvious. The device power management system can interfere with long-lived network connections, and any real-time mobile application needs to handle reconnection explicitly rather than assuming a connection will stay open indefinitely.

### 12.2.5 Keeping Scope Under Control

Midway through the project, I found myself tempted to keep adding features. Wake word detection was partially explored. Sinhala language support seemed achievable. A more sophisticated dialogue system felt like it would make the system much better. Each feature seemed like it would take just a few extra days.

Recognising this pattern and deciding to stop was one of the better decisions I made. I removed the experimental wake word code from the repository, documented these features as future work, and focused on completing and properly evaluating what I had already built. A thoroughly evaluated core system is a stronger submission than a feature-rich one with gaps in its evaluation. The temptation to keep building never fully goes away, but scope discipline is what makes the difference between finishing a project and not finishing it.

---

## 12.3 What I Would Do Differently

### 12.3.1 Start Writing Earlier

This is probably the most common reflection in any dissertation, but I genuinely experienced the consequences of leaving it too late. I delayed focused writing until the implementation was largely complete, which compressed the writing timeline more than it needed to be. The WebSocket issues in Iteration 3 pushed writing back even further, and the final weeks involved more parallel pressure than was comfortable.

Writing earlier would also have helped with the implementation itself. When I eventually wrote up the design decisions — why certain models were chosen, why the hybrid NLU pipeline was structured the way it was — some justifications required more thought than I had given them during coding. Writing those justifications earlier would probably have led to slightly better decisions at the time.

### 12.3.2 Collect Real Hotel Data Earlier

The training dataset was synthetically generated, which was practical and produced strong results — Model C achieved 99.06% accuracy on Vosk-transcribed test data. But I was always aware that real guest utterances would have made the dataset more authentic.

I had access to hotel management contacts from the requirements-gathering interviews. I could have asked informally whether staff could record a sample of real guest service requests, or whether anyone would be willing to read sample utterances aloud for recording. Even a small set of real recordings — a few hundred utterances — used to supplement the synthetic data or as an independent test set would have made the evaluation more convincing. Pursuing this earlier, before the implementation was underway, would have been much easier to organise than doing it during the busiest phase of the project.

### 12.3.3 Build in Automated Tests from the Start

I tested almost everything manually throughout the project, which worked but became increasingly time-consuming as the system grew. A basic automated test suite for the critical components — the tokeniser, the hybrid NLU pipeline, and the core API endpoints — would have caught several bugs faster. The TFLite label ordering issue and the tokeniser edge cases in particular would likely have been found much earlier if I had been running automated regression tests after every change.

The `test_api.py` script I eventually wrote for the backend was useful and straightforward to build. I should have built equivalent scripts for the NLU pipeline verification much earlier in the project.

### 12.3.4 Use Structured Logging from the Beginning

I used print statements for debugging throughout. This was fine for simple, isolated testing. For timing-related issues in the real-time communication layer — tracking exactly what the Android client was doing when the WebSocket connection dropped — it was genuinely difficult to piece together what had happened. Structured logging with timestamps and component labels from the start would have made that kind of debugging much faster.

This feels like an obvious point in retrospect, but it is the kind of thing that is easy to skip when getting started on a project because it seems like unnecessary overhead before the system is complex enough to need it. By the time the system was complex enough to need it, it was too late to retrofit easily.

---

## 12.4 Reflections on the Research Process

### 12.4.1 Iterative Development Was the Right Approach

Looking back, the iterative development structure was not just a methodology choice — it was what made the project manageable. The most consequential decisions of the entire project were made because I had a working prototype to learn from: switching to the smaller Indian English Vosk model after the large model caused memory crashes; implementing the hybrid NLU pipeline after observing that the neural model alone was unreliable for very short or keyword-dominated requests; adding the WebSocket reconnection logic after discovering the screen-off disconnection behaviour.

None of these were decisions I could have made during upfront design. They all required building something real, running it, and observing what happened. I went into the project knowing that iterative development was a sensible approach. I came out of it genuinely convinced that for any project involving technical uncertainty, it is the only reasonable approach.

### 12.4.2 The Gap Between Papers and Working Systems

Reading the MobileBERT paper and seeing hospitality NLU research naturally created an impression that the implementation would be relatively straightforward — the models exist, the frameworks exist, and people have clearly made these things work. The reality was that getting from "these models exist" to "this model runs correctly on this specific budget Android device" involved a very large number of intermediate steps, each of which could fail in its own way.

This is not a criticism of the research I read — the papers report what they are supposed to report. But I now read ML papers with a much more critical eye when it comes to deployment feasibility claims. Benchmark accuracy and on-device accuracy on your specific hardware, with your specific input pipeline, can be meaningfully different. That gap is where most of the engineering work actually lives.

### 12.4.3 Working Across Disciplines

This project sat at the intersection of machine learning, mobile development, backend engineering, real-time systems, and hospitality operations. No single course in my programme covered more than one or two of these areas. Making sensible design decisions required understanding enough about all of them to see how they connected — for example, understanding both why WordPiece tokenisation matters for MobileBERT accuracy and why reimplementing it on Android is genuinely difficult.

That interdisciplinary aspect was one of the most professionally valuable parts of the project. The ability to work across technical domains — not as a deep expert in each, but with enough understanding to make them work together — is something I will carry into my career more than any specific skill learned during the project.

---

## 12.5 Summary

This project challenged me technically, in terms of project management, and in how I think about the gap between research and real-world implementation. The main personal takeaways: test on your actual target hardware early; verify every stage of a model conversion pipeline; start writing before the implementation is finished; and maintain scope discipline even when adding more features feels tempting.

The experience of taking an idea from the literature through design, implementation, evaluation, and write-up to a working prototype has given me genuine confidence to tackle technically ambitious problems. More than any specific skill or result, the lesson I will carry forward is that the most useful decisions in a project like this come from building something real and learning from what breaks — not from getting the design perfect on paper before writing a single line of code.
