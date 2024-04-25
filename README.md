# DL-Final-Project

## AI Diving Judge

**Who:**

- Isabela Antoniuk (iantoniu)
- Rebecca Waterson (rwaterso)
- Zoe Le (zle1)

**Introduction:**
We will implement an AI model that can score dives through video analysis. The problem we are trying to solve is removing bias from judging. There are multiple ways in which bias can present itself in diving, for example, if a diver has a good reputation judges are more likely to score them higher, similarly if they compete for a reputable team. An AI judge would allow for a neutral scoring method that is solely determined by the quality of the dive performed than by outside variables that may affect the score. 

The paper we are implementing seeks to answer the following question: _What score should an athlete receive on his/her dive?_ The answer to this question lies in an accurate quantification of the quality of an action, also known as an AQA score. The AQA score is the way to measure how well an action was performed. Using AQA, the final score is determined by what was done (ex: the dive and degree of difficulty) and how it was done (the quality of execution). We chose this paper because it not only includes an existing model architecture that utilizes AQA but also suggests a hypothesis about how commentating on the action can result in more accurate score predictions. The idea is that the model needs to learn what is both wrong and right with a dive to predict an accurate score. The paper suggests that instead of using a single quality assessment, they propose a multitask learning approach that we believe will strengthen the accuracy of our model and the predicted scores.

_This is a regression problem._

**Related Work:** ["Training AI to Score Olympic Events"](https://medium.com/@logicdevildotcom/training-ai-to-score-olympic-events-649b41574160 )

The 2021 Tokyo Olympics, long-awaited and affected by pandemic postponements, sparked reflection on the subjective nature of judging decisions, leading to questions about automating scoring systems using AI. Action Quality Assessment (AQA) emerged as a solution, aiming to assess action performance impartially, applicable not only in sports but also in healthcare and music. Unlike action recognition, which identifies actions, AQA quantifies their quality, crucial for nuanced evaluations like in diving competitions. Initial studies used human pose information, but challenges in capturing vital visual cues led to exploring 3D convolutional neural networks (C3D) and a multitasking approach. Multitask AQA models, detailed in a research paper and implemented in an AI-based Olympics Judge app for diving, show promise for automated scoring in individual sports, with potential extensions to various other disciplines.

Main article to be reimplemented: ["What and How Well You Performed? A Multitask Learning Approach to
Action Quality Assessment"](https://openaccess.thecvf.com/content_CVPR_2019/papers/Parmar_What_and_How_Well_You_Performed_A_Multitask_Learning_Approach_CVPR_2019_paper.pdf)

**Data:**

We plan to use UNLV’s Dive dataset from ‘P. Parmar, B. T. Morris. Learning To Score Olympic Events. CVPR Workshops, 2017, pp. 20-28’.

If we are able to make our model work with the UNLV data (370 videos of men's individual 10 m platform diving semifinals and finals of the 2012 Olympic Games), we are considering expanding our data to include the diving videos from Brown University’s varsity dive practices. However, since this would include significant preprocessing of data with video editing, we are focusing on making sure our model works on the established UNLV dataset first.

Preprocessing:
- There are 370 samples.
- In downloading the dataset from [UNLV Olympic Scoring Dataset](http://rtis.oit.unlv.edu/datasets.html), there appears to be a folder of original length videos and a folder of the same videos normalized to 151 frames. Because of the normalized video folder we should not have to do much preprocessing with video editing.
- There are files for overall scores and degree of difficulty that correspond to each of the 370 dive videos.

**Methodology:**

Model Architecture:
- Pre-processing data: See above data section*
- We will then use 3D CNNs since they are best at capturing appearance and salient motion patterns making them the best candidates for action recognition.
- Multitask Learning: common network backbone and task-specific heads the common network backbone will learn shared representations and then is further processed through task-specific heads that obtain more task-oriented features and outputs.
- Aggregation: When an athlete gains and loses points this in an addition operation. When good representations are learned the linear combinations are meaningful. Use of averaging as the linear combination.
- C3D-AVG: Up to the average layer is the encoder, encodes input video clips into representations that when averaged will correspond to the total AQA points. The subsequent layers are decoders.
- Task-Specific Heads: For action recognition and AQA tasks. We will use separate heads for each task. For captioning, since it is a sequence-to-sequence task the clip level is input to the captioning branch before averaging.

Training the model:
- Pre-Training: We will pre-train the common network backbone using the action recognition dataset linked in the paper (UCF101)
- Captioning Module: We will utilize a GRU (Oscar or Colab). The maximum caption length will be set to 100 words, with a full vocabulary size of 5779. 
- Networks: All networks will use an Adam Optimizer for 100 epochs with a learning rate of 1e-4.

The most difficult part of this implementation is the overall multi-task learning approach we are trying to accomplish. Additionally, there are multiple losses being calculated that will help achieve the most accurate total loss and score. Essentially what this means is that we will be required to train our model to accomplish multiple tasks at once starting with pre-training our common network backbone to recognize specific actions. This can be more or less difficult depending on the data we use since outside features such as varying camera angles or different pool settings may confuse the model. We intend to use either the dataset which the existing paper used but may encounter difficulty when using video of Brown divers that have been taken from differing camera angles and locations. The target goal is to have the model work the original dataset and the reach goal is having a working model for Brown University divers. 

**Metrics:**

Planned Experiments:
Experimentation will involve training the model on various datasets, testing its performance on unseen data, and potentially fine-tuning hyperparameters.

Relevance of Accuracy:
While accuracy could be considered, metrics like correlation with human judgments, mean squared error, or F1 score may offer deeper insights given the continuous nature of action quality assessment.
We can ask for sample scores of what our dive coach would judge a specific dive video and compare that to the score of our program to determine accuracy and validity (or use the score given in the Olympic dive and compare it to that)

Existing Project Implementation:
Authors' goals: The authors aimed to develop a multitask learning (MTL) approach for action quality assessment (AQA) specifically tailored to diving events.
sought to create a model capable of accurately assessing the quality of diving performances by learning from multiple related tasks simultaneously, including action classification, commentary generation, and scoring.

Quantification of Results: the authors quantified the performance of their model using various metrics such as correlation with human judgments, mean squared error, and classification accuracy.
evaluated the model's ability to correctly classify actions, generate meaningful commentary, and accurately predict scores for diving performances.
the authors likely compared their model's performance against baseline methods and possibly existing state-of-the-art AQA models to demonstrate its superiority or competitive performance.
Base, Target, and Stretch Goals:

- Base: successfully take in one Men’s Olympic diving video and score it accurately
- Target: successfully take in multiple videos from that event
- Stretch: Allow the program to take in any diving videos (like Brown’s diving videos)

**Ethics:**
_What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?_

- The dataset is a collection of 370 dive videos from the 2012 London Olympics. It is only men’s diving events and the videos are from the semifinals and finals. The videos are all taken from the same angle (side) and are all for 10 meter platform individual diving.
- Only 370 videos, which is a relatively small dataset to be training on.
Some concerns with this data include:
- Since these divers are all at a high (Olympic) level, their scores could potentially be higher than what officials might score a college dual meet. We also might not encounter any smacks when training the data.
- Since it is all men’s dives/Olympics, is it possible that we might not have the full range of dives based on degree of difficulty? (eg. skewed more toward higher degree of difficulty, <= 3 from looking over DD data)
- Since the data is only collected from a top-tier event, it may not generalize well to less experienced divers.
- Because the scores were assigned by human judges and are subjective and can be based on the judges experience and potential bias, there can be inconsistencies in the training data.
- Since it is only men and only athletes that made it to semi-finals/finals, as stated before the dataset will be skewed toward certain groups and could lead the AI to perform better on these groups than others.
- Because the dives are all 10m platform dives (which aren’t done in college dual meets), this could affect scoring if a user were to input a 1m or 3m springboard dive video. This would be most significant with scoring the starting position/approach and the take-off between platform and springboard.

_Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?_
Stakeholders include:
- Athletes - Performances are being evaluated which directly affects the the AI’s accuracy. Incorrect scoring could impact their career progression, earnings, and motivation.
- Coaches and teams - Coaches can use this tool to improve training and strategize for competitions. Inaccurate feedback from the AI could lead to inefficient training and poor performance in competitions.
- Judging and regulatory bodies - These groups are responsible for maintaining integrity and fairness in the sport. An AI system that consistently misjudges performances could undermine their efforts and the sport’s credibility.
- Fans and viewing public - Accurate and fair judging contributes to the sport’s reputation and the audience’s enjoyment. Inconsistencies or perceived biases could alienate fans.

**Division of labor: Briefly outline who will be responsible for which part(s) of the project.**

- Bella: Model Building
- Becca: Data/Pre-processing data
- Zoe: Accuracy and interpretability
- All:  Testing & training, paper write-up

