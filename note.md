# dspy——LLM开发框架教程
![version](https://img.shields.io/badge/version-1.0.0-blue.svg) ![release date](https://img.shields.io/badge/release-2025--01-green.svg) ![author](https://img.shields.io/badge/author-FoggyStar-orange.svg)

dspy——自动化prompt优化框架

这篇文档仅仅是一个笔记、简介，详细内容、系统性学习请访问 [DSPy官网](https://dspy.ai/)

[toc]

## 为何使用DSPy？与其他框架的对比

**DSPy** 的理念和抽象与其他库和框架有很大不同，因此通常很容易判断 **DSPy** 是否适合您的用例。如果您是 NLP/AI 研究人员，答案通常是肯定的。如果您是从事其他工作的从业者，请继续阅读。

**DSPy 与提示的简单封装（OpenAI API、MiniChain、基本模板）的对比** 换句话说：为什么我不直接使用字符串提示词？ 对于非常简单的设置，这可能效果很好。（如果您熟悉神经网络，这就像用 Python for 循环来表示一个小的两层神经网络。）但是，当您需要更高的质量（或可控的成本）时，您需要迭代地探索多阶段分解、改进的提示、数据引导、仔细的微调、检索增强和/或使用更小（或更便宜，或本地）的模型。使用基础模块进行构建的真正强大之处在于这些部分的互动。但是，每次更改一个部分时，您可能会破坏（或削弱）多个其他组件。**DSPy** 清晰地抽象出（并强大地优化）这些交互中与实际系统设计无关的部分。它让您专注于设计模块级别的交互：用 10 或 20 行 **DSPy** 代码编写的代码可以轻松编译为 `GPT-4` 的多阶段指令、 `Llama2-13b` 的详细提示或 `T5-base` 的微调。您不再需要维护项目核心中那些长而脆弱的、特定于模型的字符串了。

**DSPy 与像 LangChain、LlamaIndex 这样的应用程序开发库的对比** LangChain 和 LlamaIndex 面向高级应用程序开发；它们提供了 开箱即用、预构建的应用程序模块，可以与您的数据或配置相结合。如果您乐于使用通用的、现成的prompt来对 PDF 进行问答或标准文本到 SQL，您会在这些库中得到充足的资源。**DSPy** 内部不包含针对特定应用程序的手工提示。相反，**DSPy** 引入了一小组功能更强大、用途更广泛的模块，它们可以基于您的数据学习提示（prompt）或微调您的 LM。当您更改数据、调整程序的控制流或更改目标 LM 时，**DSPy 编译器**可以将您的程序映射到一组新的提示或微调，这些提示是专门为当前流程优化的。因此，您可能会发现 **DSPy** 以最少的努力获得最高的任务质量。简而言之，**DSPy** 适用于您需要轻量级但自动优化提示词的情况，而非预定义提示和集成的库。如果您熟悉神经网络：这就像 PyTorch（即表示 **DSPy**）和 HuggingFace Transformers（即表示更高级的库）之间的区别。

**DSPy 与像 Guidance、LMQL、RELM、Outlines 这样的生成控制库的对比** 这些都是用于控制 LM 单个补全（individual completions）的新兴库，例如，如果您想强制以 JSON 格式输出，或以正则表达式限制采样。这在许多情况中很有用，但它通常侧重于对单个 LM 调用的底层、结构化控制。它不能确保您获得的 JSON（或结构化输出）对是正确或有用的。相比之下，**DSPy** 生成的prompt可以满足各种任务需求，其中还可包括结构化输出。也就是说， **DSPy** 中的 **Signatures** 可实现类正则表达式的约束。

**如何使用 DSPy？** 使用dspy本质上是一个迭代过程。首先，需要明确、定义好你要完成的任务，以及需要优化的指标（metrics）（初学时，若不在练习项目中使用提示词优化，可暂时忽略这部分），并准备一些样例数据，可以仅包含指标中要求的标签。之后，使用模块（Modules）构建程序，给予每一个模块一个签名（Signature），用于规定模块的输入、输出，之后便可调用LM运行程序。最终，使用优化器（optimizer）将代码编译为高质量的指令、自动的少样本示例或为您的 LM 更新的 LM 权重。


## 主要功能：
### 零、初始化

```python
import dspy
import os
API_Key = os.getenv('DEEPSEEK_API_KEY') #写在系统变量中
lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
dspy.configure(lm=lm)
```
### 一、使用模块指导LM完成任务 
#### 签名 Signature [源网站](https://dspy.ai/learn/programming/signatures/)
dspy包中有多个LM控制接口，最重要的传入参数为字符串，**Signature**，内涵关键字：**提供给LM的知识，要回答的问题，这个步骤要达成的目标等**。**Signature**内容会交由LM解析翻译为完整prompt，因此其仅需包含几个关键词，并使用"->"等符号指示逻辑关系，关键词意思正确即可，无需是特定词语

样例：
1. 问答: `"question -> answer"`, 等同于 `"question: str -> answer: str"`，因为默认类型是**str**
2. 语气分类: `"sentence -> sentiment: bool"`, 若语气积极则输出**True**
3. 总结: `"document -> summary"`

DSPy 提供两种定义`Signature`的方法：行内签名和类定义签名

##### 行内（简单）
直接将`Signature`作为`Modules`的参数

```python
math = dspy.ChainOfThought("question -> result")
math(question="Two dice are tossed. What is the probability that the sum equals two?")
```

##### 类定义（复杂）

此时使用自己定义的类作为`Modules`参数，需要包含以下内容：
- 输入域 **dspy.InputField**
- 输出域 **dspy.OutputField**
- 用于明确内容、要求的语句 **docstring**，以`desc="Requires"`格式作为输入、输出域的参数
```python
class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

module = dspy.Predict(ExtractInfo)

text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = module(text=text)

print(response.title)
print(response.headings)
print(response.entities)
```

通过这样的模块化、标准化编程，在LM得出的最终结果出现错误时，易于排查错误位置，节省API资源

#### 模块 [源网站](https://dspy.ai/learn/programming/modules)
DSPy提供了一些模块，其会根据`Signature`生成提示词并调用大模型，给出结果

这些模块都是基于**dspy.Predict**的，其类似于问答模型的根本功能——文章续写，或者说接龙
```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')
# 2) Call with input argument(s). 
response = classify(sentence=sentence)
# 3) Access the output.
print(response.sentiment)
```
使用其他合适的模块可以提高回答质量
```python
question = "What's something great about the ColBERT retrieval model?"
# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)
# 2) Call with input argument.
response = classify(question=question)
# 3) Access the outputs.
response.completions.answer
```
一部分其他`Modules`的介绍：
它们区别不大，`Signature -> Prompt` 的过程实现会有一定区别
1. **`dspy.Predict`**: 基本预测器。不修改`Signature`。处理学习的主要形式（即存储指令、演示和更新 LM）。

2. **`dspy.ChainOfThought`**: 让 LM 在承诺签名回应之前逐步思考。

3. **`dspy.ProgramOfThought`**: 让 LM 输出代码，代码的执行结果将决定响应。

4. **`dspy.ReAct`**: 可以使用工具执行给定函数。

5. **`dspy.MultiChainComparison`**: 可以比较来自**CoT**的多个输出，以得出最终预测结果。

#### Extra 关于缓存
缓存是DSPy调用LM后生成的结果等文件，能够在调试过程中避免重复调用LM，但偶尔会因检测错误，导致被更改的程序没有重新调用LM，致使结果没有更新
**如何关闭缓存？如何导出缓存？**

从 v2.5 开始，您可以通过在 `dspy.LM` 中将 `cache` 参数设置为 `False` 来关闭缓存：

```python
dspy.LM('openai/gpt-4o-mini',  cache=False)
```

您的本地缓存将保存到全局 env 目录 `os.environ["DSP_CACHEDIR"]` 或笔记本的 `os.environ["DSP_NOTEBOOK_CACHEDIR"]`。您通常可以将缓存目录设置为 `os.path.join(repo_path, 'cache')` 并从此处导出此缓存：
```python
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')
```


------------------------------



### 二、评估模型
在使用`Signature`和`Modules`构建好程序后，可基于自己准备的数据，构建`Example`进行测试，并设定评估标准（Metrics）用以自动评价LM程序的表现
#### 构建训练数据
DSPy 中数据的核心数据类型是 `Example`，表示训练集和测试集中的项目。

DSPy `Examples` 类似于 Python 的 `dicts`，但多一些类函数。DSPy 模块返回 `Prediction` 类型的值，它是 `Example` 的一个特殊子类。

使用 DSPy 时，会进行大量的评估和优化。每条数据都是是 `Example` 类型
##### 构造**Examples**对象
```python
# Code
qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
```bash
# Output
Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
```

##### 标记输入与标签
```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs() # 输入
non_input_key_only = article_summary.labels() # 标签

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
```
##### 构造训练集
```python
trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
```

##### 其余加载方法见附录链接

#### 构造评估标准
`Metric`只是一个函数，它可以从数据和系统输出中提取示例，并返回一个量化输出好坏的分数。

对于简单的任务，这可能只是 “准确率 ”或 “精确匹配 ”或 “F1 分数”。`Metric`可以返回 `bool`、`int` 和 `float` 类型的值。对于简单的分类或其他简短任务任务来说，这也许就够用。

但是，对于大多数应用而言，程序将输出较长内容。在这种情况下，`Metric`应该是一个较小的 DSPy 程序，用于检查输出的多个属性（使用 LM 审查）。

虽然`Metric`不可能一开始就很合适，但您应该从简单的开始，然后不断改进。
##### 简易度量函数，定义此形式的函数从而使用**dspy**内置训练函数
```python
def validate_answer(example, pred, trace=None):
return example.answer.lower() == pred.answer.lower()
```
##### 可以使用内置度量函数
```python
dspy.evaluate.metrics.answer_exact_match
dspy.evaluate.metrics.answer_passage_match
```
##### 度量函数可以更加复杂，例如检查多个属性。若 trace 为 None（即用于评估或优化），度量函数返回 float，否则将返回 bool（即用于引导演示）
```python
def validate_context_and_answer(example, pred, trace=None):
# 将真实答案和预测答案转换为小写后进行比较，确保大小写不影响匹配结果。
answer_match = example.answer.lower() == pred.answer.lower()

# 检查预测答案是否出现在提供的上下文列表中的任意一个上下文中。
context_match = any((pred.answer.lower() in c) for c in pred.context)

if trace is None: # 评估或优化模式，一个分数，表示预测的质量
    return (answer_match + context_match) / 2.0
else: # 自举模式，用于筛选高质量的示例用于训练
    return answer_match and context_match
```
##### 使用数据集
```python
# 构造训练集
def make_trainset(dataPath):
    df = pd.read_csv(dataPath)
    trainset = []
    for index,row in df.iterrows():
        trainset.append(dspy.Example(query=row['query'],analysis=row['res'] ,label=row['label']).with_inputs("query"))
    return trainset

可以编写一个函数用于评价程序表现
# 评估函数
def evaluate(trainset):
scores = []
for x in trainset:
    pred = run_model(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
return scores
```

也可以调用内置评估函数
```python
evaluator = dspy.evaluate.Evaluate(devset=evaluation_set, num_threads=3, display_progress=True, return_outputs=True)
evaluator(student, metric=metric)
```
##### 使用**AI**进行度量
```python
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

def metric(gold, pred, trace=None):
# 1. 提取问题、答案和生成的推文
question, answer, tweet = gold.question, gold.answer, pred.output

# 2. 定义两个评估标准的问题
engaging = "Does the assessed text make for a self-contained, engaging tweet?"
correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

# 3. 使用 dspy.Predict(Assess) 评估推文
correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

# 4. 提取评估结果
correct, engaging = [m.assessment_answer for m in [correct, engaging]]

# 5. 计算最终分数
# 如果推文正确且长度在280字符以内，分数为correct和engaging的和
# 否则分数为0
score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

# 6. 根据trace参数返回不同格式的结果
if trace is not None: return score >= 2  # 返回布尔值
return score / 2.0  # 返回0-1之间的分数
```

##### 使用并行提高

##### 使用 trace 参数
在评估运行期间使用您的`Metric`时，DSPy 不会尝试跟踪您的程序步骤。

但在编译（优化）过程中，DSPy 会跟踪您的 LM 调用。跟踪将包含每个 DSPy 预测器的输入/输出，您可以利用它来验证优化的中间步骤。
```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```
（这个我没看懂，故仅翻译后置于此处）

#### 自动优化提示词

##### 简介
有了LM程序和`Metric`，就可以使用 DSPy 优化器`Optimizer`来调整程序中的提示或权重。你需要构建训练集与验证集。对于训练集， 30条数据就会有不错的效果，但至少 300 个示例才算是完备。有些`Optimizer`只接受训练集。其他则要求提供训练集和验证集。对于优化器，建议将 20% 用于训练，80% 用于验证，这与 DNN 的一般做法相反。

经过前几次优化后，你要么很满意，要么取得了一些进展，但不满意。此时，请回到 DSPy 编程，重新审视主要问题。你是否很好地定义了你的任务？是否需要针对问题收集（或在线查找）更多数据？是否需要更新`Metric`？你想使用更复杂的`Optimizer`吗？您是否需要考虑像 DSPy 断言这样的高级功能？您想在 DSPy 程序中增加更多复杂性或步骤吗？您是否想按顺序使用多个`Optimizer`？

关键在于迭代开发。DSPy 为您提供了渐进式开发的工具：对数据、程序结构、断言、度量和优化步骤进行迭代。优化复杂的 LM 程序是一种全新的模式。

##### DSPy 优化器工作原理
DSPy中的不同优化器将通过以下方式调整程序的质量：为每个模块合成优秀的少量示例（如dspy.BootstrapRS1）；为每个提示符提出并智能地探索更好的自然语言指令（如dspy.MIPROv2）；为你的模块建立数据集，并利用它们来微调系统中的LM权重（如dspy.BootstrapFinetune）。

以 dspy.MIPROv2 优化器为例。首先，MIPRO 从引导阶段开始。此时程序尚未优化，它会在不同的输入条件下运行多次，收集每个`Module`的输入/输出行为轨迹`trace`。MIPRO 会对这些`trace`进行过滤，只保留那些在`Metric`中得分较高的`trace`。之后，MIPRO 会预览 DSPy 程序的代码、数据和`trace`，并利用它们为程序中的每个提示起草许多潜在指令`proposal`。第三，MIPRO 启动离散搜索阶段。它从训练集中抽取小批量样本，提出用于构建程序中每个提示的指令和`trace`组合，并在小批量上对候选程序进行评估。MIPRO 利用得出的分数更新起草模型，从而优化`proposal`

`Optimizer`之在于它们可以组合使用。你可以运行 dspy.MIPROv2，并将生成的程序作为 dspy.MIPROv2 的输入，或者作为 dspy.BootstrapFinetune 的输入，以获得更好的结果。这也是 dspy.BetterTogether 的部分精髓所在。或者，你也可以运行`Optimizer`，然后提取前 5 个候选程序，并将它们构建成一个 dspy.Ensemble 模型。以高度系统化的方式预估推理时间以及 DSPy 的预推理时间（即优化预算）。


##### 如何使用 DSPy 优化器
`Optimizer`是一种调整 DSPy 程序的参数（即提示和/或 LM 权重）的算法，以最大限度地提高`Metric`。
典型的 `Optimizer`需要三样东西：
- 你的 DSPy 程序。这可能是一个单一模块（如 dspy.Predict），也可能是一个复杂的多模块程序。
- 你的`Metric`。这是一个函数，用于评估程序的输出，并给程序打分（分数越高越好）。
- 少量训练输入。可以很少——只有 5 或 10 个示例；也可以不完整——只有输入，没有任何标签。

##### 目前有哪些 DSPy 优化器？

可以通过 `from dspy.teleprompt import *` 访问优化器。

##### 自动少量样本学习

这些优化器通过自动生成并在发送给模型的提示中包含**优化的**示例来扩展签名，从而实现少量学习。

1. `LabeledFewShot`： 从提供的带有`label`的数据点构建少量训练数据。 需要 `k`（训练数据）和 `trainset` 作为备选集。

2. `BootstrapFewShot`： 使用 `teacher` 模块（默认为您的程序）为您的程序的每个阶段生成完整的演示，并在 `trainset` 中生成带标签的示例。参数包括 `max_labeled_demos`（从 `trainset`中随机选取的示例数量）和 `max_bootstrapped_demos`（由 `teacher` 生成的额外示例数量）。引导过程会使用`Metric`来验证演示，只有通过`Metric`才会出现在结果中。高级：支持使用兼容结构的另一个 DSPy 程序作为 `teacher`，以完成更难的任务。

3. [`BootstrapFewShotWithRandomSearch`](https://dspy.ai/deep-dive/optimizers/bootstrap-fewshot)： 在生成的演示程序中多次应用 `BootstrapFewShot` 并进行随机搜索，在优化后选出最佳程序。参数与 `BootstrapFewShot` 相同，但增加了 `num_candidate_programs` 参数，用于指定在优化过程中评估的随机程序数量，包括未优化程序、`LabeledFewShot` 优化后的程序、`BootstrapFewShot` 优化后的程序（包含未打乱示例）以及`BootstrapFewShot` 编译后的程序（包含随机示例集）的 `num_candidate_programs` 参数。

4. `KNNFewShot`. 使用 k 近邻算法为给定输入示例找到最近的训练示例。然后，这些近邻演示被用作 `BootstrapFewShot` 优化过程的训练集。有关示例，请参阅 [this notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/outdated_v2.4_examples/knn.ipynb) 可能因为版本较早无法在`dspy>=2.5`使用


###### 自动指令优化

这些优化器能为提示生成最佳指令，在 MIPROv2 中还能优化 few-shot 示例集。

5. [`COPRO`](https://dspy.ai/deep-dive/optimizers/copro)： 为每一步生成和改进新指令，并通过坐标上升（使用`Metric`和 `trainset`）对其进行优化。参数包括`depth`，即`Optimizer`运行的提示改进迭代次数。

6. [`MIPROv2`](https://dspy.ai/deep-dive/optimizers/miprov2)： 在每一步中生成指令和少量示例。指令生成是数据感知和演示感知的 *编者：？*。使用贝叶斯优化（Bayesian Optimization）技术，有效地在`Modules`中搜索指令生成空间/演示空间。


###### 自动微调

该优化器用于微调底层 LLM（fine tune）

7. `BootstrapFinetune`： 将基于提示的 DSPy 程序精简为权重更新。输出的 DSPy 程序具有相同的步骤，但每一步都是由微调的模型，而非直接接受`prompt` LLM 执行的。


###### 程序转换

8. `Ensemble` 集合一组 DSPy 程序，之后或使用整组程序，或随机抽样一个子集到单个程序中。


##### 如何选取`Optimizer`

找到正确的`Optimizer`和最佳配置需要不断尝试。DSPy 的是一个迭代过程--要想在任务中获得最佳性能，您需要不断探索和迭代。 

尽管如此，以下是入门指南：

- 如果您的示例很少（约10个），请从 `BootstrapFewShot`开始。
- 如果你有多的数据（50 个或更多），请尝试`BootstrapFewShotWithRandomSearch`。
- 如果您只想进行**`prompt`优化**（0-shot learning，即零样本学习），请使用 `MIPROv2` [配置为 0-shot optimization 以进行优化](https://dspy.ai/deep-dive/optimizers/miprov2#optimizing-instructions-only-with-miprov2-0-shot)。
- 如果你愿意使用更多API调用来执行**长的优化运行**（例如 40 次尝试或更多），并且有足够的数据（例如 200 个或更多示例以防止过度拟合），那么不妨试试 `MIPROv2`。
- 如果你能够使用一个 LLM（7B 参数或以上），并且需要一个非常高效的程序，可以使用 `BootstrapFinetune` 为你的任务微调一个小型 LM。

##### 优化样例
这是一个最小但完全可运行的示例，我们设置了一个 `dspy.ChainOfThought` 模块，将短文归类为 77 个银行标签之一，然后使用 `dspy.BootstrapFinetune` 和来自 Banking77 的 2000 个文本标签对来微调 GPT-4o-mini 的权重。我们使用了变体 `dspy.ChainOfThoughtWithHint`，它在引导时接受可选的提示，以最大限度地利用训练数据。当然，测试时不会有提示。
```python
import dspy
dspy.configure(lm=dspy.LM('gpt-4o-mini-2024-07-18'))

# Define the DSPy module for classification. It will use the hint at training time, if available.
signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
classify = dspy.ChainOfThoughtWithHint(signature)

# Optimize via BootstrapFinetune.
optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
optimized = optimizer.compile(classify, trainset=trainset)

optimized(text="What does a pending cash withdrawal mean?")
```

##### 更多使用示例请见 [Cheat Sheet](https://dspy.ai/cheatsheet)

##### 保存优化好的程序

```python
# 保存最佳模型
compiled_student.save("./dspy_program/", save_program=True)
# 加载模型
loaded_student = dspy.load("./dspy_program/")
```
此方案仅在`dspy>=2.6.0`时可用，使用简易。

而先前版本的`save`函数无法保存使用类定义的结构，重新使用时需要包含程序原有结构，仅能节省评估、训练的代码，详见附录链接

---

## 引用
[Cheat Sheet](https://dspy.ai/cheatsheet)
[官方教程](https://dspy.ai/learn/)
[常见问题](https://dspy.ai/faqs/) 内部的大多数链接均失效
[API文档](https://dspy.ai/api/)
[参考文档](https://dspy.ai/tutorials/)

## 常见问题解答

**DSPy 优化器调整什么？** 或者，_编译实际上做了什么？_ 每个优化器都不同，但它们都试图通过更新提示或 LM 权重来最大化程序上的指标。当前的 DSPy `optimizers` 可以检查您的数据，模拟通过您的程序的轨迹来生成每个步骤的好/坏示例，根据过去的结果为每个步骤提出或完善指令，对自我生成的示例微调您的 LM 的权重，或结合其中几个步骤来提高质量或降低成本。我们很乐意合并新的优化器来探索更丰富的空间：您目前为提示工程、“合成数据”生成或自我改进所经历的大多数手动步骤都可以概括为作用于任意 LM 程序的 DSPy 优化器。

其他常见问题解答。我们欢迎 PR 为此处的每个问题添加正式答案。您将在现有问题、教程或所有或大部分这些问题的论文中找到答案。

### 高级用法

- **如何并行化？**
您可以通过在各自的 DSPy `optimizers` 中或在 `dspy.Evaluate` 实用函数中指定多个线程设置，在编译和评估期间并行化 DSPy 程序。

- **如何冻结模块？**

可以通过将其 `._compiled` 属性设置为 True 来冻结模块，表明该模块已通过优化器编译，不应调整其参数。这在 `dspy.BootstrapFewShot` 等优化器中内部处理，其中确保在教师在引导过程中传播收集的少样本演示之前冻结学生程序。

- **如何使用 DSPy 断言？**

    a) **如何向您的程序添加断言**：
    - **定义约束**：使用 `dspy.Assert` 和/或 `dspy.Suggest` 在您的 DSPy 程序中定义约束。这些约束基于对您想要强制执行的结果的布尔值验证检查，这些检查可以简单地是 Python 函数来验证模型输出。
    - **集成断言**：保持您的断言语句在模型生成之后（提示：在模块层之后）

    b) **如何激活断言**：
    1. **使用 `assert_transform_module`**：
        - 使用 `assert_transform_module` 函数以及 `backtrack_handler` 包装带有断言的 DSPy 模块。此函数转换您的程序以包括内部断言回溯和重试逻辑，这些逻辑也可以自定义：
        `program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)`
    2. **激活断言**：
        - 直接对带有断言的 DSPy 程序调用 `activate_assertions`： `program_with_assertions = ProgramWithAssertions().activate_assertions()`

    **注意**：要正确使用断言，您必须 **激活** 一个包含来自上述任一方法的 `dspy.Assert` 或 `dspy.Suggest` 语句的 DSPy 程序。

### 错误

- **如何处理 “上下文过长” 错误？**

如果您在 DSPy 中遇到 “上下文过长” 错误，则您可能正在使用 DSPy 优化器在提示中包含演示，而这超出了您当前的上下文窗口。尝试减少这些参数（例如 `max_bootstrapped_demos` 和 `max_labeled_demos`）。此外，您还可以减少检索到的段落/文档/嵌入的数量，以确保您的提示适合您的模型上下文长度。

一个更通用的修复方法是简单地增加为 LM 请求指定的 `max_tokens` 的数量（例如 `lm = dspy.OpenAI(model = ..., max_tokens = ...)`）。