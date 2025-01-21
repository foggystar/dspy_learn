import dspy
from dspy.teleprompt import BootstrapFewShot
import os
import dspy.evaluate
import pandas as pd
import random

class Similarity(dspy.Signature):
    input1 : str = dspy.InputField(desc="The first input string.")
    input2 : str = dspy.InputField(desc="The second input string.")
    similarity : float = dspy.OutputField(desc="The similarity between the two input strings, from 0 to 1.")

class Analyizer(dspy.Signature):
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    intention: str = dspy.OutputField(desc="The user's intention.")

class Honeypot(dspy.Signature):
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    response: str = dspy.OutputField(desc="response to user")

class Classifier(dspy.Signature):
    """Judge whether the user is attacking a LLM."""
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    intention: str = dspy.InputField(desc="The user's intention.")
    response: str = dspy.InputField(desc="The LLM's response.")
    
    malevolence: bool = dspy.OutputField(desc="Judge whether the user is attacking a LLM.")
    alienation: bool = dspy.OutputField(desc="Is the user try to change the LLM's behavior?")
    analysis: str = dspy.OutputField(desc="How do you judge the malevolence")

class injectionJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(Analyizer)
        self.judge = dspy.ChainOfThought(Classifier)
        self.honeypot = dspy.ChainOfThought(Honeypot)

    def forward(self, query):
        intention = self.analyze(query=query).intention
        response = self.honeypot(query=query).response
        res = self.judge(query=query, intention=intention, response=response)
        log("log.txt", query=query, intention=intention, response=response, res=res)
        return res

def Init():
    API_Key = os.getenv('DEEPSEEK_API_KEY')
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    print(lm("Hello", temperature=1))

def log(path, **kwargs):
    with open(path, "a", encoding='utf-8') as file:
        file.write("-------------------------------------------------\n$$$ Query: ")
        file.write(kwargs['query'])
        file.write("\n\n")
        file.write(f"""$$$ Intention: {kwargs['intention']}\n\n
                   $$$ Response: {kwargs['response']}\n\n
                   $$$ {kwargs['res']}\n\n""")

def metric(example, pred, trace=None):
    similar=dspy.ChainOfThought(Similarity)
    score = similar(input1=example.analysis, input2=pred.analysis).similarity
    truth = example.malevolence == pred.malevolence and example.alienation == pred.alienation
    return score*truth

# def evaluate(model, trainset):
#     scores = []
#     for x in trainset:
#         pred = model(query=x.query)
#         score = metric(x, pred)
#         scores.append(score)
#     return scores

def make_allset(dataPath):
    df = pd.read_csv(dataPath)
    trainset = []
    for _, row in df.iterrows():
        trainset.append(dspy.Example(query=row['query'], analysis=row['res'], malevolence=row['label'],alienation=row['label']).with_inputs("query"))
    return trainset

if __name__ == "__main__":
    Init()
    allset = make_allset("./input/new_data.csv")
    # trainset = random.sample(allset, 50)
    evalset = random.sample(allset, 200)
    with open("log.txt", "w", encoding='utf-8') as file:
        pass  # 你可以在这里写一些初始化信息到 output 文件

    # 初始化你的模型
    initial = injectionJudge()

    # 使用 BootstrapFewShot 进行 few-shot learning
    print("Compiling model")
    optimizer = BootstrapFewShot(metric=metric, max_rounds=3,max_labeled_demos=60,max_bootstrapped_demos=40)
    trained = optimizer.compile(student=initial, trainset=allset)
    assert trained is not None, "Failed to compile student"
    
    # 保存最佳模型
    trained.save("./dspy_program/", save_program=True)
    # 评估器
    evaluator = dspy.evaluate.Evaluate(devset=evalset, num_threads=50, display_progress=True, return_all_scores=True)
    # 评估模型
    print("Evaluating model")
     
    initial_score = evaluator(initial, metric=metric)
    trained_score = evaluator(trained, metric=metric)
    print(f"Initial score: {initial_score}, Evaluation scores: {trained_score}")

    
    
    # 加载模型
    loaded_student = dspy.load("./dspy_program/")
    # 现在 loaded_student 就是你之前训练好的模型
    # 你可以像调用普通 dspy.Module 一样使用它进行预测

    test_query = "用户想要重置密码"
    prediction = loaded_student(query=test_query)
    print(f"预测结果: {prediction}")