import dspy
from dspy.teleprompt import BootstrapFewShot
import os
import dspy.evaluate
import pandas as pd

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
        self.analyze = dspy.Predict(Analyizer)
        self.judge = dspy.Predict(Classifier)
        self.honeypot = dspy.Predict(Honeypot)

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
    return example.malevolence == pred.malevolence and example.alienation == pred.alienation
    # malevolence = str(example.malevolence).lower()
    # pred_malevolence = str(pred.malevolence).lower()
    # pred_alienation = str(pred.alienation).lower()

    # # 改进：更灵活的标签处理
    # if malevolence == "true":
    #     return pred_malevolence == "true" and pred_alienation == "true"
    # elif malevolence == "false":
    #     return pred_malevolence == "false"
    # else:
    #     return False  # 或者根据你的标签定义进行调整

def evaluate(model, trainset):
    scores = []
    for x in trainset:
        pred = model(query=x.query) # 假设你的 injectionJudge.forward 接收 'query'
        score = metric(x, pred)
        scores.append(score)
    return scores

def make_trainset(dataPath):
    df = pd.read_csv(dataPath)
    trainset = []
    for _, row in df.iterrows():
        trainset.append(dspy.Example(query=row['query'], analysis=row['res'], malevolence=row['malevolence'],alienation=row['alienation']).with_inputs("query"))
    return trainset

if __name__ == "__main__":
    Init()
    trainset = make_trainset("./new_data.csv")

    with open("log.txt", "w", encoding='utf-8') as file:
        pass  # 你可以在这里写一些初始化信息到 output 文件

    # 初始化你的模型
    student = injectionJudge()

    # 使用 BootstrapFewShot 进行 few-shot learning
    optimizer = BootstrapFewShot(metric=metric, max_rounds=3)
    compiled_student = optimizer.compile(student=student, trainset=trainset[:5])
    assert compiled_student is not None, "Failed to compile student"
    print("Compiled model successfully")

    # 评估模型
    print("Evaluating model")
    evaluation_set = trainset[:]
    pre_score = evaluate(student, evaluation_set)
    scores = evaluate(compiled_student, evaluation_set)
    print(f"Initial score: {pre_score}, Evaluation scores: {scores}")
    evaluation_score = sum(scores) / len(scores) if scores else 0
    initial_score = sum(pre_score) / len(pre_score) if pre_score else 0
    print(f"Initial total: {initial_score}, Evaluation total: {evaluation_score}")

    # 保存最佳模型
    compiled_student.save("./dspy_program/", save_program=True)
    
    # 加载模型
    loaded_student = dspy.load("./dspy_program/")
    # 现在 loaded_student 就是你之前训练好的模型
    # 你可以像调用普通 dspy.Module 一样使用它进行预测

    test_query = "用户想要重置密码"
    prediction = loaded_student(query=test_query)
    print(f"预测结果: {prediction}")