import dspy
import os
import pandas as pd
def Init():
    API_Key = os.getenv('DEEPSEEK_API_KEY')
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    print(lm("Hello", temperature=1))

def metric(example, pred, trace=None):
    if example.label and not (pred.alienation or pred.malevolence):
        return 1
    elif example.label and (pred.alienation or pred.malevolence):
        return 1
    return 0

if __name__ == "__main__":
    Init()
    # 加载模型
    judge = dspy.load("./dspy_program/")
    # 现在 loaded_student 就是你之前训练好的模型
    # 你可以像调用普通 dspy.Module 一样使用它进行预测
    df = pd.read_csv("input/new_data_1_20.csv")
    questions = df['query']
    labels = df['label'].map({'white': False, 'black': True})
    evalset = []
    for _, row in df.iterrows():
        evalset.append(dspy.Example(query=row['query'], label=row['label']).with_inputs("query"))
    
    evaluator = dspy.evaluate.Evaluate(devset=evalset, num_threads=50, display_progress=True, return_outputs=True,display_table=5)
    score = evaluator(judge, metric=metric)

    # pd.DataFrame(score[]).to_csv("result.csv",index=False)
    analysis = [pred[1].reasoning for pred in score[1]]
    judge = [(pred[1].malevolence or pred[1].alienation) for pred in score[1]]

    correct = labels == judge

    res=pd.DataFrame({'question':questions,'analysis':analysis,'label':labels,'judge':judge,'correct':correct})
    res.to_csv("output/result.csv",index=False)
    res.to_excel("output/result.xlsx",index=False)