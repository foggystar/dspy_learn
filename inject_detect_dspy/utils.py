import dspy
import pandas as pd
def Init(model,api_base,api_key='',max_tokens=8192,temperature=1.0,hello_test=False,path="output/log"):
    global DETECT_LOG_PATH
    DETECT_LOG_PATH = path
    with open(DETECT_LOG_PATH, "w", encoding='utf-8') as file:
        pass
    lm = dspy.LM(model, api_base=api_base, api_key=api_key,temperature=temperature,max_tokens=max_tokens)
    dspy.configure(lm=lm)
    if hello_test:
        print(lm("Who are you, briefly", temperature=1))

def log(str):
    with open(DETECT_LOG_PATH, "a", encoding='utf-8') as file:
        file.write("--------------------------------------------------------------------\n")
        file.write(f"""{str}""")

def make_allset(dataPath):
    df = pd.read_csv(dataPath)
    trainset = []
    for _, row in df.iterrows():
        trainset.append(dspy.Example(query=row['query'], analysis=row['res'], attack=row['label']).with_inputs("query"))
    return trainset