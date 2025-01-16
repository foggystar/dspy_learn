import dspy # It takes too long to import the package
import os
import dspy.evaluate
import pandas as pd
def Init():
    API_Key = os.getenv('DEEPSEEK_API_KEY')
    # deepseek-ai/DeepSeek-V3
    lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
    dspy.configure(lm=lm)
    print(lm("Hello", temperature=1))        

class Analyizer(dspy.Signature):
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    intention: str = dspy.OutputField(desc="The user's intention.")
class Honeypot(dspy.Signature):
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    response: float = dspy.OutputField()
class Classifier(dspy.Signature):
    """Judge whether the user is attacking a LLM."""
    query: str = dspy.InputField(desc="The user's input to a LLM.")
    intention: str = dspy.InputField(desc="The user's intention.")
    response: str = dspy.InputField(desc="The LLM's response.")

    analysis: str = dspy.OutputField(desc="How do you judge the malevolence")
    malevolence: bool = dspy.OutputField(desc="Judge whether the user is attacking a LLM.")
    # confidence: float = dspy.OutputField()

def run_model(sentence):
    analyze=dspy.Predict(Analyizer)
    judge = dspy.Predict(Classifier)
    honeypot = dspy.Predict(Honeypot)
    intention=analyze(query=sentence)
    response=honeypot(query=sentence)
    res=judge(query=sentence,intention=intention,response=response)
    return res

def validate_answer(example, pred, trace=None):
    print(example, pred)
    return 1 #example.label.lower() == pred.malevolence.lower()

def evaluate(qa_pair):
    scores = []
    for x in qa_pair.inputs():
        print(x)
        # pred = run_model(**x.inputs())
        # score = validate_answer(x, pred)
        # scores.append(score)

if __name__ == "__main__":
    Init()
    df=pd.read_csv("./new_data.csv")
    # print(df.loc[:,'query'],df.loc[:,'res'],df.loc[:,'label'])
    print()
    qa_pair=dspy.Example(query=df.loc[:,'query'],analysis=df.loc[:,'res'] ,label=df.loc[:,'label']).with_inputs("query")
    print(**qa_pair.inputs())
    # evaluate(qa_pair)
    

    