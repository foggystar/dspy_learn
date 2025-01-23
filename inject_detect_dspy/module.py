import dspy
from .signature import Analyizer, Classifier, Honeypot
from .utils import log
class injectionJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(Analyizer)
        self.judge = dspy.ChainOfThought(Classifier)
        self.honeypot = dspy.ChainOfThought(Honeypot)

    def forward(self, query):
        try:
            intention = self.analyze(query=query).intention
            response = self.honeypot(query=query).response
            res = self.judge(query=query, intention=intention, response=response)
            log(f"""{[query, intention, response, res]}""")
            return res
        except Exception as e:
            log(f"###### Error: {e}, Query: {query}")
            res = dspy.Prediction(analysis="Error", guilty=True, malevolence=True, alienation=True, larcenous=True)
            return res