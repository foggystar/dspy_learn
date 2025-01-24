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
        dspy.Suggest(
                len(query) <= 100,
                "Query should be short and less than 100 characters",
                target_module=self.generate_query
            )

        try:
            intention = self.analyze(query=f"""'''{query}'''""")
            dspy.Suggest(
                hasattr(intention, 'intention'),
                "Failed to find intention",
                target_module=self.analyze
            )
            intention = intention.intention

            response = self.honeypot(query=f"""'''{query}'''""")
            dspy.Suggest(
                hasattr(response, 'response'),
                "Failed to find response",
                target_module=self.honeypot
            )
            res = self.judge(query=f"""'''{query}'''""", intention=intention, response=response)
            dspy.Suggest(
                hasattr(res, 'analysis'),
                "Failed to find response",
                target_module=self.judge
            )
            log(f"""{[query, intention, response, res]}""")
            return res
        except Exception as e:
            log(f"###### Error: {e}, Query: {query}")
            res = dspy.Prediction(analysis="Error", sinful=True, malevolence=True, alienation=True, larcenous=True)
            return res