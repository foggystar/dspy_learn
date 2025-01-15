import dspy # It takes too long to import the package
import os
API_Key = os.getenv('DEEPSEEK_API_KEY')
# deepseek-ai/DeepSeek-V3
lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
dspy.configure(lm=lm)
print(lm("Say this is a test!", temperature=0.7))  # => ['This is a test!']

math = dspy.ChainOfThought("question -> answer: float")
math(question="Two dice are tossed. What is the probability that the sum equals two?")

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