import dspy # It takes too long to import the package
import os
API_Key = os.getenv('DEEPSEEK_API_KEY')
# deepseek-ai/DeepSeek-V3
lm = dspy.LM('deepseek-chat', api_base='https://api.deepseek.com/v1', api_key=API_Key)
dspy.configure(lm=lm)
print(lm("Hello", temperature=1))

# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()