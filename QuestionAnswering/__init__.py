import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from groq import Groq

class QuestionAnswering:
    def __init__(self, api_key, model_name):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def answer(self, question, documents):
        context = "\n".join(documents)
        input_text = f"""Question: {question}
        Context: {context}
        Answer:"""

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý AI..."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": input_text}
            ],
            temperature=1,
            max_tokens=1024
        )
        return completion.choices[0].message.content
