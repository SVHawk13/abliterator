from jinja2 import Template

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
PHI3_CHAT_TEMPLATE = """<|user|>\n{instruction}<|end|>\n<|assistant|>"""

class ChatTemplate:
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def format(self, instruction: str) -> str:
        return self.template.format(instruction=instruction)

    def __enter__(self):
        self.prev = self.model.chat_template
        self.model.chat_template = self
        return self

    def __exit__(self, exc, exc_value, exc_tb):
        self.model.chat_template = self.prev
        del self.prev


class TokenizerChatTemplate(ChatTemplate):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._template = Template(tokenizer.get_chat_template())

    def format(self, instruction: str) -> str:
        messages = [{"role": "user", "content": instruction}]
        return self._template.render(messages=messages)
