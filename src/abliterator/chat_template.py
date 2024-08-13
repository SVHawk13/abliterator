LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
PHI3_CHAT_TEMPLATE = """<|user|>\n{instruction}<|end|>\n<|assistant|>"""


class ChatTemplate:
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def format(self, instruction):
        return self.template.format(instruction=instruction)

    def __enter__(self):
        self.prev = self.model.chat_template
        self.model.chat_template = self
        return self

    def __exit__(self, exc, exc_value, exc_tb):
        self.model.chat_template = self.prev
        del self.prev
