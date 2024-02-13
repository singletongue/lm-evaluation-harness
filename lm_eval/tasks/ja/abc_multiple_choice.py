"""
ABC Multiple Choice
"""
import os
import warnings
import time

from lm_eval.base import MultipleChoiceTask, rf
import numpy as np


_CITATION = ""


class ABCMultipleChoice(MultipleChoiceTask):
    """
    prompt format is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """

    VERSION = 0.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "singletongue/abc-multiple-choice"
    DATASET_NAME = None
    DESCRIPTION = "[問題]に対する[答え]を[選択肢]の中から選んでください。\n\n"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "qid": doc["qid"],
            "goal": doc["question"],
            "choices": [doc[f"choice_{i + 1}"] for i in range(4)],
            "gold": doc["answer_number"] - 1,
        }

    def doc_to_text(self, doc):
        """
        [問題]:question
        [選択肢]:[choice_1, choice_2, choice_3, choice_4]
        [答え]:
        """
        return f"[問題]:{doc['goal']}\n" f"[選択肢]:[{', '.join(doc['choices'])}]\n" "[答え]:"

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]

        response = np.argmax(results)
        acc = 1.0 if response == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        out = {
            "acc": acc,
            "acc_norm": acc_norm,
        }
        # only include details if we were wrong
        if acc == 0.0:
            # without the cast it won't serialize
            response = int(response)
            out["details"] = {
                "qid": doc["qid"],
                "question": doc["goal"],
                "choices": doc["choices"],
                "gold": doc["gold"],
                "response": response,
            }
        return out


class ABCMultipleChoiceWithFintanPrompt(ABCMultipleChoice):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """

    VERSION = 0.1
    PROMPT_VERSION = 0.2
    DESCRIPTION = (
        "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号(例:0)でするものとします。 \n\n"
    )
    DID_WARNING = False

    def doc_to_text(self, doc):
        """
        質問:question
        選択肢:0.choice_1,1.choice_2, ...,3.choice_4
        回答:
        """
        if not self.DID_WARNING:
            warnings.warn(
                "#" * 100
                + "\n\nprompt version `0.2` for ABCMultipleChoice tends to output low scores! We highly recommend using `0.2.1` instead!\n\n"
                + "#" * 100
            )
            self.DID_WARNING = True
            time.sleep(5)
        choices = ",".join(
            [f"{idx}.{choice}" for idx, choice in enumerate(doc["choices"])]
        )
        return f"質問:{doc['goal']}\n" f"選択肢:{choices}\n" "回答:"

    def doc_to_target(self, doc):
        return f"{doc['gold']}"


class ABCMultipleChoiceWithFintanPromptV21(ABCMultipleChoice):
    VERSION = 0.1
    PROMPT_VERSION = "0.2.1"
    DESCRIPTION = "与えられた選択肢の中から、最適な答えを選んでください。 \n\n"

    def doc_to_text(self, doc):
        """
        与えられた選択肢の中から、最適な答えを選んでください。

        質問：{question}
        選択肢：
        - {choice_1}
        - {choice_2}
        - {choice_3}
        - {choice_4}
        回答：
        """
        choices = "\n".join([f"- {choice}" for choice in doc["choices"]])
        input_text = f"質問：{doc['goal']}\n選択肢：\n{choices}\n回答："
        return input_text


class ABCMultipleChoiceWithJAAlpacaPrompt(ABCMultipleChoice):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data.
    ```
    {
        'instruction': 'この課題では、以下の選択肢から文の出典を特定する必要があります。\n\n出力は以下から選択してください：\n- 新聞\n- 教科書\n- オンライン記事\n- 百科事典',
        'input': '彼はローマの政治家であり哲学者であり、史上最も偉大な軍事指導者の一人と考えられています。',
        'output': '百科事典'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """

    VERSION = 0.1
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられた選択肢の中から、最適な答えを選んでください。"

    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示:
        {instruction}

        ### 入力:
        {input}

        ### 応答:
        {response}
        """
        choices = "\n".join([f"- {choice}" for choice in doc["choices"]])
        instruction_text = self.INSTRUCTION + f"出力は以下から選択してください：\n{choices}"
        input_text = f"{doc['goal']}"
        return f"### 指示:\n{instruction_text}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class ABCMultipleChoiceWithRinnaInstructionSFT(ABCMultipleChoice):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """

    VERSION = 0.1
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられた選択肢の中から、最適な答えを選んでください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        choices = self.SEP.join([f"- {choice}" for choice in doc["choices"]])
        input_text = f"質問：{doc['goal']}{self.SEP}" + f"選択肢：{self.SEP}{choices}"
        return f"ユーザー: {input_text}{self.SEP}システム: "


class ABCMultipleChoiceWithRinnaBilingualInstructionSFT(
    ABCMultipleChoiceWithRinnaInstructionSFT
):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """

    PROMPT_VERSION = 0.5
    DESCRIPTION = "ユーザー: 与えられた選択肢の中から、最適な答えを選んでください。\nシステム: 分かりました。\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n"


class ABCMultipleChoiceWithLlama2(ABCMultipleChoice):
    """
    This prompt version follows the Llama2-chat's prompt format:
    ```
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    ```
    reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    PROMPT_VERSION = 0.6
    # DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    DEFAULT_SYSTEM_PROMPT = "あなたは役立つアシスタントです。"
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    DESCRIPTION = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    INSTRUCTION = "与えられた5つの選択肢の中から、最適な答えを選んでください。"
    FEWSHOT_SEP = " </s><s>[INST] "

    def doc_to_text(self, doc):
        """
        Insert the following prompt into `{{ user_msg }}`, which is based on prompt version 0.3
        ```
        与えられた選択肢の中から、最適な答えを選んでください。出力は以下から選択してください：
        - choice_1
        ...
        - choice_4

        質問：... [/INST]
        ```
        """
        choices = "\n".join([f"- {choice}" for choice in doc["choices"]])
        instruction_text = self.INSTRUCTION + f"出力は以下から選択してください：\n{choices}"
        input_text = f"質問：{doc['goal']}"
        return f"{instruction_text}\n\n{input_text} [/INST] "


VERSIONS = [
    ABCMultipleChoice,
    ABCMultipleChoiceWithFintanPrompt,
    ABCMultipleChoiceWithFintanPromptV21,
    ABCMultipleChoiceWithJAAlpacaPrompt,
    ABCMultipleChoiceWithRinnaInstructionSFT,
    ABCMultipleChoiceWithRinnaBilingualInstructionSFT,
    ABCMultipleChoiceWithLlama2,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[
            f"abc_multiple_choice-{version_class.VERSION}-{version_class.PROMPT_VERSION}"
        ] = version_class
    return tasks
