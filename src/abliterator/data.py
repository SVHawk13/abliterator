from datasets import load_dataset
from sklearn.model_selection import train_test_split


def get_harmful_instructions() -> tuple[list[str], list[str]]:
    hf_path = "Undi95/orthogonal-activation-steering-TOXIC"
    dataset = load_dataset(hf_path)
    instructions = [i["goal"] for i in dataset["test"]]

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions() -> tuple[list[str], list[str]]:
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)
    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def prepare_dataset(
    dataset: tuple[list[str], list[str]] | list[str],
) -> tuple[list[str], list[str]]:
    if len(dataset) != 2:
        # assumed to not be split into train/test
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    else:
        train, test = dataset

    return train, test
