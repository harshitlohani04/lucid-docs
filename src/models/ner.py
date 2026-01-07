from spacy.pipeline import EntityRuler
import spacy
import os
from huggingface_hub import InferenceClient
import dotenv
import re

dotenv.load_dotenv()

def normal_ner(sentences):
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN")
    )

    result = client.token_classification(
        sentences,
        model="dslim/bert-base-NER",
    )

    return result

nlp = spacy.load("en_core_web_sm")

# Add EntityRuler BEFORE NER
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    ruler.add_patterns([
        {
            "label": "NAME",
            "pattern": [
                {
                    "TEXT": {
                        "REGEX": r"^(Mr|Mrs|Miss)\.?"
                    }
                },
                {
                    "IS_ALPHA": True,
                    "OP": "{2,}"
                }
            ]
        },
        {
            "label": "AGE_",
            "pattern": [
                {"TEXT": {"REGEX": r"\d{1,3}"}},
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"REGEX": r"(Years?|Yrs?|yrs?)"}}
            ]
        }
    ])

def custom_ner(sentences: str):
    doc = nlp(sentences)
    output = []
    for ent in doc.ents:
        if ent.label_ == "NAME" or ent.label_ == "AGE_":
            if ent.label_ == "AGE_":
                if not re.search(r"\b\d{1,3}\s*(Years?|Yrs?)\b", ent.text):
                    continue
            output.append((ent.text, ent.label_))
    return output

