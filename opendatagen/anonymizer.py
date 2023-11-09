import re
import spacy
from opendatagen.model import OpenAIModel, ModelName
from opendatagen.utils import load_file

class Anonymizer:

    NER_PLACEHOLDER = {
        "PERSON": "{person}",
        "ORG": "{organization}",
        "GPE": "{location}",
        "DATE": "{date}",
        "TIME": "{time}",
        "NORP": "{group}",
        "FAC": "{facility}",
        "LOC": "{location}",
        "PRODUCT": "{product}",
        "EVENT": "{event}",
        "WORK_OF_ART": "{artwork}",
        "LAW": "{law}",
        "LANGUAGE": "{language}",
        "MONEY": "{money}",
        "PERCENT": "{percentage}",
        "ORDINAL": "{ordinal}",
        "CARDINAL": "{number}",
        # Add more if needed
    }

    REGEX_PATTERN = {
            "credit_card_pattern": "{credit_card}",
            "social_security_pattern": "{us_social_security}",
            "passport_pattern": "{passport}",
            "drivers_license_pattern": "{driver_license}",
            "ip_address_pattern": "{ip_address}",
            "date_pattern": "{date}",
            "bank_account_pattern": "{bank_account}",
            "phone_number_pattern": "{phone_number}",
            "email_pattern": "{email}",
            "url_pattern": "{url}",
            "address_pattern": "{address}"
        }

    def __init__(self, completion_model:OpenAIModel):
        
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_prompt = load_file("files/ner.txt")
        self.completion_model = completion_model

    def regex_anonymization(self, text: str) -> str:

        for pattern, replacement in self.REGEX_PATTERN.items():
            text = re.sub(pattern, replacement, text)
        
        return text

    def ner_anonymization(self, text: str) -> str:
        doc = self.nlp(text)
        for entity in doc.ents:
            placeholder = self.NER_PLACEHOLDER.get(entity.label_)
            if placeholder:
                text = text.replace(entity.text, placeholder)
        return text

    def llm_anonymization(self, text: str) -> str:

        completion = self.completion_model.ask(
            system_prompt=self.ner_prompt,
            user_prompt=text,
            max_tokens=126,
            temperature=0
        ) 

        return completion

    def anonymize(self, text: str) -> str:

        text = self.regex_anonymization(text)
        text = self.ner_anonymization(text)
        return self.llm_anonymization(text)


