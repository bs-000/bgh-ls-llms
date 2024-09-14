# creating BGH data
import spacy

remove_brackets = False
server = False
random_state = 17
no_brackets_suffix = "no_br_"
nlp = spacy.load("de_core_news_sm")  # small one
nlp_english = spacy.load("en_core_web_lg")


