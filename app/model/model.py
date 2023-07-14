import os
from torch import nn
import transformers
from transformers import AutoTokenizer
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


__version__ = "1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

max_input_length = 1024
max_target_length = 128

def download_artifacts():

    model_path = str(BASE_DIR) + "/model_artifacts"
    model_checkpoint ='t5-small'

    if os.path.isdir(model_path):


        print("Loading Model")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Model Loaded")
    else:
        print("No Model Artifacts Found")
        t5_model = None

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print("Tokenizer Loaded")
    
    return t5_model,tokenizer


t5_model,tokenizer = download_artifacts()


# input is of type list of string List[str], where each item in the list is an article which needs to be summarized
def predict_pipeline(text_input):

    text = ["summarize : " + item for item in text_input]

    inputs = tokenizer(text, max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt").input_ids

    outputs = t5_model.generate(inputs, max_length=max_target_length, do_sample=False, num_beams = 3)

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = [pred.strip() for pred in predictions]

    # result = pd.DataFrame(list(zip(text_input, predictions)))
    # result.columns = ['Text_Input','Summary']
    # result.to_csv("text_summary.csv")


    return predictions



# text_input = ["LONDON, England (Reuters) Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in Harry Potter and the Order of the Phoenix To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar, he told an Australian interviewer earlier this month. I don't think I'll be particularly extravagant. The things I like buying are things that cost about 10 pounds books and CDs and DVDs. At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film Hostel: Part II, currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. I'll definitely have some sort of party, he said in an interview. Hopefully none of you will be reading about it. Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. People are always looking to say kid star goes off the rails,he told reporters last month. But I try very hard not to go that way because it would be too easy for them. His latest outing as the boy wizard in Harry Potter and the Order of the Phoenix is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest.There is life beyond Potter, however. The Londoner has filmed a TV movie called My Boy Jack, about author Rudyard Kipling and his son, due for release later this year. He will also appear in December Boys, an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's Equus. Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: I just think I'm going to be more sort of fair game, he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."]

# print(predict_pipeline(text_input))
