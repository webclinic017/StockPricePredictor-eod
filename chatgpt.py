#https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

import json
import openai
import tiktoken
import json
import re
import ast
import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict

######################################################################################################
def CallChatGPT(prompt: str) -> dict:
    def get_completion(prompt, model="gpt-3.5-turbo"):

        system_prompt = "i will send you list of titles of articles from financial news,\
                    i want you to classify the titles whether it is positive or negative, i want you to mark  \
                    it by 3 digit number between 0 to 1, if 0 it is very negative, if it is 1 it is very positive. \
                    i want you to output rating only, no other text is needed.  \
                    i want you to output it as dictionary, keys are titles, values are ratings."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        
        response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0, 
        )
        return response.choices[0].message["content"]

    retry_count = 0
    max_retries = 5
    result = None

    while retry_count < max_retries: 
        try:
            result = get_completion(prompt)
            break
        except Exception as e:
            print("Exception popped, i m retrying again")
            retry_count += 1

    if result is None:
        print("\nFunction failed after maximum retries.")
    else:
        print("\nFunction succeeded.")   
        match = re.search(r"{.*}", result)
        if match:
            result = match.group()
           
    return result 
##############################################################################################################


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def GetTitles(news_df:pd.DataFrame,gpt_news_json_path:str,ticker:str) -> Tuple[List[str], pd.DataFrame, bool, Dict]:
    json_exists = False

    if os.path.isfile(gpt_news_json_path):
        #load json file and map it in news df
        print(f"{str.upper(ticker)} json file exists")
        json_exists = True
        # Load the JSON file
        with open(gpt_news_json_path, 'r') as json_file:
            loaded_data = json.load(json_file)
    #Map dictionary to news_df
        news_df['ChatGPT_Sentiment'] = news_df['Title'].map(loaded_data)
        news_df_empties_ = news_df[news_df['ChatGPT_Sentiment'].isnull()]
        titles_evaluate = news_df_empties_['Title']
        print("Titles to evaluate after loaded json file via ChatGPT: ",len(titles_evaluate))
        print(f"\nTitles to be evaluated: {titles_evaluate}")
        
    else:
        #Evaluate all titles from news_df
        print("json file does not exist")
        titles_evaluate = news_df['Title']
        print("Titles to evaluate via ChatGPT: ",len(titles_evaluate))
        news_df['ChatGPT_Sentiment'] = np.nan
        loaded_data = None
    return titles_evaluate, news_df, json_exists,loaded_data


def ChatGPTAnalysis(api_key_gpt: str,titles_evaluate: list,news_df:pd.DataFrame,json_exists:bool,loaded_data:dict,gpt_news_json_path:str) -> pd.DataFrame:
    dictionary_data = {}
    titles = []
    openai.api_key  = api_key_gpt 
    counter = 0
    batch = 1
    main_dict = {}

    #Split df to empties and filled data
    news_df_empties = news_df[news_df['ChatGPT_Sentiment'].isnull()].copy()
    news_df_filled = news_df[news_df['ChatGPT_Sentiment'].isnull()==False].copy()

    # ttl_tokens_text = sum([num_tokens_from_string(text, "cl100k_base") for text in titles_evaluate]) + default_text

    for text in titles_evaluate:

        counter += 1
        
        titles.append(text)

        prompt = f"My titles is followed in triple ticks as a list, do sentiment analysis between 0 to 1, do ranking in 2 digits between 0 and 1, output it as dictionary: '''{titles}'''"
        
        ttl_tokens_batch = num_tokens_from_string(prompt, "cl100k_base")

        if counter == len(titles_evaluate) and batch==1:
            dictionary_data = CallChatGPT(prompt)
            dictionary_data = ast.literal_eval(dictionary_data)
            main_dict.update(dictionary_data)

        if counter == len(titles_evaluate) and batch > 1:
            print("Last batch finished.")
            dictionary_data = CallChatGPT(prompt)
            dictionary_data = ast.literal_eval(dictionary_data)
            main_dict.update(dictionary_data)

        if ttl_tokens_batch > 1000 and counter < len(titles_evaluate):
            print("counter: ",counter)
            print("Batch over 1000 tokens: ",ttl_tokens_batch)
            dictionary_data = CallChatGPT(prompt)
            dictionary_data = ast.literal_eval(dictionary_data)
            print(f"\nNumber of passed titles to chatgpt: ",len(titles))
            print(f"Number of received from chatgpt: ",len(dictionary_data))
            main_dict.update(dictionary_data)
            
            batch += 1
            titles = []
            print("batch: ",batch)

    if json_exists == False:
        # Save the dictionary as JSON
        with open(gpt_news_json_path, 'w') as json_file:
            json.dump(main_dict, json_file)
            print("json file did not exist, chatgpt sentiment analysis done for all news, saved as json file, count: ",len(main_dict))
    else:
        # Merge the dictionaries and save new json file, overwrite initial one
        merged_dict = {}
        merged_dict = {**main_dict, **loaded_data}
        # Write the merged dictionary to a JSON file
        with open(gpt_news_json_path, 'w') as json_file:
            json.dump(merged_dict, json_file)
        print("Total news count: ",len(news_df))
        print("Loaded json file count: ",len(loaded_data))
        print(f"json file updated by {len(main_dict)}")
        print("Count of new json: ",len(merged_dict))

    #Fill sentiment analysis in news_df
    thresh = 0.05
    news_df_empties['ChatGPT_Sentiment'] = news_df_empties['Title'].map(main_dict)
    empties_count = news_df_empties['ChatGPT_Sentiment'].isnull().sum()
    ttl_count = len(news_df['ChatGPT_Sentiment'])
    null_tresh = int(round(ttl_count * thresh,0))
    news_df_empties_ = news_df[news_df['ChatGPT_Sentiment'].isnull()].copy()
    #print(news_df_empties_)
        
    if empties_count > null_tresh:
        news_df_empties_ = news_df[news_df['ChatGPT_Sentiment'].isnull()].copy()
        print(news_df_empties_)
        raise ValueError(f"The count of empty values '{empties_count}' exceeded the threshold '{int(thresh*100)}%' from ttl {ttl_count}.")
        
    else:
        print(f"Total number of  values: {ttl_count}")
        print(f"Total number of empty values: {empties_count}")

        if empties_count>0:
            print("All empty values will be populated by 0.5")
            news_df_empties['ChatGPT_Sentiment'].fillna(0.5, inplace=True)
        
        #news_df_empties_ = None
        news_df = pd.concat([news_df_filled,news_df_empties],axis=0)

    return news_df,news_df_empties_,titles_evaluate,main_dict,merged_dict