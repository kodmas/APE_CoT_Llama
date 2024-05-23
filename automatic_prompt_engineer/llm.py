"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch import nn
from openai import OpenAI

# client = OpenAI()

class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of. 
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass
class Llama2Model(LLM):
    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        # Initialize your model using the provided config
        # self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        # Initialize the model
        # model_name = 'meta-llama/Llama-2-7b-chat-hf'
        model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
        # nf4_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, quantization_config=nf4_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                             device_map="auto", 
        #                                             quantization_config=nf4_config)
        # model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                             device_map="auto", 
        #                                             torch_dtype=torch.float16)
        # model.config.pad_token_id = self.tokenizer.eos_token_id
        self.lm = model
        # self.system_prompt = "<s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being moderately concise.<</SYS>>\n\n" 
        self.system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being moderately concise.<|eot_id|>\n"
        # self.CoT = "Let's think step by step. "
        self.lm.eval()

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = 1
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                # f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text
    def make_response(self, prefix_sentences, parse=False):
        print("make_response")
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                prompt = prefix_sentences[i].replace('[APE]', '').strip()
                # prompt = self.system_prompt + prompt + " [/INST]"
                prompt = self.system_prompt + "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
                sentences.append(prompt)

            
            inputs = self.tokenizer(
                sentences, 
                return_tensors="pt", 
                padding=True,  # Ensures padding to the max_length
            ).to(self.lm.device)  
            
            generation_args = dict(temperature=0.7, max_new_tokens=256, repetition_penalty=1.1, output_scores=True, return_dict_in_generate=True, do_sample=True)  
            outputs = self.lm.generate(**inputs, **generation_args)
            print("in make response-------------")
            
            reply_strings = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            
        tmp = []
        for i in range(len(prefix_sentences)) :
            if parse == True and i == 0:
                print("Evaluation")
            print(f'reply strings {i}', reply_strings[i])
            if parse == False:
                tmp.append(reply_strings[i].split("The instruction was to")[-1].strip())
            else:
                tmp.append(reply_strings[i].split("\n\n")[-1].strip())
            print(f'tmp {i} = ',tmp[i])
            print('------\n')
        return tmp
    
    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens
    
    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if log_prob_range is not None:
            for i in range(len(text)):
                lower_index, upper_index = log_prob_range[i]
                assert lower_index < upper_index
                assert lower_index >= 0
                assert upper_index - 1 < len(text[i])
        if isinstance(text, list):
            text = [f'\n{text[i]}' for i in range(len(text))]
        else:
            text = f'\n{text}'
        response = None
        while response is None:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=200).to(self.lm.device)
                generation_args = dict(temperature=1.0, max_new_tokens=256, repetition_penalty=1.1, output_scores=True, return_dict_in_generate=True, do_sample=False)
                outputs = self.lm.generate(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'], 
                    **generation_args
                ) 
                # Access the token IDs
                token_ids = outputs['sequences'] if 'sequences' in outputs else outputs

                # Check for invalid values
                if torch.isnan(token_ids).any() or torch.isinf(token_ids).any():
                    print("Warning: Detected 'nan' or 'inf' values in outputs")
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        log_probs = [response.choices[i].logprobs.token_logprobs[1:]
                     for i in range(len(response.choices))]
        tokens = [response.choices[i].logprobs.tokens[1:]
                  for i in range(len(response.choices))]
        offsets = [response.choices[i].logprobs.text_offset[1:]
                   for i in range(len(response.choices))]

        # Subtract 1 from the offsets to account for the newline
        for i in range(len(offsets)):
            offsets[i] = [offset - 1 for offset in offsets[i]]

        if log_prob_range is not None:
            # First, we need to find the indices of the tokens in the log probs
            # that correspond to the tokens in the log_prob_range
            for i in range(len(log_probs)):
                lower_index, upper_index = self.get_token_indices(
                    offsets[i], log_prob_range[i])
                log_probs[i] = log_probs[i][lower_index:upper_index]
                tokens[i] = tokens[i][lower_index:upper_index]

        return log_probs, tokens

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            text = [prompt]
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
            print(prompt[i])
        response = None
        while response is None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=200).to(self.lm.device)
                generation_args = dict(temperature=0.7, max_new_tokens=256, repetition_penalty=1.1, output_scores=True, return_dict_in_generate=True, do_sample=False)
                outputs = self.lm.generate(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'], 
                    **generation_args
                ) 
                response = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                print("response = ",response) 
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        return response.choices

    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if log_prob_range is not None:
            for i in range(len(text)):
                lower_index, upper_index = log_prob_range[i]
                assert lower_index < upper_index
                assert lower_index >= 0
                assert upper_index - 1 < len(text[i])
        if isinstance(text, list):
            text = [f'\n{text[i]}' for i in range(len(text))]
        else:
            text = f'\n{text}'
        response = None
        while response is None:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=200).to(self.lm.device)
                generation_args = dict(temperature=0.7, max_new_tokens=256, repetition_penalty=1.1, output_scores=True, return_dict_in_generate=True, do_sample=False)
                outputs = self.lm.generate(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'], 
                    **generation_args
                ) 
                response = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                print("response = ",response) 
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        log_probs = [response.choices[i].logprobs.token_logprobs[1:]
                     for i in range(len(response.choices))]
        tokens = [response.choices[i].logprobs.tokens[1:]
                  for i in range(len(response.choices))]
        offsets = [response.choices[i].logprobs.text_offset[1:]
                   for i in range(len(response.choices))]

        # Subtract 1 from the offsets to account for the newline
        for i in range(len(offsets)):
            offsets[i] = [offset - 1 for offset in offsets[i]]

        if log_prob_range is not None:
            # First, we need to find the indices of the tokens in the log probs
            # that correspond to the tokens in the log_prob_range
            for i in range(len(log_probs)):
                lower_index, upper_index = self.get_token_indices(
                    offsets[i], log_prob_range[i])
                log_probs[i] = log_probs[i][lower_index:upper_index]
                tokens[i] = tokens[i][lower_index:upper_index]

        return log_probs, tokens

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index

class BatchSizeException(Exception):
    pass
