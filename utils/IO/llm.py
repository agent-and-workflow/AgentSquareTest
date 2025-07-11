import os
import sys

from openai import OpenAI
from tenacity import (
	retry,
	stop_after_attempt,  # type: ignore
	wait_random_exponential,  # type: ignore
)

from typing import Optional, List

if sys.version_info >= (3, 8):
	from typing import Literal
else:
	from typing_extensions import Literal

OpenAIModel = Literal["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4o"]

class OpenAILLM:
	def __init__(self,model: OpenAIModel):
		self.api_key = os.environ.get("OPENAI_API_KEY")
		self.client = OpenAI(api_key=self.api_key)
		self.completion_tokens =0
		self.prompt_tokens = 0
		self.model=model
	
	@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
	def get_completion(self, prompt: str,  temperature: float = 0.0, max_tokens: int = 1000,
	                   stop_strs: Optional[List[str]] = None, n=1) -> str:
		response = self.client.completions.create(
			model=self.model,
			prompt=prompt,
			temperature=temperature,
			max_tokens=max_tokens,
			top_p=1,
			n=n,
			frequency_penalty=0.0,
			presence_penalty=0.0,
			stop=stop_strs,
		)
		self.completion_tokens += response.usage.completion_tokens
		self.prompt_tokens += response.usage.prompt_tokens
		if n > 1:
			responses = [choice.text.replace('>', '').strip() for choice in response.choices]
			return responses
		return response.choices[0].text.replace('>', '').strip()
	
	@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
	def get_chat(self, prompt: str, model: OpenAIModel, temperature: float = 0.0, max_tokens: int = 1000,
	             stop_strs: Optional[List[str]] = None, messages=None, n=1) -> str:
		assert model != "text-davinci-003"
		if messages is None:
			messages = [
				{
					"role": "user",
					"content": prompt
				}
			]
		response = self.client.chat.completions.create(
			model=model,
			messages=messages,
			max_tokens=max_tokens,
			stop=stop_strs,
			n=n,
			temperature=temperature,
		)
		
		self.completion_tokens += response.usage.completion_tokens
		self.prompt_tokens += response.usage.prompt_tokens
		if n > 1:
			responses = [choice.message.content.replace('>', '').strip() for choice in response.choices]
			return responses
		# print(response)
		return response.choices[0].message.content.replace('>', '').strip()
	
	def llm_response(self, prompt, temperature: float = 0.0, max_tokens: int = 1000,
	                 stop_strs: Optional[List[str]] = None, n=1) -> str:
		if isinstance(prompt, str):
			if self.model == 'gpt-3.5-turbo-instruct':
				comtent = self.get_completion(prompt=prompt, model=self.model, temperature=temperature, max_tokens=max_tokens,
				                         stop_strs=stop_strs, n=n)
			else:
				comtent = self.get_chat(prompt=prompt, model=self.model, temperature=temperature, max_tokens=max_tokens,
				                   stop_strs=stop_strs, n=n)
		else:
			messages = prompt
			prompt = prompt[1]['content']
			if self.model == 'gpt-3.5-turbo-instruct':
				comtent = self.get_completion(prompt=prompt, model=self.model, temperature=temperature, max_tokens=max_tokens,
				                         stop_strs=stop_strs, n=n)
			else:
				comtent = self.get_chat(prompt=prompt, model=self.model, temperature=temperature, max_tokens=max_tokens,
				                   stop_strs=stop_strs, messages=messages, n=n)
		return comtent
	
	def get_price(self):
		return self.completion_tokens, self.prompt_tokens, self.completion_tokens * 60 / 1000000 + self.prompt_tokens * 30 / 1000000
