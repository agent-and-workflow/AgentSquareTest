import os

import requests


class Dify:
	def __init__(self, server, user, api_key):
		self.server = server
		self.user = user
		self.api_key = api_key
	
	def run_workflow(self, inputs, response_mode="blocking"):
		workflow_url = f"{self.server}/workflows/run"
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json"
		}
		body = {
			"inputs": inputs,
			"response_mode": response_mode,
			"user": self.user
		}
		try:
			response = requests.post(workflow_url, headers=headers, json=body)
			if response.status_code == 200:
				print(f"[工作流执行成功] ")
				return response.json(), True
			else:
				print(f"[工作流执行失败] ，状态码: {response.status_code}")
				return {"status": "error",
				        "message": f"Failed to execute workflow, status code: {response.status_code}"}, False
		except Exception as e:
			print(f"[工作流异常] ，错误: {str(e)}")
			return {"status": "error", "message": str(e)}, False
	
	def get_outputs(self, messages):
		return messages['data']['outputs']
	
	def get_logs(self):
		url = f"{self.server}/workflows/logs"
		querystring = {"page": "1", "limit": "20"}
		headers = {"Authorization": f"Bearer {self.api_key}"}
		response = requests.request("GET", url, headers=headers, params=querystring)
		with open("data/logs.json", "w", encoding="utf-8") as file:
			file.write(response.text)
	
	def get_run_details(self,workflow_run_id):
		url = f"https://api.dify.ai/v1/workflows/run/{workflow_run_id}"
		headers = {"Authorization": f"Bearer {self.api_key}"}
		response = requests.request("GET", url, headers=headers)
		print(response.text)
