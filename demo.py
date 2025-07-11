from uuid import UUID
from entities.workflow import Workflow
from utils.IO.data_base import DataBase
from utils.IO.dify import Dify
from utils.IO.file import Json
from utils.IO.llm import OpenAILLM

if __name__ == '__main__':
	# 开启网络代理
	import os
	os.environ['http_proxy'] = 'http://127.0.0.1:7890'
	os.environ['https_proxy'] = 'http://127.0.0.1:7890'
	os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
	
	# 预先定义一些变量
	runConfig = Json.read_json("./config/demo_run_config.json")
	inputs = {"content": runConfig["translate_str"]}  # 键为工作流的输入变量名称，值为你想设定的值
	
	# 连接数据库
	print("正在连接数据库".center(40,"="))
	dataBaseConfig = Json.read_json(runConfig["db_config_path"])
	dataBase = DataBase(dataBaseConfig["DB_USERNAME"], dataBaseConfig["DB_PASSWORD"], dataBaseConfig["DB_HOST"],
	                    dataBaseConfig["DB_PORT"], dataBaseConfig["DB_DATABASE"])

	# 获取workflow输入，这里直接从数据库中获取
	print("正在获取workflow输入".center(40, "="))
	workflow = dataBase.get_latest(Workflow)
	workflowGraph = workflow["graph"]  # json字符串
	workflowFeatures = workflow["features"]  # json字符串

	# 假如workflowGraph、workflowFeatures是输入待优化的json字符串
	# 假设使用LLM进行优化
	print("正在调用LLM优化workflow".center(40, "="))
	chatGPT = OpenAILLM(runConfig["workflow_generator"])
	graphPrompt = f'''The following content is the graph field of a Dify workflow, which is a JSON string describing the workflow's topology structure.Based on the sample graph provided below, please create a new graph JSON string following a similar structure.
Sample graph: {workflowGraph}
Please directly output the new graph JSON string.'''
	graphOutput = chatGPT.llm_response(graphPrompt)  # 优化后的graph
	featuresPrompt = f'''The following content is the features field of a Dify workflow, which is a JSON string describing the functional configuration of the workflow.
Sample features: {workflowFeatures}
Then, based on the following graph field that describes the workflow's topology, please generate a features field JSON string that corresponds to the structure and logic of the provided graph.
graph field: {graphOutput}
Please directly output the new features field as a JSON string.'''
	featuresOutput = chatGPT.llm_response(featuresPrompt)  # 优化后的features
	print("正在保存LLM的输出".center(40,"="))
	with open("new_graph.json", "w") as f:
		f.write(graphOutput)
	with open("new_features.json", "w") as f:
		f.write(featuresOutput)
	print("保存成功,请阅读文件确保json文件无误后再运行后续代码。")
	
	# 关闭代理
	os.environ.pop('http_proxy', None)
	os.environ.pop('https_proxy', None)
	os.environ.pop('all_proxy', None)
	# 修改dify数据库，为确保安全，demo中直接使用准备好的字段内容以确保格式正确
	print("正在修改dify数据库".center(40,"="))
	graphOutput = Json.read_json("./data/workflow/graph.json")
	featuresOutput = Json.read_json("./data/workflow/features.json")
	graphStr = Json.dict2str(graphOutput)
	featuresStr = Json.dict2str(featuresOutput)
	dataBase.update_by_id(Workflow, UUID(workflow["id"]), {"graph": graphStr, "features": featuresStr})
	
	# 调用dify获得输出
	print("正在调用新的工作流".center(40,"="))
	print(f"workflow id:{workflow['id']}")
	difyConfig = Json.read_json(runConfig["dify_config_path"])
	dify = Dify(difyConfig["DIFY_SERVER"], difyConfig["USER"], difyConfig["API_KEY"])
	messages, status = dify.run_workflow(inputs)
	if status:
		outputs = dify.get_outputs(messages)
	else:
		outputs = {}
	print("运行成功,即将打印工作流输出".center(40,"="))
	print(outputs)
