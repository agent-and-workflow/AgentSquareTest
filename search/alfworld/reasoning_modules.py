from utils import llm_response
from collections import Counter
import re


class ReasoningBase:
	def __init__(self, profile_type_prompt, memory, llms_type):
		self.profile_type_prompt = profile_type_prompt
		self.memory = memory
		self.llm_type = llms_type[0]
		self.task_name_cache = None
	
	def process_task_description(self, task_description):
		task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
		if self.memory is not None:
			if self.task_name_cache is not None and self.task_name_cache == task_name:
				pass
			else:
				self.task_name_cache = task_name
				self.memory_cache = self.memory(task_description)
		else:
			self.memory_cache = ''
		split_text = task_description.rsplit('You are in the', 1)
		examples = split_text[0]
		task_description = 'You are in the' + split_text[1]
		
		return examples, task_description


class ReasoningIO(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = '''Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
		prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
		reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
		
		return reasoning_result


class ReasoningCOT(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = '''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
		prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
		reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
		return reasoning_result


class ReasoningCOTSC(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = '''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
		prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
		reasoning_results = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'], n=5)
		string_counts = Counter(reasoning_results)
		reasoning_result = string_counts.most_common(1)[0][0]
		return reasoning_result


class ReasoningTOT(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = '''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
		prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
		reasoning_results = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'], n=3)
		reasoning_result = self.get_votes(task_description, reasoning_results, examples)
		return reasoning_result
	
	def get_votes(self, task_description, reasoning_results, examples):
		if 'think' in reasoning_results[0].lower():
			return reasoning_results[0]
		prompt = '''Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format. Output "The best answer is {{s}}", where s is the integer id chosen.
Here are some examples.
{examples}
Here is the task:
{task_description}

'''
		prompt = prompt.format(task_description=task_description, examples=examples)
		for i, y in enumerate(reasoning_results, 1):
			prompt += f'Answer {i}:\n{y}\n'
		vote_outputs = llm_response(prompt=prompt, model=self.llm_type, temperature=0.7, n=5)
		vote_results = [0] * len(reasoning_results)
		for vote_output in vote_outputs:
			pattern = r".*best answer is .*(\d+).*"
			match = re.match(pattern, vote_output, re.DOTALL)
			if match:
				vote = int(match.groups()[0]) - 1
				if vote in range(len(reasoning_results)):
					vote_results[vote] += 1
			else:
				print(f'vote no match: {[vote_output]}')
		ids = list(range(len(reasoning_results)))
		select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
		return reasoning_results[select_id]


class ReasoningDILU(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = [
			{
				"role": "system",
				"content": '''You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature domestic robot, who can give accurate and correct instruction in interacting with a household. You will be given a detailed description of the scenario of current frame along with your history of previous decisions.
'''
			},
			{
				"role": "user",
				"content": f'''Above messages are some examples of how you make a step successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a step for the current scenario. Your instructions must follow the examples.
Here are two examples.
{examples}{self.memory_cache}
Here is the task:
{task_description}'''
			}
		]
		reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
		return reasoning_result


class ReasoningSelfRefine(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = '''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
		prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
		reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
		reasoning_result = self.refine(reasoning_result)
		return reasoning_result
	
	def refine(self, reasoning_result):
		if 'think' in reasoning_result:
			return reasoning_result
		prompt = f'''You need to check that the syntactic structure of the step meets the requirements.
requirements: '1. take a from b 2. go to a 3. : open a 4. put a in/on b, end. 5. clean a with b, end. 6. heat a with b, end. 7. cool a with b, end. 8. use a, end.', where 'a' and 'b' are variable.
examples:
take pencil 1 from desk 2   correct
take potato 1 with fridge 1 error, The preposition with take is from. revised: take potato 1 from bridge 1
go to cabinet 3   correct
go to countertop 2 and check   error, go to countertop 2 is the complete instruction. revised: go to countertop 2
open fridge 1 and take potato 2   error, open fridge 1 is the complete instruction. revised: open fridge 1
open safe 2   correct
put mug 2 in desk 1, end   error, The preposition with put is in/on. revised: put mug 2 in/on desk 1, end
put watch 1 in/on safe 1, end   correct
clean soapbar 1 with sinkbasin 1   error, Add "end" to the clean statement. revised: clean soapbar 1 with sinkbasin 1, end
clean lettuce 4 with sinkbasin 1, end   correct
heat egg 2 with microwave 1, end   correct
heat bread 1 with stoveburner 1, end   error, microwave is what you use to heat. revised: heat bread 1 with microwave 1, end
cool potato 2 with fridge 1, end   correct
cool pan 1, end   error,  bridge is whta you ues to cool. revised: cool pan 1 with bridge 1, end
use desklamp 3 to check statue 2   error, use desklamp3 is the complete instruction. revised: use desklamp 3, end
use desklamp 2, end   correct
Just focus on syntactic structure.
step: {reasoning_result}
You can only output in two formats:
"correct" or "error, revised: your step"
'''
		feedback_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.0)
		if 'correct' in feedback_result.lower():
			if ' in ' in reasoning_result:
				reasoning_result = reasoning_result.replace(' in ', ' in/on ')
			elif ' on ' in reasoning_result:
				reasoning_result = reasoning_result.replace(' on ', ' in/on ')
			return reasoning_result.replace('.', '')
		else:
			if ' in ' in feedback_result:
				feedback_result = feedback_result.replace(' in ', ' in/on ')
			elif ' on ' in feedback_result:
				feedback_result = feedback_result.replace(' on ', ' in/on ')
			return feedback_result.split(':')[-1].replace('.', '').strip()


class ReasoningStepBack(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		if task_description.split('Your')[-1].count('>') == 1:
			self.principle = self.stepback(task_description)
		
		prompt = f'''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{self.memory_cache}{self.principle}
Here is the task:
{task_description}'''
		reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
		return reasoning_result
	
	def stepback(self, task_description):
		last_index = task_description.rfind('>')
		task_description = task_description[:last_index]
		stepback_prompt = f'''What common sense, instruction structure is involved in solving this task?
{task_description}'''
		principle = llm_response(prompt=stepback_prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
		return principle


class ReasoningSelfReflectiveTOT(ReasoningBase):
	def __call__(self, task_description: str, feedback: str = ''):
		examples, task_description = self.process_task_description(task_description)
		prompt = '''Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
		prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
		reasoning_results = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'], n=3)
		reasoning_result = self.get_votes(task_description, reasoning_results, examples)
		reasoning_result = self.refine(reasoning_result)
		return reasoning_result
	
	def get_votes(self, task_description, reasoning_results, examples):
		if 'think' in reasoning_results[0].lower():
			return reasoning_results[0]
		prompt = '''Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format, which outputs "The best answer is {{s}}", where s is the integer id chosen.
Here are some examples.
{examples}
Here is the task:
{task_description}

'''
		prompt = prompt.format(task_description=task_description, examples=examples)
		for i, y in enumerate(reasoning_results, 1):
			prompt += f'Answer {i}:\n{y}\n'
		vote_outputs = llm_response(prompt=prompt, model=self.llm_type, temperature=0.3, n=5)
		vote_results = [0] * len(reasoning_results)
		for vote_output in vote_outputs:
			pattern = r".*best answer is .*(\d+).*"
			match = re.match(pattern, vote_output, re.DOTALL)
			if match:
				vote = int(match.groups()[0]) - 1
				if vote in range(len(reasoning_results)):
					vote_results[vote] += 1
			else:
				print(f'vote no match: {[vote_output]}')
		ids = list(range(len(reasoning_results)))
		select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
		return reasoning_results[select_id]
	
	def refine(self, reasoning_result):
		if 'think' in reasoning_result:
			return reasoning_result
		prompt = f'''You need to check that the syntactic structure of the step meets the requirements.
requirements: '1. take a from b 2. go to a 3. : open a 4. put a in/on b, end. 5. clean a with b, end. 6. heat a with b, end. 7. cool a with b, end. 8. use a, end.', where 'a' and 'b' are variable.
examples:
take pencil 1 from desk 2   correct
take potato 1 with fridge 1 error, The preposition with take is from. revised: take potato 1 from bridge 1
go to cabinet 3   correct
go to countertop 2 and check   error, go to countertop 2 is the complete instruction. revised: go to countertop 2
open fridge 1 and take potato 2   error, open fridge 1 is the complete instruction. revised: open fridge 1
open safe 2   correct
put mug 2 in desk 1, end   error, The preposition with put is in/on. revised: put mug 2 in/on desk 1, end
put watch 1 in/on safe 1, end   correct
clean soapbar 1 with sinkbasin 1   error, Add "end" to the clean statement. revised: clean soapbar 1 with sinkbasin 1, end
clean lettuce 4 with sinkbasin 1, end   correct
heat egg 2 with microwave 1, end   correct
heat bread 1 with stoveburner 1, end   error, microwave is what you use to heat. revised: heat bread 1 with microwave 1, end
cool potato 2 with fridge 1, end   correct
cool pan 1, end   error,  bridge is whta you ues to cool. revised: cool pan 1 with bridge 1, end
use desklamp 3 to check statue 2   error, use desklamp3 is the complete instruction. revised: use desklamp 3, end
use desklamp 2, end   correct
Just focus on syntactic structure.
step: {reasoning_result}
You can only output in two formats:
"correct" or "error, revised: your step"
'''
		
		feedback_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.0)
		if 'correct' in feedback_result.lower():
			if ' in ' in reasoning_result:
				reasoning_result = reasoning_result.replace(' in ', ' in/on ')
			elif ' on ' in reasoning_result:
				reasoning_result = reasoning_result.replace(' on ', ' in/on ')
			return reasoning_result
		else:
			if ' in ' in feedback_result:
				feedback_result = feedback_result.replace(' in ', ' in/on ')
			elif ' on ' in feedback_result:
				feedback_result = feedback_result.replace(' on ', ' in/on ')
			return feedback_result.split(':')[-1].strip()



class SequentialDependencyPlanner():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]
    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]
        # Creating a prompt that emphasizes task dependencies and reasoning
        if feedback == '':
            prompt = '''You are an advanced planning agent tasked with breaking down a {task_type} into detailed sub-tasks.
Your goal is to not only enumerate the steps but also to explain why each step is necessary and how they depend on each other.
Make sure to provide clear reasoning instructions and tool use instructions for each sub-task. Your output format should follow the example below:
The following are some examples:
Task: {example}

Task: {task_description}
'''  
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are an advanced planning agent tasked with breaking down a {task_type} into detailed sub-tasks.
Your goal is to not only enumerate the steps but also to explain why each step is necessary and how they depend on each other.
Make sure to provide clear reasoning instructions and tool use instructions for each sub-task. Your output format should follow the example below:
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''  
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan






class HybridPlanningModule():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]
    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]
        # The prompt words of the planning module
        if feedback == '':
            prompt = '''You are an advanced planner who needs to break down a {task_type} task into clear, actionable subtasks.
Your goal is to create a concise plan that minimizes the number of steps while ensuring clarity and completeness. For each subtask, provide a brief description, reasoning instructions, and tool use instructions if applicable.
Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''                
prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are an advanced planner who needs to break down a {task_type} task into clear, actionable subtasks.
Your goal is to create a concise plan that minimizes the number of steps while ensuring clarity and completeness. For each subtask, provide a brief description, reasoning instructions, and tool use instructions if applicable.
Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''                
prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan






class ContextualPlanning():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]

    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]

        # Constructing the prompt with a focus on contextual awareness
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks by considering the surrounding environment.
            You must factor in the objects visible and their relevant attributes.
            For each subtask, provide a detailed description, reasoning instructions, and instructions for using any tools necessary.
            Your output format should follow the example below.
            The following are some examples:
            Task: {example}

            Task: {task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks by considering the surrounding environment.
            You must factor in the objects visible and their relevant attributes.
            For each subtask, provide a detailed description, reasoning instructions, and instructions for using any tools necessary.
            Your output format should follow the example below.
            The following are some examples:
            Task: {example}

            end
            --------------------
            Reflexion:{feedback}
            Task:{task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan






class ContextualPlanning():
    def __init__(self, llms_type):
        self.plan = []
        self.llm_type = llms_type[0]
    def __call__(self, task_type, task_description, feedback):
        few_shot = planning_prompt[task_type]
        if feedback == '':
            prompt = '''You are an advanced planner who specializes in decomposing complex {task_type} tasks into actionable subtasks.
Your goal is to understand the environment and past interactions to optimize the subtask decomposition.
Provide clear reasoning for each subtask and specific tool use instructions where necessary.
Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''              
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are an advanced planner who specializes in decomposing complex {task_type} tasks into actionable subtasks.
Consider the feedback and refine your approach. Use your understanding of the environment and previous interactions to optimize the subtask decomposition.
Provide clear reasoning for each subtask and specific tool use instructions where necessary.
Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''              
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        dict_strings = re.findall(r'\{[^{}]*\}', string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan






class ContextualPlanning():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]
    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]
        # The prompt words of the planning module
        if feedback == '':
            prompt = '''You are a sophisticated planner that divides a {task_type} task into several meaningful subtasks while considering the context and dependencies involved in the task. For each subtask, you need to provide:
1. A clear description of the subtask.
2. Reasoning instructions that explain why this subtask is necessary and how it relates to the overall goal.
3. Any tool use instructions that may be relevant.
Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
''' 
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a sophisticated planner that divides a {task_type} task into several meaningful subtasks while considering the context and dependencies involved in the task. For each subtask, you need to provide:
1. A clear description of the subtask.
2. Reasoning instructions that explain why this subtask is necessary and how it relates to the overall goal.
3. Any tool use instructions that may be relevant.
Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
''' 
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan






class StructuredTaskPlanner():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]
    
    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]
        
        if feedback == '':
            prompt = '''You are a detailed planner who breaks down a {task_type} task into coherent and logical subtasks. For each subtask, provide a clear description of the action needed, the reasoning behind it, and any tools that may be utilized. Your output format should follow the example below.
    The following are some examples:
    Task: {example}

    Task: {task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a detailed planner who breaks down a {task_type} task into coherent and logical subtasks. For each subtask, provide a clear description of the action needed, the reasoning behind it, and any tools that may be utilized. Your output format should follow the example below.
    The following are some examples:
    Task: {example}

    end
    --------------------
    Reflexion:{feedback}
    Task:{task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        
        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan








class IterativePlanning():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]
    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]
        # The prompt words of the planning module
        if feedback == '':
            prompt = '''You are a thoughtful planner tasked with breaking down a {task_type} into clear, actionable sub-tasks. For each sub-task, provide detailed reasoning and anticipate any tools needed. Consider potential challenges and how to address them. Your output should follow the example below.
            The following are some examples:
            Task: {example}

            Task: {task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a thoughtful planner tasked with breaking down a {task_type} into clear, actionable sub-tasks. Based on the previous feedback, refine your approach to ensure clarity and effectiveness. For each sub-task, provide detailed reasoning and anticipate any tools needed. Consider potential challenges and how to address them. Your output should follow the example below.
            The following are some examples:
            Task: {example}

            end
            --------------------
            Reflexion:{feedback}
            Task:{task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan








class HierarchicalPlanning():
    def __init__(self, llms_type):
        # Initialization of the class, do not modify this part
        self.plan = []
        self.llm_type = llms_type[0]

    def __call__(self, task_type, task_description, feedback):
        # Assign few_shot based on the task type, do not modify this part.
        few_shot = planning_prompt[task_type]

        # The prompt words of the planning module, the difference between different modules is also mainly here.
        if feedback == '':
            prompt = '''You are a hierarchical planner who divides a {task_type} task into several subtasks. 
            Consider the dependencies between tasks and order them accordingly. 
            For each subtask, provide a description, reasoning instructions, and tool use instructions. 
            Your output format should follow the example below.
            The following are some examples:
            Task: {example}

            Task: {task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a hierarchical planner who divides a {task_type} task into several subtasks. 
            Consider the dependencies between tasks and order them accordingly. 
            For each subtask, provide a description, reasoning instructions, and tool use instructions. 
            Your output format should follow the example below.
            The following are some examples:
            Task: {example}

            end
            --------------------
            Reflexion:{feedback}
            Task:{task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

        # Invoke the large language model
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)

        # String parsing, do not modify this part.
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan






class ReasoningTest(ReasoningBase):
    pass
