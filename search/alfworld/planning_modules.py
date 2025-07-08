import re
import ast
from utils import llm_response
from planning_prompt import *

class PlanningBase():
    def __init__(self, llms_type):
        self.plan = []
        self.llm_type = llms_type[0]
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        # raise NotImplementedError("Subclasses should implement this method")
        pass
    
    def __call__(self, task_type, task_description, feedback):
        few_shot = planning_prompt[task_type]
        prompt = self.create_prompt(task_type, task_description, feedback, few_shot)
        string = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1)
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        dicts = [ast.literal_eval(ds) for ds in dict_strings]
        self.plan = dicts
        return self.plan
    
class PlanningIO(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class PlanningDEPS(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class PlanningTD(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format must follow the example below, specifying the order and dependencies.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format should follow the example below, specifying the order and dependencies.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

class PlanningVoyager(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a  list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
reflexion:{feedback}
task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

class PlanningOPENAGI(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.  
Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.
Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

class PlanningHUGGINGGPT(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)



    class EnhancedSelfRefine():
        # Initialization of the class, do not modify this part
        def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
            self.feedback = ''
            self.profile_type_prompt = profile_type_prompt
            self.memory = memory
            self.llm_type = llms_type[0]
            self.tooluse = tooluse
            self.task_name_cache = None
        def __call__(self, task_description: str, tool_instruction :str='', feedback :str=''):
            # Call tools use modules and memory modules, do not modify this part
            task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)        
            if self.memory is not None:
                if self.task_name_cache is not None and self.task_name_cache == task_name:
                    pass
                else:
                    self.task_name_cache = task_name
                    self.memory_cache = self.memory(task_description)
            else:
                self.memory_cache = ''
            if self.tooluse is not None:
                tooluse = self.tooluse(task_description, tool_instruction)
            else:
                tooluse = ''
            # Split into two parts, on is the task solution track example, the other is the current task
            split_text = task_description.rsplit('You are in the', 1)
            # task solution track examples
            examples = split_text[0]
            # current task
            task_description = 'You are in the' + split_text[1]
            # Step 1: Generate initial reasoning step
            prompt_step = '''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.\nHere are some examples.\n{examples}{memory}\nHere is the task:\n{task_description}'''
            reasoning_result = llm_response(prompt=prompt_step.format(task_description=task_description, examples=examples, memory=self.memory_cache), model=self.llm_type, temperature=0.1)
            # Step 2: Evaluate reasoning process and refine
            feedback_prompt = f'''You need to check that the reasoning step meets the requirements.\nStep: {reasoning_result}\nExamples of correct steps include: 'go to cabinet 1', 'put object 1 in/on surface 1, end'.\nIs the step correct? If not, please revise it.'''  
            feedback_result = llm_response(prompt=feedback_prompt, model=self.llm_type, temperature=0.0)
            # Step 3: Process feedback and ensure final output meets format
            if 'correct' in feedback_result.lower():
                final_result = reasoning_result.replace('.', '')
            else:
                final_result = feedback_result.split(':')[-1].strip()  
            return final_result




class SelfEval():
    # Initialization of the class, do not modify this part
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None

    def __call__(self, task_description: str, tool_instruction: str='', feedback: str=''):
        # Call tools use modules and memory modules, do not modify this part
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''

        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''

        # Split into two parts, one is the task solution track example, the other is the current task
        split_text = task_description.rsplit('You are in the', 1)
        examples = split_text[0]
        task_description = 'You are in the' + split_text[1]

        # Generate multiple potential next steps
        potential_steps = []
        for i in range(3):  # Generate 3 potential next steps
            prompt = '''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''
            prompt = prompt.format(task_description=task_description, examples=examples, memory=self.memory_cache)
            potential_steps.append(llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n']))

        # Evaluate potential steps against previous successful steps
        best_step = self.evaluate_steps(potential_steps, examples)

        return best_step

    def evaluate_steps(self, steps, examples):
        # Evaluate the generated steps against the historical successful examples
        # Implement logic to determine which of the steps is most aligned with past successful actions
        # This could involve checking for specific keywords or structures that match previous successful commands
        best_step = steps[0]  # Placeholder for the best step logic
        return best_step






class FeedbackEnhancedSelfRefine():
    # Initialization of the class, do not modify this part
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None
        self.action_history = []  # Store historical actions and feedback

    def __call__(self, task_description: str, tool_instruction: str = '', feedback: str = ''):
        # Call tools use modules and memory modules, do not modify this part
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''
        # Split into two parts, one is the task solution track example, the other is the current task
        split_text = task_description.rsplit('You are in the', 1)
        # Task solution track examples
        examples = split_text[0]
        # Current task
        task_description = 'You are in the' + split_text[1]
        # Execute action
        action = self.perform_action(task_description)  # Logic for determining the next action
        # Evaluate action performance
        feedback_result = self.evaluate_action(action)  # Logic for checking if the action was successful
        self.action_history.append((action, feedback_result))  # Store the action and its feedback
        # Generate output based on refined learning
        refined_output = self.refine_output(action, feedback_result)
        return refined_output

    def perform_action(self, task_description):
        prompt = f'''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{self.memory_cache}
Here is the task:
{task_description}'''
        action_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
        return action_result.strip()

    def evaluate_action(self, action):
        # Logic to evaluate the action's effectiveness
        evaluation_prompt = f'Was the action "{action}" effective? Explain why.'
        evaluation_result = llm_response(prompt=evaluation_prompt, model=self.llm_type, temperature=0.1)
        return evaluation_result.strip()

    def refine_output(self, action, feedback_result):
        if 'error' in feedback_result.lower():
            # Logic to refine the action based on feedback
            return f'Error detected: {feedback_result}. Consider revising action.'
        return action






class COTSelfRefineHybrid():
    # Initialization of the class, do not modify this part
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None
    
    def __call__(self, task_description: str, tool_instruction :str='', feedback :str=''):
        # Call tools use modules and memory modules, do not modify this part
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''
        # Split into two parts, on is the task solution track example, the other is the current task
        split_text = task_description.rsplit('You are in the', 1)
        # task solution track examples
        examples = split_text[0]
        # current task
        task_description = 'You are in the' + split_text[1]
        
        # Generate reasoning output using COT
        reasoning_prompt = '''Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{memory}
Here is the task:
{task_description}'''.format(examples=examples, memory=self.memory_cache, task_description=task_description)
        reasoning_result = llm_response(prompt=reasoning_prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
        
        # Refinement process
        refinement_prompt = f'''You need to check that the syntactic structure of the step meets the requirements.
requirements: '1. take a from b 2. go to a 3. open a 4. put a in/on b, end. 5. clean a with b, end.', where 'a' and 'b' are variable.
step: {reasoning_result}'''
        feedback_result = llm_response(prompt=refinement_prompt, model=self.llm_type, temperature=0.0)
        if 'correct' in feedback_result.lower():
            return reasoning_result.replace('.', '')
        else:
            return feedback_result.split(':')[-1].replace('.', '').strip()






class HybridCOTSelfRefine():
    # Initialization of the class, do not modify this part
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None
    def __call__(self, task_description: str, tool_instruction :str='', feedback :str=''):
        # Call tools use modules and memory modules, do not modify this part
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)        
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''
        # Split into two parts, on is the task soleution track example, the other is the current task
        split_text = task_description.rsplit('You are in the', 1)
        examples = split_text[0]
        task_description = 'You are in the' + split_text[1]
        
        # Construct the prompt for reasoning
        prompt = f'''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.\nHere are some examples.\n{examples}{self.memory_cache}\nHere is the task:\n{task_description}'''
        reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
        
        # Refinement process to ensure command structure
        if 'think' not in reasoning_result:
            refinement_prompt = f'''You need to check if the output follows the required command format.\nOutput: {reasoning_result}\nCheck and refine if necessary.\n'''  
            feedback_result = llm_response(prompt=refinement_prompt, model=self.llm_type, temperature=0.0)
            reasoning_result = feedback_result if 'error' in feedback_result else reasoning_result
        return reasoning_result






class StructuredRefine():
    # Initialization of the class, do not modify this part
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None
    def __call__(self, task_description: str, tool_instruction :str='', feedback :str=''):
        # Call tools use modules and memory modules, do not modify this part
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''
        # Split into two parts, on is the task solution track example, the other is the current task
        split_text = task_description.rsplit('You are in the', 1)
        # task solution track examples
        examples = split_text[0]
        # current task
        task_description = 'You are in the' + split_text[1]
        
        prompt = f'''Follow the structured approach to solve the task step by step. Pay attention to the examples provided to guide your reasoning. 
Here are some examples:
{examples}{self.memory_cache}Here is the task:
{task_description}'''
        reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
        # Refining the output based on the correct structure
        refined_result = self.refine(reasoning_result)
        return refined_result
    def refine(self, reasoning_result):
        if 'think' in reasoning_result:
            return reasoning_result
        prompt = f'''Evaluate the following step according to the task structure requirements. 
Requirements: '1. take a from b 2. go to a 3. open a 4. put a in/on b, end. 5. clean a with b, end.'
Step: {reasoning_result}
Please indicate if it is correct or provide a revised step. Output in the format: 'correct' or 'error, revised: your step'.'''
        feedback_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.0)
        if 'correct' in feedback_result.lower():
            return reasoning_result.replace('.', '')
        else:
            return feedback_result.split(':')[-1].replace('.', '').strip()






class ContextAwareFeedback():
    # Initialization of the class, do not modify this part
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None

    def __call__(self, task_description: str, tool_instruction: str='', feedback: str=''):
        # Call tools use modules and memory modules, do not modify this part
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)        
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''
        # Split into two parts, on is the task soleution track example, the other is the current task
        split_text = task_description.rsplit('You are in the', 1)
        # task solution track examples
        examples = split_text[0]
        # currect task
        task_description = 'You are in the' + split_text[1]

        # Constructing the prompt with an emphasis on task context
        prompt = f'''Evaluate your next action based on the task goal. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples:
{examples}{self.memory_cache}
Here is the task:
{task_description}

Consider the overall goal of the task and ensure your response logically leads to that goal. You may provide feedback or revise your action if necessary. 
'''  

        # Call the LLM with the prompt
        reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])

        return reasoning_result








class ReflectiveIterativeReasoning():
    def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
        self.feedback = ''
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm_type = llms_type[0]
        self.tooluse = tooluse
        self.task_name_cache = None

    def __call__(self, task_description: str, tool_instruction: str='', feedback: str=''):
        task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)
        if self.memory is not None:
            if self.task_name_cache is not None and self.task_name_cache == task_name:
                pass
            else:
                self.task_name_cache = task_name
                self.memory_cache = self.memory(task_description)
        else:
            self.memory_cache = ''
        if self.tooluse is not None:
            tooluse = self.tooluse(task_description, tool_instruction)
        else:
            tooluse = ''

        split_text = task_description.rsplit('You are in the', 1)
        examples = split_text[0]
        task_description = 'You are in the' + split_text[1]

        # Creating a prompt that encourages reflective reasoning
        prompt = f'''Reflect on your previous actions and the current task. Consider what has worked in the past and how you can improve.
Here are some examples:
{examples}{self.memory_cache}
Your task is to:
{task_description}'''

        # Get reasoning result from LLM
        reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])

        # Implement feedback loop to refine reasoning if necessary
        if feedback:
            refined_prompt = f'''Given the previous reasoning step: {reasoning_result}, and the feedback: {feedback}, refine your reasoning to ensure it aligns with task requirements.''' 
            refined_result = llm_response(prompt=refined_prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
            return refined_result

        return reasoning_result








    class HybridSelfRefine():
        # Initialization of the class, do not modify this part
        def __init__(self, profile_type_prompt, memory, tooluse, llms_type):
            self.feedback = ''
            self.profile_type_prompt = profile_type_prompt
            self.memory = memory
            self.llm_type = llms_type[0]
            self.tooluse = tooluse
            self.task_name_cache = None

        def __call__(self, task_description: str, tool_instruction :str='', feedback :str=''):
            # Call tools use modules and memory modules, do not modify this part
            task_name = re.findall(r'Your task is to:\s*(.*?)\s*>', task_description)        
            if self.memory is not None:
                if self.task_name_cache is not None and self.task_name_cache == task_name:
                    pass
                else:
                    self.task_name_cache = task_name
                    self.memory_cache = self.memory(task_description)
            else:
                self.memory_cache = ''
            if self.tooluse is not None:
                tooluse = self.tooluse(task_description, tool_instruction)
            else:
                tooluse = ''
            # Split into two parts, one is the task solution track example, the other is the current task
            split_text = task_description.rsplit('You are in the', 1)
            # task solution track examples
            examples = split_text[0]
            # current task
            task_description = 'You are in the' + split_text[1]
            # Step 1: Attempt to reason through the task
            prompt = f'''Solve the task step by step. Interact with a household to solve a task. Your instructions must follow the examples.
Here are some examples.
{examples}{self.memory_cache}
Here is the task:
{task_description}'''
            reasoning_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.1, stop_strs=['\n'])
            # Step 2: Refine the result if necessary
            reasoning_result = self.refine(reasoning_result)
            return reasoning_result

        def refine(self, reasoning_result):
            if 'think' in reasoning_result:
                return reasoning_result
            prompt = f'''You need to check that the syntactic structure of the step meets the requirements.
requirements: '1. take a from b 2. go to a 3. : open a 4. put a in/on b, end. 5. clean a with b, end. 6. heat a with b, end. 7. cool a with b, end. 8. use a, end.', where 'a' and 'b' are variable.
examples:
take pencil 1 from desk 2   correct
take potato 1 with fridge 1 error, The preposition with take is from. revised: take potato 1 from fridge 1
...
step: {reasoning_result}
You can only output in two formats:
"correct" or "error, revised: your step"
'''    
            feedback_result = llm_response(prompt=prompt, model=self.llm_type, temperature=0.0)
            if 'correct' in feedback_result.lower():
                return reasoning_result.replace('.', '')
            else:
                return feedback_result.split(':')[-1].replace('.', '').strip()





class PlanningTest(PlanningBase):
    pass
