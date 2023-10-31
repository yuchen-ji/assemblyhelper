import os
import re
import copy
import yaml
import openai
import gradio as gr

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
openai.api_base = "https://api.ai-yyds.com/v1"
openai.api_key = os.getenv(
    "OPENAI_KEY", default="sk-GM3AyFSCFHwbJdnC4c1a2637E4Bf4433AcFcAc8c3e976cFe"
)


class CodeGenerator:
    def __init__(
        self,
        role="robot",
        file_path=None,
        preprompt=None,
        model="gpt-3.5-turbo",
        oncecall=False,
    ):
        """
        通过file或者str初始化llm的prompt, file的优先级高
        """

        robot_role = "You are a desktop robotic arm with 6 degrees of freedom and a gripper end effector. You need to understand my actions/language and assist me in completing the assembly of the parts."
        scene_role = "You should act as an scene detector used to detect new scene observations after a desktop robot completes its actions."
        valid_role = "You need to act as a validator and answer the validation questions based on given robot code."

        if role == "robot":
            role = robot_role
        elif role == "scene":
            role = scene_role
        elif role == "valid":
            role = valid_role
        else:
            role = ""

        self.model = model
        self.file_path = file_path
        self.preprompt = preprompt
        self.system = [
            {
                "role": "system",
                "content": role,
            },
        ]
        self.history = copy.deepcopy(self.system)
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.preprompt = file.read()

        if oncecall:
            self.get_llm_response(user_input=self.preprompt)

    def get_llm_response(self, user_input=None):
        """
        获取llm的反馈api
        """
        if user_input:
            self.history.append({"role": "user", "content": user_input})

        completion = openai.ChatCompletion.create(
            model=self.model,       # model="gpt-3.5-turbo",
            messages=self.history,  # prompt
            temperature=0.2,        # 0~2, 数字越大越有想象空间, 越小答案越确定
            n=1,                    # 生成的结果数
            # top_p=0.1,            # 结果采样策略，0.1只采样前10%可能性的结果
            # presence_penalty=0,   # 主题的重复度 default 0, between -2 and 2. 控制围绕主题程度，越大越可能谈论新主题。
            # frequency_penalty=0,  # 重复度惩罚因子 default 0, between -2 and 2. 减少重复生成的字。
            # stream=False,
            # logprobs=1,           # Modify the likelihood of specified tokens appearing in the completion.
            # stop="\n"             # 结束字符标记
        )

        answer = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        # print(f"ChatGPT: {answer}")
        return answer

    def clear_history(self):
        """
        clear the history
        """
        print("CLEAR")
        self.history = copy.deepcopy(self.system)
        self.get_llm_response(user_input=self.preprompt)


def read_dataset(dataset_path, interval=5):
    with open(dataset_path, 'r') as file:
        lines = file.readlines()
    
    tasks = []
    tasks_name = []
    task = ""
    for i in range(len(lines)):
        if i % interval == 0:
            tasks_name.append(lines[i])
        if i % interval in [1, 2, 3]:
            task += lines[i]
        if i % interval == 3:
            tasks.append(task)
            task = ""
    
    return tasks, tasks_name
         
         
         
if __name__ == '__main__':   
    
    interval = 5
    dataset_path = 'eval/experiments/dataset_hard.yml'
    prompt_path = "eval/experiments/prompts/cot_1shot_comment.yml"
    result_path = 'eval/experiments/feedback/gpt4_origin.yml'

    tasks, tasks_name = read_dataset(dataset_path, interval)
    print(tasks[1])
    
    # codeg = CodeGenerator(role="robot",file_path=prompt_path, model="gpt-3.5-turbo", oncecall=True)
    codeg = CodeGenerator(role="robot",file_path=prompt_path, model="gpt-4-0613", oncecall=True)
    
    for task, task_name in zip(tasks[:], tasks_name[:]):    
        result = task_name
        answer = codeg.get_llm_response(task)
        answer = re.sub(r'\n\s*\n', '\n', answer)
        
        result += task + answer + '\n' + '\n'
        with open(result_path, "a") as f:
            f.write(result)
            
        codeg.clear_history()
    

    # easy_35 = [2, 9, 12, 14, 15, 18, 23, 25]
    # middle_35 = [12, 13, 15, 16, 17, 19, 20, 21, 22, 24]
    # hard_35 = [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # codeg = CodeGenerator(role="scene",file_path="src/workspaces/scene_description_prompt.yml", model="gpt-3.5-turbo", oncecall=True)
    # codeg = CodeGenerator(role="scene",file_path="src/workspaces/scene_description_prompt.yml", model="gpt-4-0613", oncecall=True)

    # codeg = CodeGenerator(role="robot",file_path="eval/prompts/robot_prompt_update9.yml", model="gpt-3.5-turbo", oncecall=True)
    # codeg = CodeGenerator(role="robot",file_path="eval/prompts/robot_prompt_update9.yml", model="gpt-4-0613", oncecall=True)

    # codeg = CodeGenerator(role="valid", file_path="eval/prompts/validation_prompt.yml", model="gpt-3.5-turbo", oncecall=True)
    # codeg = CodeGenerator(role="valid", file_path="eval/prompts/validation_prompt.yml", model="gpt-4-0613", oncecall=True)

    # codeg = CodeGenerator()

