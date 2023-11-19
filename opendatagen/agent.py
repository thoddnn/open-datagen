from openai import OpenAI
import os 
import json 
import pandas as pd 

class DataAgent:

    data_frame:pd.DataFrame = None
    data_to_correct:str = None 
    initial_issue:str = None 
    current_row_to_correct:int = None 
    columns_to_analyse:str = None
    column_to_modify:str = None  
    
    current_correction:str = None 
    successful_conversation_list:list = None 

    system_prompt = """
        You are CSVGPT, a GPT specialized in evaluating and correcting CSV files. Your key functions involve:
        1. Requesting the CSV file path. (use ask_user_for_file_path then load_csv)
        2. Asking users about the specific evaluations or corrections they need. (use ask_user_for_evaluation_criteria)
        3. Identifying issues. (use identify_issue)
        4. Confirming detected issues and proposed corrections with users. No verbose. (use confirm_and_propose_corrections)
        5. Applying corrections across similar lines after obtaining user consent. (use apply_corrections)
        6. Focusing on data accuracy and maintaining user control in the process.

        Process step by step.

        If you need precision please ask. (use ask_user_for_precision)

        You communicate in a professional yet approachable tone, making technical concepts accessible without being overly casual.
        This balance ensures clarity and fosters a positive user experience.
        In situations of ambiguity, you actively seek clarifications to provide precise assistance.
        Your interactions are always consent-driven, emphasizing clarity and user preferences.
    """

    functions = [
            {
                
                "name": "ask_user_for_precision",
                "description": "Ask the user for precision",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "message": {"type": "string", "description": "Simple sentence to ask for precision from the user"}
                    },
                    "required": ["message"]
                }
            
            },

             {
                
                "name": "ask_user_for_file_path",
                "description": "Ask the user for the CSV file path",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            },
            {
                
                "name": "load_csv",
                "description": "Load a CSV file into a DataFrame from the file path provided by the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "file_path": {"type": "string", "description": "The CSV file path"}
                    },
                    "required": ["file_path"]
                }
            
            },
            {
                "name": "ask_user_for_evaluation_criteria",
                "description": "Ask the user about the specific evaluations or corrections they need.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                
                "name": "identify_issue",
                "description": "Identify issues in the CSV data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                            "columns_name": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Must be an array of the column's name to analyse.",
                            },
                            "column_to_correct": {"type": "string", "description": "Must be the column name to correct"}
                        },
                        "required": ["columns_name", "column_to_correct"]
                }
            
            },
            {
                
                "name": "confirm_and_propose_corrections",
                "description": "Correct issues in the CSV data.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                        "required": []
                }
            
            },
            {
                
                "name": "apply_corrections",
                "description": "Apply corrections to the DataFrame based on user consent.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            },
        
    ]

    client = OpenAI()

    #gpt-4-1106-preview
    def __init__(self, model_name="gpt-4-1106-preview"):
        self.model_name = model_name
        self.client.api_key = os.environ.get("OPENAI_API_KEY")

    def load_csv(self, file_path:dict):
        """
        Load a CSV file into a DataFrame.
        """

        file_path = file_path.get("file_path")
        
        try:

            self.data_frame = pd.read_csv(file_path)

            self.csv_path = file_path

            return "CSV successfully loaded."
        
        except Exception as e:
            return f"Error loading file: {e}"
        
    def ask_user_for_file_path(self):
        """
        Ask the user about the CSV file path.
        """
        user_input = "Here is the CSV file path:" + input("Please specify the file path of your CSV:")

        return user_input
    
    def ask_user_for_precision(self, message):
        """
        Ask the user for precision
        """
        user_input = input(message)

        return user_input
        
    def ask_user_for_evaluation_criteria(self):
        """
        Ask the user about the specific evaluations or corrections they need.
        
        Returns:
        str: User input regarding the required evaluations or corrections.
        """

        user_input = input("Please specify the evaluations or corrections needed: ")

        self.initial_issue = user_input 

        return user_input + ". Now let's identify issues"

    def identify_issue(self, issue:dict):
        """
        Identify issues in the CSV data.
        """

        #issue_string = issue["issue_string"]
        self.columns_to_analyse = issue["columns_name"]
        self.column_to_modify = issue["column_to_correct"]
        
        """
        issue_system_prompt = 
            You are DataDetectiveGPT. Your role is to analyze a diverse range of data types. The focus of the analysis will be directed by the user's input.
            Based on this input, you must determine if an issue exists within the provided data, responding exclusively with either 'issue identified' or 'issue not identified'.
            This requires careful examination of the data details provided by the user to accurately ascertain the presence or absence of issues.
        """

        issue_system_prompt = """
            The user has provided a general issue to correct in a CSV file. 
            Given this issue, your job is to detect if the issue occurs for the given CSV Data provided.
            Answer by explaining why you detect an issue or not and finish your explanation by 'issue detected' or 'issue not detected'.
        """

        #header_line = "|| ".join(self.columns_to_analyse)
        
        # Iterate over each row in the DataFrame
        for index, row in self.data_frame.iterrows():

            # Iterate over each column in the row
            #csv_line = "|| ".join([f"{col}: {row[col]}" for col in self.columns_to_analyse if col in row])
            #csv_line = row.to_string()
            csv_data_str = "\n\n".join([f"{col} value:\n'''\n{row[col]}\n'''" for col in self.columns_to_analyse if col in row])


            issue_user_prompt = f"""
                                    Issue: 
                                    '''{self.initial_issue}'''
                                    
                                    CSV Data: 
                                    {csv_data_str}
                                    
                                """

            messages = [

                {"role":"system", "content":issue_system_prompt},
                {"role":"user","content": issue_user_prompt}

            ]

            completion = self.askgpt(messages, max_tokens=2024, functions=None)

            answer = completion.choices[0].message.content

            if "issue detected" in answer.lower():
                
                self.data_to_correct = csv_data_str

                self.current_row_to_correct = index

                return f"I have detected an issue at the line {index} of the CSV. Now propose a correction"
            
            else:

                log_to_print = f"No issue detected for the line {index}"

                print(log_to_print)


        return "I haven't detected any issue. End the conversation."

    def confirm_and_propose_corrections(self):
        """
        Confirm detected issues with the user and propose corrections.
        """

        """
        confirm_and_correct_system_prompt = 
            You are DataCorrectorGPT. DataCorrectorGPT is equipped to handle a wide range of data types and errors, as specified by the user.
            It is adept at analyzing various data formats and understanding different kinds of errors that can occur in data sets, such as formatting mistakes, missing values, statistical inaccuracies, or inconsistencies.
            The GPT will rely on user input to identify the specific type of data and the nature of the error, then use its expertise to suggest appropriate corrections or modifications.
            This broad capability allows it to be versatile and responsive to diverse user needs in data correction.

        """

        confirm_and_correct_system_prompt = """ 
            Correct the issue declared by the user.
        """

        confirm_and_correct_user_prompt = f""" 
            Issue detected and confirmed by the user: 
            {self.initial_issue}

            Data to correct: '''
            {self.data_to_correct}
            '''

        """


        messages = [

            {"role":"system", "content":confirm_and_correct_system_prompt},
            {"role":"user","content": confirm_and_correct_user_prompt}

        ]

        completion = self.askgpt(messages)

        answer = completion.choices[0].message.content

        while True:

            user_input = input(answer + "\n\n Do you confirm the correction ? (y/n)").strip().lower()
            if user_input in ['y', 'n']:

                if user_input == "y":

                    non_verbose_answer = self.extract_answer_from_verbose_answer(verbose_answer=answer)

                    self.current_correction = non_verbose_answer

                    message_to_return = f"""
                        Here are the correction you have provided:
                        '''
                        {non_verbose_answer}
                        '''

                        That's great! The answer is correct now let's apply change to the corresponding row.
                    """

                    return message_to_return

                else:

                    debug_input = input("Please provide precision on why the answer is not good.")

                    return f"No the answer provided is not correct '''{answer}'''  because '{debug_input}'. Please submit another correction."

            else:

                print("Please answer with 'y' for yes or 'n' for no.")



    def extract_answer_from_verbose_answer(self, verbose_answer:str):

        old_answer = self.data_frame.loc[self.current_row_to_correct, self.column_to_modify]

        extract_answer_system_prompt = """ 
        Given the old answer and the new answer provided by the user. 
        You must rewrite the new answer with the same format as the old. 

        Example:

        Old answer: 
        '''
        {"sentence":"Hello I'm Bryan"}
        '''

        Verbose answer:
        '''
        Based on the issue you have detected the answer is
        {"sentence":"Hello I'm Thomas"}
        '''

        New answer:
        {"sentence":"Hello I'm Thomas"}
            
        """

        extract_answer_user_prompt = f"""

            Old answer:
            '''
            {old_answer}
            '''

            Verbose answer:
            '''
            {verbose_answer}
            '''

            New answer:
        
        """
    
        messages = [

            {"role":"system", "content":extract_answer_system_prompt},
            {"role":"user","content": extract_answer_user_prompt}

        ]

        completion = self.askgpt(messages)

        answer = completion.choices[0].message.content

        return answer 


    def apply_corrections(self):

        """
        Apply corrections to the DataFrame based on user consent.
        """

        if self.current_correction is not None and self.column_to_modify is not None and self.current_row_to_correct is not None:

            self.data_frame.loc[self.current_row_to_correct, self.column_to_modify] = self.current_correction
            self.data_frame.to_csv(self.csv_path, index=False)

            return "The data is corrected in the file. Great work. End of the conversation."
        
        else: 
            
            return "An error occured, please debug the process. End of the conversation."

  

    def askgpt(self, messages:list,functions:list = None, temperature:int = 0, max_tokens:int=512):
        
        param = {
            "model": self.model_name,
            "messages": messages
        }

        if functions:
            param["functions"] = functions 

        completion = self.client.chat.completions.create(**param)

        return completion

    def function_to_call(self, parameter_to_pass, function_to_call):

        #parameter_to_pass = json.loads(completion.choices[0].message.function_call.arguments)
        #function_to_call = completion.choices[0].message.function_call.name

        func = getattr(self, function_to_call)

        if len(parameter_to_pass) == 0:
            result = func()
        else:
            result = func(parameter_to_pass)

        return result
    
    def run(self):

        #RUN AGENT 
        messages = [
            {"role":"system", "content":self.system_prompt},
        ]
        
        while True:

            completion = self.askgpt(messages, functions=self.functions) 

            print(completion)
            
            if completion.choices[0].finish_reason == "function_call":

                #answer = completion.choices[0].message.content

                parameters = json.loads(completion.choices[0].message.function_call.arguments)
                function_name = completion.choices[0].message.function_call.name
                
                result = self.function_to_call(parameters, function_name)

                #messages.append({"role":"assistant", "content": answer})
                messages.append({"role":"user", "content":result})

            elif completion.choices[0].finish_reason == "stop":

                print(completion.choices[0].message.content)
                print("END OF CONVERSATION")
                break

            else:

                print(completion.choices[0].message.content)
                print("END OF CONVERSATION")
                break 
