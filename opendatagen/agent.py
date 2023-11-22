from openai import OpenAI
import os 
import json 
import pandas as pd 
import copy 

class DataAgent:

    initial_df:pd.DataFrame = None 
    data_frame:pd.DataFrame = None
    data_to_correct:str = None 
    initial_issue:str = None 
    current_row_to_correct:int = None 
    columns_to_analyse:str = None
    column_to_modify:str = None  
    
    current_correction:str = None 
    successful_conversation_list:list = None 
    good_examples:list = None 
    good_example_column:str = None

    start_line_to_analyse:int = None 
    last_line_to_analyse:int = None 
    specific_lines_to_analyse:list = None 


    system_prompt = """
        You are CSVGPT, a GPT specialized in evaluating and correcting CSV files. Your key functions involve:
        1. Requesting the CSV file path. (use ask_user_for_file_path then load_csv function)
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

    messages = [
            {"role":"system", "content":system_prompt},
        ]

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
                "description": "Load a CSV file into a DataFrame from the file path provided by the user. If the user provide only one line to process, you must return a value for start_line and end_line",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "The CSV file path"},
                        "start_line": {"type": "integer", "description": "The first CSV line where the user want to start the process"},
                        "end_line": {"type": "integer", "description": "The last CSV line where the user want to end the process."},
                        "delimiter": {"type": "string", "description": "The delimiter for the CSV file. Default is ','", "enum": [";", ","]},
                        "specific_lines": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                },
                                "description": "Must be an array of the CSV line index to process, if specified",
                            }
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
                            "column_to_correct": {"type": "string", "description": "Must be the column name to correct"},
                            "good_examples": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                },
                                "description": "Must be an array of the CSV line index where the issue is correctly handled.",
                            },
                            "good_example_column": {"type": "string", "description": "Must be the column name where the good examples are"},
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

    def load_csv(self, params:dict):
        """
        Load a CSV file into a DataFrame.
        """
        file_path_str = params.get("file_path")
        self.delimiter = params.get("delimiter", ",")

        #self.initial_df = pd.read_csv(file_path_str, delimiter=self.delimiter, header=None)
        
        try:
            
            self.data_frame = pd.read_csv(file_path_str, delimiter=self.delimiter)

            self.start_line_to_analyse = params.get("start_line", None)

            self.last_line_to_analyse = min(params.get("end_line", len(self.data_frame.index) + 1) , len(self.data_frame.index) + 1)

            self.specific_lines_to_analyse = params.get("specific_lines", None)

            self.csv_path = file_path_str

            return "CSV successfully loaded. Now let's ask the user about the specific evaluations or corrections they need."
        
        except Exception as e:

            print(e)
            return f"Error loading file: {e}. Please re-ask for the filepath." 
        
    def ask_user_for_file_path(self):
        """
        Ask the user about the CSV file path.
        """

        file_path = input("Please specify the file path of your CSV: ")

        delimiter_input = input("Please specify the delimiter (optional): ")

        process_detail = input("Please specify the lines you want to process. Line 1 is the header (optional): ")

        if delimiter_input.strip == "":
            delimiter_input = "The delimiter to use is ','."
        else: 
            delimiter_input = f"The delimiter to use is '{delimiter_input}'."

        if process_detail:

            m = f""" Here is the CSV file path: '{file_path}'
            And the CSV lines to process: {process_detail}.
            {delimiter_input}
            """

            return m
        
        else:
            
            m = f""" Here is the CSV file path: '{file_path}'.
            {delimiter_input}
            """

            return m
            
       
    def ask_user_for_precision(self, message):
        """
        Ask the user for precision
        """
        user_input = input(message)

        return user_input
        
    def ask_user_for_evaluation_criteria(self):
        """
        Ask the user about the specific evaluations or corrections they need.
        """

        user_input = input("Please specify the evaluations or corrections needed: ")

        self.initial_issue = user_input 

        m = ""

        while True:

            is_good_examples = input("Is there any rows where this issue is correctly handled (y/n)")

            if is_good_examples.lower() in ['y', 'n']:

                if is_good_examples.lower() == 'y':

                    good_examples_input = input("Please provide a line where the issue is correctly handled: ")

                    good_example_column_input = input("Please provide the column where the issue is correctly handled: ")
                
                    m = f"""Here is the evaluation and correction needed:
                        '{user_input}'

                        Here are the lines where the issue is correctly handled:
                        '{good_examples_input}'

                        Here is the column where good examples are:
                        '{good_example_column_input}'

                        Now let's identify issues with the function identify_issue
                        """
                    
                    break
                
                else:

                    m = f"""Here is the evaluation and correction needed:
                        '{user_input}'

                        There is no lines where the issue is correctly handle.

                        Now let's identify issues with the function identify_issue
                    """

                    break 

        
        return m

    def identify_issue(self, issue:dict):
        """
        Identify issues in the CSV data.
        """

        #issue_string = issue["issue_string"]
        self.columns_to_analyse = issue.get("columns_name", None)
        self.column_to_modify = issue.get("column_to_correct", None) 
        self.good_examples = issue.get("good_examples", None) 
        self.good_example_column = issue.get("good_example_column", None) 

        if self.good_examples:

            example_line = int(self.good_examples[0]) - 2
            
            example = self.data_frame.loc[example_line, self.good_example_column]
            
            issue_system_prompt = f"""
                The user has provided a general issue to correct in a CSV file. 
                Given this issue, your job is to detect if the issue occurs for the given CSV Data provided.
                Answer by explaining why you detect an issue or not and finish your explanation by 'issue detected' or 'issue not detected'.
                Here is one example where the issue is correctly handled:
                {example}
            """

        else: 

             issue_system_prompt = """
                The user has provided a general issue to correct in a CSV file. 
                Given this issue, your job is to detect if the issue occurs for the given CSV Data provided.
                Answer by explaining why you detect an issue or not and finish your explanation by 'issue detected' or 'issue not detected'.
            """
             
        indices = self.get_indices_to_analyze(start_line=self.start_line_to_analyse, last_line=self.last_line_to_analyse, specific_lines=self.specific_lines_to_analyse)

        # Iterate over each row in the DataFrame
        for index, row in self.data_frame.iterrows():
            
            adjusted_index = index + 2
            
            if adjusted_index in indices:

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

                    log_to_print = f"I have detected an issue at line {adjusted_index} of the CSV. Now proposing a correction."

                    print(log_to_print)

                    # Creating a copy of self.messages for recursive processing
                    messages_copy = copy.deepcopy(messages)
                    messages_copy.append({"role": "assistant", "content": log_to_print})

                    # Recursive call to run method with the copied messages
                    self.run(messages=messages_copy)

                    # After returning from the recursive call, continue processing the next lines
                    continue

                else:
                    
                    log_to_print = f"No issue detected for the line {adjusted_index}"

                    print(log_to_print)
 
        return "All the CSV row has been processed. End the conversation."

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
            self.data_frame.to_csv(self.csv_path, index=False, sep=self.delimiter)

            return "The data is corrected in the file. Great work. End of the conversation."
        
        else: 
            
            return "An error occured, please debug the process. End of the conversation."

  

    def askgpt(self, messages:list,functions:list = None, temperature:int = 0, max_tokens:int=512):
        
        param = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
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
    
    def run(self, messages=None):

        if messages is None:
            messages = self.messages

        #RUN AGENT 
        while True:

            completion = self.askgpt(messages, functions=self.functions) 

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

    def indices_to_ignore(self, dataframe, rows_to_keep):
        """
        Returns a list of indices to ignore for a given pandas DataFrame.

        :param dataframe: pandas DataFrame from which indices are derived.
        :param rows_to_keep: List of integer indices of rows that should be kept.
        :return: List of integer indices of rows to ignore.
        """
        all_indices = set(range(len(dataframe)))  # Generate a set of all row indices
        indices_to_keep = set(rows_to_keep)       # Convert rows_to_keep to a set for efficient removal
        return list(all_indices - indices_to_keep)  # Return the difference as a list

    def get_indices_to_analyze(self, start_line, last_line, specific_lines):
        indices_to_analyze = set()

        # Adjust for the offset in the loop
        offset = 0

        # Check if all parameters are None
        if start_line is None and last_line is None and specific_lines is None:
            # Add all indices adjusted by the offset
            indices_to_analyze.update(range(offset, len(self.data_frame) + offset))

        else:
            # Prioritize specific lines if specified
            if specific_lines is not None:
                adjusted_specific_lines = [line + offset for line in specific_lines]
                indices_to_analyze.update(adjusted_specific_lines)

            # If only start_line is specified
            elif start_line is not None and last_line is None:
                indices_to_analyze.add(start_line + offset)

            # If only last_line is specified
            elif start_line is None and last_line is not None:
                indices_to_analyze.update(range(offset, last_line + 1 + offset))

            # If both start_line and last_line are specified
            elif start_line is not None and last_line is not None:
                # Only add range if specific_lines is None
                if specific_lines is None:
                    indices_to_analyze.update(range(start_line + offset, last_line + 1 + offset))

            else:

                return indices_to_analyze

        return indices_to_analyze
