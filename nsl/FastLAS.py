import re
import uuid
import os
import subprocess
import sys
import time
from os.path import dirname, realpath


class FastLASSession:
    def __init__(self,
                 examples=None,
                 background_knowledge=None,
                 mode_declarations=None,
                 load_from_cache=False,
                 cached_lt=None):

        if not load_from_cache:
            self.learning_task = '%Examples\n'
            for e in examples:
                self.learning_task += e+'\n'

            self.learning_task += '\n% Background Knowledge\n' + background_knowledge\
                                  + '\n% Mode Declarations' + mode_declarations
        else:
            self.learning_task = cached_lt
        self.output_info = {}
        self.rules = ''

    def parse_output(self, output, learning_time):
        self.rules = output

        # Count number of rules/predicates for interpretability
        num_rules = len(self.rules.split('\n'))
        total_predicates = 0
        for rule in self.rules.split('\n'):
            if ':- ' in rule:
                rule = rule.split(':- ')[1]
            total_predicates += len(rule.split(';'))

        self.output_info = {
            "learning_time": float(learning_time),
            "interpretability": {
                "num_rules": num_rules,
                "total_predicates": total_predicates
            },
            "raw_output": output
        }
        return self.rules, self.output_info


class FastLASSystem:
    def __init__(self,
                 tmp_dir_name='_tmp_ILP_working_dir',
                 executable='FastLAS',
                 cmd_line_args=''):
        file_path = realpath(__file__)
        file_dir = dirname(file_path)
        parent_dir = dirname(file_dir)

        self.tmp_dir = os.path.join(parent_dir, tmp_dir_name)
        self.executable = executable
        self.cmd_line_args = cmd_line_args

    def run(self, session):
        # Save learning task to temporary file
        file_name = str(uuid.uuid1())
        file_input_name = file_name + '_input.las'
        file_input_path = os.path.join(self.tmp_dir, file_input_name)
        file = open(file_input_path, "w")
        file.write(session.learning_task)
        file.close()

        # Run using ILP System
        ilp_cmd = self.executable + ' ' + file_input_path + self.cmd_line_args

        start_time = time.time()
        result = subprocess.run(ilp_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True,
                                executable='/bin/bash')
        learning_time = time.time() - start_time
        output = result.stdout

        # Cleanup files
        os.remove(file_input_path)
        return session.parse_output(output.decode(), learning_time)
