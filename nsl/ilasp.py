import re
import uuid
import os
import subprocess
import sys
from os.path import dirname, realpath


# Helper to determine if running inside a virtualenv, if so deactivate first to avoid clashing with ILASP
def is_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


class ILASPSession:
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

    def parse_output(self, output):
        self.rules = output.split('\n\n%')[0]
        # Parse output information
        output_info = '%' + output.split('\n%')[-2]
        match = re.search('% Total\s+:\s(.+)s', output_info)
        learning_time = match.group(1)

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


class ILASPSystem:
    def __init__(self,
                 tmp_dir_name='_tmp_ILP_working_dir',
                 executable='ILASP',
                 cmd_line_args='--restarts --strict-types -nc',
                 run_with_pylasp=True):
        file_path = realpath(__file__)
        file_dir = dirname(file_path)
        parent_dir = dirname(file_dir)

        self.tmp_dir = os.path.join(parent_dir, tmp_dir_name)
        self.pylasp_script_path = os.path.join(os.path.join(parent_dir, 'nsl'), 'pylasp_run_script.py')
        self.executable = executable
        self.cmd_line_args = cmd_line_args
        self.run_with_pylasp = run_with_pylasp

        # When not running with pylasp, add version=4 to the cmd line args.
        if not self.run_with_pylasp:
            self.pylasp_script_path = '--version=4'

    def run(self, session):
        # Save learning task to temporary file
        file_name = str(uuid.uuid1())
        file_input_name = file_name + '_input.las'
        file_input_path = os.path.join(self.tmp_dir, file_input_name)
        file = open(file_input_path, "w")
        file.write(session.learning_task)
        file.close()

        # Run using ILP System
        # deactivate to remove virtualenv from path to avoid clashing with ILASP.
        ilp_cmd = self.executable + ' ' + file_input_path + ' ' + self.pylasp_script_path + ' ' + self.cmd_line_args
        print(ilp_cmd)
        if is_venv():
            ilp_cmd = 'deactivate && '+ilp_cmd

        result = subprocess.run(ilp_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True,
                                executable='/bin/bash')
        output = result.stdout

        # Cleanup files
        os.remove(file_input_path)
        return session.parse_output(output.decode())
