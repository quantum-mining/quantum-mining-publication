import json

class Configuration:
    def __init__(self, filename):
        self.import_config(filename)

    def import_config(self, filename):
        with open(filename, 'r') as f:
            self.settings = json.load(f)
        self.problem_options = self.settings['problem_settings']
        self.solver_options = self.settings['solver_settings']

    def get_problem_options(self):
        return self.problem_options

    def get_solver_options(self):
        return self.solver_options

if __name__ == '__main__':
    config = Configuration('configs/config.json')

