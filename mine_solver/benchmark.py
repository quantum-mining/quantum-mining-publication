import matplotlib.pyplot as plt
from config import Configuration
from problem import OpenPitMiningProblem
from solver import VariationalSolver
from solver import ExactSolver
from utils import *
import seaborn as sns
import random
import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
sns.set()

class Benchmark():
    """
    Benchmark results

    Attributes:
        config_filename (string): the configuration filename
        verbose (bool): if True, print progress
    """
    
    def __init__(self, config_filename, verbose=True):
        """
        Initializes the Benchmark with the attributes.
        """
        self.verbose = verbose
        self.config_filename = config_filename
        self.configuration = Configuration(self.config_filename)
        self.problem = OpenPitMiningProblem(self.configuration.get_problem_options())
        self.solver_options = self.configuration.get_solver_options()
        
    def run_benchmark_for_optimizers(self, mean_iter, file_plots, file_results):
        """
        Run the solver for different circuit ansatzes
        
        Args:
            mean_iter (int): the number of iterations for finding the average loss
        """

        optimizer_options = ['SPSA','COBYLA','NELDER_MEAD', 'CG', 'L-BFGS-B']  
        #optimizer_options = ['AQGD','SPSA']
        f=open(file_results, "a+")
        colours = {"SPSA": 'blue', "COBYLA": 'orange', "AQGD": 'red', "NELDER_MEAD": 'yellow', 'CG': 'cyan',
                   'L-BFGS-B': 'magenta'}
        for optimizer in optimizer_options:
            optimizer_loss = []
            for i in range(mean_iter):
                if self.verbose:
                    print("Optimizer: " + optimizer + ", Iteration: " + str(i))
                # Change the ansatz of the solver by creating a local copy of the solver options
                local_solver_options = dict(self.solver_options)
                local_solver_options['optimizer'] = optimizer
                # Create the solver
                solver = VariationalSolver(self.problem, local_solver_options)
                # Run the solver
                result = solver.solve(verbose=self.verbose, plot=False, visualize=False)
                # Get the loss data results
                loss_data = result.get_loss_data()
                optimizer_loss.append(loss_data)
            # Find the average of the loss data across all the iterations
            optimizer_mean = np.mean(optimizer_loss, axis=0)
            # Find the standard deviation of the loss data across all the iterations
            optimizer_error = np.std(optimizer_loss, axis = 0) / np.sqrt(mean_iter)
            # Plot the data
            f.write("Results for " + optimizer + "\n")
            f.write(str(optimizer_loss)+"\n \n")
            self.plot(optimizer_loss, optimizer_mean, optimizer_error, optimizer, colours)
        f.close()
        # Show the plot
        self.show_plot(file_plots=file_plots)
        
    def run_benchmark_for_ansatz(self, mean_iter, filename):
        """
        Run the solver for different circuit ansatzes
        
        Args:
            mean_iter (int): the number of iterations for finding the average loss
        """
        # The different possible ansatz options
        ansatz_options = ["undug","dug","superposition"]
        # Generate random initial parameters for the circuit
        random_parameters = [[random.random()*2*np.pi for x in range(60)] for x in range(mean_iter)]
        f=open(filename, "a+")
        for ansatz in ansatz_options:
            ansatz_loss = []
            # Average out over the random parameters
            for i in range(mean_iter):
                if self.verbose:
                    print("Ansatz: " + ansatz + ", Iteration: " + str(i))
                # Change the ansatz of the solver by creating a local copy of the solver options
                local_solver_options = dict(self.solver_options)
                local_solver_options['ansatz'] = ansatz
                # Consistently iterate over the parameters
                parameters = random_parameters[i]
                # Create the solver
                solver = VariationalSolver(self.problem, local_solver_options, parameters)
                # Run the solver
                result = solver.solve(verbose=self.verbose, plot=False, visualize=False)
                # Get the loss data results
                loss_data = result.get_loss_data()
                ansatz_loss.append(loss_data)
            # Find the average of the loss data across all the iterations
            ansatz_mean = np.mean(ansatz_loss, axis=0)
            # Find the standard deviation of the loss data across all the iterations
            ansatz_error = np.std(ansatz_loss, axis = 0) / np.sqrt(mean_iter)
            # Plot the data
            f.write("Results for " + ansatz + "\n")
            f.write(str(ansatz_loss)+"\n \n")
            self.plot(optimizer_loss, optimizer_mean, optimizer_error, optimizer)
        f.close()
        # Show the plot
        self.show_plot()
        
    def run_on_hardware(self, file_plots, file_results, file_params_results):
        """
        Run on hardware
        
        """
        f=open(file_results, "a+")
        f_params = open(file_params_results, "a+")
        e_solver = ExactSolver(self.problem, self.solver_options)
        e_result = e_solver.solve(verbose=True)
        print(e_result)
        e_loss = e_result.get_final_loss()
        f.write("Results report for exact solver: \n \n")
        f.write("Seconds to solve: " + str(e_result.elapsed_time) + "\n \n")
        f.write("Final loss: " + str(e_result.min_val) + "\n \n")
        f.write("Final Configuration: ")
        f.write(str(e_result.grid.tolist()))
        f.write("\n \n")
        backend_options = ["simulator", "hardware", "hardware_with_em"]
        #backend_options = ["simulator"]
        for backend_option in backend_options:
            print("Running on :", backend_option)
            solver = VariationalSolver(self.problem, self.solver_options)
            # Run the solver
            f.write("Individual losses  for " + backend_option + "\n \n")
            result = solver.solve(verbose=self.verbose, backend_option=backend_option, filename=f, params_filename=f_params)
            loss_data = result.get_loss_data()
            f.write("\n \n Results report for " + backend_option + ": \n \n")
            f.write("Loss data: " + str(loss_data)+"\n \n")
            f.write("Seconds to solve: " + str(result.elapsed_time) + "\n \n")
            f.write("Final loss: " + str(result.min_val) + "\n \n")
            f.write("Optimal Parameters: " + str(result.opt_params.tolist()) + "\n \n")
            f.write("Final Configuration: ")
            f.write(str(result.grid.tolist()))
            f.write("\n \n")
            self.simple_plot(loss_data, e_loss, backend_option)
            self.show_plot(file_plots)
        f.close()
        
    def simple_plot(self, loss_data, e_loss, backend_option):
        """
        Get the plot results
        """
        colours = {"simulator": 'blue', "hardware": 'orange', "hardware_with_em": 'green'}
        plt.plot(list(range(1,len(loss_data)+1)), loss_data, color=colours[backend_option], label=backend_option)
        if backend_option == "simulator":
            plt.axhline(y=e_loss, color='red', linestyle='--', label="Theoretical minimum loss")


    def plot(self, ansatz_loss, ansatz_mean, ansatz_error, ansatz, colours):
        """
        Plot a graph of the iterations vs the loss
        
        Args:
            ansatz_loss (array): an array of the loss data for each ansatz
            ansatz_mean (array): an array of the average of the loss data for each ansatz
            ansatz_error (array): an array of the standard deviation of the loss data for each ansatz
            ansatz (string): the ansatz
        """
        # Specify the colours of the plot
        #colours = {"undug": 'blue', "dug": 'orange', "superposition": 'green'}
        # x-axis data is the number of iterations
        x = list(range(1,len(ansatz_mean)+1))
        # Plot the data
        plt.plot(x,ansatz_mean, color=colours[ansatz], label=ansatz)
        # Plot the error
        plt.fill_between(x, np.subtract(ansatz_mean,ansatz_error), np.add(ansatz_mean, ansatz_error), color=colours[ansatz], alpha=0.2)
        
    def show_plot(self, file_plots="../plots/plot.png"):
        """
        Show the plot
        """
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(file_plots)
        plt.show()
        
def run_hardware_main():
    # from benchmark import Benchmark
    #config_filename = '../configs/config2.json'
    config_filename = '../configs/config1.json'
    benchmark_results = Benchmark(config_filename, verbose=True)
    #benchmark_results.run_benchmark_for_ansatz(mean_iter=1, filename="../results/results2.txt")
    benchmark_results.run_on_hardware(file_plots="../plots/plot_2021_05_03_14_00.png", file_results="../results/result_2021_05_03_14_00.txt", file_params_results="../results/params_result_2021_05_03_14_00.txt")
    
def run_optimizer_main():
    # from benchmark import Benchmark
    config_filename = '../configs/config1.json'
    benchmark_results = Benchmark(config_filename, verbose=True)
    benchmark_results.run_benchmark_for_optimizers(mean_iter = 3, file_plots="../plots/plot_optimizers5.png", file_results="../results/result_optimizers5.txt")
        
if __name__ == "__main__":
    run_hardware_main()
    
    