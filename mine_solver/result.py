from utils import *
import time

class Result():
    """
    Class which carries the results of an experiment. It contains
    data such as the dimension of the problem, and has a callback
    that can be placed into an iterative algorithm so it collects
    results and data in real time.

    Usage:

    result = Result(dims)
    algo = Algorithm(..., callback = result.iteration_callback, ...)
    output = algo.run()
    result.add_solver_output(output)
    """

    def __init__(self, dims, verbose=True, print_every=2, filename=None, params_filename=None):
        self.dims = dims
        self.loss_data = []
        self.verbose = verbose
        self.print_every = print_every
        self.params = []
        self.start_time = time.time()
        self.filename=filename
        self.params_filename=params_filename

    def iteration_callback(self, *argv):
        """
        A function which is called on every iteration of
        optimization. At this point, it just collects data.
        """
        iter_n, params = argv[0], argv[1]
        mean, std = argv[2], argv[3]
        self.params.append(params)
        self.loss_data.append(mean)
        if self.filename:
            self.filename.write(str(mean) + ", ")
        if self.params_filename:
            self.filename.write(str(iter_n) + "\n")
            self.filename.write(str(params) + "\n")
        if self.verbose: #and iter_n % self.print_every - 1 == 0:
            print("Evaluation: {}, Loss: {}".format(iter_n, mean))

    def add_solver_output(self, solver_output, fragment=None):
        """
        A function adds the data from the output of a solver
        to the state.
        If function is called for a fragment then provide the fragment list.
        """
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.ground_state = read_ground_state(solver_output['aux_ops'][0])
        self.min_val = solver_output['min_val']
        self.opt_params = solver_output['opt_params']
        self.clean_grid()
        self.build_result_grid(fragment)

    def clean_grid(self):
        """
        Makes it so that the only points that are valid
        to reach are reachable. Creates a "reachable_points"
        set that can be used as a dictionary to map to qubits.

        Creates dictionary _reachable_points {qubit_index : point}
        """
        self.reachable_points = {}
        counter = 0
        self.W, self.H = self.dims
        for i in range(self.W):
            for j in range(i, self.H-i):
                self.reachable_points[counter] = (i, j)
                counter += 1

    def build_result_grid(self, fragment=None):
        """
        Builds a result that easily visualizes what sites
        are dug at the grid.
        If function is called for a fragment then provide the fragment list.
        """
        self.grid = np.zeros(self.dims)
        #self.ground_state = self.ground_state[::-1]
        for i, val in enumerate(self.ground_state):
            if fragment is None:
                self.grid[self.reachable_points[i]] = val
            else:
                self.grid[fragment[i]] = val
                
    def get_loss_data(self):
        """
        Get the loss data
        """
        return self.loss_data
            
    def get_final_loss(self):
        """
        Get the final loss
        """
        return self.min_val


    def print_results(self):
        print("Results Report:")
        print("Solving Took {} seconds".format(self.elapsed_time))
        print("Loss Function: {}".format(self.min_val))
        print("Optimal Parameters: {}".format(self.opt_params))
        print("Final Configuration:")
        print_grid(self.grid)
