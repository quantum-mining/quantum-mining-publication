import numpy as np
import sys
from utils import generate_pauli_Z, print_grid
from qiskit.aqua.operators import WeightedPauliOperator
from config import Configuration


class OpenPitMiningProblem():
	"""
	Class definition for an open pit mining problem.
	Contains functionality for loading the problem,
	generating the Hamiltonian, building options, etc.
	"""
	def __init__(self, config):
		self.config = config
		self.load_data(config['pit_profile_filename'])
		print_grid(self._grid)
		self.clean_grid()
		self.get_all_sites()
		#self.build_hamiltonian()

	def load_data(self, filename):
		"""
		Loads data from a specified
		The expected format for defining a pit profile will come as:
		H W (D)
		x1 y1 (z1) v1
		x2 y2 (z2) v2
		...
		xN yN (zN) vN
		"""
		with open(filename, 'r') as f:
			first_line = f.readline().split()
			if len(first_line) == 2:
				self.dim = 2
				self.W, self.H = int(first_line[0]), int(first_line[1])
				self._grid = np.zeros((self.W,self.H))
				self.D = None
			elif len(first_line) == 3:
				self.dim = 3
				self.W, self.H, self.D = int(first_line[0]),\
										int(first_line[1]),\
										int(first_line[2])
				self._grid = np.zeros((self.W,self.H,self.D))
			for line in f.readlines():
				line = line.split()
				if self.dim == 2:
					x,y = int(line[0]), int(line[1])
					self._grid[x,y] = float(line[2])
				elif self.dim == 3:
					x,y,z = int(line[0]), int(line[1]), int(line[2])
					self._grid[x,y,z] = float(line[3])


	def get_dims(self):
		if self.dim == 2:
			return (self.W, self.H)
		if self.dim == 3:
			return (self.W, self.H, self.D)

	def clean_grid(self):
		"""
		Makes it so that the only points that are valid
		to reach are reachable. Creates a "reachable_points"
		set that can be used as a dictionary to map to qubits.

		Creates dictionary _reachable_points {point : qubit_index}
		"""
		self._reachable_points = {}
		counter = 0
		for i in range(self.W):
			for j in range(i,self.H-i):
				self._reachable_points[(i,j)] = counter
				counter += 1

	def get_all_sites(self, verbose = False):
		"""
		Returns all points to be considered.
		A dictionary of {point : qubit_index}
		"""
		if verbose:
			print('Reachable points:', self._reachable_points)
		return self._reachable_points

	def get_parents(self, point, verbose=False):
		"""
		Given a point in the grid, return the set of points
		that are "above" it, i.e. the set of points that
		would need to be dug before ~point~.
		"""
		parents = []
		x,y = point
		for y_prime in range(y-1, y+2):
			if y_prime >= 0 and y_prime < self.W and x > 0 \
				and (x-1, y_prime) in self._reachable_points.keys():
				parents.append((x-1, y_prime))
		if verbose:
			print('Parents of', point, 'are:', parents)
		return parents

	def get_children(self, point, verbose=False):
		"""
		Return the list of points that a given point is a parent to.
		Needed for build_fragment_hamiltonian.
		"""
		children = []
		x,y = point
		for y_prime in range(y-1, y+2):
			if y_prime >= 0 and y_prime < self.W \
				and (x+1, y_prime) in self._reachable_points.keys():
				children.append((x+1, y_prime))
		if verbose:
			print('Children of', point, 'are:', children)
		return children

	def build_hamiltonian(self, verbose=False):
		"""
		Builds Hamiltonian for the problem. Right now we are using
		the qubit approach. This method should actually be abstract
		and implemented in subclasses.

		Essentially cost function
		"""
		n = len(self._reachable_points)
		H = []
		penalty = self.config['penalty']
		offset = 0.0
		# Number of parent-child pairs
		pc = 0
		for ind, point in enumerate(self._reachable_points):
			val = self._grid[point]
			# Change the signs because we minimize -HV
			H.append((0.5*val, generate_pauli_Z([ind], n)))
			H.append((-0.5*val, generate_pauli_Z([], n)))
			# Below is HS
			parents = self.get_parents(point)
			for parent in parents:
				pc += 1
				par_ind = self._reachable_points[parent]
				H.append((0.25*penalty, generate_pauli_Z([par_ind], n)))
				H.append((-0.25*penalty, generate_pauli_Z([ind], n)))
				H.append((-0.25*penalty, generate_pauli_Z([par_ind, ind], n)))
				H.append((0.25*penalty, generate_pauli_Z([], n)))

		qubit_op = WeightedPauliOperator(H)
		if verbose:
			print('WeightedPauliOperator details:')
			print(qubit_op.print_details())

		return n, qubit_op, offset


	def build_fragment_hamiltonian(self, fragments, which_fragment, expectations, verbose=True):
		"""
		Generalized to build the Hamiltonian for any fragment.
		fragments: list of lists
		which_fragment: integer
		expectations: list of lists
		"""
		fragment_points = fragments[which_fragment]  # E.g. [(0,0), (0,1), (0,2)]
		n = len(fragment_points)  # E.g. 3

		# Neighbors are all points whose expected values will contribute
		# to the Hamiltonian. Thus they are either children or parents of
		# the fragment points that are OUTSIDE the fragment itself.
		neighbors = []  # E.g. [(1,1)]
		for point in fragment_points:
			children = self.get_children(point, verbose=False)
			for child in children:
				if child not in neighbors and child not in fragment_points:
					neighbors.append(child)
			parents = self.get_parents(point, verbose=False)
			for parent in parents:
				if parent not in neighbors and parent not in fragment_points:
					neighbors.append(parent)
		if verbose:
			print('The points whose expected values contribute to the Hamiltonian are:', neighbors)

		means = []  # E.g. [<z4>]
		for neighbor in neighbors:
			for i, fragment in enumerate(fragments):
				for j, point in enumerate(fragment):
					if neighbor == point:
						means.append(expectations[i][j])
		if verbose:
			print('And the corresponding expected values are:', means)

		# We also need the parents inside the fragment
		inside_parents = []
		for point in fragment_points:
			parents = self.get_parents(point, verbose=False)
			for parent in parents:
				if parent not in neighbors and parent not in inside_parents:
					inside_parents.append(parent)
		if verbose:
			print('Parents inside the fragment are:', inside_parents)

		H = []
		offset = 0.0
		penalty = self.config['penalty']
		for i, point in enumerate(fragment_points):
			val = self._grid[point]
			# Change the signs because we minimize -HV
			H.append((+0.5*val, generate_pauli_Z([i], n)))
			H.append((-0.5*val, generate_pauli_Z([], n)))
			offset -= 0.5*val
			# Below is HS
			parents = self.get_parents(point)
			for parent in parents:
				if parent in inside_parents:
					par_ind = fragment_points.index(parent)
					H.append((0.25*penalty, generate_pauli_Z([par_ind], n)))
					H.append((-0.25*penalty, generate_pauli_Z([i], n)))
					H.append((-0.25*penalty, generate_pauli_Z([par_ind, i], n)))
					H.append((0.25*penalty, generate_pauli_Z([], n)))
					offset += 0.25*penalty

		for i, neighbor in enumerate(neighbors):
			# Still contribution from HV
			val = self._grid[neighbor]
			#H.append((-0.5*val*(1-means[i]), generate_pauli_Z([], n)))
			offset -= 0.5*val*(1-means[i])
			# Below is HS
			parents = self.get_parents(neighbor)
			for parent in parents:
				if parent in fragment_points:
					par_ind = fragment_points.index(parent)
					H.append((0.25*penalty*(1-means[i]), generate_pauli_Z([par_ind], n)))
					H.append((0.25*penalty*(1-means[i]), generate_pauli_Z([], n)))
					offset += 0.25*penalty*(1-means[i])
			children = self.get_children(neighbor)
			for child in children:
				if child in fragment_points:
					child_ind = fragment_points.index(child)
					H.append((0.25*penalty*(-1-means[i]), generate_pauli_Z([child_ind], n)))
					H.append((0.25*penalty*(1+means[i]), generate_pauli_Z([], n)))
					offset += 0.25*penalty*(1+means[i])

		qubit_op = WeightedPauliOperator(H)
		if verbose:
			print('Penalty:', penalty)
			print('Offset:', offset)
			print('WeightedPauliOperator details:')
			print(qubit_op.print_details())

		return qubit_op, offset


if __name__ == "__main__":
	config_filename = '../configs/config2.json'
	configuration = Configuration(config_filename)
	problem = OpenPitMiningProblem(configuration.get_problem_options())
