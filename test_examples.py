import random
import time  # For unique timestamp identifiers
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import state_fidelity, Statevector
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import state_fidelity, Statevector
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time  # For unique timestamp identifiers
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import state_fidelity, Statevector
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.circuit.random import random_circuit
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def run_quantum_optimization(run_id=None):


    num_qubits = 16
    depth=25
    gate_set = ['u3', 'cx']
    random_qc = random_circuit(num_qubits, depth, max_operands=2, measure=False)
    qc = transpile(random_qc, basis_gates = gate_set)
    qc = random_qc.decompose()

    num_circuits = 100
    population = [qc.copy() for _ in range(num_circuits)]

    generations = 200

    ### K-L Algorithm #######################################
    def create_graph_from_circuit(circuit):
        dag = circuit_to_dag(circuit)
        graph = nx.Graph()

        for qubit in circuit.qubits:
            graph.add_node(qubit.index)

        for gate in dag.two_qubit_ops():
            control, target = [q.index for q in gate.qargs]
            graph.add_edge(control, target, weight=2)
            graph.add_edge(target, control, weight=1)

        return graph

    def partition_graph(graph):
        partition_1, partition_2 = nx.algorithms.community.kernighan_lin_bisection(graph, weight='weight')
        return (list(partition_1), list(partition_2))

    def count_global_gates_with_kl_algorithm(circuit, partition):
        global_count = 0
        for gate in circuit.data:
            if gate[0].name == "save_statevector":
                continue
            qubits = [q.index for q in gate[1]]
            if (min(qubits) in partition[0] and max(qubits) in partition[1]) or (min(qubits) in partition[1] and max(qubits) in partition[0]):
                global_count += 1
        return global_count

    graph = create_graph_from_circuit(qc)
    partition_kl = partition_graph(graph)
    initial_global_gates_kl = count_global_gates_with_kl_algorithm(qc, partition_kl)

    def evaluate_fidelity(circuit, qc, backend):
        circuit = circuit.copy()
        circuit.save_statevector()
        job = execute(circuit, backend)
        result = job.result()
        optimized_state = result.get_statevector()
        initial_state = Statevector.from_instruction(qc)
        fidelity = state_fidelity(initial_state, optimized_state)
        return fidelity

    def fitness_calculation(circuit, partition, qc, backend, weight_gates=0.5, weight_fidelity=0.5, fidelity_threshold=0.99):
        global_count = count_global_gates_with_kl_algorithm(circuit, partition)
        fidelity = evaluate_fidelity(circuit, qc, backend)
        if fidelity < fidelity_threshold:
            penalty = 30
            return weight_gates * global_count + penalty
        return (weight_gates * global_count) + (weight_fidelity * (1 - fidelity))

    backend = Aer.get_backend('aer_simulator_statevector')
    initial_depth = qc.depth()
    current_best_circuit = qc
    current_best_fitness = fitness_calculation(qc, partition_kl, qc, backend)
    current_best_fidelity = 1.0

    fitness_values_per_generation = []
    mean_fitness_per_generation = []

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        # Mutation example
        mutated_population = []
        for circuit in population:
            dag = circuit_to_dag(circuit)
            layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
            random.shuffle(layers)
            mutated_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
            for layer in layers:
                mutated_circuit.compose(layer, inplace=True)
            mutated_population.append(mutated_circuit)

        scored_population = [(fitness_calculation(circuit, partition_kl, qc, backend), circuit) for circuit in mutated_population]
        scored_population.sort(key=lambda x: x[0])
        population = [circuit for _, circuit in scored_population[:num_circuits]]

        best_circuit = scored_population[0][1]
        best_fitness = scored_population[0][0]
        best_fidelity = evaluate_fidelity(best_circuit, qc, backend)

        if best_fidelity >= 0.99 and best_fitness < current_best_fitness:
            current_best_circuit = best_circuit
            current_best_fitness = best_fitness
            current_best_fidelity = best_fidelity

        fitness_values_per_generation.append([score for score, _ in scored_population[:num_circuits]])
        mean_fitness_per_generation.append(np.mean(fitness_values_per_generation[-1]))

    optimized_depth = current_best_circuit.depth()
    print(f"Initial Depth: {initial_depth}")
    print(f"Optimized Depth: {optimized_depth}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), mean_fitness_per_generation, label='Mean Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_quantum_optimization, f"Run_{i}") for i in range(1)]
        for future in futures:
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")
