
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

def run_quantum_optimization(run_id=None):
    # Generate a unique run ID if not provided
    if run_id is None:
        run_id = f"Run_{int(time.time())}"
    
    num_qubits = 16

    qc = QuantumCircuit.from_qasm_file("C:/Users/Pasarle/Desktop/Abasin/Experiment/MQTBench_2024-09-05-12-48-08/qft_indep_qiskit_16.qasm")
    qc = qc.decompose()


    num_circuits = 100
    population = [qc.copy() for _ in range(num_circuits)]


    generations = 200



    ###  K-L-Algorithm  #######################################


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
        partition = (list(partition_1), list(partition_2))
        
        print(f"Partition 1: {partition_1}")
        print(f"Partition 2: {partition_2}")
        
        return partition


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

    print(f"\nInitial global gates: {initial_global_gates_kl}")


    def evaluate_fidelity(circuit, qc, backend):

        circuit = circuit.copy()

        circuit.save_statevector()

        job = execute(circuit, backend)
        result = job.result()
        optimized_state = result.get_statevector()

        initial_state = Statevector.from_instruction(qc)
        fidelity = state_fidelity(initial_state, optimized_state)


        if fidelity >= 1.0:
            fidelity = 1.0


        return fidelity




    def fitness_calculation(circuit, partition, qc, backend, weight_gates=0.5, weight_fidelity=0.5, fidelity_threshold=0.99):

        global_count = count_global_gates_with_kl_algorithm(circuit, partition)
        
        fidelity = evaluate_fidelity(circuit, qc, backend)

        if fidelity < fidelity_threshold:
            penalty = 30
            return weight_gates * global_count + penalty
        
        return (weight_gates * global_count) + (weight_fidelity * (1 - fidelity) + 0.0000001)



    def roulette_wheel_selection(population, num_circuits, qc, backend):
        scored_population = [(fitness_calculation(circuit, partition_kl, qc, backend), circuit) for circuit in population]
        
        inverse_sum_score = sum(1.0 / score for score, _ in scored_population)

        probabilities = [(1.0 / score) / inverse_sum_score for score, _ in scored_population]

        selected_circuits = random.choices(
            population=[circuit for _, circuit in scored_population],
            weights=probabilities,
            k=num_circuits)
        
        p1 = []
        p2 = []

        for i, circuit in enumerate(selected_circuits):
            if i % 2 == 0:
                p1.append(circuit)
            else:
                p2.append(circuit)

        return p1, p2, scored_population[:num_circuits]



    #### Crossover Function ##############################



    r_cross = 0.8


    def crossover(p1, p2, r_cross):
        children = []
        for i in range(min(len(p1), len(p2))):
            parent1 = p1[i]
            parent2 = p2[i]
            c1 = parent1.copy()
            c2 = parent2.copy()

            p1_gates = len(parent1.data)
            p2_gates = len(parent2.data)

            if p1_gates <= p2_gates:
                p_gates = p1_gates
            else:
                p_gates = p2_gates

            select_crossover = random.randint(1, 2)

            if select_crossover == 1:
                if random.random() < r_cross: 
                    pt = random.randint(1, p_gates - 1)

                    num_qubits_p1 = c1.num_qubits
                    num_clbits_p1 = c1.num_clbits
                    num_qubits_p2 = c2.num_qubits
                    num_clbits_p2 = c2.num_clbits

                    new_c1 = QuantumCircuit(num_qubits_p1, num_clbits_p1)
                    new_c2 = QuantumCircuit(num_qubits_p2, num_clbits_p2)

                    new_c1.data = parent1.data[:pt] + parent2.data[pt:]
                    new_c2.data = parent2.data[:pt] + parent1.data[pt:]

                    c1 = new_c1
                    c2 = new_c2

            if select_crossover == 2:

                if random.random() < r_cross: 
                    pt1 = random.randint(1, p_gates - 2)
                    pt2 = random.randint(pt1 + 1, p_gates - 1)

                    num_qubits_p1 = c1.num_qubits
                    num_clbits_p1 = c1.num_clbits
                    num_qubits_p2 = c2.num_qubits
                    num_clbits_p2 = c2.num_clbits

                    new_c1 = QuantumCircuit(num_qubits_p1, num_clbits_p1)
                    new_c2 = QuantumCircuit(num_qubits_p2, num_clbits_p2)

                    new_c1.data = parent1.data[:pt1] + parent2.data[pt1:pt2] + parent1.data[pt2:]
                    new_c2.data = parent2.data[:pt1] + parent1.data[pt1:pt2] + parent2.data[pt2:]

                    c1 = new_c1
                    c2 = new_c2

            children.append(c1)
            children.append(c2)

        return children


    #### Mutation Function ###############################

    r_mut = 0.6

    def mutation(circuit, r_mut):
        select_mutation = random.randint(1, 3)

        if select_mutation == 1:  
            if random.random() < r_mut: 
                dag = circuit_to_dag(circuit)
                my_list = []

                for layer in dag.layers():
                    layer_as_circuit = dag_to_circuit(layer['graph'])
                    my_list.append(layer_as_circuit)

                my_list.reverse()

                reversed_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

                for circ in my_list:
                    reversed_circuit.compose(circ, inplace=True)

                return reversed_circuit
            else:
                return circuit
            
        if select_mutation == 2:  
            if random.random() < r_mut:
                dag = circuit_to_dag(circuit)
                my_list = []

                for layer in dag.layers():
                    layer_as_circuit = dag_to_circuit(layer['graph'])
                    my_list.append(layer_as_circuit)

                random.shuffle(my_list)

                shuffled_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

                for circ in my_list:
                    shuffled_circuit.compose(circ, inplace=True)

                return shuffled_circuit
            else:
                return circuit
            
        if select_mutation == 3:  
            if random.random() < r_mut:
                dag = circuit_to_dag(circuit)
                my_list = []

                for layer in dag.layers():
                    layer_as_circuit = dag_to_circuit(layer['graph'])
                    my_list.append(layer_as_circuit)

                remove_element = random.choice(my_list)

                my_list.remove(remove_element)

                remove_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

                for circ in my_list:
                    remove_circuit.compose(circ, inplace=True)

                return remove_circuit
            else:
                return circuit




    backend = Aer.get_backend('aer_simulator_statevector')




    backend = Aer.get_backend('aer_simulator')

    # Depth of Initial Circuit
    initial_depth = qc.depth()



    # Optimized Circuit of each Generation
    final_optimized_circuit = None

    # Initial Circuit
    current_best_circuit = qc
    current_best_fitness = fitness_calculation(qc, partition_kl, qc, backend)
    current_best_fidelity = 1.0  


    results_per_generation = []
    fitness_values_per_generation = []
    best_fidelity_per_generation = []
    best_fitness_per_generation = []
    mean_fitness_per_generation = []
    current_best_fitness_per_generation = []

    #current_best_fitness_per_generation.append(current_best_fitness)

    for generation in range(generations):

        print(f"Generation {generation + 1}/{generations}")
        
        p1, p2, _ = roulette_wheel_selection(population, num_circuits, qc, backend)
        children = crossover(p1, p2, r_cross)
        mutated_children = [mutation(child, r_mut) for child in children]

        new_population = mutated_children + population

        scored_population = [(fitness_calculation(circuit, partition_kl, qc, backend), circuit) for circuit in new_population]
        scored_population.sort(key=lambda x: x[0])
        population = [circuit for _, circuit in scored_population[:num_circuits]]

        # Select best Circuit
        best_circuit = scored_population[0][1]
        best_score = scored_population[0][0]
        best_fidelity = evaluate_fidelity(best_circuit, qc, backend)

        optimized_global_gates = count_global_gates_with_kl_algorithm(best_circuit, partition_kl)
    
        if best_fidelity >= 0.99 and best_score < current_best_fitness and optimized_global_gates >= 1:
            current_best_circuit = best_circuit
            current_best_fitness = best_score
            current_best_fidelity = best_fidelity

        current_best_fitness_per_generation.append(current_best_fitness)

        fitness_values_per_generation.append([score for score, _ in scored_population[:num_circuits]])

        best_fitness = min(fitness_values_per_generation[-1])
        best_fitness_per_generation.append(best_fitness)
        best_fidelity_per_generation.append(current_best_fidelity)

        mean_fitness = np.mean(fitness_values_per_generation[-1])
        mean_fitness_per_generation.append(mean_fitness)


        if generation == generations - 1:
            final_optimized_circuit = current_best_circuit


    print("\nErgebnisse pro Generation:")
    for result in results_per_generation:
        print(result)

    if final_optimized_circuit:

        print("\nOptimierter QFT Circuit:")
        print(final_optimized_circuit.draw('text'))

        optimized_global_gates = count_global_gates_with_kl_algorithm(final_optimized_circuit, partition_kl)
        optimized_depth = final_optimized_circuit.depth()

        print(f"\nInitial global gates (K-L): {initial_global_gates_kl}")
        print(f"Optimized global gates: {optimized_global_gates}")
        print(f"\nInitial depth: {initial_depth}")
        print(f"Optimized depth: {optimized_depth}")


    generations_list = list(range(1, len(best_fitness_per_generation) + 1))
    current_best_fitness_per_generation = current_best_fitness_per_generation[-len(generations_list):]
    plt.figure(figsize=(10, 6))

    # 1. Lightblue Fläche: Min- und Max-Bereich der Fitnesswerte pro Generation
    plt.fill_between(
        generations_list, 
        np.min(fitness_values_per_generation, axis=1), 
        np.max(fitness_values_per_generation, axis=1), 
        color='lightblue', 
        alpha=0.5, 
        label='Spannweite'
    )

    # 2. Blaue Linie: Mean Fitness pro Generation
    plt.plot(
        generations_list, 
        mean_fitness_per_generation, 
        color='blue', 
        linewidth=2, 
        label='Mean Fitness'
    )


    # Plot anzeigen
    plt.title("Fitness per generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.2, 0.1, f"\nK-L algorithm: {initial_global_gates_kl} |    Optimized global gates: {optimized_global_gates} |    Number of qubits: {num_qubits} \n"
                f"Initial depth: {initial_depth} |     Optimized depth: {optimized_depth}",
                ha="left", fontsize=10)
    plt.tight_layout(rect=[0.15, 0.15, 0.90, 0.85])
    plt.show()




    #plt.tight_layout(rect=[0.15, 0.15, 0.90, 0.85])

    plt.figure(figsize=(10, 6))
    plt.plot(generations_list, best_fidelity_per_generation, color='green', label='Best Fidelity', linewidth=2)
    plt.title(f"Best Fidelity per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Submit multiple runs with unique IDs
        futures = [executor.submit(run_quantum_optimization, f"Run_{i}") for i in range(2)]
        for future in futures:
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")


"""
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
    # Generate a unique run ID if not provided
    if run_id is None:
        run_id = f"Run_{int(time.time())}"
    
    # Your existing setup here
    # Define number of Qubits
    num_qubits = 4
    depth = 20  # depth 20 mit 4 num qubits       | depth 25 mit 6 num qubits
    gate_set = ['u3', 'cx']
    random_qc = random_circuit(num_qubits, depth, max_operands=2, measure=False)
    qc = transpile(random_qc, basis_gates = gate_set)
    # Decompose the circuit to ensure only single-qubit and CNOT gates are used
    qc = random_qc.decompose()


    num_circuits = 100
    population = [qc.copy() for _ in range(num_circuits)]


    generations = 200 



    ###  K-L-Algorithm  #######################################


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
        partition = (list(partition_1), list(partition_2))
        
        print(f"Partition 1: {partition_1}")
        print(f"Partition 2: {partition_2}")
        
        return partition


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

    print(f"\nInitial global gates: {initial_global_gates_kl}")


    def evaluate_fidelity(circuit, qc, backend):

        circuit = circuit.copy()

        circuit.save_statevector()

        job = execute(circuit, backend)
        result = job.result()
        optimized_state = result.get_statevector()

        initial_state = Statevector.from_instruction(qc)
        fidelity = state_fidelity(initial_state, optimized_state)


        if fidelity >= 1.0:
            fidelity = 1.0


        return fidelity




    def fitness_calculation(circuit, partition, qc, backend, weight_gates=0.5, weight_fidelity=0.5, fidelity_threshold=0.9):

        global_count = count_global_gates_with_kl_algorithm(circuit, partition)
        
        fidelity = evaluate_fidelity(circuit, qc, backend)

        if fidelity < fidelity_threshold:
            penalty = 30
            return weight_gates * global_count + penalty
        
        return (weight_gates * global_count) + (weight_fidelity * (1 - fidelity) + 0.0000001)



    def roulette_wheel_selection(population, num_circuits, qc, backend):
        scored_population = [(fitness_calculation(circuit, partition_kl, qc, backend), circuit) for circuit in population]
        
        inverse_sum_score = sum(1.0 / score for score, _ in scored_population)

        probabilities = [(1.0 / score) / inverse_sum_score for score, _ in scored_population]

        selected_circuits = random.choices(
            population=[circuit for _, circuit in scored_population],
            weights=probabilities,
            k=num_circuits)
        
        p1 = []
        p2 = []

        for i, circuit in enumerate(selected_circuits):
            if i % 2 == 0:
                p1.append(circuit)
            else:
                p2.append(circuit)

        return p1, p2, scored_population[:num_circuits]



    #### Crossover Function ##############################



    r_cross = 0.8


    def crossover(p1, p2, r_cross):
        children = []
        for i in range(min(len(p1), len(p2))):
            parent1 = p1[i]
            parent2 = p2[i]
            c1 = parent1.copy()
            c2 = parent2.copy()

            p1_gates = len(parent1.data)
            p2_gates = len(parent2.data)

            if p1_gates <= p2_gates:
                p_gates = p1_gates
            else:
                p_gates = p2_gates

            select_crossover = random.randint(1, 2)

            if select_crossover == 1:
                if random.random() < r_cross: 
                    pt = random.randint(1, p_gates - 1)

                    num_qubits_p1 = c1.num_qubits
                    num_clbits_p1 = c1.num_clbits
                    num_qubits_p2 = c2.num_qubits
                    num_clbits_p2 = c2.num_clbits

                    new_c1 = QuantumCircuit(num_qubits_p1, num_clbits_p1)
                    new_c2 = QuantumCircuit(num_qubits_p2, num_clbits_p2)

                    new_c1.data = parent1.data[:pt] + parent2.data[pt:]
                    new_c2.data = parent2.data[:pt] + parent1.data[pt:]

                    c1 = new_c1
                    c2 = new_c2

            if select_crossover == 2:

                if random.random() < r_cross: 
                    pt1 = random.randint(1, p_gates - 2)
                    pt2 = random.randint(pt1 + 1, p_gates - 1)

                    num_qubits_p1 = c1.num_qubits
                    num_clbits_p1 = c1.num_clbits
                    num_qubits_p2 = c2.num_qubits
                    num_clbits_p2 = c2.num_clbits

                    new_c1 = QuantumCircuit(num_qubits_p1, num_clbits_p1)
                    new_c2 = QuantumCircuit(num_qubits_p2, num_clbits_p2)

                    new_c1.data = parent1.data[:pt1] + parent2.data[pt1:pt2] + parent1.data[pt2:]
                    new_c2.data = parent2.data[:pt1] + parent1.data[pt1:pt2] + parent2.data[pt2:]

                    c1 = new_c1
                    c2 = new_c2

            children.append(c1)
            children.append(c2)

        return children


    #### Mutation Function ###############################

    r_mut = 0.6

    def mutation(circuit, r_mut):
        select_mutation = random.randint(1, 3)

        if select_mutation == 1:  
            if random.random() < r_mut: 
                dag = circuit_to_dag(circuit)
                my_list = []

                for layer in dag.layers():
                    layer_as_circuit = dag_to_circuit(layer['graph'])
                    my_list.append(layer_as_circuit)

                my_list.reverse()

                reversed_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

                for circ in my_list:
                    reversed_circuit.compose(circ, inplace=True)

                return reversed_circuit
            else:
                return circuit
            
        if select_mutation == 2:  
            if random.random() < r_mut:
                dag = circuit_to_dag(circuit)
                my_list = []

                for layer in dag.layers():
                    layer_as_circuit = dag_to_circuit(layer['graph'])
                    my_list.append(layer_as_circuit)

                random.shuffle(my_list)

                shuffled_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

                for circ in my_list:
                    shuffled_circuit.compose(circ, inplace=True)

                return shuffled_circuit
            else:
                return circuit
            
        if select_mutation == 3:  
            if random.random() < r_mut:
                dag = circuit_to_dag(circuit)
                my_list = []

                for layer in dag.layers():
                    layer_as_circuit = dag_to_circuit(layer['graph'])
                    my_list.append(layer_as_circuit)

                remove_element = random.choice(my_list)

                my_list.remove(remove_element)

                remove_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

                for circ in my_list:
                    remove_circuit.compose(circ, inplace=True)

                return remove_circuit
            else:
                return circuit




    backend = Aer.get_backend('aer_simulator_statevector')




    backend = Aer.get_backend('aer_simulator')

    # Depth of Initial Circuit
    initial_depth = qc.depth()



    # Optimized Circuit of each Generation
    final_optimized_circuit = None

    # Initial Circuit
    current_best_circuit = qc
    current_best_fitness = fitness_calculation(qc, partition_kl, qc, backend)
    current_best_fidelity = 1.0  


    results_per_generation = []
    fitness_values_per_generation = []
    best_fidelity_per_generation = []
    best_fitness_per_generation = []
    mean_fitness_per_generation = []
    current_best_fitness_per_generation = []

    #current_best_fitness_per_generation.append(current_best_fitness)

    for generation in range(generations):

        print(f"Generation {generation + 1}/{generations}")
        
        p1, p2, _ = roulette_wheel_selection(population, num_circuits, qc, backend)
        children = crossover(p1, p2, r_cross)
        mutated_children = [mutation(child, r_mut) for child in children]

        new_population = mutated_children + population

        scored_population = [(fitness_calculation(circuit, partition_kl, qc, backend), circuit) for circuit in new_population]
        scored_population.sort(key=lambda x: x[0])
        population = [circuit for _, circuit in scored_population[:num_circuits]]

        # Select best Circuit
        best_circuit = scored_population[0][1]
        best_score = scored_population[0][0]
        best_fidelity = evaluate_fidelity(best_circuit, qc, backend)

        #optimized_global_gates = count_global_gates_with_kl_algorithm(best_circuit, partition_kl)
    
        if best_fidelity >= 0.9 and best_score < current_best_fitness: #and optimized_global_gates >= 1:
            current_best_circuit = best_circuit
            current_best_fitness = best_score
            current_best_fidelity = best_fidelity

        current_best_fitness_per_generation.append(current_best_fitness)

        fitness_values_per_generation.append([score for score, _ in scored_population[:num_circuits]])

        best_fitness = min(fitness_values_per_generation[-1])
        best_fitness_per_generation.append(best_fitness)
        best_fidelity_per_generation.append(current_best_fidelity)

        mean_fitness = np.mean(fitness_values_per_generation[-1])
        mean_fitness_per_generation.append(mean_fitness)


        if generation == generations - 1:
            final_optimized_circuit = current_best_circuit


    print("\nErgebnisse pro Generation:")
    for result in results_per_generation:
        print(result)
    
    if final_optimized_circuit:

        print("\nOptimierter QFT Circuit:")
        print(final_optimized_circuit.draw('text'))

        optimized_global_gates = count_global_gates_with_kl_algorithm(final_optimized_circuit, partition_kl)
        optimized_depth = final_optimized_circuit.depth()

        print(f"\nInitial global gates (K-L): {initial_global_gates_kl}")
        print(f"Optimized global gates: {optimized_global_gates}")
        print(f"\nInitial depth: {initial_depth}")
        print(f"Optimized depth: {optimized_depth}")


    generations_list = list(range(1, len(best_fitness_per_generation) + 1))
    current_best_fitness_per_generation = current_best_fitness_per_generation[-len(generations_list):]
    plt.figure(figsize=(10, 6))

    # 1. Lightblue Fläche: Min- und Max-Bereich der Fitnesswerte pro Generation
    plt.fill_between(
        generations_list, 
        np.min(fitness_values_per_generation, axis=1), 
        np.max(fitness_values_per_generation, axis=1), 
        color='lightblue', 
        alpha=0.5, 
        label='Spannweite'
    )

    # 2. Blaue Linie: Mean Fitness pro Generation
    plt.plot(
        generations_list, 
        mean_fitness_per_generation, 
        color='blue', 
        linewidth=2, 
        label='Mean Fitness'
    )


    # Plot anzeigen
    plt.title("Fitness per generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(0.2, 0.1, f"\nK-L algorithm: {initial_global_gates_kl} |    Optimized global gates: {optimized_global_gates} |    Number of qubits: {num_qubits} \n"
                f"Initial depth: {initial_depth} |     Optimized depth: {optimized_depth}",
                ha="left", fontsize=10)
    plt.tight_layout(rect=[0.15, 0.15, 0.90, 0.85])
    plt.show()





    plt.figure(figsize=(10, 6))
    plt.plot(generations_list, best_fidelity_per_generation, color='green', label='Best Fidelity', linewidth=2)
    plt.title(f"Best Fidelity per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Submit multiple runs with unique IDs
        futures = [executor.submit(run_quantum_optimization, f"Run_{i}") for i in range(3)]
        for future in futures:
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")

"""