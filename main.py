'''
TODO:
    1- Make Node class store a Set object rather than set value
    2- Make the code work with CSV and XLSX files
    3- Improve the way in which non-functioning nodes are set
    4- look if there are any chances of improving the efficiency of the code (if necessary)
    5- Clean up the code (Always store backup)
'''


import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os


class Node:
    node_dict = {}

    def __init__(self, label, x, y):
        self.label = label
        self.is_partly_switch = False
        self.node_set = None
        self.neighbor_switch = None
        self.is_visited = False
        self.y = y
        self.x = x
        self.add_to_node_dict()

    def set_as_switch(self, neighbor_node):
        self.is_partly_switch = True
        self.neighbor_switch = neighbor_node

    def set_node_set(self, node_set):
        self.node_set = node_set

    def set_y(self, y):
        self.y = y

    def set_x(self, x):
        self.x = x

    def get_node_set(self):
        return self.node_set

    def get_node_set_value(self):
        return self.node_set

    def get_value(self):
        return self.label

    def get_y(self):
        return self.y

    def get_x(self):
        return self.x

    def is_switch(self, neighbor_node):
        if self.is_partly_switch and (self.neighbor_switch == neighbor_node):
            return True
        return False

    def set_as_visited(self):
        self.is_visited = True

    def get_if_visited(self):
        return self.is_visited

    def add_to_node_dict(self):
        Node.node_dict[self.label] = self


class Set:
    set_dict = {}

    def __init__(self, value):
        self.value = value
        self.isConnected = True
        self.isFunctioning = True
        self.is_visited = False
        self.there_is_a_path = False
        self.add_to_set_dict()

    def set_as_not_functioning(self):
        self.isFunctioning = False

    def set_as_not_connected(self):
        if self.there_is_a_path:
            self.isConnected = True
        self.isConnected = False

    def get_functionality(self):
        return self.isFunctioning

    def get_connectivity(self):
        return self.isConnected

    def get_value(self):
        return self.value

    def set_as_visited(self):
        self.is_visited = True

    def get_if_visited(self):
        return self.is_visited

    def set_path_state(self):
        self.there_is_a_path = True

    def get_path_state(self):
        return self.there_is_a_path

    def add_to_set_dict(self):
        Set.set_dict[self.value] = self


def create_graph(file_path, sheet_name):
    _, ext = os.path.splitext(file_path)
    if ext == '.xlsx' or True:
        return create_graph_from_xlsx(file_path, sheet_name)
    else:
        raise ValueError('Unsupported file format')


def create_graph_from_xlsx(file_path, sheet_name):
    draw_graph(file_path, drawing_sheet_name)
    temp_node_graph = nx.Graph()  # Initialize an empty graph
    node_dict = {}  # Dictionary to track nodes by their values

    df = pd.read_excel(file_path,  keep_default_na=False, sheet_name=sheet_name, skiprows=1, header=None)
    for row in df.itertuples(index=False):
        nodes = [cell for cell in row]

        value1 = nodes[0]
        value2 = nodes[1]

        if value1 == '':
            continue

        # Check if the nodes with the same values already exist in the graph
        if value1 in node_dict:
            node1 = node_dict[value1]
        else:
            if get_node_by_value(value1) is None:
                node1 = Node(value1, None, None)
            else:
                node1 = get_node_by_value(value1)
                node_dict[value1] = node1

        if value2 in node_dict:
            node2 = node_dict[value2]
        else:
            if get_node_by_value(value2) is None:
                node2 = Node(value1, None, None)
            else:
                node2 = get_node_by_value(value2)
                node_dict[value2] = node2

        temp_node_graph.add_edge(node1, node2)

        if nodes[2] == 'sw':
            # Perform additional operations here
            node1.set_as_switch(node2)
            node2.set_as_switch(node1)

    return temp_node_graph


def draw_graph(drawing_file_path, sheet_name):
    df = pd.read_excel(drawing_file_path,  keep_default_na=False, sheet_name=sheet_name, skiprows=2, header=None)

    nodes_list = []
    x_values = df.iloc[:, 1]
    y_values = df.iloc[:, 2]
    labels = df.iloc[:, 0]
    plt.scatter(x_values, y_values)
    for i, label in enumerate(labels):
        x = x_values[i]
        y = y_values[i]
        node = Node(label, x, y)
        nodes_list.append(node)
        plt.annotate(label, (x, y))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Points')
    plt.show()


def add_edge(graph, vertex1, vertex2):
    if vertex1 not in graph.nodes:
        graph.add_node(vertex1)
    if vertex2 not in graph.nodes:
        graph.add_node(vertex2)
    graph.add_edge(vertex1, vertex2)
    

def group_nodes_between_switches(graph, start_node, set_value):
    start_node.set_as_visited()
    start_node.set_node_set(set_value)

    for neighbor in graph.neighbors(start_node):
        if not neighbor.get_if_visited():
            if neighbor.is_switch(start_node):
                continue
            print(start_node.get_value(), neighbor.get_value(), set_value)
            group_nodes_between_switches(graph, neighbor, set_value)

    for neighbor in graph.neighbors(start_node):
        if not neighbor.get_if_visited():
            if neighbor.is_switch(start_node):
                global adder
                adder = adder + 1
                neighbor_set_value = set_list[len(set_list) - 1] + 1
                if set_value in set_dict:
                    set_node1 = set_dict[set_value]
                else:
                    set_node1 = Set(set_value)
                    set_dict[set_value] = set_node1

                if neighbor_set_value in set_dict:
                    set_node2 = set_dict[neighbor_set_value]
                else:
                    set_node2 = Set(neighbor_set_value)
                    set_dict[neighbor_set_value] = set_node2
                add_edge(sets_graph, set_node1, set_node2)
                print(start_node.get_value(), neighbor.get_value(), set_list, set_value, neighbor_set_value)
                set_list.append(neighbor_set_value)
                group_nodes_between_switches(graph, neighbor, neighbor_set_value)


def check_connected_set(graph, start_set):
    start_set.set_as_visited()
    for neighbor in graph.neighbors(start_set):
        if not neighbor.get_if_visited():
            if not (start_set.get_connectivity() and start_set.get_functionality()):
                neighbor.set_as_not_connected()
            else:
                neighbor.set_path_state()
            check_connected_set(graph, neighbor)


# Function to get a node by its value
def get_node_by_value(value):
    return Node.node_dict.get(value, None)


# Function to get a node by its value
def get_set_by_value(value):
    return Set.set_dict.get(value, None)

def create_excel_file(nodes_graph):
    data = []
    for node in nodes_graph.nodes:
        temp_set = get_set_by_value(node.get_node_set())
        if not temp_set.get_functionality():
            color = 2
        elif not temp_set.get_connectivity():
            color = 1
        else:
            color = 0
        data.append([node.get_value(), color, node.get_node_set()])

    df = pd.DataFrame(data, columns=["Node Value", "Functionality", "Node Set"])
    df.to_excel("nodes_data.xlsx", index=False)


# Enter the file path to read
file_path = "EPRI_Ckt5_Data2.xlsx"
drawing_sheet_name = "38-buss"
create_sheet_name = "38-connection (2)"
#file_path = "ResearchSample.xlsx"

#draw_graph(file_path)
# Call the function to create the graph
nodes_graph = create_graph(file_path, create_sheet_name)


sets_graph = nx.Graph()

# Perform DFS traversal on the graph
start_node = list(nodes_graph.nodes)[0]  # Choose any starting node
set_value = 1
adder = 0
set_list = []
set_list.append(1)
set_dict = {}
group_nodes_between_switches(nodes_graph, start_node, set_value)

visual_node_graph = nx.Graph()

nonFunctioning_node = ""

while True:
    nonFunctioning_node = input("Enter the non-functioning nodes (or 'exit' to quit): ")
    if nonFunctioning_node == "exit":
        break

    nodes = re.split(r'-', nonFunctioning_node.strip())
    if len(nodes) != 2:
        print("Invalid input. Please provide two nodes separated by a hyphen.")
        continue

    try:
        node1value = int(nodes[0])
        node2value = int(nodes[1])
        node1 = get_node_by_value(node1value)
        node2 = get_node_by_value(node2value)

        # Check if the nodes are neighbors in the graph
        if not nodes_graph.has_edge(node1, node2):
            print("Invalid input. The specified nodes are not neighbors in the graph.")
            continue

        set1value = node1.get_node_set_value()
        set2value = node2.get_node_set_value()
        set1 = get_set_by_value(set1value)
        set2 = get_set_by_value(set2value)
        set1.set_as_not_functioning()
        set2.set_as_not_functioning()
    except KeyError:
        print("Invalid input. The specified nodes or sets do not exist in the graph.")

start_set = list(sets_graph.nodes)[0]  # Choose any starting node
check_connected_set(sets_graph, start_set)

# Print the grouped nodes and their respective sets
for node in nodes_graph.nodes:
    print(f"Node {node.get_value()} - Set: {node.get_node_set()}")


print("State of sets:")
# Print the state of a set
for set in sets_graph.nodes:
    if not set.get_functionality():
        print(f"Set{set.get_value()} is not functioning")
    elif not set.get_connectivity():
        print(f"Set{set.get_value()} is not connected but functioning")
    else:
        print(f"Set{set.get_value()} is connected and functioning")


print("State of nodes:")
for node in nodes_graph.nodes:
    temp_set = get_set_by_value(node.get_node_set())
    if not temp_set.get_functionality():
        print(f"Node{node.get_value()} is not functioning and therefore not connected")
    elif not temp_set.get_connectivity():
        print(f"Node{node.get_value()} is not connected but functioning")
    else:
        print(f"Node{node.get_value()} is connected and functioning")


# Create a new graph for visualization
fig = plt.figure(figsize=(40, 40))
visual_node_graph = nx.Graph()

# Add nodes and edges from the original graph
for node in nodes_graph.nodes:
    visual_node_graph.add_node(node.get_value())

    temp_set = get_set_by_value(node.get_node_set())
    if not temp_set.get_functionality():
        visual_node_graph.nodes[node.get_value()]["color"] = "red"
    elif not temp_set.get_connectivity():
        visual_node_graph.nodes[node.get_value()]["color"] = "yellow"
    else:
        visual_node_graph.nodes[node.get_value()]["color"] = "green"

for edge in nodes_graph.edges:
    visual_node_graph.add_edge(edge[0].get_value(), edge[1].get_value())

# Create a dictionary to store the positions of nodes
pos = {}
for node in nodes_graph.nodes:
    x = node.x if node.x is not None else 0.0
    y = node.y if node.y is not None else 0.0
    pos[node.get_value()] = (x, y)

# Draw the graph with nodes positioned based on x and y coordinates
node_colors = [visual_node_graph.nodes[node]["color"] for node in visual_node_graph.nodes]
nx.draw(visual_node_graph, with_labels=True, node_color=node_colors, edge_color='gray', font_weight='bold', node_size=500, pos=pos)


'''
# Create a new graph for visualization
fig = plt.figure(figsize=(40, 40))
visual_node_graph = nx.Graph()

# Add nodes and edges from the original graph
for node in nodes_graph.nodes:
    visual_node_graph.add_node(node.get_value())

    temp_set = get_set_by_value(node.get_node_set())
    if not temp_set.get_functionality():
        visual_node_graph.nodes[node.get_value()]["color"] = "red"
    elif not temp_set.get_connectivity():
        visual_node_graph.nodes[node.get_value()]["color"] = "yellow"
    else:
        visual_node_graph.nodes[node.get_value()]["color"] = "green"


for edge in nodes_graph.edges:
    visual_node_graph.add_edge(edge[0].get_value(), edge[1].get_value())

# Draw the graph
node_colors = [visual_node_graph.nodes[node]["color"] for node in visual_node_graph.nodes]
nx.draw(visual_node_graph, with_labels = True, node_color = node_colors, edge_color = 'gray', font_weight = 'bold', node_size=500)
'''


# Create a new graph for visualization
visual_set_graph = nx.Graph()
# Add nodes and edges from the original graph
for set_node in sets_graph.nodes:
    visual_set_graph.add_node(set_node.get_value())

    if not set_node.get_functionality():
        visual_set_graph.nodes[set_node.get_value()]["color"] = "red"
    elif not set_node.get_connectivity():
        visual_set_graph.nodes[set_node.get_value()]["color"] = "yellow"
    else:
        visual_set_graph.nodes[set_node.get_value()]["color"] = "green"

for edge in sets_graph.edges:
    visual_set_graph.add_edge(edge[0].get_value(), edge[1].get_value())

# Draw the graph
#node_colors = [visual_set_graph.nodes[node]["color"] for node in visual_set_graph.nodes]
#nx.draw(visual_set_graph, with_labels = True, node_color = node_colors, edge_color = 'gray', font_weight = 'bold')

# Show the plot
plt.show()

# Save the figure as an SVG file, replacing the file if it already exists
fig.savefig('circuit.svg')




#Green - 0
#Yellow - 1
#Red - 2

create_excel_file(nodes_graph)
