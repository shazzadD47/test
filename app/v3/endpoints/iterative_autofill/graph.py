class Node:
    def __init__(self, value):
        self.value = value
        self.adjacent = []  # List of connected nodes


class Graph:
    def __init__(self):
        self.nodes = {}  # Dictionary to store nodes by value

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = Node(value)

    def add_edge(self, from_value, to_value):
        # Ensure both nodes exist
        if from_value not in self.nodes:
            self.add_node(from_value)
        if to_value not in self.nodes:
            self.add_node(to_value)

        # Add unidirectional edges
        self.nodes[from_value].adjacent.append(self.nodes[to_value])

    def is_circular(self):
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in node.adjacent:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        # Check for cycles in all nodes
        return any(node not in visited and dfs(node) for node in self.nodes.values())

        return False

    def __str__(self):
        result = []
        for node in self.nodes.values():
            adjacent_values = [neighbor.value for neighbor in node.adjacent]
            result.append(f"{node.value}: {', '.join(adjacent_values)}")
        return "\n".join(result)
