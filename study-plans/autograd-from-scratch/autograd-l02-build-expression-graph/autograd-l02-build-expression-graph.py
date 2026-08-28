import numpy as np

def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """
    nodes = []
    values = {}

    for leaf in leaves:
        node_id = leaf['id']
        data = np.float64(leaf['data'])

        node = {
            "id": node_id,
            "data": data,
            "grad": np.float64(0.0),
            "op": "",
            "parents": [],
        }
        nodes.append(node)
        values[node_id] = data


    for operation in operations:
        node_id = operation["id"]
        op = operation["op"]
        left_id = operation["left"]
        right_id = operation["right"]

        left_data = values[left_id]
        right_data = values[right_id]

        if op == "+":
            data = np.float64(left_data + right_data)
        else:  # op == "*"
            data = np.float64(left_data * right_data)

        node = {
            "id": node_id,
            "data": data,
            "grad": np.float64(0.0),
            "op": op,
            "parents": [left_id, right_id],
        }

        nodes.append(node)
        values[node_id] = data

    # 3. Final node
    if operations:
        final_id = operations[-1]["id"]
    else:
        final_id = leaves[-1]["id"]

    return nodes, final_id
        
