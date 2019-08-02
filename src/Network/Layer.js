const Node = require('./Node');

class Layer {
	constructor(numNodes) {
		this.Nodes = [];

		for (let i = 0; i < numNodes; ++i) {
			this.Nodes.push(new Node());
		}
	}

	getNodeVals() {
		const output = [];

		for (const node of this.Nodes) {
			output.push(node.value);
		}

		return output;
	}

	getWeights() {
		if (this.Nodes[0].Connections) {
			const output = [];

			for (const node of this.Nodes) {
				const connections = [];

				for (const connection of node.Connections) {
					connections.push(connection.weight);
				}

				output.push(connections);
			}

			return output;
		} else {
			return [];
		}
	}

	connectNodes(inputLayer) {
		for (const node of this.Nodes) {
			node.connect(inputLayer.Nodes);
		}
	}

	calculate() {
		for (const node of this.Nodes) {
			node.calculate();
		}
	}
}

module.exports = Layer;
