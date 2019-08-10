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

	getNodeValsTransposed() {
		const output = [];

		for (const node of this.Nodes) {
			output.push([node.value]);
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

	setWeights(weights) {
		if (weights.length !== this.Nodes.length) {
			throw new Error("Invalid input weights: dimensions must match layer's.");
		} else {
			for (let i = 0; i < weights.length; ++i) {
				this.Nodes[i].setWeights(weights[i]);
			}
		}
	}

	getBiases() {
		const output = [];

		for (const node of this.Nodes) {
			output.push([node.bias]);
		}

		return output;
	}

	setBiases(biases) {
		if (biases.length !== this.Nodes.length) {
			throw new Error("Invalid input biases: dimensions must match layer's.");
		} else {
			for (let i = 0; i < biases.length; ++i) {
				this.Nodes[i].bias = biases[i];
			}
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
