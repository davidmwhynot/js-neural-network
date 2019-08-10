const Connection = require('./Connection');

class Node {
	constructor() {
		this.value = Math.random() / 4 - 0.125;
		this.Connections = [];
		this.bias = Math.random();
	}

	connect(inputLayer) {
		for (const node of inputLayer) {
			this.Connections.push(
				new Connection({
					inputNode: node,
					// weight: 1
					weight: Math.random() / 4 - 0.125
				})
			);
		}
	}

	setWeights(weights) {
		if (weights.length !== this.Connections.length) {
			throw new Error("Invalid input weights: dimensions must match node's.");
		} else {
			for (let i = 0; i < weights.length; ++i) {
				this.Connections[i].weight = weights[i];
			}
		}
	}

	calculate() {
		let rawSum = 0;

		for (const connection of this.Connections) {
			rawSum += connection.calculate();
		}

		rawSum += this.bias;

		this.value = this.sigmoid(rawSum);
	}

	sigmoid(input) {
		return 1 / (1 + Math.E ** (input * -1));
	}
}

module.exports = Node;
