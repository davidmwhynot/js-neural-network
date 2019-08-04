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
