const Connection = require('./Connection');

class Node {
	constructor() {
		this.value = 1;
		this.Connections = [];
	}

	connect(inputLayer) {
		for (const node of inputLayer) {
			this.Connections.push(
				new Connection({
					inputNode: node,
					// weight: 1
					weight: Math.random() * 10
				})
			);
		}
	}

	calculate() {
		let rawSum = 0;

		for (const connection of this.Connections) {
			rawSum += connection.calculate();
		}

		this.value = this.sigmoid(rawSum);
	}

	sigmoid(input) {
		return 1 / (1 + Math.E ** (input * -1));
	}
}

module.exports = Node;
