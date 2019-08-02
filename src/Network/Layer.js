const Node = require('./Node');

class Layer {
	constructor(numNodes, output) {
		this.Nodes = [];

		for (let i = 0; i < numNodes; ++i) {
			this.Nodes.push(new Node(output));
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
