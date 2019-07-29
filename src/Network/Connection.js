class Connection {
	constructor({ inputNode, weight }) {
		this.inputNode = inputNode;
		this.weight = weight;
	}

	calculate() {
		return this.weight * this.inputNode.value;
	}
}

module.exports = Connection;
