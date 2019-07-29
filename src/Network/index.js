const Layer = require('./Layer');

class Network {
	constructor({ layers, learningRate }) {
		this.Layers = [];
		this.learningRate = learningRate;

		for (const layer of layers) {
			this.Layers.push(new Layer(layer.numNodes));
		}

		this.connectLayers();
	}

	print() {
		for (const layer of this.Layers) {
			let output = '';
			for (const node in layer.Nodes) {
				const val = layer.Nodes[node].value;
				output += `n${node}: ${Math.round(val * 1000) / 1000}   `;
			}
			// console.log(output);
		}
	}

	connectLayers() {
		for (let i = 1; i < this.Layers.length; ++i) {
			this.Layers[i].connectNodes(this.Layers[i - 1]);
		}
	}

	setInputs(values) {
		for (let i = 0; i < values.length; ++i) {
			this.Layers[0].Nodes[i].value = values[i];
		}
	}

	calculate() {
		for (let i = 1; i < this.Layers.length; ++i) {
			this.Layers[i].calculate();
		}
	}

	getOutputs() {
		let output = [];

		for (const node of this.Layers[this.Layers.length - 1].Nodes) {
			output.push(node.value);
		}

		return output;
	}

	train(data) {
		let deltas = [];
		let results = [];
		for (const example of data) {
			this.setInputs(example.inputs);
			this.calculate();

			let outputs = this.getOutputs();

			// errors
			let errors = [];
			let outputErrors = [];
			for (let i = 0; i < outputs.length; ++i) {
				outputErrors.push(example.outputs[i] - outputs[i]);
				// outputErrors.push(0.5 * (example.outputs[i] - outputs[i]) ** 2);
			}
			errors.push(outputErrors);

			// hidden errors
			// for (let j = this.Layers.length - 1; j > 0; --j) {
			for (let j = this.Layers.length - 2; j > 0; --j) {
				const layer = this.Layers[j];

				let hiddenErrorsRow = [];

				// loop through hidden layer nodes
				for (const node of layer.Nodes) {
					// get total weight for this node's connections
					let totalWeight = 0;

					for (const connection of node.Connections) {
						totalWeight += connection.weight;
					}

					let errorSum = 0;
					let lastHiddenErrorsRow = errors[errors.length - 1];
					// loop through nodes connections
					for (const connection of node.Connections) {
						// loop through last hidden layer's errors
						for (let i = 0; i < lastHiddenErrorsRow.length; ++i) {
							errorSum +=
								lastHiddenErrorsRow[i] *
								(connection.weight / totalWeight);
							// errorSum +=
							// 	lastHiddenErrorsRow[i] * connection.weight;
						}
					}

					hiddenErrorsRow.push(errorSum);
				}

				errors.push(hiddenErrorsRow);
			}

			errors.reverse();
			// log(errors);

			let exampleDeltas = [];
			for (let i = 1; i < this.Layers.length; ++i) {
				let layerDeltas = [];

				const layer = this.Layers[i];
				for (let j = 0; j < layer.Nodes.length; ++j) {
					let nodeDeltas = [];

					const node = layer.Nodes[j];
					let x = 0;
					for (const connection of node.Connections) {
						++x;

						connection.weight +=
							this.learningRate *
							errors[i - 1][j] *
							// sigmoid(
							// 	connection.inputNode.value * connection.weight
							// ) *
							// (1 -
							// 	sigmoid(
							// 		connection.inputNode.value *
							// 			connection.weight
							// 	)) *
							connection.inputNode.value;
					}

					layerDeltas.push(nodeDeltas);
				}

				exampleDeltas.push(layerDeltas);
			}
			deltas.push(exampleDeltas);

			// log('exampleDeltas');
			// log(exampleDeltas);

			// log('network outputs');
			// log(outputs);

			// log('desired outputs');
			// log(example.outputs);
			// log('\n\n');

			results.push({ guess: outputs, answer: example.outputs });
		}
		for (let i = 0; i < deltas.length; ++i) {
			// log('deltas');
			// log(i, deltas[i]);
			// log('...for results ', results[i]);
		}

		// for (let i = 1; i < this.Layers.length; ++i) {
		// 	const layer = this.Layers[i];

		// 	for (let j = 0; j < layer.Nodes.length; ++j) {
		// 		const node = layer.Nodes[j];

		// 		for (let k = 0; k < node.Connections.length; ++k) {
		// 			const connection = node.Connections[k];

		// 			let sum = 0;
		// 			for (let l = 0; l < data.length; ++l) {
		// 				sum += deltas[l][i - 1][j][k];
		// 			}
		// 			// log(
		// 			// 	`\nadjustments for layer${i} node${j} connection${k}`
		// 			// );
		// 			// log('pre connection.weight');
		// 			// log(connection.weight);
		// 			// log('sum / data.length');
		// 			// log(sum / data.length);

		// 			connection.weight += sum / data.length;
		// 			// log('post connection.weight');
		// 			// log(connection.weight);
		// 		}
		// 	}
		// }
	}
}

function log(...s) {
	console.log(...s);
}

function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

module.exports = Network;

/*
x = weight
a = inputActivation
inside = sum(1 / (1 + e ^ -(x * a))) - example.output

costGradient = (inside) ^ 2

derivative of costGradient for x =
2(inside)(derivative of inside)

derivative of inside for x
((weight * e ^ -(ax)) / (e ^ (-ax) + 1) ^ 2)

dx / d(costGradient) = 2(sum(1 / (1 + e ^ -(x * a))) - example.output) * (((weight * e ^ -(ax)) / (e ^ (-ax) + 1) ^ 2))



			let newWeights = [];
			for (let i = 1; i < this.Layers.length; ++i) {
				let newWeightsLayer = [];

				const layer = this.Layers[i];
				for (const node in layer.Nodes) {
					let newWeightsNode = [];

					let output = `n${node}\n`;
					for (const connection of layer.Nodes[node].Connections) {
						output += 'w: ' + connection.weight;
						newWeightsNode.push(connection.weight);
					}
					// console.log(output + '\n');

					newWeightsLayer.push(newWeightsNode);
				}
				// console.log('\n\n\n');

				newWeights.push(newWeightsLayer);
			}

			for (let i = 1; i < this.Layers.length; ++i) {
				for (let j = 0; j < this.Layers[i].Nodes.length; ++j) {
					for (
						let k = 0;
						k < this.Layers[i].Nodes[j].Connections.length;
						++k
					) {
						// console.log(
							oldWeights[i - 1][j][k] - newWeights[i - 1][j][k]
						);
					}
				}
			}


			let oldWeights = [];
			for (let i = 1; i < this.Layers.length; ++i) {
				let oldWeightsLayer = [];

				const layer = this.Layers[i];
				for (const node in layer.Nodes) {
					let oldWeightsNode = [];

					let output = `n${node}\n`;
					for (const connection of layer.Nodes[node].Connections) {
						output += 'w: ' + connection.weight;
						oldWeightsNode.push(connection.weight);
					}
					// console.log(output + '\n');

					oldWeightsLayer.push(oldWeightsNode);
				}
				// console.log('\n\n\n');

				oldWeights.push(oldWeightsLayer);
			}




nodeDeltas.push(
	this.learningRate *
		errors[i - 1][j] *
		sigmoid(
			connection.inputNode.value *
				connection.weight
		) *
		(1 -
			sigmoid(
				connection.inputNode.value *
					connection.weight
			)) *
		connection.inputNode.value
);
log(
	`\nweight adjustment for layer${i} node${j} connection${x}:`
);
log('connection.weight', connection.weight);
log('errors[i - 1][j]', errors[i - 1][j]);
log('connection.weight', connection.weight);

nodeDeltas.push(
	this.learningRate *
		errors[i - 1][j] *
		node.value *
		connection.inputNode.value
);
*/
