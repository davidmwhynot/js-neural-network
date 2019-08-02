const Layer = require('./Layer');

class Network {
	constructor({ layers, learningRate }) {
		this.Layers = [];
		this.learningRate = learningRate;

		for (const layer of layers) {
			this.Layers.push(new Layer(layer.numNodes, layer.output));
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

	getNodes() {
		const output = [];
		for (const layer in this.Layers) {
			const layerNodes = [];
			for (const node of this.Layers[layer].Nodes) {
				const nodeObj = {
					layer: layer,
					val: node.value,
					conns: []
				};
				for (const connection of node.Connections) {
					nodeObj.conns.push({
						inputVal: connection.inputNode.value,
						weight: connection.weight
					});
				}
				layerNodes.push(nodeObj);
			}
			output.push(layerNodes);
		}
		return output;
	}

	getOutputs() {
		let output = [];

		for (const node of this.Layers[this.Layers.length - 1].Nodes) {
			output.push(node.value);
		}

		return output;
	}

	train(data, logFlag) {
		let deltas = [];
		let results = [];
		for (const example of data) {
			log2('\n\n');
			this.setInputs(example.inputs);
			this.calculate();

			let outputs = this.getOutputs();

			// errors
			let errors = [];
			let outputErrors = [];
			for (let i = 0; i < outputs.length; ++i) {
				outputErrors.push(
					(example.outputs[i] - outputs[i]) * (outputs[i] * (1 - outputs[i]))
				);
				// outputErrors.push(0.5 * (example.outputs[i] - outputs[i]) ** 2);
			}
			errors.push(outputErrors);

			log2('outputErrors');
			table(outputErrors);

			// hidden errors
			// for (let j = this.Layers.length - 1; j > 0; --j) {
			for (let j = this.Layers.length - 2; j > 0; --j) {
				const layer = this.Layers[j];

				let hiddenErrorsRow = [];

				// loop through hidden layer nodes
				for (const node of layer.Nodes) {
					// get total weight for this node's connections
					// let totalWeight = 0;

					// for (const connection of node.Connections) {
					// 	totalWeight += connection.weight;
					// }

					let errorSum = 0;
					let lastHiddenErrorsRow = errors[errors.length - 1];
					// loop through nodes connections
					for (const connection of node.Connections) {
						// loop through last hidden layer's errors
						for (let i = 0; i < lastHiddenErrorsRow.length; ++i) {
							// errorSum +=
							// 	lastHiddenErrorsRow[i] *
							// 	(connection.weight / totalWeight);
							errorSum += lastHiddenErrorsRow[i] * connection.weight;
						}
						errorSum += node.value * (1 - node.value);
					}

					hiddenErrorsRow.push(errorSum);
				}

				errors.push(hiddenErrorsRow);
			}

			errors.reverse();
			log2('errors');
			table(errors);

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

						// const delta =
						// 	this.learningRate * errors[i - 1][j] * sigmoid(node.value);
						const delta =
							this.learningRate * errors[i - 1][j] * connection.inputNode.value;

						nodeDeltas.push(delta);

						const deltaNeg = delta < 0;
						let logflag = false;
						if (deltaNeg && example.outputs[0] > outputs[0]) {
							log('\nwrong direction: positive');
							logflag = true;
						}
						if (!deltaNeg && example.outputs[0] < outputs[0]) {
							logflag = true;
							log('\nwrong direction: negative');
						}
						if (logflag) {
							log('delta');
							log(delta);
							log('this.learningRate');
							log(this.learningRate);
							log('errors[i-1][j]');
							log(errors[i - 1][j]);
							log('connection.inputNode.value');
							log(connection.inputNode.value);
						}

						// connection.weight += delta;
					}

					layerDeltas.push(nodeDeltas);
				}

				exampleDeltas.push(layerDeltas);
			}
			deltas.push(exampleDeltas);

			if (logFlag) {
				console.log('errors');
				console.table(errors);

				console.log('nodes');
				const getNodesOutput = this.getNodes();
				console.log(getNodesOutput);
				// for (const node in getNodesOutput) {
				// 	console.log(`layer ${node.layer} node ${node} val ${node.val}`);
				// 	console.table(node.conns);
				// 	console.log('\n');
				// }

				console.log('exampleDeltas');
				console.table(exampleDeltas);

				console.log('network outputs');
				console.log(outputs);

				console.log('desired outputs');
				console.log(example.outputs);
				console.log('\n\n\n\n');
			}
			log2('nodes');
			table(this.getNodes());

			log2('exampleDeltas');
			table(exampleDeltas);

			log2('network outputs');
			log2(outputs);

			log2('desired outputs');
			log2(example.outputs);
			log('\n\n');

			results.push({ guess: outputs, answer: example.outputs });
		}
		// for (let i = 0; i < deltas.length; ++i) {
		// log('deltas');
		// log(i, deltas[i]);
		// log('...for results ', results[i]);
		// }

		for (let i = 1; i < this.Layers.length; ++i) {
			const layer = this.Layers[i];

			for (let j = 0; j < layer.Nodes.length; ++j) {
				const node = layer.Nodes[j];

				for (let k = 0; k < node.Connections.length; ++k) {
					const connection = node.Connections[k];

					let sum = 0;
					for (let l = 0; l < data.length; ++l) {
						sum += deltas[l][i - 1][j][k];
					}
					// log(
					// 	`\nadjustments for layer${i} node${j} connection${k}`
					// );
					// log('pre connection.weight');
					// log(connection.weight);
					// log2('sum / data.length');
					if (logFlag) {
						console.log(`layer ${i}  node ${j}  conn ${k}`);
						console.log(sum / data.length);
					}
					log2(`layer ${i}  node ${j}  conn ${k}`);
					log2(sum / data.length);

					connection.weight += sum / data.length;
					// log('post connection.weight');
					// log(connection.weight);
				}
			}
		}
	}

	train2(data, logFlag) {
		function log(...s) {
			if (logFlag) {
				console.log(...s);
			}
		}

		let gradients = {
			outputGradients: [],
			hiddenGradients: []
		};
		let results = [];
		for (const example of data) {
			let exampleGradients = [];
			log('example', example);

			this.setInputs(example.inputs);
			this.calculate();

			let outputs = this.getOutputs();

			log('outputs', outputs);

			// output layer
			let outputGradients = [];
			let outputDeltas = [];

			for (
				let i = 0;
				i < this.Layers[this.Layers.length - 1].Nodes.length;
				++i
			) {
				log('i', i);
				let node = this.Layers[this.Layers.length - 1].Nodes[i];
				log('node.value', node.value);

				const delta =
					-(example.outputs[i] - node.value) *
					(sigmoid(node.value) - (sigmoid(node.value) ^ 2));
				log('delta', delta);
				for (const connection of node.Connections) {
					log('connection.inputNode.value', connection.inputNode.value);
					outputGradients.push(connection.inputNode.value * delta);
				}

				outputDeltas.push(delta);

				log('\n');
			}

			gradients.outputGradients.push(outputGradients);

			// hidden layer

			let hiddenDeltas = [];
			for (
				let i = 0;
				i < this.Layers[this.Layers.length - 1].Nodes.length;
				++i
			) {
				let node = this.Layers[this.Layers.length - 1].Nodes[i];

				for (const connection of node.Connections) {
					hiddenDeltas.push(
						connection.weight *
							outputDeltas[i] *
							(connection.inputNode.value - (connection.inputNode.value ^ 2))
					);
				}
			}

			log('hiddenDeltas', hiddenDeltas);
			let hiddenGradients = [];

			for (
				let i = 0;
				i < this.Layers[this.Layers.length - 2].Nodes.length;
				++i
			) {
				const node = this.Layers[this.Layers.length - 2].Nodes[i];

				const nodeHiddenGradients = [];
				for (const connection of node.Connections) {
					nodeHiddenGradients.push(
						connection.inputNode.value * hiddenDeltas[i]
					);
				}
				hiddenGradients.push(nodeHiddenGradients);
			}

			gradients.hiddenGradients.push(hiddenGradients);

			log('hiddenGradients', hiddenGradients);

			// let hiddenGradients = [];
			// let hiddenDeltas = [];

			// for(let i = 0; i < this.Layers[this.Layers.length-2].Nodes.length; ++i) {
			// 	log('i', i);
			// 	let node = this.Layers[this.Layers.length - 2].Nodes[i];

			// 	log('node.value', node.value);

			// 	const deltas = [];

			// 	deltas.push();
			// 		-(example.outputs[i] - node.value) *
			// 		(node.value - (node.value ^ 2));
			// 	log('delta', delta);

			// 	for (const connection of node.Connections) {
			// 		log(
			// 			'connection.inputNode.value',
			// 			connection.inputNode.value
			// 		);
			// 		outputGradients.push(connection.inputNode.value * delta);
			// 	}

			// 	outputDeltas.push(delta);

			// log('\n');
			// }

			log('outputGradients', outputGradients);
			log('outputDeltas', outputDeltas);
			log('\n\n\n\n');

			// gradients.push(exampleGradients);
		}

		for (const node of this.Layers[2].Nodes) {
			for (let i = 0; i < node.Connections.length; ++i) {
				const connection = node.Connections[i];
				let sum = 0;
				for (let k = 0; k < data.length; ++k) {
					sum += gradients.outputGradients[k][i];
				}
				connection.weight -= this.learningRate * (sum / data.length);
			}
		}

		for (let i = 0; i < this.Layers[1].Nodes.length; ++i) {
			const node = this.Layers[1].Nodes[i];

			for (let j = 0; j < node.Connections.length; ++j) {
				const connection = node.Connections[j];
				let sum = 0;

				for (let k = 0; k < data.length; ++k) {
					sum += gradients.hiddenGradients[k][i][j];
				}
				connection.weight -= this.learningRate * (sum / data.length);
			}
		}

		// console.table(gradients);

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
		// 			// log2('sum / data.length');
		// 			if (logFlag) {
		// 				log(`layer ${i}  node ${j}  conn ${k}`);
		// 				log(sum / data.length);
		// 			}
		// 			log2(`layer ${i}  node ${j}  conn ${k}`);
		// 			log2(sum / data.length);

		// 			connection.weight += sum / data.length;
		// 			// log('post connection.weight');
		// 			// log(connection.weight);
		// 		}
		// 	}
		// }
	}
}

// x =   1 / ( 1 + ( e ^ -(  x * ( 1 / ( 1 + ( e ^ -(  h * a ) ) ) ) ) ) );

// x = -(e^(w/(e^(-a h) + 1) + a h) ((o - 1) e^(w/(e^(-a h) + 1)) + o))/((e^(a h) + 1) (e^(w/(e^(-a h) + 1)) + 1)^3)

function log1(...s) {
	// console.log(...s);
}

function log2(...s) {
	// console.log(...s);
}

function table(x) {
	// table(x);
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
