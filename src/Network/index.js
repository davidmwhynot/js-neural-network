const Layer = require('./Layer');
const Matrix = require('./Matrix');

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

	train(data, logFlag) {
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
				outputErrors.push(
					-(example.outputs[i] - outputs[i]) *
						(outputs[i] * (1 - outputs[i]))
				);
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
							// errorSum +=
							// 	lastHiddenErrorsRow[i] *
							// 	(connection.weight / totalWeight);
							errorSum +=
								lastHiddenErrorsRow[i] * connection.weight;
						}
						errorSum +=
							connection.inputNode.value *
							(1 - connection.inputNode.value);
					}

					hiddenErrorsRow.push(errorSum);
				}

				errors.push(hiddenErrorsRow);
			}

			errors.reverse();
			log1(errors);

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

						const delta =
							this.learningRate *
							errors[i - 1][j] *
							connection.inputNode.value;

						nodeDeltas.push(delta);

						const deltaNeg = delta < 0;
						let logflag = false;
						if (deltaNeg && example.outputs[0] > outputs[0]) {
							log1('\nwrong direction: positive');
							logflag = true;
						}
						if (!deltaNeg && example.outputs[0] < outputs[0]) {
							logflag = true;
							log1('\nwrong direction: negative');
						}
						if (logflag) {
							log1('delta');
							log1(delta);
							log1('this.learningRate');
							log1(this.learningRate);
							log1('errors[i-1][j]');
							log1(errors[i - 1][j]);
							log1('connection.inputNode.value');
							log1(connection.inputNode.value);
						}

						// connection.weight += delta;
					}

					layerDeltas.push(nodeDeltas);
				}

				exampleDeltas.push(layerDeltas);
			}
			deltas.push(exampleDeltas);

			log1('exampleDeltas');
			log1(exampleDeltas);

			log1('network outputs');
			log1(outputs);

			log1('desired outputs');
			log1(example.outputs);
			log1('\n\n');

			results.push({ guess: outputs, answer: example.outputs });
		}
		// for (let i = 0; i < deltas.length; ++i) {
		// log1('deltas');
		// log1(i, deltas[i]);
		// log1('...for results ', results[i]);
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
					// log1(
					// 	`\nadjustments for layer${i} node${j} connection${k}`
					// );
					// log1('pre connection.weight');
					// log1(connection.weight);
					// log1('sum / data.length');
					// log1(sum / data.length);

					connection.weight += sum / data.length;
					// log1('post connection.weight');
					// log1(connection.weight);
				}
			}
		}
	}

	// -(y-yHat) == yHat - y
	// -(4 - 2) == 2 - 4
	// 2 - 4 == -2

	train3(data, logFlag) {
		const log = (...s) => {
			logFlag ? console.log(...s) : '';
		};

		const dJdW1s = [];
		const dJdW2s = [];

		for (const example of data) {
			this.setInputs(example.inputs);
			this.calculate();

			const yHat = Matrix.fromArray(this.getOutputs());
			const y = Matrix.fromArray(example.outputs);
			const yHatMinusy = Matrix.subtract(yHat, y);

			const layer3nodeVals = this.Layers[2].Nodes.map(n => n.value);
			const z3 = Matrix.fromArray(layer3nodeVals);
			z3.map(sigmoidPrime);

			const delta3 = Matrix.multiply(yHatMinusy, z3);
			log('delta3', delta3);

			const layer2nodeVals = this.Layers[1].Nodes.map(n => n.value);
			const a2 = Matrix.fromArray(layer2nodeVals);

			const dJdW2 = Matrix.multiply(a2, delta3);

			log('dJdW2', dJdW2);
			log('\n\n');

			dJdW2s.push(dJdW2.data);

			const w2array = this.Layers[1].Nodes.map(n =>
				n.Connections.map(c => c.weight)
			);

			const w2 = new Matrix(w2array.length, w2array[0].length);

			for (let i = 0; i < w2array.length; ++i) {
				for (let j = 0; j < w2array[i].length; ++j) {
					w2.data[i][j] = w2array[i][j];
				}
			}
			log('w2', w2);

			w2.multiply(delta3.data[0][0]);
			log('w2 multiplied', w2);

			const z2 = Matrix.fromArray(layer2nodeVals);
			z2.map(sigmoidPrime);

			const delta2 = Matrix.multiply(w2, z2);
			log('delta2', delta2);

			const X = Matrix.fromArray(example.inputs);
			log('X', X);

			const XTranspose = Matrix.transpose(X);
			log('XTranspose', XTranspose);

			const dJdW1 = Matrix.multiply(delta2, XTranspose);
			log('dJdW1', dJdW1);

			dJdW1s.push(dJdW1.data);

			log('\n\n');
			log('example.inputs', example.inputs);
			log('example.outputs', example.outputs);
			log('outputs', this.getOutputs());
			log('\n\n');
			log('\n\n');
		}
		log('dJdW1s');
		console.table(dJdW1s);
		log('dJdW2s');
		console.table(dJdW2s);
		log('\n\n');

		for (let i = 0; i < this.Layers[2].Nodes.length; ++i) {
			const node = this.Layers[2].Nodes[i];

			for (let j = 0; j < node.Connections.length; ++j) {
				const connection = node.Connections[j];

				let sum = 0;
				for (let k = 0; k < data.length; ++k) {
					sum += dJdW2s[k][j];
				}

				connection.weight -= this.learningRate * sum;
			}
		}

		for (let i = 0; i < this.Layers[1].Nodes.length; ++i) {
			const node = this.Layers[1].Nodes[i];

			for (let j = 0; j < node.Connections.length; ++j) {
				const connection = node.Connections[j];

				let sum = 0;
				for (let k = 0; k < data.length; ++k) {
					sum += dJdW2s[k][i][j];
				}

				connection.weight -= this.learningRate * sum;
			}
		}
	}
}

function log1(...s) {
	// console.log(...s);
}

function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x) {
	return x * (1 - x);
}

// function sigmoidPrime(x) {
// 	return sigmoid(x) * (1 - sigmoid(x));
// }

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
log1(
	`\nweight adjustment for layer${i} node${j} connection${x}:`
);
log1('connection.weight', connection.weight);
log1('errors[i - 1][j]', errors[i - 1][j]);
log1('connection.weight', connection.weight);

nodeDeltas.push(
	this.learningRate *
		errors[i - 1][j] *
		node.value *
		connection.inputNode.value
);
*/
