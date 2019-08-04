const { GPU, input } = require('gpu.js');

const Layer = require('./Layer');
const Matrix = require('./Matrix');

const gpu = new GPU();

// common GPU kernel settings
const kernelSettings = {
	optimizeFloatMemory: true,
	// precision: 'single',
	loopMaxIterations: 8192
	// dynamicOutput: true,
	// dynamicInput: true
};

class Network {
	constructor({ layers, learningRate }) {
		this.Layers = [];
		this.learningRate = learningRate;

		for (const layer of layers) {
			this.Layers.push(new Layer(layer.numNodes));
		}

		// create GPU Kernels
		// feed forward kernels
		const forward_multiply_kernels = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			forward_multiply_kernels[i] = gpu.createKernel(
				function(a, b) {
					let sum = 0;
					// i = y = a.rows
					// j = x = b.cols
					for (let k = 0; k < this.constants.size; ++k) {
						sum += a[this.thread.y][k] * b[k][this.thread.x];
					}
					return sum;
				},
				{
					...kernelSettings,
					output: [1, this.Layers[i].Nodes.length],
					constants: {
						size: this.Layers[i - 1].Nodes.length
					}
				}
			);
		}

		const forward_calculate_kernels = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			forward_calculate_kernels[i] = gpu.createKernel(
				function(a, b) {
					return (
						1 /
						(1 +
							Math.exp(
								-(
									a[this.thread.x][this.thread.y] +
									b[this.thread.x][this.thread.y]
								)
							))
					);
				},
				{
					...kernelSettings,
					output: {
						x: 1,
						y: this.Layers[i].Nodes.length
					}
				}
			);
		}

		this.forward_kernels = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			const forward_multiply_kernel = forward_multiply_kernels[i];
			const forward_calculate_kernel = forward_calculate_kernels[i];
			this.forward_kernels[i] = gpu.combineKernels(
				forward_multiply_kernel,
				forward_calculate_kernel,
				function(w, i, b) {
					return forward_calculate_kernel(forward_multiply_kernel(w, i), b);
				}
			);
		}

		this.forward_layer_kernel = gpu.create;

		// training kernels
		this.weight_ho_deltas_kernel = gpu.createKernel(
			function(a, b) {
				let sum = 0;
				// i = y = a.rows
				// j = x = b.cols
				for (let k = 0; k < this.constants.size; ++k) {
					sum += a[this.thread.x][k] * b[k][this.thread.y];
				}
				return sum;
			},
			{
				...kernelSettings,
				output: [
					this.Layers[this.Layers.length - 2].Nodes.length,
					this.Layers[this.Layers.length - 1].Nodes.length
				],
				constants: { size: 1 }
			}
		);

		this.hidden_errors_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.hidden_errors_kernels[z] = gpu.createKernel(
				function(a, b) {
					let sum = 0;
					// i = y = a.rows
					// j = x = b.cols
					for (let k = 0; k < this.constants.size; ++k) {
						sum += a[this.thread.x][k] * b[k][this.thread.y];
					}
					return sum;
				},
				{
					...kernelSettings,
					output: [1, this.Layers[z - 2].Nodes.length],
					constants: {
						size: this.Layers[z - 1].Nodes.length
					}
				}
			);
		}

		this.weight_deltas_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.weight_deltas_kernels[z] = gpu.createKernel(
				function(a, b) {
					let sum = 0;
					// i = y = a.rows
					// j = x = b.cols
					for (let k = 0; k < this.constants.size; ++k) {
						sum += a[this.thread.x][k] * b[k][this.thread.y];
					}
					return sum;
				},
				{
					...kernelSettings,
					output: [
						this.Layers[z - 3].Nodes.length,
						this.Layers[z - 2].Nodes.length
					],
					constants: {
						size: 1
					}
				}
			);
		}

		this.connectLayers();
	}

	feedForward(inputArray) {
		let inputs = Matrix.fromArray(inputArray);
		for (let i = 1; i < this.Layers.length; ++i) {
			const weights = new Matrix(
				this.Layers[i].Nodes.length,
				this.Layers[i - 1].Nodes.length
			);
			weights.data = this.Layers[i].getWeights();

			const biases = new Matrix(this.Layers[i].Nodes.length);
			biases.data = this.Layers[i].getBiases();

			console.time('cpu');
			// calculate values
			inputs = Matrix.multiply(weights, inputs);
			inputs.add(biases);
			inputs.map(sigmoid);
			console.timeEnd('cpu');
		}
		return inputs.toArray();
	}

	feedForwardGPU(inputArray) {
		let inputs = inputArray;

		for (let i = 1; i < this.Layers.length; ++i) {
			const weights = this.Layers[i].getWeights();
			const biases = this.Layers[i].getBiases();

			console.time('gpu');
			inputs = this.forward_kernels[i](
				weights,
				input(new Float32Array(inputs.flat()), [1, inputs.length]),
				biases
			);
			console.timeEnd('gpu');
		}
		return inputs;
	}

	inspect() {
		const inspect = [];
		for (let i = 0; i < this.Layers.length; ++i) {
			const layer = this.Layers[i];

			for (let j = 0; j < layer.Nodes.length; ++j) {
				const node = layer.Nodes[j];

				const inspectNode = {
					layer: i,
					node: j,
					val: node.value,
					connections: []
				};

				for (let k = 0; k < node.Connections.length; ++k) {
					const connection = node.Connections[k];

					inspectNode.connections.push({
						num: k,
						weight: connection.weight,
						inputVal: connection.inputNode.value
					});
				}
				inspect.push(inspectNode);
			}
		}
		return inspect;
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

	trainGPU2(data, chunkSize, round, timeLogFlag, logFlag) {
		const log = (...s) => {
			logFlag ? console.log(...s) : '';
		};
		const table = (s, x) => {
			if (logFlag) {
				console.log(s);
				console.table(x);
			}
		};

		const chunks = Math.floor(data.length / chunkSize);

		for (let chunk = 0; chunk < chunks; ++chunk) {
			if (timeLogFlag) {
				console.time('chunk');
			}

			const examples = [];

			for (let i = 0; i < chunkSize; ++i) {
				// log('i + chunk * chunkSize', i + chunk * chunkSize);
				examples.push(data[i + chunk * chunkSize]);
			}

			// if (examples[0]) {
			// if (false) {
			const exampleHiddenDeltas = [];
			const outputDeltas = [];

			const exampleHiddenBiasGradients = [];
			const outputBiasGradients = [];

			for (const example of examples) {
				this.setInputs(example.inputs);
				this.calculate();

				// let nodes = this.inspect();
				// for (const node of nodes) {
				// 	log(node);
				// }
				// log('\n\n');

				// back prop
				let inputs = this.Layers[0].getNodeValsTransposed();
				let hidden = this.Layers[
					this.Layers.length - 2
				].getNodeValsTransposed();

				let outputs = this.Layers[
					this.Layers.length - 1
				].getNodeValsTransposed();
				let targets = Matrix.fromArray(example.outputs).data;

				// calculate the error
				// ERROR = TARGETS - OUTPUTS
				// let output_errors = Matrix.subtract(targets, outputs);

				let output_errors_kernel = gpu.createKernel(
					function(a, b) {
						return (
							a[this.thread.x][this.thread.y] - b[this.thread.x][this.thread.y]
						);
					},
					{
						...kernelSettings,
						output: {
							x: this.Layers[this.Layers.length - 1].Nodes.length,
							y: 1
						},
						pipeline: true
					}
				);
				let output_errors = output_errors_kernel(targets, outputs);

				// Calculate gradient
				// let gradients = Matrix.map(outputs, sigmoidPrime);
				let gradients_kernel = gpu.createKernel(
					function(a) {
						return (
							a[this.thread.x][this.thread.y] *
							(1 - a[this.thread.x][this.thread.y])
						);
					},
					{
						...kernelSettings,
						output: {
							x: this.Layers[this.Layers.length - 1].Nodes.length,
							y: 1
						},
						pipeline: true
					}
				);

				let gradients = gradients_kernel(outputs);

				// gradients.multiply(output_errors);
				let gradients_multply_kernel = gpu.createKernel(
					function(a, b) {
						return (
							a[this.thread.x][this.thread.y] * b[this.thread.x][this.thread.y]
						);
					},
					{
						...kernelSettings,
						output: {
							x: 1,
							y: this.Layers[this.Layers.length - 1].Nodes.length
						},
						pipeline: true
					}
				);
				let gradients_multiply = gradients_multply_kernel(
					gradients,
					output_errors
				);

				// gradients.multiply(this.learningRate);
				let gradients_learningRate_kernel = gpu.createKernel(
					function(a, b) {
						return a[this.thread.x][this.thread.y] * b;
					},
					{
						...kernelSettings,
						output: {
							x: this.Layers[this.Layers.length - 1].Nodes.length,
							y: 1
						},
						pipeline: true
					}
				);
				let gradients_multiply_learningRate = gradients_learningRate_kernel(
					gradients_multiply,
					this.learningRate
				);
				log(
					'gradients_multiply_learningRate',
					gradients_multiply_learningRate.toArray()
				);

				// Calculate deltas
				// let hidden_T = Matrix.transpose(hidden);
				let hidden_T_kernel = gpu.createKernel(function(a, b) {}, {
					...kernelSettings,
					output: {
						x: this.Layers[1].Nodes.length,
						y: 1
					}
				});
				let weight_ho_deltas = Matrix.gpuMultiplyKernel(
					gradients,
					hidden_T,
					this.weight_ho_deltas_kernel
				);

				// Adjust the weights by deltas
				let exampleOutputDeltas = [];
				for (
					let i = 0;
					i < this.Layers[this.Layers.length - 1].Nodes.length;
					++i
				) {
					const node = this.Layers[this.Layers.length - 1].Nodes[i];

					const nodeDeltas = [];
					for (let j = 0; j < node.Connections.length; ++j) {
						nodeDeltas.push(weight_ho_deltas.data[i][j]);
					}
					exampleOutputDeltas.push(nodeDeltas);
				}
				outputDeltas.push(exampleOutputDeltas);

				// Adjust the output bias by its deltas (which is just the gradients)
				outputBiasGradients.push(gradients.toArray());

				// XXX HIDDEN LAYER XXX
				let errors = output_errors;

				const layerDeltas = [];
				const layerBiasGradients = [];

				for (let z = this.Layers.length; z > 2; --z) {
					hidden = Matrix.fromArray(this.Layers[z - 2].getNodeVals());
					inputs = Matrix.fromArray(this.Layers[z - 3].getNodeVals());
					//  create weights matrix
					const weights = new Matrix(
						this.Layers[z - 1].Nodes.length,
						this.Layers[z - 2].Nodes.length
					);

					// populate weights matrix
					for (let i = 0; i < this.Layers[z - 1].Nodes.length; ++i) {
						const node = this.Layers[z - 1].Nodes[i];

						for (let j = 0; j < node.Connections.length; ++j) {
							const connection = node.Connections[j];

							// set values in weight matrix's data array to the corresponding connection's weight
							weights.data[i][j] = connection.weight;
						}
					}

					// Calculate the hidden layer errors
					let w_t = Matrix.transpose(weights);
					let hidden_errors = Matrix.gpuMultiplyKernel(
						w_t,
						errors,
						this.hidden_errors_kernels[z]
					);
					errors = hidden_errors;

					// Calculate hidden gradient
					let hidden_gradient = Matrix.map(hidden, sigmoidPrime);
					hidden_gradient.multiply(errors);
					hidden_gradient.multiply(this.learningRate);

					// Calculate hidden deltas
					let i_T = Matrix.transpose(inputs);
					let weight_deltas = Matrix.gpuMultiplyKernel(
						hidden_gradient,
						i_T,
						this.weight_deltas_kernels[z]
					);
					// log('a: hidden_gradient', hidden_gradient);
					// log('b: i_T', i_T);
					// log('weight_deltas', hidden_errors);

					const nodeDeltas = [];
					for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
						const node = this.Layers[z - 2].Nodes[i];

						const weightDeltas = [];
						for (let j = 0; j < node.Connections.length; ++j) {
							weightDeltas.push(weight_deltas.data[i][j]);
						}
						nodeDeltas.push(weightDeltas);
					}
					layerDeltas[z - 2] = nodeDeltas;

					// Adjust the hidden bias by its deltas (which is just the hidden gradients)
					layerBiasGradients[z - 2] = hidden_gradient.toArray();
					// log('\n\n');
				}
				exampleHiddenDeltas.push(layerDeltas);

				exampleHiddenBiasGradients.push(layerBiasGradients);

				// log('\n\n');
				// nodes = this.inspect();
				// for (const node of nodes) {
				// 	log(node);
				// }
				log('\n\n');
				log('\n\n');
				log('\n\n');
			}

			for (
				let i = 0;
				i < this.Layers[this.Layers.length - 1].Nodes.length;
				++i
			) {
				const node = this.Layers[this.Layers.length - 1].Nodes[i];

				let biasSum = 0;
				for (let k = 0; k < chunkSize; ++k) {
					biasSum += outputBiasGradients[k][i];
				}
				node.bias += biasSum / chunkSize;

				for (let j = 0; j < node.Connections.length; ++j) {
					const connection = node.Connections[j];

					let sum = 0;

					for (let k = 0; k < chunkSize; ++k) {
						sum += Number(outputDeltas[k][i][j]);
					}

					connection.weight += sum / chunkSize;
				}
			}

			for (let z = this.Layers.length; z > 2; --z) {
				for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
					const node = this.Layers[z - 2].Nodes[i];

					let biasSum = 0;
					for (let k = 0; k < chunkSize; ++k) {
						biasSum += Number(exampleHiddenBiasGradients[k][z - 2][i]);
					}
					node.bias += biasSum / chunkSize;

					for (let j = 0; j < node.Connections.length; ++j) {
						const connection = node.Connections[j];

						let sum = 0;

						for (let k = 0; k < chunkSize; ++k) {
							sum += Number(exampleHiddenDeltas[k][z - 2][i][j]);
						}

						connection.weight += sum / chunkSize;
					}
				}
			}

			if (timeLogFlag) {
				console.log(`round: ${round}\tchunk: ${chunk + 1} / ${chunks}`);
				console.timeEnd('chunk');
			}

			// log('\n\n');
			// let nodesOutside = this.inspect();
			// for (const node of nodesOutside) {
			// 	log(node);
			// }
			// log('\n\n');
			// log('\n\n');
			// log('\n\n');
			// log('\n\n');
			// }
		}
	}

	trainGPU(data, chunkSize, round, timeLogFlag, logFlag) {
		const log = (...s) => {
			logFlag ? console.log(...s) : '';
		};
		const table = (s, x) => {
			if (logFlag) {
				console.log(s);
				console.table(x);
			}
		};

		const chunks = Math.floor(data.length / chunkSize);

		for (let chunk = 0; chunk < chunks; ++chunk) {
			if (timeLogFlag) {
				console.time('chunk');
			}

			const examples = [];

			for (let i = 0; i < chunkSize; ++i) {
				// log('i + chunk * chunkSize', i + chunk * chunkSize);
				examples.push(data[i + chunk * chunkSize]);
			}

			// if (examples[0]) {
			// if (false) {
			const exampleHiddenDeltas = [];
			const outputDeltas = [];

			const exampleHiddenBiasGradients = [];
			const outputBiasGradients = [];

			for (const example of examples) {
				this.setInputs(example.inputs);
				this.calculate();

				// let nodes = this.inspect();
				// for (const node of nodes) {
				// 	log(node);
				// }
				// log('\n\n');

				// back prop
				let inputs = Matrix.fromArray(example.inputs);
				let hidden = Matrix.fromArray(
					this.Layers[this.Layers.length - 2].getNodeVals()
				);

				let outputs = Matrix.fromArray(this.getOutputs());
				let targets = Matrix.fromArray(example.outputs);

				// calculate the error
				// ERROR = TARGETS - OUTPUTS
				let output_errors = Matrix.subtract(targets, outputs);

				// Calculate gradient
				let gradients = Matrix.map(outputs, sigmoidPrime);
				gradients.multiply(output_errors);
				gradients.multiply(this.learningRate);

				// Calculate deltas
				let hidden_T = Matrix.transpose(hidden);
				let weight_ho_deltas = Matrix.gpuMultiplyKernel(
					gradients,
					hidden_T,
					this.weight_ho_deltas_kernel
				);

				// Adjust the weights by deltas
				let exampleOutputDeltas = [];
				for (
					let i = 0;
					i < this.Layers[this.Layers.length - 1].Nodes.length;
					++i
				) {
					const node = this.Layers[this.Layers.length - 1].Nodes[i];

					const nodeDeltas = [];
					for (let j = 0; j < node.Connections.length; ++j) {
						nodeDeltas.push(weight_ho_deltas.data[i][j]);
					}
					exampleOutputDeltas.push(nodeDeltas);
				}
				outputDeltas.push(exampleOutputDeltas);

				// Adjust the output bias by its deltas (which is just the gradients)
				outputBiasGradients.push(gradients.toArray());

				// XXX HIDDEN LAYER XXX
				let errors = output_errors;

				const layerDeltas = [];
				const layerBiasGradients = [];

				for (let z = this.Layers.length; z > 2; --z) {
					hidden = Matrix.fromArray(this.Layers[z - 2].getNodeVals());
					inputs = Matrix.fromArray(this.Layers[z - 3].getNodeVals());
					//  create weights matrix
					const weights = new Matrix(
						this.Layers[z - 1].Nodes.length,
						this.Layers[z - 2].Nodes.length
					);

					// populate weights matrix
					for (let i = 0; i < this.Layers[z - 1].Nodes.length; ++i) {
						const node = this.Layers[z - 1].Nodes[i];

						for (let j = 0; j < node.Connections.length; ++j) {
							const connection = node.Connections[j];

							// set values in weight matrix's data array to the corresponding connection's weight
							weights.data[i][j] = connection.weight;
						}
					}

					// Calculate the hidden layer errors
					let w_t = Matrix.transpose(weights);
					let hidden_errors = Matrix.gpuMultiplyKernel(
						w_t,
						errors,
						this.hidden_errors_kernels[z]
					);
					errors = hidden_errors;

					// Calculate hidden gradient
					let hidden_gradient = Matrix.map(hidden, sigmoidPrime);
					hidden_gradient.multiply(errors);
					hidden_gradient.multiply(this.learningRate);

					// Calculate hidden deltas
					let i_T = Matrix.transpose(inputs);
					let weight_deltas = Matrix.gpuMultiplyKernel(
						hidden_gradient,
						i_T,
						this.weight_deltas_kernels[z]
					);
					// log('a: hidden_gradient', hidden_gradient);
					// log('b: i_T', i_T);
					// log('weight_deltas', hidden_errors);

					const nodeDeltas = [];
					for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
						const node = this.Layers[z - 2].Nodes[i];

						const weightDeltas = [];
						for (let j = 0; j < node.Connections.length; ++j) {
							weightDeltas.push(weight_deltas.data[i][j]);
						}
						nodeDeltas.push(weightDeltas);
					}
					layerDeltas[z - 2] = nodeDeltas;

					// Adjust the hidden bias by its deltas (which is just the hidden gradients)
					layerBiasGradients[z - 2] = hidden_gradient.toArray();
					// log('\n\n');
				}
				exampleHiddenDeltas.push(layerDeltas);

				exampleHiddenBiasGradients.push(layerBiasGradients);

				// log('\n\n');
				// nodes = this.inspect();
				// for (const node of nodes) {
				// 	log(node);
				// }
				log('\n\n');
				log('\n\n');
				log('\n\n');
			}

			for (
				let i = 0;
				i < this.Layers[this.Layers.length - 1].Nodes.length;
				++i
			) {
				const node = this.Layers[this.Layers.length - 1].Nodes[i];

				let biasSum = 0;
				for (let k = 0; k < chunkSize; ++k) {
					biasSum += outputBiasGradients[k][i];
				}
				node.bias += biasSum / chunkSize;

				for (let j = 0; j < node.Connections.length; ++j) {
					const connection = node.Connections[j];

					let sum = 0;

					for (let k = 0; k < chunkSize; ++k) {
						sum += Number(outputDeltas[k][i][j]);
					}

					connection.weight += sum / chunkSize;
				}
			}

			for (let z = this.Layers.length; z > 2; --z) {
				for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
					const node = this.Layers[z - 2].Nodes[i];

					let biasSum = 0;
					for (let k = 0; k < chunkSize; ++k) {
						biasSum += Number(exampleHiddenBiasGradients[k][z - 2][i]);
					}
					node.bias += biasSum / chunkSize;

					for (let j = 0; j < node.Connections.length; ++j) {
						const connection = node.Connections[j];

						let sum = 0;

						for (let k = 0; k < chunkSize; ++k) {
							sum += Number(exampleHiddenDeltas[k][z - 2][i][j]);
						}

						connection.weight += sum / chunkSize;
					}
				}
			}

			if (timeLogFlag) {
				console.log(`round: ${round}\tchunk: ${chunk + 1} / ${chunks}`);
				console.timeEnd('chunk');
			}

			// log('\n\n');
			// let nodesOutside = this.inspect();
			// for (const node of nodesOutside) {
			// 	log(node);
			// }
			// log('\n\n');
			// log('\n\n');
			// log('\n\n');
			// log('\n\n');
			// }
		}
	}

	train(data, chunkSize, round, timeLogFlag, logFlag) {
		const log = (...s) => {
			logFlag ? console.log(...s) : '';
		};
		const table = (s, x) => {
			if (logFlag) {
				console.log(s);
				console.table(x);
			}
		};

		const chunks = Math.floor(data.length / chunkSize);

		for (let chunk = 0; chunk < chunks; ++chunk) {
			if (timeLogFlag) {
				console.time('chunk');
			}

			const examples = [];

			for (let i = 0; i < chunkSize; ++i) {
				// log('i + chunk * chunkSize', i + chunk * chunkSize);
				examples.push(data[i + chunk * chunkSize]);
			}

			// if (examples[0]) {
			// if (false) {
			const exampleHiddenDeltas = [];
			const outputDeltas = [];

			const exampleHiddenBiasGradients = [];
			const outputBiasGradients = [];

			for (const example of examples) {
				this.setInputs(example.inputs);
				this.calculate();

				// let nodes = this.inspect();
				// for (const node of nodes) {
				// 	log(node);
				// }
				// log('\n\n');

				// back prop
				let inputs = Matrix.fromArray(example.inputs);
				let hidden = Matrix.fromArray(
					this.Layers[this.Layers.length - 2].getNodeVals()
				);

				let outputs = Matrix.fromArray(this.getOutputs());
				let targets = Matrix.fromArray(example.outputs);

				// calculate the error
				// ERROR = TARGETS - OUTPUTS
				let output_errors = Matrix.subtract(targets, outputs);

				// Calculate gradient
				let gradients = Matrix.map(outputs, sigmoidPrime);
				gradients.multiply(output_errors);
				gradients.multiply(this.learningRate);

				// Calculate deltas
				let hidden_T = Matrix.transpose(hidden);
				log('hidden_T', hidden_T);
				let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

				// Adjust the weights by deltas
				let exampleOutputDeltas = [];
				for (
					let i = 0;
					i < this.Layers[this.Layers.length - 1].Nodes.length;
					++i
				) {
					const node = this.Layers[this.Layers.length - 1].Nodes[i];

					const nodeDeltas = [];
					for (let j = 0; j < node.Connections.length; ++j) {
						nodeDeltas.push(weight_ho_deltas.data[i][j]);
					}
					exampleOutputDeltas.push(nodeDeltas);
				}
				outputDeltas.push(exampleOutputDeltas);

				// Adjust the output bias by its deltas (which is just the gradients)
				outputBiasGradients.push(gradients.toArray());

				// XXX HIDDEN LAYER XXX
				let errors = output_errors;

				const layerDeltas = [];
				const layerBiasGradients = [];

				for (let z = this.Layers.length; z > 2; --z) {
					hidden = Matrix.fromArray(this.Layers[z - 2].getNodeVals());
					inputs = Matrix.fromArray(this.Layers[z - 3].getNodeVals());
					//  create weights matrix
					const weights = new Matrix(
						this.Layers[z - 1].Nodes.length,
						this.Layers[z - 2].Nodes.length
					);

					// populate weights matrix
					for (let i = 0; i < this.Layers[z - 1].Nodes.length; ++i) {
						const node = this.Layers[z - 1].Nodes[i];

						for (let j = 0; j < node.Connections.length; ++j) {
							const connection = node.Connections[j];

							// set values in weight matrix's data array to the corresponding connection's weight
							weights.data[i][j] = connection.weight;
						}
					}

					// Calculate the hidden layer errors
					let w_t = Matrix.transpose(weights);
					let hidden_errors = Matrix.multiply(w_t, errors);
					errors = hidden_errors;

					// Calculate hidden gradient
					let hidden_gradient = Matrix.map(hidden, sigmoidPrime);
					hidden_gradient.multiply(errors);
					hidden_gradient.multiply(this.learningRate);

					// Calculate hidden deltas
					let i_T = Matrix.transpose(inputs); // FLAG:
					let weight_deltas = Matrix.multiply(hidden_gradient, i_T);

					const nodeDeltas = [];
					for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
						const node = this.Layers[z - 2].Nodes[i];

						const weightDeltas = [];
						for (let j = 0; j < node.Connections.length; ++j) {
							weightDeltas.push(weight_deltas.data[i][j]);
						}
						nodeDeltas.push(weightDeltas);
					}
					layerDeltas[z - 2] = nodeDeltas;

					// Adjust the hidden bias by its deltas (which is just the hidden gradients)
					layerBiasGradients[z - 2] = hidden_gradient.toArray();
				}
				exampleHiddenDeltas.push(layerDeltas);

				exampleHiddenBiasGradients.push(layerBiasGradients);

				// log('\n\n');
				// nodes = this.inspect();
				// for (const node of nodes) {
				// 	log(node);
				// }
				// log('\n\n');
			}

			// for (
			// 	let i = 0;
			// 	i < this.Layers[this.Layers.length - 1].Nodes.length;
			// 	++i
			// ) {
			// 	const node = this.Layers[this.Layers.length - 1].Nodes[i];

			// 	let biasSum = 0;
			// 	for (let k = 0; k < chunkSize; ++k) {
			// 		biasSum += outputBiasGradients[k][i];
			// 	}
			// 	node.bias += biasSum / chunkSize;

			// 	for (let j = 0; j < node.Connections.length; ++j) {
			// 		const connection = node.Connections[j];

			// 		let sum = 0;

			// 		for (let k = 0; k < chunkSize; ++k) {
			// 			sum += Number(outputDeltas[k][i][j]);
			// 		}

			// 		connection.weight += sum / chunkSize;
			// 	}
			// }

			// for (let z = this.Layers.length; z > 2; --z) {
			// 	for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
			// 		const node = this.Layers[z - 2].Nodes[i];

			// 		let biasSum = 0;
			// 		for (let k = 0; k < chunkSize; ++k) {
			// 			biasSum += Number(exampleHiddenBiasGradients[k][z - 2][i]);
			// 		}
			// 		node.bias += biasSum / chunkSize;

			// 		for (let j = 0; j < node.Connections.length; ++j) {
			// 			const connection = node.Connections[j];

			// 			let sum = 0;

			// 			for (let k = 0; k < chunkSize; ++k) {
			// 				sum += Number(exampleHiddenDeltas[k][z - 2][i][j]);
			// 			}

			// 			connection.weight += sum / chunkSize;
			// 		}
			// 	}
			// }

			if (timeLogFlag) {
				console.log(`round: ${round}\tchunk: ${chunk + 1} / ${chunks}`);
				console.timeEnd('chunk');
			}

			// log('\n\n');
			// let nodesOutside = this.inspect();
			// for (const node of nodesOutside) {
			// 	log(node);
			// }
			// log('\n\n');
			// log('\n\n');
			// log('\n\n');
			// log('\n\n');
			// }
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
