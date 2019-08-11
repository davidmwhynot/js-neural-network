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

		this.connectLayers();

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

		// gpu 2 kernels
		this.output_errors_kernel = gpu.createKernel(
			function(a, b) {
				return (
					a[this.thread.x][this.thread.y] - b[this.thread.x][this.thread.y]
				);
			},
			{
				...kernelSettings,
				output: {
					y: this.Layers[this.Layers.length - 1].Nodes.length,
					x: 1
				},
				pipeline: true
			}
		);

		this.gradients_kernel = gpu.createKernel(
			function(a) {
				return (
					a[this.thread.x][this.thread.y] *
					(1 - a[this.thread.x][this.thread.y])
				);
			},
			{
				...kernelSettings,
				output: {
					y: this.Layers[this.Layers.length - 1].Nodes.length,
					x: 1
				},
				pipeline: true
			}
		);

		this.gradients_multply_kernel = gpu.createKernel(
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

		this.gradients_learningRate_kernel = gpu.createKernel(
			function(a, b) {
				return a[this.thread.x][this.thread.y] * b;
			},
			{
				...kernelSettings,
				output: {
					y: this.Layers[this.Layers.length - 1].Nodes.length,
					x: 1
				},
				pipeline: true
			}
		);

		// this.gradient_kernels = gpu.combineKernels(
		// 	this.gradients_kernel,
		// 	this.gradients_multply_kernel,
		// 	this.gradients_learningRate_kernel

		// )

		this.hidden_T_kernel = gpu.createKernel(
			function(a) {
				return a[this.thread.x][this.thread.y];
			},
			{
				...kernelSettings,
				output: {
					x: this.Layers[this.Layers.length - 2].Nodes.length,
					y: 1
				},
				pipeline: true
			}
		);

		this.weight_ho_deltas_kernel = gpu.createKernel(
			function(a, b) {
				return a[this.thread.y][0] * b[0][this.thread.x];
			},
			{
				...kernelSettings,
				output: {
					x: this.Layers[this.Layers.length - 2].Nodes.length,
					y: this.Layers[this.Layers.length - 1].Nodes.length
				}
			}
		);

		// hidden layer kernels
		this.w_t_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.w_t_kernels[z] = gpu.createKernel(
				function(a) {
					return a[this.thread.x][this.thread.y];
				},
				{
					...kernelSettings,
					output: {
						x: this.Layers[z - 1].Nodes.length,
						y: this.Layers[z - 2].Nodes.length
					},
					pipeline: true
				}
			);
		}

		this.hidden_errors_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.hidden_errors_kernels[z] = gpu.createKernel(
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
					// output: [1, this.Layers[z - 2].Nodes.length],
					output: {
						x: 1,
						y: this.Layers[z - 2].Nodes.length
					},
					constants: {
						size: this.Layers[z - 1].Nodes.length
					},
					pipeline: true
				}
			);
		}

		this.hidden_gradient_map_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.hidden_gradient_map_kernels[z] = gpu.createKernel(
				function(a) {
					return (
						a[this.thread.x][this.thread.y] *
						(1 - a[this.thread.x][this.thread.y])
					);
				},
				{
					...kernelSettings,
					output: {
						y: this.Layers[z - 2].Nodes.length,
						x: 1
					},
					pipeline: true
				}
			);
		}

		this.hidden_gradient_multiply_errors_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.hidden_gradient_multiply_errors_kernels[z] = gpu.createKernel(
				function(a, b) {
					return (
						a[this.thread.x][this.thread.y] * b[this.thread.x][this.thread.y]
					);
				},
				{
					...kernelSettings,
					output: {
						x: 1,
						y: this.Layers[z - 2].Nodes.length
					},
					pipeline: true
				}
			);
		}

		this.hidden_gradient_multiply_learningRate_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.hidden_gradient_multiply_learningRate_kernels[z] = gpu.createKernel(
				function(a, b) {
					return a[this.thread.x][this.thread.y] * b;
				},
				{
					...kernelSettings,
					output: {
						y: this.Layers[z - 2].Nodes.length,
						x: 1
					},
					pipeline: true
				}
			);
		}

		this.i_T_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.i_T_kernels[z] = gpu.createKernel(
				function(a) {
					return a[this.thread.x][this.thread.y];
				},
				{
					...kernelSettings,
					output: {
						x: this.Layers[z - 3].Nodes.length,
						y: 1
					},
					pipeline: true
				}
			);
		}

		this.weight_deltas_kernels = [];
		for (let z = this.Layers.length; z > 2; --z) {
			this.weight_deltas_kernels[z] = gpu.createKernel(
				function(a, b) {
					return a[this.thread.y][0] * b[0][this.thread.x];
				},
				{
					...kernelSettings,
					output: {
						x: this.Layers[z - 3].Nodes.length,
						y: this.Layers[z - 2].Nodes.length
					}
				}
			);
		}
	}

	trainGPU2({
		data,
		chunkSize,
		round,
		rounds,
		timeLogFlag,
		logFlag,
		memoryLogFlag
	}) {
		const log = (...s) => {
			logFlag ? console.log(...s) : '';
		};
		const table = (s, x) => {
			if (logFlag) {
				console.log(s);
				console.table(x);
			}
		};
		const memLog = () => {
			if (memoryLogFlag) {
				console.log('memory usage:');
				const usage = process.memoryUsage();
				console.log('rss: ' + usage.rss / 1000000 + 'MB');
				console.log('heapTotal: ' + usage.heapTotal / 1000000 + 'MB');
				console.log('heapUsed: ' + usage.heapUsed / 1000000 + 'MB');
				console.log('external: ' + usage.external / 1000000 + 'MB');
				console.log('\n');
			}
		};

		memLog();

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

			const exampleHiddenDeltas = [];
			const outputDeltas = [];

			const exampleHiddenBiasGradients = [];
			const outputBiasGradients = [];

			if (timeLogFlag) {
				console.time('calculate');
			}

			// initiate deltas
			for (
				let i = 0;
				i < this.Layers[this.Layers.length - 1].Nodes.length;
				++i
			) {
				const node = this.Layers[this.Layers.length - 1].Nodes[i];

				const nodeDeltas = [];
				for (let j = 0; j < node.Connections.length; ++j) {
					nodeDeltas.push(0);
				}
				outputDeltas.push(nodeDeltas);
			}

			let weightsLayers = [];
			for (let z = this.Layers.length; z > 2; --z) {
				weightsLayers[z] = this.Layers[z - 1].getWeights();

				const nodeDeltas = [];
				for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
					const node = this.Layers[z - 2].Nodes[i];

					const weightDeltas = [];
					for (let j = 0; j < node.Connections.length; ++j) {
						weightDeltas.push(0);
					}
					nodeDeltas.push(weightDeltas);
				}
				exampleHiddenDeltas[z - 2] = nodeDeltas;
			}

			// time data
			let outputSum = 0;
			let hiddenSum = 0;

			for (const example of examples) {
				// forward prop
				this.setInputs(example.inputs);
				this.calculate();

				// back prop
				const outputStart = Date.now();
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
				let output_errors = this.output_errors_kernel(targets, outputs);

				// Calculate gradient
				let gradients = this.gradients_kernel(outputs);
				let gradients_multiply = this.gradients_multply_kernel(
					gradients,
					output_errors
				);
				let gradients_multiply_learningRate = this.gradients_learningRate_kernel(
					gradients_multiply,
					this.learningRate
				);

				// Calculate deltas
				let hidden_T = this.hidden_T_kernel(hidden);
				let weight_ho_deltas = this.weight_ho_deltas_kernel(
					gradients_multiply_learningRate,
					hidden_T
				);

				// Adjust the weights by deltas
				for (
					let i = 0;
					i < this.Layers[this.Layers.length - 1].Nodes.length;
					++i
				) {
					const node = this.Layers[this.Layers.length - 1].Nodes[i];
					for (let j = 0; j < node.Connections.length; ++j) {
						outputDeltas[i][j] += weight_ho_deltas[i][j];
					}
				}
				// Adjust the output bias by its deltas (which is just the gradients)
				let biasGradients = gradients_multiply_learningRate
					.toArray()
					.map(i => i[0]);
				outputBiasGradients.push(biasGradients);

				outputSum += Date.now() - outputStart;

				// XXX HIDDEN LAYER XXX
				let errors = output_errors;

				const layerBiasGradients = [];

				const hiddenStart = Date.now();
				for (let z = this.Layers.length; z > 2; --z) {
					hidden = this.Layers[z - 2].getNodeValsTransposed();
					inputs = this.Layers[z - 3].getNodeValsTransposed();

					// Calculate the hidden layer errors
					let w_t = this.w_t_kernels[z](weightsLayers[z]);
					let hidden_errors = this.hidden_errors_kernels[z](w_t, errors);
					errors = hidden_errors;

					// Calculate hidden gradient
					let hidden_gradient_map = this.hidden_gradient_map_kernels[z](hidden);

					let hidden_gradient_multiply_errors = this.hidden_gradient_multiply_errors_kernels[
						z
					](hidden_gradient_map, errors);

					let hidden_gradient_multiply_learningRate = this.hidden_gradient_multiply_learningRate_kernels[
						z
					](hidden_gradient_multiply_errors, this.learningRate);

					// Calculate hidden deltas
					let i_T = this.i_T_kernels[z](inputs);
					let weight_deltas = this.weight_deltas_kernels[z](
						hidden_gradient_multiply_learningRate,
						i_T
					);

					for (let i = 0; i < this.Layers[z - 2].Nodes.length; ++i) {
						const node = this.Layers[z - 2].Nodes[i];
						for (let j = 0; j < node.Connections.length; ++j) {
							exampleHiddenDeltas[z - 2][i][j] += weight_deltas[i][j];
						}
					}

					// Adjust the hidden bias by its deltas (which is just the hidden gradients)
					layerBiasGradients[
						z - 2
					] = hidden_gradient_multiply_learningRate.toArray().map(i => i[0]);
				}

				exampleHiddenBiasGradients.push(layerBiasGradients);

				hiddenSum += Date.now() - hiddenStart;

				// log('chunk:', chunk);
				memLog();
			}
			if (timeLogFlag) {
				console.timeEnd('calculate');

				console.log('outputSum', outputSum);
				console.log('output average', outputSum / chunkSize);

				console.log('hiddenSum', hiddenSum);
				console.log('hidden average', hiddenSum / chunkSize);

				console.time('adjust');
			}

			memLog();

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

					connection.weight += outputDeltas[i][j] / chunkSize;
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

						connection.weight += exampleHiddenDeltas[z - 2][i][j] / chunkSize;
					}
				}
			}

			if (timeLogFlag) {
				console.timeEnd('adjust');
				console.timeEnd('chunk');
				console.log(
					`round: ${round + 1} / ${rounds}\tchunk: ${chunk + 1} / ${chunks}`
				);
				console.log('\n\n');
			}
		}
	}

	setWeights(weights) {
		if (this.Layers.length - 1 !== weights.length) {
			throw new Error(
				"Invalid input weights: dimensions must match network's."
			);
		} else {
			for (let i = 1; i < this.Layers.length; ++i) {
				this.Layers[i].setWeights(weights[i - 1]);
			}
		}
	}

	setBiases(biases) {
		if (this.Layers.length - 1 !== biases.length) {
			throw new Error("Invalid input biases: dimensions must match network's.");
		} else {
			for (let i = 1; i < this.Layers.length; ++i) {
				this.Layers[i].setBiases(biases[i - 1]);
			}
		}
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

			// console.time('gpu');
			inputs = this.forward_kernels[i](
				weights,
				input(new Float32Array(inputs.flat()), [1, inputs.length]),
				biases
			);
			// console.timeEnd('gpu');
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
					bias: node.bias,
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
			console.log(output);
		}
	}

	getWeights() {
		const output = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			output.push(this.Layers[i].getWeights());
		}
		return output;
	}

	getBiases() {
		const output = [];
		for (let i = 1; i < this.Layers.length; ++i) {
			output.push(this.Layers[i].getBiases());
		}
		return output;
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

	trainGPU(data, chunkSize, round, rounds, timeLogFlag, logFlag) {
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
				console.log(`round: ${round + 1}\tchunk: ${chunk + 1} / ${chunks}`);
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

	train(data, chunkSize, round, rounds, timeLogFlag, logFlag) {
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

			if (timeLogFlag) {
				console.time('calculate');
			}
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
				// log('output_errors', output_errors);

				// Calculate gradient
				let gradients = Matrix.map(outputs, sigmoidPrime);
				// log('gradients', gradients);
				gradients.multiply(output_errors);
				// log('gradients', gradients);
				gradients.multiply(this.learningRate);
				// log('gradients', gradients);

				// Calculate deltas
				let hidden_T = Matrix.transpose(hidden);
				// log('hidden_T', hidden_T);
				let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);
				// log('weight_ho_deltas', weight_ho_deltas);

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
				// console.log('gradients', gradients.toArray());
				outputBiasGradients.push(gradients.toArray());

				// log('\n');
				// XXX HIDDEN LAYER XXX
				let errors = output_errors;

				const layerDeltas = [];
				const layerBiasGradients = [];

				for (let z = this.Layers.length; z > 2; --z) {
					hidden = Matrix.fromArray(this.Layers[z - 2].getNodeVals());
					// log('hidden', hidden);
					inputs = Matrix.fromArray(this.Layers[z - 3].getNodeVals());
					// log('inputs', inputs);

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

					// log('weights', weights);

					// Calculate the hidden layer errors
					let w_t = Matrix.transpose(weights);
					// log('w_t', w_t);
					// log('errors', errors);
					let hidden_errors = Matrix.multiply(w_t, errors);
					// log('hidden_errors', hidden_errors);
					errors = hidden_errors;

					// Calculate hidden gradient
					let hidden_gradient = Matrix.map(hidden, sigmoidPrime);
					// log('hidden_gradient', hidden_gradient);
					hidden_gradient.multiply(errors);
					// log('hidden_gradient', hidden_gradient);
					hidden_gradient.multiply(this.learningRate);
					// log('hidden_gradient', hidden_gradient);

					// Calculate hidden deltas
					let i_T = Matrix.transpose(inputs);
					// log('i_T', i_T);
					let weight_deltas = Matrix.multiply(hidden_gradient, i_T);
					// log('weight_deltas', weight_deltas);

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
			if (timeLogFlag) {
				console.timeEnd('calculate');
				console.time('adjust');
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
				console.timeEnd('adjust');
				console.timeEnd('chunk');
				console.log(
					`round: ${round + 1} / ${rounds}\tchunk: ${chunk + 1} / ${chunks}`
				);
				console.log('\n');
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
