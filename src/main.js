const Network = require('./Network');

module.exports = (train, test) => {
	const network = new Network({
		learningRate: 0.1,
		layers: [
			{
				numNodes: 784
			},
			{
				numNodes: 64
			},
			{
				numNodes: 16
			},
			{
				numNodes: 10
			}
		]
	});

	const TRAINING_ROUNDS = 50;
	const TRAINING_ITERATIONS_ROUND = 1000;

	const TEST_ROUNDS = 1;
	const TEST_ITERATIONS_ROUND = 1;
	for (let round = 0; round < TRAINING_ROUNDS; ++round) {
		let trainingData = [];
		for (
			let i = TRAINING_ITERATIONS_ROUND * round;
			i < TRAINING_ITERATIONS_ROUND + TRAINING_ITERATIONS_ROUND * round;
			++i
		) {
			const example = train[i];
			let output = {
				inputs: example.image,
				outputs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			};

			output.outputs[example.label] = 1;

			// console.log(example.label);
			// console.log(output.outputs);

			trainingData.push(output);
		}

		console.log(`\n\ntraining round ${round + 1}`);
		// network.train(trainingData);
		network.train(trainingData);

		let testExample = test[1];

		network.setInputs(testExample.image);

		network.calculate();

		const outputs = network.getOutputs();

		const guess = outputs.indexOf(Math.max(...outputs));

		console.log('guess:');
		console.log(guess);

		if (guess == -1) {
			console.log(network.Layers);

			for (let i = 0; i < network.Layers.length; ++i) {
				const layer = network.Layers[i];

				console.log('\n\nlayer' + i);

				for (let j = 0; j < layer.Nodes.length; ++j) {
					const node = layer.Nodes[j];

					if (node.value == NaN) {
						console.log('node' + j);
						if (i > 0) {
							let output = '';
							for (let k = 0; k < node.Connections.length; ++k) {
								const connection = node.Connections[k];
								output += `c${k}: ${connection.weight} `;
								nanflag = true;
							}
							console.log(output);
						}
					}

					console.log('node' + j, node.value);
					if (i > 0) {
						let nanflag = false;
						let output = '';
						for (let k = 0; k < node.Connections.length; ++k) {
							const connection = node.Connections[k];
							if (connection.weight == NaN) {
								output += `c${k}: ${connection.weight} `;
								nanflag = true;
							}
						}
						if (nanflag) {
							console.log(output);
						}
					}
				}
			}
			break;
		}

		console.log('label:');
		console.log(testExample.label);

		for (const output in outputs) {
			console.log('output ' + output + ': ', outputs[output]);
		}
	}
	let hits = 0;
	let tries = 0;
	for (let round = 0; round < TEST_ROUNDS; ++round) {
		let trainingData = [];
		for (
			let i = TEST_ITERATIONS_ROUND * round;
			i < TEST_ITERATIONS_ROUND + TEST_ITERATIONS_ROUND * round;
			++i
		) {
			const example = test[i];
			let output = {
				inputs: example.image,
				outputs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				label: example.label
			};

			output.outputs[example.label] = 1;

			// console.log(example.label);
			// console.log(output.outputs);

			trainingData.push(output);
		}

		for (const data of trainingData) {
			++tries;
			console.log(`\n\ntest round ${round + 1}`);
			// network.train(trainingData);

			let testExample = data;

			network.setInputs(testExample.inputs);

			network.calculate();

			const outputs = network.getOutputs();

			console.log('guess:');
			// console.log(outputs);
			// const max = Math.max(1, 2, 3);
			// console.log(max);
			const guess = outputs.indexOf(Math.max(...outputs));
			console.log(guess);
			console.log('label:');
			console.log(testExample.label);

			if (guess == testExample.label) {
				++hits;
				console.log('hit!');
			}

			console.log('misses: ', tries - hits);
			console.log('hits: ', hits);
			console.log('tries: ', tries);
			console.log(
				'percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%'
			);

			// for (const output in outputs) {
			// 	console.log('output ' + output + ': ', outputs[output]);
			// }
		}
	}
};
