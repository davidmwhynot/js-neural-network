const Network = require('./Network');

const train = require('../train.json');
const test = require('../test.json');

const args = process.argv;
console.log(args);

const TRAINING_DATA_PERCENTAGE = 100;

const TRAINING_ROUNDS = Number(process.argv[2]) || 1;
const TRAINING_ITERATIONS_ROUND = 1;
const TRAINING_CHUNK_SIZE = Number(process.argv[3]) || 200;

const TEST_ROUNDS = 9999;
const TEST_ITERATIONS_ROUND = 1;

const HIDDEN_LAYER_SIZE = Number(process.argv[4]) || 32;
const LEARNING_RATE = Number(process.argv[5]) || 1;

const TRAIN_GPU = Number(process.argv[6]) || 0;

/*



*/

let trainingData = [];

for (
	let i = 0;
	i < Math.floor((train.length - 1) * (TRAINING_DATA_PERCENTAGE / 100));
	++i
) {
	const example = train[i];
	let output = {
		inputs: example.image,
		outputs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	};

	output.outputs[example.label] = 1;

	trainingData.push(output);
}

const network = new Network({
	learningRate: LEARNING_RATE,
	layers: [
		{
			numNodes: 784
		},
		{
			numNodes: HIDDEN_LAYER_SIZE
		},
		{
			numNodes: 10
		}
	]
});

module.exports = () => {
	console.time('cpu');
	for (let round = 0; round < TRAINING_ROUNDS; ++round) {
		console.log(
			'=========================================================================================================================================='
		);
		console.log('\n\n\n\n\n\n\n');
		console.log(
			'=========================================================================================================================================='
		);
		console.log(`training round ${round + 1}`);
		// network.train(trainingData);

		console.time('time');

		for (let i = 0; i < TRAINING_ITERATIONS_ROUND; ++i) {
			if (TRAIN_GPU === 1) {
				network.trainGPU(
					shuffle(trainingData),
					TRAINING_CHUNK_SIZE,
					round,
					true,
					false
				);
			} else {
				network.train(
					shuffle(trainingData),
					TRAINING_CHUNK_SIZE,
					round,
					true,
					false
				);
			}
			console.timeLog('time');
		}

		// nodes = network.inspect();
		// let outputFlag = false;
		// let inputLayerVals = '';
		// for (const node of nodes) {
		// 	switch (node.layer) {
		// 		case 0:
		// 			inputLayerVals += ` n${node.node}: ${node.val}`;
		// 			break;
		// 		case 1:
		// 			if (!outputFlag) {
		// 				console.log('inputLayerVals\n', inputLayerVals);
		// 				outputFlag = true;
		// 			}
		// 			console.log('node ', node.node, '\tval ', node.val);
		// 			break;
		// 		case 2:
		// 			console.log(node);
		// 			break;
		// 	}
		// }

		console.log(`training round ${round + 1}`);
		console.log('time:');
		console.timeEnd('time');

		for (let i = 0; i < 2; ++i) {
			let testExample = test[i];

			network.setInputs(testExample.image);

			network.calculate();

			const outputs = network.getOutputs();

			const guess = outputs.indexOf(Math.max(...outputs));

			console.log('guess:');
			console.log(guess);

			console.log('label:');
			console.log(testExample.label);

			console.table(outputs);
			// for (const output in outputs) {
			// 	console.log('output ' + output + ': ', outputs[output]);
			// }
		}
		let hits = 0;
		let tries = 0;
		for (let round = 0; round < TEST_ROUNDS; ++round) {
			let testingData = [];
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

				testingData.push(output);
			}

			for (const data of testingData) {
				++tries;
				// console.log(`\n\ntest round ${round + 1}`);
				// network.train(testingData);

				let testExample = data;

				network.setInputs(testExample.inputs);

				network.calculate();

				const outputs = network.getOutputs();

				// console.log('guess:');
				// console.log(outputs);
				// const max = Math.max(1, 2, 3);
				// console.log(max);
				const guess = outputs.indexOf(Math.max(...outputs));
				// console.log(guess);
				// console.log('label:');
				// console.log(testExample.label);

				if (guess == testExample.label) {
					++hits;
					// console.log('hit!');
				}

				// console.table(outputs);
				// for (const output in outputs) {
				// 	console.log('output ' + output + ': ', outputs[output]);
				// }
			}
		}

		console.log('misses: ', tries - hits);
		console.log('hits: ', hits);
		console.log('tries: ', tries);
		console.log(
			'percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%'
		);
	}
	console.timeEnd('cpu');
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
