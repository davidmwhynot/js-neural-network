const Network = require('./Network');

const train = require('../train.json');
const test = require('../test.json');

const args = process.argv;
console.log(args);

const TRAINING_DATA_PERCENTAGE = 100;

const TRAINING_ROUNDS = 1;
const TRAINING_ITERATIONS_ROUND = 1;
const TRAINING_CHUNK_SIZE = 600;

const HIDDEN_LAYER_SIZE = 256;
const LEARNING_RATE = 1;

// const TRAINING_ROUNDS = Number(process.argv[2]) || 2;
// const TRAINING_ITERATIONS_ROUND = 1;
// const TRAINING_CHUNK_SIZE = Number(process.argv[3]) || 200;

// const HIDDEN_LAYER_SIZE = Number(process.argv[4]) || 512;
// const LEARNING_RATE = Number(process.argv[5]) || 1;

// const TRAIN_GPU = Number(process.argv[6]) || 1;

/*



*/

let trainingData = [];

for (
	let i = 0;
	i < Math.floor(train.length * (TRAINING_DATA_PERCENTAGE / 100));
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
	let gpuTestTimes = [];

	const gpuStart = Date.now();
	console.time('gpu');
	for (let round = 0; round < TRAINING_ROUNDS; ++round) {
		console.log(
			'==============================================================================='
		);
		console.log('\n\n\n\n\n\n\n');
		console.log(
			'==============================================================================='
		);
		console.log(`training round ${round + 1}`);
		// network.train(trainingData);

		for (let i = 0; i < TRAINING_ITERATIONS_ROUND; ++i) {
			// if (TRAIN_GPU === 1) {
			network.trainGPU2({
				data: shuffle(trainingData),
				chunkSize: TRAINING_CHUNK_SIZE,
				round: round,
				rounds: TRAINING_ROUNDS,
				timeLogFlag: true,
				logFlag: false,
				memoryLogFlag: false
			});
			console.timeLog('gpu');
			// } else {
			// }
		}

		console.log(`training round ${round + 1}`);

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

		const gpuTestStart = Date.now();
		for (const data of test) {
			++tries;

			network.setInputs(data.image);

			network.calculate();

			const outputs = network.getOutputs();
			// const outputs = network.feedForwardGPU(data.image);
			const guess = outputs.indexOf(Math.max(...outputs));

			if (guess == data.label) {
				++hits;
			}
		}
		gpuTestTimes.push(Date.now() - gpuTestStart);

		console.log('misses: ', tries - hits);
		console.log('hits: ', hits);
		console.log('tries: ', tries);
		console.log(
			'percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%'
		);
	}
	console.timeEnd('gpu');
	const gpuTime = Date.now() - gpuStart;
	console.log(gpuTime);

	// console.log('\n\n\n');

	// let cpuTestTimes = [];
	// const cpuStart = Date.now();
	// console.time('cpu');
	// for (let round = 0; round < TRAINING_ROUNDS; ++round) {
	// 	console.log(
	// 		'==============================================================================='
	// 	);
	// 	console.log('\n\n\n\n');
	// 	console.log(
	// 		'==============================================================================='
	// 	);
	// 	console.log(`training round ${round + 1}`);
	// 	// network.train(trainingData);

	// 	for (let i = 0; i < TRAINING_ITERATIONS_ROUND; ++i) {
	// 		// if (TRAIN_GPU === 1) {
	// 		// } else {
	// 		network.train(
	// 			shuffle(trainingData),
	// 			TRAINING_CHUNK_SIZE,
	// 			round,
	// 			TRAINING_ROUNDS,
	// 			true,
	// 			false
	// 		);
	// 		console.timeLog('cpu');
	// 		// }
	// 	}

	// 	console.log(`training round ${round + 1}`);

	// 	for (let i = 0; i < 2; ++i) {
	// 		let testExample = test[i];

	// 		network.setInputs(testExample.image);

	// 		network.calculate();

	// 		const outputs = network.getOutputs();

	// 		const guess = outputs.indexOf(Math.max(...outputs));

	// 		console.log('guess:');
	// 		console.log(guess);

	// 		console.log('label:');
	// 		console.log(testExample.label);

	// 		console.table(outputs);
	// 		// for (const output in outputs) {
	// 		// 	console.log('output ' + output + ': ', outputs[output]);
	// 		// }
	// 	}
	// 	let hits = 0;
	// 	let tries = 0;

	// 	const cpuTestStart = Date.now();
	// 	for (const data of test) {
	// 		++tries;

	// 		network.setInputs(data.image);

	// 		network.calculate();

	// 		const outputs = network.getOutputs();
	// 		const guess = outputs.indexOf(Math.max(...outputs));

	// 		if (guess == data.label) {
	// 			++hits;
	// 		}
	// 	}
	// 	cpuTestTimes.push(Date.now() - cpuTestStart);

	// 	console.log('misses: ', tries - hits);
	// 	console.log('hits: ', hits);
	// 	console.log('tries: ', tries);
	// 	console.log(
	// 		'percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%'
	// 	);
	// }
	// console.timeEnd('cpu');
	// const cpuTime = Date.now() - cpuStart;
	// console.log(cpuTime);
	// console.log('cpuTime - gpuTime:', Number(cpuTime - gpuTime));
	// console.log('gpuTestTimes:', gpuTestTimes);
	// console.log('cpuTestTimes:', cpuTestTimes);
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
