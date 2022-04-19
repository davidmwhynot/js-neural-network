// !!! imports !!!
const chalk = require('chalk');
const { writeFileSync, mkdirSync } = require('fs');

const getDirectories = require('./utils/getDirectories');
const Network = require('./Network');

const train = require('../train.json');
const test = require('../test.json');

// !!! config !!!
const args = process.argv;
console.log(args);

const TRAINING_DATA_PERCENTAGE = 10;

const TRAINING_ROUNDS = 1;
const TRAINING_CHUNK_SIZE = 6000;

const HIDDEN_LAYER_SIZE = 64;
const LEARNING_RATE = 1;

// const TRAINING_ROUNDS = Number(process.argv[2]) || 2;
// const TRAINING_CHUNK_SIZE = Number(process.argv[3]) || 200;

// const HIDDEN_LAYER_SIZE = Number(process.argv[4]) || 512;
// const LEARNING_RATE = Number(process.argv[5]) || 1;

// const TRAIN_GPU = Number(process.argv[6]) || 1;

// get weights directory
let computeOutputDir = './output/';

const outputDirs = getDirectories('./output');

if (outputDirs.length > 0) {
	const outputDirNumbers = outputDirs.map(dir => Number(dir[dir.length - 1]));
	const highestOutputDir = Math.max(...outputDirNumbers);
	computeOutputDir += '' + (Number(highestOutputDir) + 1);
} else {
	computeOutputDir += '0';
}

const OUTPUT_DIRECTORY = computeOutputDir;

// create output directories
mkdirSync(OUTPUT_DIRECTORY);
mkdirSync(OUTPUT_DIRECTORY + '/weights');
mkdirSync(OUTPUT_DIRECTORY + '/biases');

log('CONFIG');
log('TRAINING_DATA_PERCENTAGE', TRAINING_DATA_PERCENTAGE);
log('TRAINING_ROUNDS', TRAINING_ROUNDS);
log('TRAINING_CHUNK_SIZE', TRAINING_CHUNK_SIZE);
log('HIDDEN_LAYER_SIZE', HIDDEN_LAYER_SIZE);
log('LEARNING_RATE', LEARNING_RATE);
log('\n');

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

// !!! exports !!!
module.exports = () => {
	let gpuTestTimes = [];

	const gpuStart = Date.now();
	console.time('gpu');
	for (let round = 0; round < TRAINING_ROUNDS; ++round) {
		log(
			'==============================================================================='
		);
		log('\n\n\n\n\n\n\n');
		log(
			'==============================================================================='
		);
		log(`training round ${round + 1}`);
		// network.train(trainingData);

		// if (TRAIN_GPU === 1) {
		network.trainGPU({
			data: shuffle(trainingData),
			chunkSize: TRAINING_CHUNK_SIZE,
			round: round,
			rounds: TRAINING_ROUNDS,
			timeLogFlag: true,
			logFlag: false,
			memoryLogFlag: true
		});
		console.timeLog('gpu');
		// } else {
		// }

		log(`training round ${round + 1}`);

		for (let i = 0; i < 2; ++i) {
			let testExample = test[i];

			network.setInputs(testExample.image);

			network.calculate();

			const outputs = network.getOutputs();

			const guess = outputs.indexOf(Math.max(...outputs));

			log('guess:');
			log(guess);

			log('label:');
			log(testExample.label);
			let x = 0;
			let output = '';
			for (let j = 0; j < 28; ++j) {
				for (let k = 0; k < 28; ++k) {
					output += chalk.bgRgb(
						testExample.image[x],
						testExample.image[x],
						testExample.image[x]
					)('  ');
					++x;
				}
				output += '\n';
			}
			console.log(output);

			console.table(outputs);
			log('\n\n');
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

			if (tries % 500 === 0) {
				log('guess:');
				log(guess);

				log('label:');
				log(data.label);

				let output = '';
				let x = 0;
				for (let j = 0; j < 28; ++j) {
					for (let k = 0; k < 28; ++k) {
						output += chalk.bgRgb(
							data.image[x],
							data.image[x],
							data.image[x]
						)('  ');
						++x;
					}
					output += '\n';
				}
				console.log(output);

				console.table(outputs);
				log('\n\n');
			}
		}
		gpuTestTimes.push(Date.now() - gpuTestStart);

		log('misses: ', tries - hits);
		log('hits: ', hits);
		log('tries: ', tries);
		log('percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%');

		const weightsFileName =
			OUTPUT_DIRECTORY + '/weights/' + round + '.json';
		const biasesFileName = OUTPUT_DIRECTORY + '/biases/' + round + '.json';

		log('file names');
		log('weights file', weightsFileName);
		log('biases file', biasesFileName);

		writeFileSync(
			weightsFileName,
			JSON.stringify(network.getWeights()),
			'utf8'
		);
		writeFileSync(
			biasesFileName,
			JSON.stringify(network.getBiases()),
			'utf8'
		);
	}
	console.timeEnd('gpu');
	const gpuTime = Date.now() - gpuStart;
	log(gpuTime);

	log('\n\n\n');

	// log('network.inspect:', network.inspect());
};

// ! functions !
function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}

function log(...s) {
	console.log(...s);
}
