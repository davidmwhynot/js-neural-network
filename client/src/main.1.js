const Network = require('./Network');

const train = require('../train.json');
// const test = require('../test.json');

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
			numNodes: 2048
		},
		{
			numNodes: 2048
		}
	]
});

module.exports = () => {
	for (let i = 0; i < 10; ++i) {
		network.setInputs(trainingData[i].inputs);

		console.time('old');
		network.calculate();
		console.timeEnd('old');

		network.feedForward(trainingData[i].inputs);

		network.feedForwardGPU(trainingData[i].inputs);

		console.log('\n');
	}
	console.time('cputotal');
	for (let i = 0; i < 10; ++i) {
		network.feedForward(trainingData[i].inputs);
	}
	console.timeEnd('cputotal');

	console.time('gputotal');
	for (let i = 0; i < 10; ++i) {
		network.feedForwardGPU(trainingData[i].inputs);
	}
	console.timeEnd('gputotal');
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
