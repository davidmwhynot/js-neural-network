const Network = require('./Network');

const train = require('../train.json');
// const test = require('../test.json');

const args = process.argv;
console.log(args);

const TRAINING_DATA_PERCENTAGE = 1;

const TRAINING_ROUNDS = Number(process.argv[2]) || 1;
const TRAINING_CHUNK_SIZE = Number(process.argv[3]) || 200;

const HIDDEN_LAYER_SIZE = Number(process.argv[4]) || 32;
const LEARNING_RATE = Number(process.argv[5]) || 1;

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
	console.time('gpu');
	network.trainGPU({
		data: shuffle(trainingData),
		chunkSize: TRAINING_CHUNK_SIZE,
		round: 0,
		rounds: TRAINING_ROUNDS,
		timeLogFlag: true,
		logFlag: false,
		memoryLogFlag: true
	});
	console.timeEnd('gpu');
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
