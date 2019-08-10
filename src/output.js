const fs = require('fs');

const Network = require('../src/Network');

const network = new Network({
	learningRate: 1,
	layers: [
		{
			numNodes: 784
		},
		{
			numNodes: 128
		},
		{
			numNodes: 10
		}
	]
});

const TRAINING_DATA_PERCENTAGE = 100;

const training = require('../train.json');
const testing = require('../test.json');

let trainingData = [];

for (
	let i = 0;
	i < Math.floor((training.length - 1) * (TRAINING_DATA_PERCENTAGE / 100));
	++i
) {
	const example = training[i];
	if (
		i ==
		Math.floor((training.length - 1) * (TRAINING_DATA_PERCENTAGE / 100) - 1)
	) {
		console.log(example.label);
	}
	let output = {
		inputs: example.image,
		outputs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	};

	output.outputs[example.label] = 1;

	trainingData.push(output);
}
// for (let i = 0; i < 1; ++i) {
for (let i = 0; i < 10; ++i) {
	// network.train(shuffle(trainingData), 10, i, 1, true, false);
	network.train(shuffle(trainingData), 500, i, 10, true, false);
	let tries = 0;
	let hits = 0;
	for (const data of testing) {
		++tries;
		// console.log(`\n\ntest round ${round + 1}`);
		// network.train(testingData);

		network.setInputs(data.image);

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

		if (guess == data.label) {
			++hits;
			// console.log('hit!');
		}

		// console.table(outputs);
		// for (const output in outputs) {
		// 	console.log('output ' + output + ': ', outputs[output]);
		// }
	}

	console.log('misses: ', tries - hits);
	console.log('hits: ', hits);
	console.log('tries: ', tries);
	console.log('percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%');
	console.log('\n\n');

	const weightsJSON = JSON.stringify(network.getWeights());
	fs.writeFileSync('weights.0.' + i + '.json', weightsJSON, 'utf8');

	const biasesJSON = JSON.stringify(network.getBiases());
	fs.writeFileSync('biases.0.' + i + '.json', biasesJSON, 'utf8');
}

const network1 = new Network({
	learningRate: 1,
	layers: [
		{
			numNodes: 784
		},
		{
			numNodes: 128
		},
		{
			numNodes: 64
		},
		{
			numNodes: 10
		}
	]
});

for (let i = 0; i < 10; ++i) {
	// network.train(shuffle(trainingData), 10, i, 1, true, false);
	network1.train(shuffle(trainingData), 500, i, 10, true, false);
	let tries = 0;
	let hits = 0;
	for (const data of testing) {
		++tries;
		// console.log(`\n\ntest round ${round + 1}`);
		// network1.train(testingData);

		network1.setInputs(data.image);

		network1.calculate();

		const outputs = network1.getOutputs();

		// console.log('guess:');
		// console.log(outputs);
		// const max = Math.max(1, 2, 3);
		// console.log(max);
		const guess = outputs.indexOf(Math.max(...outputs));
		// console.log(guess);
		// console.log('label:');
		// console.log(testExample.label);

		if (guess == data.label) {
			++hits;
			// console.log('hit!');
		}

		// console.table(outputs);
		// for (const output in outputs) {
		// 	console.log('output ' + output + ': ', outputs[output]);
		// }
	}

	console.log('misses: ', tries - hits);
	console.log('hits: ', hits);
	console.log('tries: ', tries);
	console.log('percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%');
	console.log('\n\n');
	const weightsJSON = JSON.stringify(network1.getWeights());
	fs.writeFileSync('weights.1.' + i + '.json', weightsJSON, 'utf8');

	const biasesJSON = JSON.stringify(network1.getBiases());
	fs.writeFileSync('biases.1.' + i + '.json', biasesJSON, 'utf8');
}

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
