const fs = require('fs');

const Network = require('../src/Network');

const testing = require('../test.json');

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

for (let i = 0; i < 10; ++i) {
	console.log('i: ', i);
	// load
	const inputWeights = JSON.parse(
		fs.readFileSync('weights/weights.0.' + i + '.json', 'utf8')
	);
	const inputBiases = JSON.parse(
		fs.readFileSync('biases/biases.0.' + i + '.json', 'utf8')
	);

	network.setWeights(inputWeights);
	network.setBiases(inputBiases);

	let tries = 0;
	let hits = 0;
	for (let i = 0; i < 10000; ++i) {
		const data = testing[i];
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
	console.log('error: ', 1 - hits / tries);
	console.log('percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%');
	console.log('\n\n');
}
