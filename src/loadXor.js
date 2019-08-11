// import
const fs = require('fs');
const Network = require('./Network');

// config
const network = new Network({
	learningRate: 1,
	layers: [
		{
			numNodes: 2
		},
		{
			numNodes: 2
		},
		{
			numNodes: 1
		}
	]
});

let trainingData = [
	{
		inputs: [0, 0],
		outputs: [0]
	},
	{
		inputs: [0, 1],
		outputs: [0]
	},
	{
		inputs: [1, 0],
		outputs: [0]
	},
	{
		inputs: [1, 1],
		outputs: [1]
	}
];

// main
console.table(shuffle(trainingData));

// train
for (let i = 0; i < 10000; ++i) {
	network.train(shuffle(trainingData), 4, i, 10000, false, false);
}

// test
network.setInputs([0, 0]);
network.calculate();
console.log(network.getOutputs());
network.setInputs([0, 1]);
network.calculate();
console.log(network.getOutputs());
network.setInputs([1, 0]);
network.calculate();
console.log(network.getOutputs());
network.setInputs([1, 1]);
network.calculate();
console.log(network.getOutputs());

console.log('\n');

// log
const weights = network.getWeights();
const biases = network.getBiases();

for (const layer in weights) {
	console.log('weights for layer ' + layer);
	console.table(weights[layer]);
}
for (const layer in biases) {
	console.log('biases for layer ' + layer);
	console.table(biases[layer]);
}

console.log('\n');

// output
const weightsJSON = JSON.stringify(weights);
fs.writeFileSync('weights.json', weightsJSON, 'utf8');

const biasesJSON = JSON.stringify(biases);
fs.writeFileSync('biases.json', biasesJSON, 'utf8');

// load
const inputWeights = JSON.parse(fs.readFileSync('weights.json', 'utf8'));
const inputBiases = JSON.parse(fs.readFileSync('biases.json', 'utf8'));

// console.log('inputWeights', inputWeights);
// console.log('inputBiases', inputBiases);

const network2 = new Network({
	learningRate: 1,
	layers: [
		{
			numNodes: 2
		},
		{
			numNodes: 2
		},
		{
			numNodes: 1
		}
	]
});

network2.setWeights(inputWeights);
network2.setBiases(inputBiases);

// debug
// const network2Weights = network2.getWeights();
// const network2Biases = network2.getBiases();

// for (const layer in network2Weights) {
// 	console.log('weights for layer ' + layer);
// 	console.table(network2Weights[layer]);
// }
// for (const layer in network2Biases) {
// 	console.log('biases for layer ' + layer);
// 	console.table(network2Biases[layer]);
// }

// const inspect = network2.inspect();
// for (const node of inspect) {
// 	console.log(node);
// }

// test
network2.setInputs([0, 0]);
network2.calculate();
console.log(network2.getOutputs());
console.log(network2.feedForwardGPU([0, 0]));
console.log(network2.feedForward([0, 0]));
network2.setInputs([0, 1]);
network2.calculate();
console.log(network2.getOutputs());
console.log(network2.feedForwardGPU([0, 1]));
console.log(network2.feedForward([0, 1]));
network2.setInputs([1, 0]);
network2.calculate();
console.log(network2.getOutputs());
console.log(network2.feedForwardGPU([1, 0]));
console.log(network2.feedForward([1, 0]));
network2.setInputs([1, 1]);
network2.calculate();
console.log(network2.getOutputs());
console.log(network2.feedForwardGPU([1, 1]));
console.log(network2.feedForward([1, 1]));

// functions
function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
