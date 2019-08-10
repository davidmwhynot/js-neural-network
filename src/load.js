// const fs = require('fs');

const Network = require('./Network');

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

const weights = network.getWeights();

for (const layer in weights) {
	console.log('layer ' + layer);
	console.table(weights[layer]);
}
