const Network = require('./Network');

module.exports = () => {
	const network = new Network({
		learningRate: 0.1,
		layers: [
			{
				numNodes: 2
			},
			{
				numNodes: 16
			},
			{
				numNodes: 16
			},
			{
				numNodes: 16
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
			outputs: [1]
		},
		{
			inputs: [1, 0],
			outputs: [1]
		},
		{
			inputs: [1, 1],
			outputs: [0]
		},
		{
			inputs: [0, 0],
			outputs: [0]
		},
		{
			inputs: [0, 1],
			outputs: [1]
		},
		{
			inputs: [1, 0],
			outputs: [1]
		},
		{
			inputs: [1, 1],
			outputs: [0]
		},
		{
			inputs: [0, 0],
			outputs: [0]
		},
		{
			inputs: [0, 1],
			outputs: [1]
		},
		{
			inputs: [1, 0],
			outputs: [1]
		},
		{
			inputs: [1, 1],
			outputs: [0]
		},
		{
			inputs: [0, 0],
			outputs: [0]
		},
		{
			inputs: [0, 1],
			outputs: [1]
		},
		{
			inputs: [1, 0],
			outputs: [1]
		},
		{
			inputs: [1, 1],
			outputs: [0]
		}
	];

	network.setInputs([0, 0]);
	network.calculate();
	let zeroOutputs = network.getOutputs();
	console.log(zeroOutputs);
	console.log('=========');

	for (let j = 0; j < 100; ++j) {
		for (let i = 0; i < 10000; ++i) {
			// for (let j = 0; j < 1; ++j) {
			// 	for (let i = 0; i < 1; ++i) {
			network.train(shuffle(trainingData), 4, false, false);
			// network.train2(shuffle(trainingData), 4, false, true);
			// network.train3(shuffle(trainingData), true);
			// network.train3(trainingData, true);
			// network.train(trainingData);
			// network.train(shuffle(trainingData), i == 299999);
		}

		network.setInputs([0, 0]);
		network.calculate();
		let outputs = network.getOutputs();
		console.log(outputs);

		network.setInputs([0, 1]);
		network.calculate();
		outputs = network.getOutputs();
		console.log(outputs);

		network.setInputs([1, 0]);
		network.calculate();
		outputs = network.getOutputs();
		console.log(outputs);

		network.setInputs([1, 1]);
		network.calculate();
		outputs = network.getOutputs();
		console.log(outputs);
		console.log('\n');
	}
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
