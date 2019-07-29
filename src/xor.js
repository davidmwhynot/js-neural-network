const Network = require('./Network');

module.exports = () => {
	const network = new Network({
		learningRate: 2,
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

	for (let i = 0; i < 3000000; ++i) {
		// for (let i = 0; i < 1; ++i) {
		network.train(shuffle(trainingData));
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
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
