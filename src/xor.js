const Network = require('./Network');

module.exports = () => {
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

	for (let k = 0; k < 5; ++k) {
		const network = new Network({
			learningRate: 0.1,
			layers: [
				{
					numNodes: 2
				},
				{
					numNodes: 4
				},
				{
					numNodes: 1,
					output: true
				}
			]
		});
		console.log('\n\n\n', k);
		for (let j = 0; j < 5; ++j) {
			for (let i = 0; i < 500; ++i) {
				network.train2(shuffle(trainingData), false);
				// network.train2(shuffle(trainingData), i == 49 || i == 9);
			}
			console.log(network.getNodes());

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
	}

	// network.Layers[1].Nodes[0].Connections[0].weight = 0;
	// network.Layers[1].Nodes[0].Connections[1].weight = 0;

	// network.Layers[1].Nodes[1].Connections[0].weight = 1;
	// network.Layers[1].Nodes[1].Connections[1].weight = 1;

	// network.Layers[2].Nodes[0].Connections[0].weight = -5;
	// network.Layers[2].Nodes[0].Connections[1].weight = 5;

	// network.setInputs([0, 0]);
	// network.calculate();
	// let outputs = network.getOutputs();
	// console.log(outputs);
	// // console.log('=========');

	// network.setInputs([0, 1]);
	// network.calculate();
	// outputs = network.getOutputs();
	// console.log(outputs);
	// // console.log('=========');

	// network.setInputs([1, 0]);
	// network.calculate();
	// outputs = network.getOutputs();
	// console.log(outputs);
	// // console.log('=========');

	// network.setInputs([1, 1]);
	// network.calculate();
	// outputs = network.getOutputs();
	// console.log(outputs);
	// // console.log('=========');

	// for (let j = 0; j < 50; ++j) {
	// 	for (let i = 0; i < 30000; ++i) {
	// 		// network.train(shuffle(trainingData));
	// 		// network.train(shuffle(trainingData), false);
	// 		network.train(shuffle(trainingData), i == 29999);
	// 	}

	// 	network.setInputs([0, 0]);
	// 	network.calculate();
	// 	let outputs = network.getOutputs();
	// 	console.log(outputs);

	// 	network.setInputs([0, 1]);
	// 	network.calculate();
	// 	outputs = network.getOutputs();
	// 	console.log(outputs);

	// 	network.setInputs([1, 0]);
	// 	network.calculate();
	// 	outputs = network.getOutputs();
	// 	console.log(outputs);

	// 	network.setInputs([1, 1]);
	// 	network.calculate();
	// 	outputs = network.getOutputs();
	// 	console.log(outputs);
	// 	console.log('\n');
	// }
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
