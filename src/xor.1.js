const Network = require('./Network');

module.exports = () => {
	const network = new Network({
		learningRate: 0.1,
		layers: [
			{
				numNodes: 2
			},
			{
				numNodes: 3
			},
			{
				numNodes: 4
			}
		]
	});

	let trainingData = [
		{
			inputs: [0, 0],
			outputs: [0, 0, 0, 0]
		}
		// {
		// 	inputs: [0, 1],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 0],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 1],
		// 	outputs: [0]
		// }
		// {
		// 	inputs: [0, 0],
		// 	outputs: [0]
		// },
		// {
		// 	inputs: [0, 1],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 0],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 1],
		// 	outputs: [0]
		// },
		// {
		// 	inputs: [0, 0],
		// 	outputs: [0]
		// },
		// {
		// 	inputs: [0, 1],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 0],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 1],
		// 	outputs: [0]
		// },
		// {
		// 	inputs: [0, 0],
		// 	outputs: [0]
		// },
		// {
		// 	inputs: [0, 1],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 0],
		// 	outputs: [1]
		// },
		// {
		// 	inputs: [1, 1],
		// 	outputs: [0]
		// }
	];

	network.train(trainingData, 1, 1, false, true);
	network.trainGPU2(trainingData, 1, 1, false, true);
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
