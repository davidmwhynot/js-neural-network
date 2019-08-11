const Network = require('./Network');

const ROUNDS = 1000;
const CHUNK_SIZE = 4;

module.exports = () => {
	const network = new Network({
		learningRate: 1,
		layers: [
			{
				numNodes: 4
			},
			{
				numNodes: 5
			},
			{
				numNodes: 3
			},
			{
				numNodes: 2
			}
		]
	});

	let trainingData = [];

	for (let i = 0; i < 4; ++i) {
		// for (let i = 0; i < 40; ++i) {
		trainingData.push({
			inputs: [0, 0, 0, 0],
			outputs: [0, 0]
		});
		trainingData.push({
			inputs: [1, 1, 0, 0],
			outputs: [1, 1]
		});
		trainingData.push({
			inputs: [0, 0, 1, 1],
			outputs: [1, 1]
		});
		trainingData.push({
			inputs: [1, 1, 1, 1],
			outputs: [0, 0]
		});
		// trainingData.push({
		// 	inputs: [0, 0],
		// 	outputs: [0]
		// });
		// trainingData.push({
		// 	inputs: [0, 1],
		// 	outputs: [1]
		// });
		// trainingData.push({
		// 	inputs: [1, 0],
		// 	outputs: [1]
		// });
		// trainingData.push({
		// 	inputs: [1, 1],
		// 	outputs: [0]
		// });
	}

	// gpu: 207895.571ms
	// 207898

	// const cpuStart = Date.now();
	// console.time('cpu');
	// for (let i = 0; i < ROUNDS; ++i) {
	// 	network.train(trainingData, CHUNK_SIZE, i, ROUNDS, true, false);

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
	// console.timeEnd('cpu');
	// const cpuTime = Date.now() - cpuStart;
	// console.log(cpuTime);
	// console.log('\n\n');

	const gpuStart = Date.now();
	console.time('gpu');
	for (let i = 0; i < ROUNDS; ++i) {
		if (i % 100 === 0) {
			console.log('i', i);
		}
		network.trainGPU2({
			data: shuffle(trainingData),
			chunkSize: CHUNK_SIZE,
			round: i,
			rounds: ROUNDS,
			timeLogFlag: false,
			logFlag: false,
			memoryLogFlag: false
		});
		// network.train(shuffle(trainingData), CHUNK_SIZE, i, ROUNDS, false, false);
	}
	network.setInputs([0, 0, 0, 0]);
	network.calculate();
	let outputs = network.getOutputs();
	console.log(outputs);

	network.setInputs([1, 1, 0, 0]);
	network.calculate();
	outputs = network.getOutputs();
	console.log(outputs);

	network.setInputs([0, 0, 1, 1]);
	network.calculate();
	outputs = network.getOutputs();
	console.log(outputs);

	network.setInputs([1, 1, 1, 1]);
	network.calculate();
	outputs = network.getOutputs();
	console.log(outputs);
	console.log('\n');
	console.timeEnd('gpu');
	const gpuTime = Date.now() - gpuStart;
	console.log(gpuTime);
	// console.log('cpuTime - gpuTime:', Number(cpuTime - gpuTime));
};

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
