let training = require('../train.json');
const testing = require('../test.json');

const TRAINING_DATA_PERCENTAGE = 2;

const main = async (
	emitter,
	network,
	training_rounds = 1,
	training_chunk_size = 200,
	test_rounds = 10
) => {
	let trainingData = [];

	training = shuffle(training);

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
	emitter.emit('training_start');
	await train(network, trainingData, training_rounds, training_chunk_size);
	emitter.emit('training_finish');

	for (let i = 0; i < test_rounds; ++i) {
		const testResults = await test(network, testing[i].image);

		const guess = testResults.indexOf(Math.max(...testResults));
		console.log('label', testing[i].label);
		console.log('guess', guess);
		console.log('testResults', testResults);
		console.log('\n');

		emitter.emit('test', {
			...testing[i],
			round: i,
			guess
		});
	}
	emitter.emit('testing_finish');
};

function train(network, trainingData, training_rounds, training_chunk_size) {
	return new Promise(resolve => {
		for (let i = 0; i < training_rounds; ++i) {
			network.train(
				shuffle(trainingData),
				training_chunk_size,
				i,
				training_rounds,
				true,
				false
			);
		}

		resolve();
	});
}

function test(network, data) {
	return new Promise(resolve => {
		network.setInputs(data);
		network.calculate();
		resolve(network.getOutputs());
	});
}

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}

module.exports = main;
