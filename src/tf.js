const tf = require('@tensorflow/tfjs-node-gpu');
const chalk = require('chalk');

const train = require('../train.json');
const test = require('../test.json');

// tf.setBackend('cpu');

console.log('\n');
console.log('backend');
console.log(tf.getBackend());
console.log('\n');

let rawData = [];
let rawLabels = [];

shuffle(train);

for (let i = 0; i < 60000; ++i) {
	rawData.push(train[i].image);
	let outputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

	outputs[train[i].label] = 1;
	rawLabels.push(outputs);
}

const data = tf.tensor(rawData);
const labels = tf.tensor(rawLabels);

async function main() {
	const model = tf.sequential({
		layers: [
			tf.layers.dense({
				inputShape: [784],
				units: 4096,
				activation: 'sigmoid'
			}),
			tf.layers.dense({ units: 4096, activation: 'sigmoid' }),
			tf.layers.dense({ units: 4096, activation: 'sigmoid' }),
			tf.layers.dense({ units: 4096, activation: 'sigmoid' }),
			// tf.layers.dense({ units: 512, activation: 'sigmoid' }),
			// tf.layers.dense({ units: 1024, activation: 'sigmoid' }),
			// tf.layers.dense({ units: 256, activation: 'sigmoid' }),
			tf.layers.dense({ units: 10, activation: 'sigmoid' })
		]
	});

	// print info
	model.weights.forEach(w => {
		console.log(w.name, w.shape);
	});

	// compile with training params
	model.compile({
		optimizer: 'sgd',
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	});

	// train
	// const data = tf.randomNormal([100, 784]);
	// const labels = tf.randomUniform([100, 10]);

	log('training');
	console.time('training');
	console.time('batch');
	function onBatchEnd(batch, logs) {
		console.timeEnd('batch');
		log(batch + '\tAccuracy: ' + logs.acc * 100 + '%\n');
		console.time('batch');
	}
	console.timeEnd('batch');
	console.timeEnd('training');

	// Train for 5 epochs with batch size of 32.
	const info = await model.fit(data, labels, {
		epochs: 100,
		batchSize: 256,
		callbacks: { onBatchEnd },
		verbose: 0
	});

	let tries = 0;
	let hits = 0;
	// model.predict(tf.tensor(test)).print();
	for (const data of test) {
		++tries;

		// network.setInputs(data.image);
		// network.calculate();
		// const outputs = network.getOutputs();
		// const outputs = network.feedForwardGPU(data.image);

		// tf predict
		// const { data: outputs } = model.predict(tf.tensor(data.image, [1, 784]));
		const outputs = await model.predict(tf.tensor(data.image, [1, 784])).data();
		// outputs.print();

		const guess = outputs.indexOf(Math.max(...outputs));
		if (guess == data.label) {
			++hits;
		}

		// if (tries % 1000 === 0) {
		// 	log('try', tries);
		// 	log('guess:');
		// 	log(guess);

		// 	log('label:');
		// 	log(data.label);

		// 	let output = '';
		// 	let x = 0;
		// 	for (let j = 0; j < 28; ++j) {
		// 		for (let k = 0; k < 28; ++k) {
		// 			output += chalk.bgRgb(data.image[x], data.image[x], data.image[x])(
		// 				'  '
		// 			);
		// 			++x;
		// 		}
		// 		output += '\n';
		// 	}
		// 	console.log(output);

		// 	console.table(outputs);
		// 	log('\n\n');
		// }
	}

	log('misses: ', tries - hits);
	log('hits: ', hits);
	log('tries: ', tries);
	log('percentage: ' + Math.round((hits / tries) * 10000) / 100 + '%');
	log('Final accuracy', info.history.acc);
}

tf.tidy(main);

// ! functions !
function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}

function log(...s) {
	console.log(...s);
}
