const csv = require('csvtojson');
const moment = require('moment');
const { writeFileSync } = require('fs');

const transactionsOutput = [];
function transactions(j) {
	return new Promise(async resolve => {
		const json = await csv({
			noheader: true,
			headers: ['time', 'price', 'amount']
		}).fromFile('data/33-bitstamp.csv');
		// }).fromFile('data/bitstampUSD.csv');

		console.log('data loaded... length', json.length);

		let x = 0;
		let inputs = [];
		let outputs = [];

		// for (let i = 0; i < 33000; ++i) {
		for (let i = 0; i < 1000; ++i) {
			for (let j = 0; j < 999; ++j) {
				inputs.push(Number(json[x].price));
				++x;
			}

			outputs.push(Number(json[x].price));
			++x;
		}

		console.log('inputs/outputs fetched');

		console.log('inputs', inputs);
		console.log('outputs', outputs);

		// normalize data
		// let inputsMin = Math.min(...inputs);
		let inputsMin = inputs.reduce(function(a, b) {
			return Math.min(a, b);
		});
		log('inputsMin', inputsMin);
		// let inputsMax = Math.max(...inputs);
		let inputsMax = inputs.reduce(function(a, b) {
			return Math.max(a, b);
		});
		log('inputsMax', inputsMax);

		// let outputsMin = Math.min(...outputs);
		let outputsMin = outputs.reduce(function(a, b) {
			return Math.min(a, b);
		});
		log('outputsMin', outputsMin);
		// let outputsMax = Math.max(...outputs);
		let outputsMax = outputs.reduce(function(a, b) {
			return Math.max(a, b);
		});
		log('outputsMax', outputsMax);

		let inputsMaxMinusMin = inputsMax - inputsMin;
		log('inputsMaxMinusMin', inputsMaxMinusMin);
		let outputsMaxMinusMin = outputsMax - outputsMin;
		log('outputsMaxMinusMin', outputsMaxMinusMin);

		for (let i = 0; i < inputs.length - 1; ++i) {
			inputs[i] = (inputs[i] - inputsMin) / inputsMaxMinusMin;
		}

		for (let i = 0; i < outputs.length - 1; ++i) {
			outputs[i] = (outputs[i] - outputsMin) / outputsMaxMinusMin;
		}

		console.log('inputs/outputs normalized');

		console.log('generating data');
		const output = [];
		x = 0;
		// for (let i = 0; i < (j.length - (j.length % 1000)) / 1000; ++i) {
		// for (let i = 0; i < 33000; ++i) {
		for (let i = 0; i < 999; ++i) {
			let segment = {
				inputs: [],
				outputs: []
			};

			for (let j = 0; j < 999; ++j) {
				segment.inputs.push(inputs[x]);
				++x;
			}

			// for (let j = 0; j < 100; ++j) {
			segment.outputs.push(outputs[i]);
			++x;
			// }

			output.push(segment);

			// const usage = process.memoryUsage();
			// console.log('rss: ' + (usage.rss / 1000000 / 8192) * 100 + '%');
			// console.log(
			// 	'heapTotal: ' + (usage.heapTotal / 1000000 / 8192) * 100 + '%'
			// );
			// console.log('heapUsed: ' + (usage.heapUsed / 1000000 / 8192) * 100 + '%');
			// console.log('external: ' + (usage.external / 1000000 / 8192) * 100 + '%');
			// console.log('\n\n');
		}

		resolve(output);

		/*



		*/
		// const json = await csv({
		// 	noheader: true,
		// 	headers: ['time', 'price', 'amount']
		// }).fromFile('data/' + j + '-bitstamp.csv');

		// const output = [];
		// let x = 0;
		// // for (let i = 0; i < (j.length - (j.length % 1000)) / 1000 - 1; ++i) {
		// for (let i = 0; i < 1000; ++i) {
		// 	let segment = {
		// 		inputs: [],
		// 		outputs: []
		// 	};

		// 	for (let j = 0; j < 999; ++j) {
		// 		segment.inputs.push(Number(json[x].price));
		// 		++x;
		// 	}

		// 	// for (let j = 0; j < 100; ++j) {
		// 	segment.outputs.push(Number(json[x].price));
		// 	++x;
		// 	// }

		// 	output.push(segment);
		// }

		// resolve(output);
	});
}

function minutes(j) {
	return new Promise(async resolve => {
		const json = await csv({
			noheader: true,
			headers: ['time', 'price', 'amount']
		}).fromFile('data/' + j + '-bitstamp.csv');
		// console.log('json', json);

		const output = [];

		for (let i = 0; i < json.length; ++i) {
			let minutes = 0;
		}

		writeFileSync('');

		resolve();
	});
}

function stats(j) {
	return new Promise(async resolve => {
		const json = await csv({
			noheader: true,
			headers: ['time', 'price', 'amount']
		}).fromFile('data/' + j + '-bitstamp.csv');
		// console.log('json', json);

		console.log('entries: ', json.length);

		let sum = 0;
		let min = 9999999;
		let max = json[1].time - json[0].time;
		let zeros = 0;

		let a = Number(json[0].time);
		let b = 0;
		let sixtys = 0;

		for (let i = 1; i < json.length; ++i) {
			if (i % 100000 === 0) {
				console.log('i', i);
			}
			// const date = new Date(Number(data.time) * 1000);
			// console.log(date.toString());

			const data = json[i];

			b = Number(data.time);
			// console.log(b - a);
			let difference = b - a;
			sum += difference;

			if (difference > max) max = difference;
			if (difference < min && difference > 0) min = difference;
			if (difference === 0) ++zeros;
			if (difference > 60) ++sixtys;

			a = b;
		}

		console.log('average', sum / (json.length - 1));
		console.log('max', max);
		console.log('min', min);
		console.log('zeros', zeros);
		console.log('sixtys', sixtys);
		console.log('sixtys percent: ', (sixtys / json.length) * 100 + '%');

		resolve();
	});
}

// async function runStats() {
// 	for (let i = 0; i < 34; ++i) {
// 		console.log('i', i);
// 		await stats(i + 1);
// 		console.log('i', i);
// 		console.log('\n\n');
// 	}
// }
// runStats();

// async function runMinutes() {
// 	for (let i = 0; i < 34; ++i) {
// 		console.log('i', i);
// 		await minutes(i + 1);
// 		console.log('i', i);
// 		console.log('\n\n');
// 	}
// }

// runMinutes();

async function runTransactions() {
	const output = [];
	for (let i = 0; i < 1; ++i) {
		console.log('i', i);
		const segmentOutput = await transactions(i + 1);
		output.push(...segmentOutput);
		console.log('i', i);

		const usage = process.memoryUsage();
		console.log('rss: ' + usage.rss / 1000000 + 'MB');
		console.log('heapTotal: ' + usage.heapTotal / 1000000 + 'MB');
		console.log('heapUsed: ' + usage.heapUsed / 1000000 + 'MB');
		console.log('external: ' + usage.external / 1000000 + 'MB');
		console.log('\n\n');
	}

	// for (const data of output) {
	// 	console.log('inputs');
	// 	console.log(data.inputs);

	// 	console.log('outputs');
	// 	console.log(data.outputs);
	// 	console.log('');
	// }

	writeFileSync('data/transactions-all.json', JSON.stringify(output), 'utf8');
}

runTransactions();

function log(...s) {
	console.log(...s);
}
