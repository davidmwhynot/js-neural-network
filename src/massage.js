const csv = require('csvtojson');
const moment = require('moment');
const { writeFileSync } = require('fs');

const transactionsOutput = [];
function transactions(j) {
	return new Promise(async resolve => {
		const json = await csv({
			noheader: true,
			headers: ['time', 'price', 'amount']
		}).fromFile('data/' + j + '-bitstamp.csv');

		const output = [];
		let inputs = 0;
		let outputs = 0;
		let segment = {
			inputs: [],
			outputs: []
		};

		for (let i = 0; i < json.length; ++i) {
			if (inputs < 900) {
				// generate inputs
				segment.inputs.push(Number(json[i].price));
				++inputs;
			} else if (outputs < 100) {
				// generate outputs
				segment.outputs.push(Number(json[i].price));
				++outputs;
			} else {
				// reset
				output.push(segment);
				inputs = 0;
				outputs = 0;
				segment = {
					inputs: [],
					outputs: []
				};
			}
		}

		resolve(output);
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
		console.log('\n\n');
	}

	for(const data of output) {
		console.log('inputs');
		console.log(data.inputs);
		
		console.log('outputs');
		console.log(data.outputs);
		console.log('');
	}

	writeFileSync('data/transactions.json', JSON.stringify(output), 'utf8');
}

runTransactions();
