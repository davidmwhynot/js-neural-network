const csv = require('csvtojson');

async function main() {
	const json = await csv({
		noheader: true,
		headers: ['time', 'price', 'amount']
	}).fromFile('data/tail1000000.csv');
	// console.log('json', json);

	console.log('entries: ', json.length);

	let sum = 0;
	let min = 9999999;
	let max = json[1].time - json[0].time;
	let zeros = 0;

	let a = Number(json[0].time);
	let b = 0;

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

		a = b;
	}
	console.log('average', sum / (json.length - 1));
	console.log('max', max);
	console.log('min', min);
	console.log('zeros', zeros);
}

main();
// 1000000
