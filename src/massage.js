const csv = require('csvtojson');

async function main() {
	const json = await csv({ noheader: true }).fromFile('data/tail1000.csv');
	console.log('json', json);

	console.log('entries: ', json.length);

	let sum = 0;
	let a = Number(json[0].field1);
	let b = 0;

	for (let i = 1; i < json.length; ++i) {
		if (i % 1000 === 0) {
			console.log('i', i);
		}
		// const date = new Date(Number(data.field1) * 1000);
		// console.log(date.toString());

		const data = json[i];

		b = Number(data.field1);
		// console.log(b - a);
		sum += b - a;
		a = b;
	}
	console.log('average', sum / (json.length - 1));
}

main();
