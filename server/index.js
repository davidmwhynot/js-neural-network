const express = require('express');
const path = require('path');
const http = require('http');
const socketio = require('socket.io');
const EventEmitter = require('events');

const Network = require('../src/Network');

const network = new Network({
	learningRate: 3,
	layers: [
		{
			numNodes: 784
		},
		{
			numNodes: 64
		},
		{
			numNodes: 10
		}
	]
});

const TRAINING_DATA_PERCENTAGE = 100;

const training = require('../train.json');

let trainingData = [];

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
for (let i = 0; i < 3; ++i) {
	network.train(shuffle(trainingData), 500, i, 3, true, false);
}

class Emitter extends EventEmitter {}

// const images = require('../test.json');

const nn = require('../src');

const PORT = 3001;

const app = express();
const server = http.createServer(app);
const io = socketio(server);

app.use(express.static(path.resolve(__dirname, '..', 'client', 'build')));

io.on('connection', async socket => {
	console.log('client connected');
	const emitter = new Emitter();
	const data = [];

	emitter.on('training_start', () => {
		console.log('training start emit');
		io.emit('training_start');
	});

	emitter.on('training_finish', () => {
		console.log('training finish emit');
		io.emit('training_finish');
	});

	emitter.on('testing_finish', () => {
		console.log('testing finish emit');
		io.emit('testing_finish');
	});

	emitter.on('test', results => {
		console.log('test emit');
		data.push(results);
		io.emit('data', data);
	});

	nn(emitter, network, 2, 300, 4);

	// setInterval(() => {
	// 	++x;
	// }, 500);
});

server.listen(PORT, () => console.log(`App listening on port: ${PORT}`));

function shuffle(array) {
	return array.sort(() => Math.random() - 0.5);
}
