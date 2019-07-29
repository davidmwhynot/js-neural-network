# __**THIS IS BROKEN. HELP WANTED**__
# JS Neural Network

Run:
```
node index.js
```

`src/Network/index.js` is where `train()` function is located.

# example usage
```js
// /example.js
const Network = require('./src/Network');

const network = new Network({
	learningRate: 0.1,
	// you will need at least 3 layers
	// 1 - input
	// 1 - hidden
	// 1 - output
	layers: [
		{
			numNodes: 2
		},
		{
			numNodes: 2
		},
		{
			numNodes: 1
		}
	]
});

// define training data in this format
// each element of the array is a training "example"
const trainingData = [
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

// train the network
network.train(trainingData);

// test the results of training
network.setInputs([0, 1]);
network.calculate();

// get test results output layer
const outputs = network.getOutputs();
console.log(outputs);

```