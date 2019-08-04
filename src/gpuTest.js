const { GPU } = require('gpu.js');
const gpu = new GPU({
	// mode: 'cpu'
});
const Matrix = require('./Network/Matrix');

const SIZE = 4;

module.exports = () => {
	const settings = {
		output: [SIZE, SIZE],
		precision: 'single',
		optimizeFloatMemory: true,
		constants: {
			size: SIZE
		}
	};
	const multiplyMatrx = gpu.createKernel(function(a, b) {
		let sum = 0;
		for (let i = 0; i < this.constants.size; ++i) {
			sum += a[this.thread.y][i] * b[i][this.thread.x];
		}
		return sum;
	}, settings);

	const aMatrix = new Matrix(SIZE, SIZE);
	const bMatrix = new Matrix(SIZE, SIZE);

	aMatrix.randomize();
	bMatrix.randomize();

	// for (let i = 0; i < SIZE; ++i) {
	// 	for (let j = 0; j < SIZE; ++j) {
	// 		aMatrix.data[i][j] = Math.round(aMatrix.data[i][j] * 10);
	// 		bMatrix.data[i][j] = Math.round(bMatrix.data[i][j] * 10);
	// 	}
	// }

	const { data: a } = aMatrix;
	const { data: b } = bMatrix;

	console.log('aMatrix', aMatrix);
	console.log('bMatrix', bMatrix);
	console.log('a', a);
	console.log('b', b);

	const c = multiplyMatrx(a, b);
	const cMatrix = Matrix.multiply(aMatrix, bMatrix);
	const cGpuMatrix = Matrix.gpuMultiply(aMatrix, bMatrix);

	for (let i = 0; i < SIZE; ++i) {
		for (let j = 0; j < SIZE; ++j) {
			if (cMatrix.data[i][j] !== cGpuMatrix.data[i][j]) {
				console.log('not equiv');
				console.log('i', i);
				console.log('j', j);
				console.log('cMatrix.data[i][j]', cMatrix.data[i][j]);
				console.log('cGpuMatrix.data[i][j]', cGpuMatrix.data[i][j]);
				break;
			}
		}
	}

	console.time('gpu');
	for (let i = 0; i < 10; ++i) {
		multiplyMatrx(a, b);
	}
	console.timeEnd('gpu');

	setTimeout(function() {
		console.time('matrix');
		for (let i = 0; i < 10; ++i) {
			Matrix.gpuMultiply(aMatrix, bMatrix);
		}
		console.timeEnd('matrix');
	}, 500);

	// console.log('cMatrix', cMatrix);
	// console.log('c', c);
};
