const { GPU } = require('gpu.js');
const gpu = new GPU();

const settings = {
	// precision: 'single'
	optimizeFloatMemory: true,
	loopMaxIterations: 2048
};

class Matrix {
	constructor(rows, cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = [];

		for (let i = 0; i < this.rows; i++) {
			this.data[i] = [];
			for (let j = 0; j < this.cols; j++) {
				this.data[i][j] = 0;
			}
		}
	}

	static fromArray(arr) {
		let m = new Matrix(arr.length, 1);
		for (let i = 0; i < arr.length; i++) {
			m.data[i][0] = arr[i];
		}
		return m;
	}

	static subtract(a, b) {
		// Return a new Matrix a-b
		let result = new Matrix(a.rows, a.cols);
		for (let i = 0; i < result.rows; i++) {
			for (let j = 0; j < result.cols; j++) {
				result.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}
		return result;
	}

	toArray() {
		let arr = [];
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				arr.push(this.data[i][j]);
			}
		}
		return arr;
	}

	randomize() {
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				this.data[i][j] = Math.random() * 2 - 1;
			}
		}
	}

	add(n) {
		if (n instanceof Matrix) {
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.cols; j++) {
					this.data[i][j] += n.data[i][j];
				}
			}
		} else {
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.cols; j++) {
					this.data[i][j] += n;
				}
			}
		}
	}

	gpuAdd(n) {
		if (n instanceof Matrix) {
			const kernel = gpu.createKernel(
				function(a, b) {
					return (
						a[this.thread.x][this.thread.y] + b[this.thread.x][this.thread.y]
					);
				},
				{
					...settings,
					output: {
						x: this.rows,
						y: this.cols
					}
				}
			);

			this.data = kernel(this.data, n.data);
		} else {
			const kernel = gpu.createKernel(
				function(a, b) {
					return a[this.thread.x][this.thread.y] + b;
				},
				{
					...settings,
					output: {
						x: this.rows,
						y: this.cols
					}
				}
			);

			this.data = kernel(this.data, n);
		}
	}

	static transpose(matrix) {
		let result = new Matrix(matrix.cols, matrix.rows);
		for (let i = 0; i < matrix.rows; i++) {
			for (let j = 0; j < matrix.cols; j++) {
				result.data[j][i] = matrix.data[i][j];
			}
		}
		return result;
	}

	static gpuMultiplyKernel(a, b, kernel) {
		// Matrix product
		if (a.cols !== b.rows) {
			console.log('a', a);
			console.log('b', b);
			console.error(new Error('Columns of A must match rows of B.'));
			return undefined;
		} else {
			let result = new Matrix(a.rows, b.cols);
			result.data = kernel(a.data, b.data);
			// console.log('result', result);
			// console.log('result.data[0][0]', result.data[0][0]);
			return result;
		}
	}

	static gpuMultiply(a, b) {
		// Matrix product
		if (a.cols !== b.rows) {
			console.log('a', a);
			console.log('b', b);
			console.error(new Error('Columns of A must match rows of B.'));
			return undefined;
		} else {
			let result = new Matrix(a.rows, b.cols);
			const kernel = gpu.createKernel(
				function(a, b) {
					let sum = 0;
					// i = y = a.rows
					// j = x = b.cols
					for (let k = 0; k < this.constants.size; ++k) {
						sum += a[this.thread.x][k] * b[k][this.thread.y];
					}
					return sum;
				},
				{
					...settings,
					output: [b.cols, a.rows],
					constants: { size: a.cols }
				}
			);

			result.data = kernel(a.data, b.data);
			// console.log('result', result);
			// console.log('result.data[0][0]', result.data[0][0]);
			return result;
		}
	}

	static multiply(a, b) {
		// Matrix product
		if (a.cols !== b.rows) {
			console.log('a', a);
			console.log('b', b);
			console.error(new Error('Columns of A must match rows of B.'));
			return undefined;
		}
		let result = new Matrix(a.rows, b.cols);
		for (let i = 0; i < result.rows; i++) {
			for (let j = 0; j < result.cols; j++) {
				// Dot product of values in col
				let sum = 0;
				for (let k = 0; k < a.cols; k++) {
					sum += a.data[i][k] * b.data[k][j];
				}
				result.data[i][j] = sum;
			}
		}
		return result;
	}

	multiply(n) {
		if (n instanceof Matrix) {
			// hadamard product
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.cols; j++) {
					this.data[i][j] *= n.data[i][j];
				}
			}
		} else {
			// Scalar product
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.cols; j++) {
					this.data[i][j] *= n;
				}
			}
		}
	}

	map(func) {
		// Apply a function to every element of matrix
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				let val = this.data[i][j];
				this.data[i][j] = func(val);
			}
		}
	}

	static map(matrix, func) {
		let result = new Matrix(matrix.rows, matrix.cols);
		// Apply a function to every element of matrix
		for (let i = 0; i < matrix.rows; i++) {
			for (let j = 0; j < matrix.cols; j++) {
				let val = matrix.data[i][j];
				result.data[i][j] = func(val);
			}
		}
		return result;
	}

	print() {
		console.table(this.data);
	}
}

export default Matrix;
