import React, { Component } from 'react';
import Navbar from './components/Navbar';
import io from 'socket.io-client/dist/socket.io.js';

class App extends Component {
	constructor() {
		super();

		this.state = { results: [], state: 'Loading...' };
		const socket = io();

		socket.on('data', data => {
			// console.log(data);
			this.setState({ results: data });
		});
		socket.on('training_start', () => {
			this.setState({ state: 'Training...' });
		});
		socket.on('training_finish', () => {
			this.setState({ state: 'Testing...' });
		});
		socket.on('testing_finish', () => {
			this.setState({ state: '' });
		});
	}

	componentDidMount() {}
	componentDidUpdate() {
		const results = this.state.results;
		if (results.length > 0) {
			for (const result of results) {
				const image = result.image;
				const canvas = document.getElementById('number' + result.round);
				const context = canvas.getContext('2d');
				let x = 0;
				for (let i = 0; i < 28; ++i) {
					for (let j = 0; j < 28; ++j) {
						context.fillStyle = `rgb(${image[x]}, ${image[x]}, ${image[x]})`;
						// context.fillRect(i * 1, j * 1, 1, 1);
						context.fillRect(j * 5, i * 5, 5, 5);
						++x;
					}
				}
			}
		}
	}

	render() {
		const { state, results } = this.state;
		return (
			<div className='App'>
				<Navbar />
				<div className='container' style={{ marginTop: 100 }}>
					<h1>{state}</h1>
					<div className='row'>
						{results.map(result => {
							return (
								<div className='col-3'>
									<h1>Label: {result.label}</h1>
									<h1>Guess: {result.guess}</h1>
									<canvas
										id={'number' + result.round}
										width='140'
										height='140'
										style={{ border: '1px solid black' }}
									/>
								</div>
							);
						})}
					</div>
				</div>
			</div>
		);
	}
}

export default App;
