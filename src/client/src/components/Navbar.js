import React from 'react';

const Navbar = () => {
	return (
		<nav className='navbar navbar-expand-md navbar-dark bg-dark fixed-top'>
			<div className='container'>
				<a className='navbar-brand' href='#'>
					JS Neural Network
				</a>
				<button
					className='navbar-toggler'
					type='button'
					data-toggle='collapse'
					data-target='#navbarsExampleDefault'
					aria-controls='navbarsExampleDefault'
					aria-expanded='false'
					aria-label='Toggle navigation'
				>
					<span className='navbar-toggler-icon' />
				</button>
				<div className='collapse navbar-collapse' id='navbarsExampleDefault'>
					<ul className='navbar-nav mr-auto'>
						<li className='nav-item active'>
							<a className='nav-link' href='#'>
								Home <span className='sr-only'>(current)</span>
							</a>
						</li>
					</ul>
				</div>
			</div>
		</nav>
	);
};

export default Navbar;
