(function() {

var c = document.getElementById('visualizationCanvas_neuralNet');
var ctx = c.getContext('2d');

// Set canvas to full screen
c.width = window.innerWidth;
c.height = window.innerHeight;


var w = c.width = window.innerWidth,
		h = c.height = window.innerHeight,
		ctx = c.getContext( '2d' ),
		
		opts = {
			
			range: 200,
			baseConnections: 8,
			addedConnections: 10,
			baseSize: 3,
			minSize: 1,
			dataToConnectionSize: .5,
			sizeMultiplier: .7,
			allowedDist: 40,
			baseDist: 40,
			addedDist: 30,
			connectionAttempts: 130,
			
			dataToConnections: 1,
			baseSpeed: .01,
			addedSpeed: .001,
			baseGlowSpeed: .05,
			addedGlowSpeed: .05,
			
			rotVelX: .001,
			rotVelY: .0005,
			tick_incr: 1,
			repaintColor: '#111',
			connectionColor: 'hsla(30,0%,100%, alp)', //blue
			rootColor: 'hsla(35,0%,100%, 1)', // red
			endColor: 'hsla(40,0%,100%, 1)', // cyan
			dataColor: 'hsla(30,100%,50%, 1)',  // orange

			// connectionColor: 'hsla(200,60%,light%, alp)', //blue
			// rootColor: 'hsla(0,60%,light%, alp)', // red
			// endColor: 'hsla(160,20%,light%,alp)', // cyan
			// dataColor: 'hsla(40,80%,light%,alp)',  // orange
			
			wireframeWidth: .1,
			wireframeColor: '#fff',
			
			depth: 250,
			focalLength: 250,
			vanishPoint: {
				x: w / 2,
				y: h / 2
			}
		},
		
		squareRange = opts.range * opts.range,
		squareAllowed = opts.allowedDist * opts.allowedDist,
		mostDistant = opts.depth + opts.range,
		sinX = sinY = 0,
		cosX = cosY = 0,
		
		connections = [],
		toDevelop = [],
		data = [],
		all = [],
		tick = 0,
		totalProb = 0,
		
		animating = false,
		
		Tau = Math.PI * 2;

  // Title and Subtitle (you can replace this with dynamic values if needed)
  var title = 'Kha Vo' // '{{ site.title }}'; // Title from Jekyll
  var subtitle = 'The no.1 Australian AI scientist on Kaggle' // '{{ site.subtitle }}'; // Subtitle from Jekyll, optional
  // Set font styles

ctx.fillRect( 0, 0, w, h );

//ctx.fillText( 'Calculating Nodes', w / 2 - ctx.measureText( 'Calculating Nodes' ).width / 2, h / 2 - 15 );

window.setTimeout( init, 4 ); // to render the loading screen

function click_canva(){
 opts.tick_incr = -opts.tick_incr;
}

function init(){
	
	connections.length = 0;
	data.length = 0;
	all.length = 0;
	toDevelop.length = 0;
	
	var connection = new Connection( 0, 0, 0, opts.baseSize );
	connection.step = Connection.rootStep;
	connections.push( connection );
	all.push( connection );
	connection.link();
	
	while( toDevelop.length > 0 ){
	
		toDevelop[ 0 ].link();
		toDevelop.shift();
	}
	
	if( !animating ){
		animating = true;
		anim();
	}
}
function Connection( x, y, z, size ){
	
	this.x = x;
	this.y = y;
	this.z = z;
	this.size = size;
	
	this.screen = {};
	
	this.links = [];
	this.probabilities = [];
	this.isEnd = false;
	
	this.glowSpeed = opts.baseGlowSpeed + opts.addedGlowSpeed * Math.random();
}
Connection.prototype.link = function(){
	
	if( this.size < opts.minSize )
		return this.isEnd = true;
	
	var links = [],
			connectionsNum = opts.baseConnections + Math.random() * opts.addedConnections |0,
			attempt = opts.connectionAttempts,
			
			alpha, beta, len,
			cosA, sinA, cosB, sinB,
			pos = {},
			passedExisting, passedBuffered;
	
	while( links.length < connectionsNum && --attempt > 0 ){
		
		alpha = Math.random() * Math.PI;
		beta = Math.random() * Tau;
		len = opts.baseDist + opts.addedDist * Math.random();
		
		cosA = Math.cos( alpha );
		sinA = Math.sin( alpha );
		cosB = Math.cos( beta );
		sinB = Math.sin( beta );
		
		pos.x = this.x + len * cosA * sinB;
		pos.y = this.y + len * sinA * sinB;
		pos.z = this.z + len *        cosB;
		
		if( pos.x*pos.x + pos.y*pos.y + pos.z*pos.z < squareRange ){
		
			passedExisting = true;
			passedBuffered = true;
			for( var i = 0; i < connections.length; ++i )
				if( squareDist( pos, connections[ i ] ) < squareAllowed )
					passedExisting = false;

			if( passedExisting )
				for( var i = 0; i < links.length; ++i )
					if( squareDist( pos, links[ i ] ) < squareAllowed )
						passedBuffered = false;

			if( passedExisting && passedBuffered )
				links.push( { x: pos.x, y: pos.y, z: pos.z } );
			
		}
		
	}
	
	if( links.length === 0 )
		this.isEnd = true;
	else {
		for( var i = 0; i < links.length; ++i ){
			
			var pos = links[ i ],
					connection = new Connection( pos.x, pos.y, pos.z, this.size * opts.sizeMultiplier );
			
			this.links[ i ] = connection;
			all.push( connection );
			connections.push( connection );
		}
		for( var i = 0; i < this.links.length; ++i )
			toDevelop.push( this.links[ i ] );
	}
}
Connection.prototype.step = function(){
	
	this.setScreen();
	this.screen.color = ( this.isEnd ? opts.endColor : opts.connectionColor ).replace( 'light', 30 + ( ( tick * this.glowSpeed ) % 30 ) ).replace( 'alp', .2 + ( 1 - this.screen.z / mostDistant ) * .8 );
	
	for( var i = 0; i < this.links.length; ++i ){
		ctx.moveTo( this.screen.x, this.screen.y );
		ctx.lineTo( this.links[ i ].screen.x, this.links[ i ].screen.y );
	}
}
Connection.rootStep = function(){
	this.setScreen();
	this.screen.color = opts.rootColor.replace( 'light', 30 + ( ( tick * this.glowSpeed ) % 30 ) ).replace( 'alp', ( 1 - this.screen.z / mostDistant ) * .8 );
	
	for( var i = 0; i < this.links.length; ++i ){
		ctx.moveTo( this.screen.x, this.screen.y );
		ctx.lineTo( this.links[ i ].screen.x, this.links[ i ].screen.y );
	}
}
Connection.prototype.draw = function(){
	ctx.fillStyle = this.screen.color;
	ctx.beginPath();
	ctx.arc( this.screen.x, this.screen.y, this.screen.scale * this.size, 0, Tau );
	ctx.fill();
}
function Data( connection ){
	
	this.glowSpeed = opts.baseGlowSpeed + opts.addedGlowSpeed * Math.random();
	this.speed = opts.baseSpeed + opts.addedSpeed * Math.random();
	
	this.screen = {};
	
	this.setConnection( connection );
}
Data.prototype.reset = function(){
	
	this.setConnection( connections[ 0 ] );
	this.ended = 2;
}
Data.prototype.step = function(){
	
	this.proportion += this.speed;
	
	if( this.proportion < 1 ){
		this.x = this.ox + this.dx * this.proportion;
		this.y = this.oy + this.dy * this.proportion;
		this.z = this.oz + this.dz * this.proportion;
		this.size = ( this.os + this.ds * this.proportion ) * opts.dataToConnectionSize;
	} else 
		this.setConnection( this.nextConnection );
	
	this.screen.lastX = this.screen.x;
	this.screen.lastY = this.screen.y;
	this.setScreen();
	this.screen.color = opts.dataColor.replace( 'light', 40 + ( ( tick * this.glowSpeed ) % 50 ) ).replace( 'alp', .2 + ( 1 - this.screen.z / mostDistant ) * .6 );
	
}
Data.prototype.draw = function(){
	
	if( this.ended )
		return --this.ended; // not sre why the thing lasts 2 frames, but it does
	
	ctx.beginPath();
	ctx.strokeStyle = this.screen.color;
	ctx.lineWidth = this.size * this.screen.scale;
	ctx.moveTo( this.screen.lastX, this.screen.lastY );
	ctx.lineTo( this.screen.x, this.screen.y );
	ctx.stroke();
}
Data.prototype.setConnection = function( connection ){
	
	if( connection.isEnd )
		this.reset();
	
	else {
		
		this.connection = connection;
		this.nextConnection = connection.links[ connection.links.length * Math.random() |0 ];
		
		this.ox = connection.x; // original coordinates
		this.oy = connection.y;
		this.oz = connection.z;
		this.os = connection.size; // base size
		
		this.nx = this.nextConnection.x; // new
		this.ny = this.nextConnection.y;
		this.nz = this.nextConnection.z;
		this.ns = this.nextConnection.size;
		
		this.dx = this.nx - this.ox; // delta
		this.dy = this.ny - this.oy;
		this.dz = this.nz - this.oz;
		this.ds = this.ns - this.os;
		
		this.proportion = 0;
	}
}
Connection.prototype.setScreen = Data.prototype.setScreen = function(){
	
	var x = this.x,
			y = this.y,
			z = this.z;
	
	// apply rotation on X axis
	var Y = y;
	y = y * cosX - z * sinX;
	z = z * cosX + Y * sinX;
	
	// rot on Y
	var Z = z;
	z = z * cosY - x * sinY;
	x = x * cosY + Z * sinY;
	
	this.screen.z = z;
	
	// translate on Z
	z += opts.depth;
	
	this.screen.scale = opts.focalLength / z;
	this.screen.x = opts.vanishPoint.x + x * this.screen.scale;
	this.screen.y = opts.vanishPoint.y + y * this.screen.scale;
	
}
function squareDist( a, b ){
	
	var x = b.x - a.x,
			y = b.y - a.y,
			z = b.z - a.z;
	
	return x*x + y*y + z*z;
}

function anim(){
	
	window.requestAnimationFrame( anim );
	
	ctx.globalCompositeOperation = 'source-over';
	ctx.fillStyle = opts.repaintColor;
	ctx.fillRect( 0, 0, w, h );
	
	tick = tick + opts.tick_incr;
	
	var rotX = tick * opts.rotVelX,
		rotY = tick * opts.rotVelY;
	
	cosX = Math.cos( rotX );
	sinX = Math.sin( rotX );
	cosY = Math.cos( rotY );
	sinY = Math.sin( rotY );
	
	if( data.length < connections.length * opts.dataToConnections ){
		var datum = new Data( connections[ 0 ] );
		data.push( datum );
		all.push( datum );
	}
	
	ctx.globalCompositeOperation = 'lighter';
	ctx.beginPath();
	ctx.lineWidth = opts.wireframeWidth;
	ctx.strokeStyle = opts.wireframeColor;
	all.map( function( item ){ item.step(); } );
	ctx.stroke();
	ctx.globalCompositeOperation = 'source-over';
	all.sort( function( a, b ){ return b.screen.z - a.screen.z } );
	all.map( function( item ){ item.draw(); } );
	
	/*ctx.beginPath();
	ctx.strokeStyle = 'red';
	ctx.arc( opts.vanishPoint.x, opts.vanishPoint.y, opts.range * opts.focalLength / opts.depth, 0, Tau );
	ctx.stroke();*/

	// WRITE TEXT
	ctx.textAlign = 'center'; // Center text horizontally
	ctx.fillStyle = isHovering ? '#BE5504': '#FC6A03'; // Title text color
	ctx.font = isHovering ? 'bold 49px "Fira Code", monospace': 'bold 48px "Fira Code", monospace' // 'bold 50px "IBM Plex Mono", monospace', '50px Verdana'
	// ctx.fillText(title, w/2, 7*h/15);  // Adjust y position as needed
    ctx.fillText(title, opts.vanishPoint.x, opts.vanishPoint.y - 20);

 
    ctx.font = 'italic 20px "IBM Plex Mono", monospace'; // Smaller font for subtitle
    ctx.fillStyle = isHovering ? '#ED820E':  '#FDA172'; 
    ctx.textAlign = 'center'; // Center text horizontally
    //   ctx.fillText(subtitle, w/2, 8*h/15);  // Subtitle below the title
    ctx.fillText(subtitle, opts.vanishPoint.x, opts.vanishPoint.y + 30);

}

// window.addEventListener( 'resize', function(){
	
// 	opts.vanishPoint.x = ( w = c.width = window.innerWidth ) / 2;
// 	opts.vanishPoint.y = ( h = c.height = window.innerHeight ) / 2;
// 	ctx.fillRect( 0, 0, w, h );
// });

window.addEventListener( 'click', click_canva );


// Update canvas size when window is resized
window.addEventListener('resize', function() {
    c.width = window.innerWidth;
    c.height = window.innerHeight;
});



let titleArea = { x: opts.vanishPoint.x, y: opts.vanishPoint.y - 20, width: 300, height: 50 };
let isHovering = false;

// Mouse move to detect hover
c.addEventListener('mousemove', (e) => {
    const rect = c.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Check if mouse is over title area
    isHovering = mouseX >= titleArea.x - titleArea.width / 2 &&
                  mouseX <= titleArea.x + titleArea.width / 2 &&
                  mouseY >= titleArea.y - titleArea.height / 2 &&
                  mouseY <= titleArea.y + titleArea.height / 2;
});

// Click event to navigate
c.addEventListener('click', () => {
    if (isHovering) {
        window.location.href = 'https://khavo.ai/about';
    }
});

})();
