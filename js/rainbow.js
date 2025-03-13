var c = document.getElementById('visualizationCanvas_rainbow');
var ctx = c.getContext('2d');
// Set canvas to full screen
c.width = window.innerWidth;
c.height = window.innerHeight;


var w = c.width = window.innerWidth,
		h = c.height = window.innerHeight,
		ctx = c.getContext( '2d' ),
		
		opts = {
			
			particles: 50,
			gravity: .1,
			baseSize: 7,
			addedSize: 3,
			sizeVelocityMultiplier: .8,
			connectionDistX: w / 5, 
			connectionDistY: h / 5,
			boundaryY: h * 3 / 2,
			lineWidth: 0.4,
			connectTime: 1,
			
			cx: w / 2,
			cy: h / 2,
			
			colorMultiplier: 360 / w,
			tickIncrementer: 5 ,
			growthIncrement: .2,
		},
		
		particles = [],
		tick = ( Math.random() * 360 ) |0;

function Particle(){
	
	this.reset();
}
Particle.prototype.reset = function(){
	
	this.finalSize = ( opts.baseSize + opts.addedSize * Math.random() ) |0;
	this.size = 0;
	
	var rad = -Math.random() * Math.PI;
	this.vx = Math.cos( rad ) * this.finalSize * opts.sizeVelocityMultiplier;
	this.vy = Math.sin( rad ) * this.finalSize * opts.sizeVelocityMultiplier;
	
	this.x = opts.cx;
	this.y = opts.cy;
	
	this.connected = false;
	this.connectTime = 0;
}
Particle.prototype.step = function(){
	
	this.connected = false;
	
	this.x += this.vx *= .999;
	this.y += this.vy += opts.gravity;
	
	if( this.size < this.finalSize )
		this.size += opts.growthIncrement;
	else this.size = this.finalSize;
	
	if( this.y > opts.boundaryY )
		this.reset();
	
}
Particle.prototype.draw = function(){
	
	if( this.connected && this.connectTime < opts.connectTime )
		++this.connectTime;
	else if( !this.connected && this.connectTime > 0 )
		--this.connectTime;
	
	ctx.fillStyle = 'hsl( hue, 50%, 50% )'.replace( 'hue', this.x * opts.colorMultiplier + tick );
	ctx.beginPath();
	ctx.arc( this.x, this.y, this.size, 0, Math.PI * 2 );
	ctx.fill();
	
	if( this.connectTime > 0 ){
		
		ctx.beginPath();
		ctx.lineWidth = this.connectTime / 4;
		ctx.arc( this.x, this.y, this.size, 0, Math.PI * 2 );
		ctx.stroke();
	}
}

function anim(){
	
	window.requestAnimationFrame( anim );
	
	tick += opts.tickIncrementer;
	
	ctx.fillStyle = '#222';
	ctx.fillRect( 0, 0, w, h );
	
	if( particles.length < opts.particles && Math.random() < .2)
		particles.push( new Particle );
	
	particles.map( function( particle ){ particle.step(); } );
	
	ctx.lineWidth = opts.lineWidth;
	ctx.strokeStyle = '#bbb';
	ctx.beginPath();
	
	for( var i = 0; i < particles.length; ++i )
		for( var j = i + 1; j < particles.length; ++j )
			if( Math.abs( particles[ i ].x - particles[ j ].x ) < opts.connectionDistX && Math.abs( particles[ i ].y - particles[ j ].y ) < opts.connectionDistY ){
				
				var p1 = particles[ i ],
						p2 = particles[ j ];
				
				p1.connected = p2.connected = true;
				
				ctx.moveTo( p1.x, p1.y );
				ctx.lineTo( p2.x, p2.y );
			}
	
	ctx.stroke();
	
	particles.map( function( particle ){ particle.draw(); } );
}
anim();

window.addEventListener( 'resize', function(){
	
	w = c.width = window.innerWidth;
	h = c.height = window.innerHeight;
	
	opts.cx = w / 2;
	opts.cy = h / 2;
});
window.addEventListener( 'click', function( e ){
	
	opts.cx = e.clientX;
	opts.cy = e.clientY;
})


// Update canvas size when window is resized
window.addEventListener('resize', function() {
    c.width = window.innerWidth;
    c.height = window.innerHeight;
});
