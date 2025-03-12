document.addEventListener('DOMContentLoaded', () => {
    const canvases = document.querySelectorAll('canvas');
    const activeAnimations = new Map();
    let currentCanvas = null;
    let fadeOutCanvas = null;
    let fadeOpacity = 1;
    let isTabVisible = !document.hidden;

    // Create intersection observer to handle canvas animations based on viewport visibility
    const observer = new IntersectionObserver((entries) => {
        entries.sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        
        entries.forEach(entry => {
            const canvas = entry.target;

            if (entry.isIntersecting) {
                // Start animation if not already running
                if (!activeAnimations.has(canvas.id)) {
                    currentCanvas = canvas;
                    initializeCanvas(canvas);
                }
            } else if (!entry.isIntersecting) {
                // Clean up when out of view
                if (activeAnimations.has(canvas.id)) {
                    cancelAnimationFrame(activeAnimations.get(canvas.id));
                    activeAnimations.delete(canvas.id);
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    canvas.style.opacity = '';
                }
            }
        });
    }, {
        threshold: [0, 0.1, 0.5, 1]
    });

    // Handles the fade out animation of the previous canvas
    function fadeOutPreviousCanvas() {
        if (fadeOutCanvas && fadeOpacity > 0) {
            fadeOpacity -= 0.05; // Adjust this value to control fade speed
            fadeOutCanvas.style.opacity = fadeOpacity;

            if (fadeOpacity <= 0) {
                // Clean up the old canvas
                if (activeAnimations.has(fadeOutCanvas.id)) {
                    cancelAnimationFrame(activeAnimations.get(fadeOutCanvas.id));
                    activeAnimations.delete(fadeOutCanvas.id);
                    const ctx = fadeOutCanvas.getContext('2d');
                    ctx.clearRect(0, 0, fadeOutCanvas.width, fadeOutCanvas.height);
                }
                fadeOutCanvas.style.opacity = ''; // Reset opacity
                fadeOutCanvas = null;
            } else {
                requestAnimationFrame(fadeOutPreviousCanvas);
            }
        }
    }

    // Initializes and starts the canvas animation when it becomes visible
    function initializeCanvas(canvas) {
        // Start with 0 opacity and fade in
        canvas.style.opacity = 0;
        let fadeInOpacity = 0;

        function fadeIn() {
            fadeInOpacity += 0.05; // Adjust this value to control fade speed
            canvas.style.opacity = fadeInOpacity;

            if (fadeInOpacity < 1) {
                requestAnimationFrame(fadeIn);
            }
        }

        fadeIn();
        fadeOutPreviousCanvas();

        if (canvas) {
            const ctx = canvas.getContext('2d');

            // Make canvas full screen
            function resizeCanvas() {
                const pixelRatio = window.devicePixelRatio || 1;
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;

                // Set actual size in memory (scaled to account for extra pixel density)
                // canvas.width = width * pixelRatio;
                // canvas.height = height * pixelRatio;
                
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
                
                // Style size (CSS pixels)
                canvas.style.width = width + 'px';
                canvas.style.height = height + 'px';

                // Scale all drawing operations by the dpr
                ctx.scale(pixelRatio, pixelRatio);
            }
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);

            // Particle class: Manages individual particle behavior, appearance, and connections
            class Particle {
                constructor(x, y, angle, patternRadius, centerX, centerY, color, settings) {
                    this.turn = 0
                    this.x = x;
                    this.y = y;
                    this.angle = angle;
                    this.patternRadius = patternRadius;
                    this.centerX = centerX;
                    this.centerY = centerY;
                    this.radius = 0;
                    this.maxRadius = 0.6 + Math.random() * 0.8;
                    this.color = color;
                    this.opacity = 0;
                    // Use settings for speeds
                    this.fadeInSpeed = (settings.fadeInSpeed || 0.01) + Math.random() * (settings.fadeInVariation || 0.02);
                    this.fadeOutSpeed = (settings.fadeOutSpeed || 0.005) + Math.random() * (settings.fadeOutVariation || 0.01);
                    this.pulsationState = 'fadeIn';
                    this.orbitSpeed = (settings.orbitSpeed || 0.002) + Math.random() * (settings.orbitVariation || 0.003);
                    this.orbitRadius = this.patternRadius + Math.random() * 80;
                    this.rotationSpeed = settings.rotationSpeed || 0.001;
                    this.currentRotation = 0;
                    this.lineWidth = settings.lineWidth || 0.5;
                    this.lineOpacity = settings.lineOpacity || 0.3;

                    this.connectedParticles = [];

                    // Pre-calculate constants
                    this.TWO_PI = Math.PI * 2;
                    this.baseAngle = angle;
                    this.orbitRadiusSquared = this.orbitRadius * this.orbitRadius;

                    // Add initialization for wave pattern properties
                    this.movement_style = settings.movement_style;
                    if (this.movement_style === "gravity") {
                        this.cycleLength = settings.cycleLength || 3000;
                        this.maxVelocity = settings.maxVelocity || 0.5;
                        this.randomization = settings.randomization || 0.9;
                        // Initialize velocities
                        this.xVelocity = 0;
                        this.yVelocity = 0;
                    }
                }

                update() {
                    // Optimize angle calculation by using modulo to keep it in range
                    // this.angle = (this.angle + this.orbitSpeed) % this.TWO_PI;

                    // Calculate position
                    const angleCalc = this.angle;
                    this.x = this.centerX + this.orbitRadius * Math.cos(angleCalc);
                    this.y = this.centerY + this.orbitRadius * Math.sin(angleCalc);

                    this.turn += 1

                    // Simplified pulsation logic with fewer conditionals
                    if (this.pulsationState === 'fadeIn') {
                        this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                        this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                        
                        if (this.opacity >= 1) {
                            this.opacity = 1;
                            this.pulsationState = 'fadeOut';
                        }
                    } else {
                        this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                        this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                        
                        if (this.opacity <= 0) {
                            this.opacity = 0;
                            this.pulsationState = 'fadeIn';
                            this.radius = 0;
                        }
                    }
                }

                draw() {
                    // Only draw if particle is visible
                    if (this.opacity <= 0) return;
                    if (this.turn < this.orbitRadius * 2) return;

                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
                    ctx.fill();
                }

                connect(otherParticle) {
                    this.connectedParticles.push(otherParticle);
                }

                clearConnections() {
                    this.connectedParticles = [];
                }

                drawConnections() {
                    // Skip if no connections or particle is invisible
                    if (!this.connectedParticles.length || this.opacity <= 0) return;

                    this.connectedParticles.forEach(otherParticle => {
                        // Skip if other particle is invisible
                        if (otherParticle.opacity <= 0) return;

                        if (this.turn < this.orbitRadius * 2.2) return;

                        const distance = Math.hypot(this.x - otherParticle.x, this.y - otherParticle.y);
                        const maxDistance = 200;

                        if (distance < maxDistance) {
                            // Cache the opacity calculation
                            const opacity = (1 - distance / maxDistance) * this.lineOpacity;
                            
                            // Only draw if the line would be visible
                            if (opacity > 0.01) {
                                ctx.beginPath();
                                ctx.moveTo(this.x, this.y);
                                ctx.lineTo(otherParticle.x, otherParticle.y);
                                ctx.strokeStyle = `rgba(50, 50, 50, ${opacity})`;
                                ctx.lineWidth = this.lineWidth;
                                ctx.stroke();
                            }
                        }
                    });
                }
            }

            // SpatialGrid class: Optimizes particle proximity calculations using a grid-based system
            class SpatialGrid {
                constructor(width, height, cellSize) {
                    this.cellSize = cellSize;
                    this.cols = Math.ceil(width / this.cellSize);
                    this.rows = Math.ceil(height / this.cellSize);
                    this.grid = new Array(this.cols * this.rows).fill().map(() => []);
                }

                clear() {
                    this.grid.forEach(cell => cell.length = 0);
                }

                insert(particle) {
                    const cellX = Math.floor(particle.x / this.cellSize);
                    const cellY = Math.floor(particle.y / this.cellSize);
                    if (cellX >= 0 && cellX < this.cols && cellY >= 0 && cellY < this.rows) {
                        this.grid[cellY * this.cols + cellX].push(particle);
                    }
                }

                getNearbyParticles(particle, radius) {
                    const nearby = [];
                    const cellX1 = Math.max(0, Math.floor((particle.x - radius) / this.cellSize));
                    const cellY1 = Math.max(0, Math.floor((particle.y - radius) / this.cellSize));
                    const cellX2 = Math.min(this.cols - 1, Math.floor((particle.x + radius) / this.cellSize));
                    const cellY2 = Math.min(this.rows - 1, Math.floor((particle.y + radius) / this.cellSize));

                    for (let cy = cellY1; cy <= cellY2; cy++) {
                        for (let cx = cellX1; cx <= cellX2; cx++) {
                            nearby.push(...this.grid[cy * this.cols + cx]);
                        }
                    }
                    return nearby;
                }
            }

            let particles = [];

            // Creates particles based on the specified pattern
            function createParticles(pattern = 'fibonacci', settings = {}) {
                // Default animation settings
                const defaultSettings = {
                    fadeInSpeed: 0.01,
                    fadeInVariation: 0.02,
                    fadeOutSpeed: 0.005,
                    fadeOutVariation: 0.01,
                    orbitSpeed: 0.002,
                    orbitVariation: 0.003,
                    rotationSpeed: 0.001,
                    lineWidth: 0.5,
                    lineOpacity: 0.3
                };

                // Merge default settings with provided settings
                settings = { ...defaultSettings, ...settings };

                particles = [];
                const centerX = canvas.clientWidth / 2;
                const centerY = canvas.clientHeight / 2;
                const minDimension = Math.min(canvas.clientWidth, canvas.clientHeight);

                if (pattern === 'asteroids') {
                    const numClusters = 7;
                    const baseParticlesPerCluster = 100;
                    
                    // Calculate the maximum safe orbit radius
                    // Need to account for cluster size (0.15 * minDimension) plus padding
                    const clusterSize = minDimension * 0.1;
                    const maxOrbitRadius = (Math.min(canvas.clientWidth, canvas.clientHeight) / 2) - clusterSize;
                    
                    const clusterSpeeds = [
                        { orbit: 0.4, particle: 0.3 },
                        { orbit: -0.2, particle: -0.2 },
                        { orbit: 0.15, particle: 0.4 },
                        { orbit: -0.3, particle: 0.1 },
                        { orbit: 0.25, particle: -0.3 },
                        { orbit: 0.25, particle: -0.3 },
                        { orbit: 0.25, particle: -0.3 },
                    ];
                    
                    // Create clusters
                    for (let c = 0; c < numClusters; c++) {
                        const clusterSizeFactor = 1;
                        const particleCountFactors = [0.5, 0.6, 0.8, 0.4, 0.7, 0.8, 0.9];
                        const particlesPerCluster = Math.floor(baseParticlesPerCluster * particleCountFactors[c]);
                        
                        const baseAngle = (c / numClusters) * Math.PI * 2;
                        // Scale orbit radii to ensure clusters stay in bounds
                        const orbitRadii = [
                            maxOrbitRadius * 0,  // Innermost
                            maxOrbitRadius * 0.6, // Inner-middle
                            maxOrbitRadius * 0.6,  // Middle
                            maxOrbitRadius * 0.7, // Outer-middle
                            maxOrbitRadius * 0.8, // Outer-middle
                            maxOrbitRadius * 0.9, // Outer-middle
                            maxOrbitRadius * 1   // Outermost
                        ];
                        const orbitRadius = orbitRadii[c];
                        
                        for (let i = 0; i < particlesPerCluster; i++) {
                            const angle = Math.random() * Math.PI * 2;
                            const radius = Math.sqrt(Math.random()) * (minDimension * 0.15 * clusterSizeFactor);
                            
                            // Calculate initial position
                            const clusterX = centerX + orbitRadius * Math.cos(baseAngle);
                            const clusterY = centerY + orbitRadius * Math.sin(baseAngle);
                            const x = clusterX + radius * Math.cos(angle);
                            const y = clusterY + radius * Math.sin(angle);
                            
                            const particle = new Particle(
                                x, y, angle, radius,
                                clusterX, clusterY,
                                null,
                                {
                                    ...settings,
                                    fadeInSpeed: 0.02,
                                    fadeOutSpeed: 0.01,
                                    lineOpacity: 0.4
                                }
                            );

                            // Add cluster-specific properties
                            particle.clusterId = c;
                            particle.clusterSizeFactor = clusterSizeFactor;
                            particle.orbitRadius = orbitRadius;
                            particle.baseAngle = baseAngle;
                            particle.particleRadius = radius;
                            particle.orbitOffset = Math.random() * Math.PI * 2;
                            
                            // Update method remains the same as before
                            particle.update = function() {
                                const time = Date.now() * 0.001;
                                const speeds = clusterSpeeds[this.clusterId];
                                
                                // Calculate cluster center position with independent orbit
                                const clusterAngle = this.baseAngle + time * speeds.orbit;
                                this.centerX = centerX + this.orbitRadius * Math.cos(clusterAngle);
                                this.centerY = centerY + this.orbitRadius * Math.sin(clusterAngle);
                                
                                // Calculate particle position within cluster with independent rotation
                                const particleAngle = this.angle + time * speeds.particle;
                                const wobbleAmount = 2 * this.clusterSizeFactor;
                                const wobbleX = Math.sin(time + this.orbitOffset) * wobbleAmount;
                                const wobbleY = Math.cos(time + this.orbitOffset) * wobbleAmount;
                                
                                this.x = this.centerX + this.particleRadius * Math.cos(particleAngle) + wobbleX;
                                this.y = this.centerY + this.particleRadius * Math.sin(particleAngle) + wobbleY;

                                // Standard pulsation logic
                                if (this.pulsationState === 'fadeIn') {
                                    this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                    this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                                    if (this.opacity >= 1) {
                                        this.opacity = 1;
                                        this.pulsationState = 'fadeOut';
                                    }
                                } else {
                                    this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                                    this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                                    if (this.opacity <= 0) {
                                        this.opacity = 0;
                                        this.pulsationState = 'fadeIn';
                                        this.radius = 0;
                                    }
                                }
                            };

                            particles.push(particle);
                        }
                    }
                } else if (pattern === 'donut') {
                    const patternRadius = minDimension * 0.2;
                    const numRings = 3;
                    const ringSpacing = 50;
                    const ringThickness = 35;
                    
                    // Base number of particles for the innermost ring
                    const baseParticles = 50;
                    
                    // Create particles for each ring
                    for (let ring = 0; ring < numRings; ring++) {
                        const ringRadius = patternRadius + (ring * ringSpacing);
                        const ringOffset = (Math.PI * 2) / numRings * ring;
                        
                        const particlesPerRing = Math.floor(baseParticles * (1 + ring * 1));
                        const layersInRing = 3;

                        for (let layer = 0; layer < layersInRing; layer++) {
                            const layerRadius = ringRadius - ringThickness/2 + (layer * (ringThickness/layersInRing));
                            
                            for (let i = 0; i < particlesPerRing; i++) {
                                const angle = (i / particlesPerRing) * Math.PI * 2 + ringOffset;
                                
                                const radiusVariation = Math.sin(angle * 2) * 3;
                                const currentRadius = layerRadius + radiusVariation;
                                
                                const x = centerX + currentRadius * Math.cos(angle);
                                const y = centerY + currentRadius * Math.sin(angle);
                                
                                const particle = new Particle(
                                    x, y, angle, currentRadius,
                                    centerX, centerY,
                                    null,
                                    {
                                        ...settings,
                                        orbitSpeed: 0.001 * (numRings - ring) * (i % 2 === 0 ? 1 : -1),
                                        fadeInSpeed: 0.015 + (ring * 0.002),
                                        fadeOutSpeed: 0.008 + (ring * 0.001),
                                        lineOpacity: 0.45 - (ring * 0.07),
                                        lineWidth: 0.35 * (baseParticles / particlesPerRing),
                                    }
                                );

                                // Add custom properties for organic movement
                                particle.angularVelocity = (Math.random() * 0.002) - 0.001;
                                particle.baseRadius = currentRadius;
                                particle.radiusVariation = 0;
                                particle.wobbleOffset = Math.random() * Math.PI * 2;
                                particle.wobbleSpeed = 0.5 + Math.random() * 0.5;

                                // Custom update function for organic movement
                                particle.update = function() {
                                    const time = Date.now() * 0.001;
                                    
                                    // Update angular velocity with random variation
                                    this.angularVelocity += (Math.random() - 0.5) * 0.0001;
                                    this.angularVelocity = Math.max(-0.003, Math.min(0.003, this.angularVelocity));
                                    
                                    // Update angle with varying speed
                                    this.angle += this.orbitSpeed + this.angularVelocity;
                                    
                                    // Update radius variation
                                    this.radiusVariation += (Math.random() - 0.5) * 0.2;
                                    this.radiusVariation *= 0.95; // Damping
                                    this.radiusVariation = Math.max(-5, Math.min(5, this.radiusVariation));
                                    
                                    // Calculate breathing effect
                                    const breathingEffect = Math.sin(time * this.wobbleSpeed + this.wobbleOffset) * 3;
                                    
                                    // Calculate final radius with all variations
                                    const currentRadius = this.baseRadius + this.radiusVariation + breathingEffect;
                                    
                                    // Add vertical wobble
                                    const verticalWobble = Math.sin(time * 2 + this.angle) * 2;
                                    
                                    // Update position
                                    this.x = this.centerX + currentRadius * Math.cos(this.angle);
                                    this.y = this.centerY + currentRadius * Math.sin(this.angle) + verticalWobble;

                                    // Standard pulsation logic
                                    if (this.pulsationState === 'fadeIn') {
                                        this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                        this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                                        if (this.opacity >= 1) {
                                            this.opacity = 1;
                                            this.pulsationState = 'fadeOut';
                                        }
                                    } else {
                                        this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                                        this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                                        if (this.opacity <= 0) {
                                            this.opacity = 0;
                                            this.pulsationState = 'fadeIn';
                                            this.radius = 0;
                                        }
                                    }
                                };

                                particles.push(particle);
                            }
                        }
                    }
                } else if (pattern === 'fibonacci') {
                    const numParticles = 1200;
                    // Calculate maxRadius based on canvas dimensions
                    const maxRadius = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.28;
                    const goldenAngle = Math.PI * (3 - Math.sqrt(5));

                    for (let i = 0; i < numParticles; i++) {
                        const distance = Math.sqrt(i / numParticles) * maxRadius;
                        const angle = i * goldenAngle + Math.random();
                        const x = centerX + distance * Math.cos(angle);
                        const y = centerY + distance * Math.sin(angle);
                        particles.push(new Particle(x, y, angle, distance, centerX, centerY, null, settings));
                    }
                } else if (pattern === 'wave') {
                    const numParticles = 800;  // Increased for denser water effect
                    const gridCols = 40;
                    const gridRows = 30;
                    
                    // Calculate grid cell size based on canvas dimensions
                    const cellWidth = canvas.clientWidth / gridCols;
                    const cellHeight = canvas.clientHeight / gridRows;
                    
                    for (let row = 0; row < gridRows; row++) {
                        for (let col = 0; col < gridCols; col++) {
                            const x = col * cellWidth + (cellWidth / 2);
                            const y = row * cellHeight + (cellHeight / 2);
                            
                            const particle = new Particle(
                                x, y, 0, 0,
                                x, y,  // Each particle's center is its starting position
                                null,
                                {
                                    ...settings,
                                    lineOpacity: 0.15,
                                    lineWidth: 0.4,
                                    fadeInSpeed: 0.03,
                                    fadeOutSpeed: 0.02
                                }
                            );
                            
                            // Store grid position for wave calculations
                            particle.gridX = col;
                            particle.gridY = row;
                            particle.baseX = x;
                            particle.baseY = y;
                            
                            // Custom update function for water ripple effect
                            particle.update = function() {
                                const time = Date.now() * 0.001;
                                
                                // Create multiple wave sources
                                const wave1 = Math.sin(time * 1.5 + this.gridX * 0.2 + this.gridY * 0.3) * 4;
                                const wave2 = Math.cos(time * 2.0 + this.gridX * 0.3 - this.gridY * 0.2) * 3;
                                const wave3 = Math.sin(time * 2.5 - this.gridX * 0.4 + this.gridY * 0.4) * 2;
                                
                                // Combine waves with distance-based dampening
                                const centerDistX = (this.gridX - gridCols/2) / gridCols;
                                const centerDistY = (this.gridY - gridRows/2) / gridRows;
                                const distanceFromCenter = Math.sqrt(centerDistX * centerDistX + centerDistY * centerDistY);
                                const dampening = 1 - (distanceFromCenter * 0.5);
                                
                                // Apply combined waves to position
                                this.x = this.baseX + (wave1 + wave2 + wave3) * dampening;
                                this.y = this.baseY + (wave2 + wave3 + wave1) * dampening;
                                
                                // Standard pulsation logic
                                if (this.pulsationState === 'fadeIn') {
                                    this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                    this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                                    if (this.opacity >= 1) {
                                        this.opacity = 1;
                                        this.pulsationState = 'fadeOut';
                                    }
                                } else {
                                    this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                                    this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                                    if (this.opacity <= 0) {
                                        this.opacity = 0;
                                        this.pulsationState = 'fadeIn';
                                        this.radius = 0;
                                    }
                                }
                            };
                            
                            // Custom connect function to create grid-based connections
                            particle.connect = function(otherParticle) {
                                const gridDiffX = Math.abs(this.gridX - otherParticle.gridX);
                                const gridDiffY = Math.abs(this.gridY - otherParticle.gridY);
                                
                                // Connect only to immediate neighbors in the grid
                                if (gridDiffX <= 1 && gridDiffY <= 1) {
                                    this.connectedParticles.push(otherParticle);
                                }
                            };

                            particles.push(particle);
                        }
                    }
                } else if (pattern === 'gravitydispersion') {
                    const numParticles = 900;  // Updated from 800 to 900
                    const maxRadius = minDimension * 0.4;
                    const goldenAngle = Math.PI * (3 - Math.sqrt(5));

                    for (let i = 0; i < numParticles; i++) {
                        const distance = Math.sqrt(i / numParticles) * maxRadius;
                        const angle = i * goldenAngle + Math.random();
                        const x = centerX + distance * Math.cos(angle);
                        const y = centerY + distance * Math.sin(angle);

                        const particle = new Particle(x, y, angle, distance, centerX, centerY, null, {
                            ...settings,
                            movement_style: "gravity",
                            cycleLength: 3000,
                            maxVelocity: 0.5,
                            randomization: 0.9
                        });

                        particle.update = function() {
                            let xVelocity = this.xVelocity || 0;
                            let yVelocity = this.yVelocity || 0;

                            if (!this.cycleLength) {
                                var forceFactorDirection = Math.round(Math.random()) * 2 - 1;
                                var forceFactor = Math.cos(Date.now() * 0.0001);
                            } else {
                                var date = Date.now();
                                if (date % this.cycleLength === Math.round(Math.random())) {
                                    this.cycleLength = this.cycleLength * Math.round(Math.random() * 2);
                                }
                                if (date % (this.cycleLength * 2) < this.cycleLength) {
                                    var forceFactorDirection = 1;
                                } else {
                                    var forceFactorDirection = -1;
                                }
                                var forceFactor = (date % this.cycleLength) * 0.0001;
                            }

                            const maxDistance = 100;
                            const maxVelocity = this.maxVelocity;
                            
                            // Apply gravity (attraction to other particles)
                            for (let i = 0; i < particles.length; i++) {
                                if (particles[i] !== this) {
                                    const dx = particles[i].x - this.x;
                                    const dy = particles[i].y - this.y;
                                    const distance = Math.sqrt(dx * dx + dy * dy);
                                    const minDistance = 100; // Minimum distance to avoid strong forces
                                    
                                    if (distance > 0 && distance < maxDistance) {
                                        const forceDirectionX = dx / distance;
                                        const forceDirectionY = dy / distance;
                                        const force = forceFactor * (minDistance / distance) * forceFactorDirection;

                                        xVelocity = forceDirectionX * force + (0.5 - Math.random()) * this.randomization;
                                        yVelocity = forceDirectionY * force + (0.5 - Math.random()) * this.randomization;
                                    }
                                }
                            }
                            
                            // Limit velocities
                            if (xVelocity < 0) {
                                xVelocity = Math.max(xVelocity, -maxVelocity);
                            } else {
                                xVelocity = Math.min(xVelocity, maxVelocity);
                            }
                            if (yVelocity < 0) {
                                yVelocity = Math.max(yVelocity, -maxVelocity);
                            } else {
                                yVelocity = Math.min(yVelocity, maxVelocity);
                            }

                            // Update position based on velocity
                            this.x += xVelocity;
                            this.y += yVelocity;

                            // Wrap around edges
                            if (this.x + this.radius > canvas.width) {
                                this.x = 0;
                            }
                            if (this.x - this.radius < 0) {
                                this.x = canvas.width;
                            }
                            if (this.y + this.radius > canvas.height) {
                                this.y = 0;
                            }
                            if (this.y - this.radius < 0) {
                                this.y = canvas.height;
                            }

                            // Standard pulsation logic
                            if (this.pulsationState === 'fadeIn') {
                                this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                                if (this.opacity >= 1) {
                                    this.opacity = 1;
                                    this.pulsationState = 'fadeOut';
                                }
                            } else {
                                this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                                this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                                if (this.opacity <= 0) {
                                    this.opacity = 0;
                                    this.pulsationState = 'fadeIn';
                                    this.radius = 0;
                                }
                            }
                        };
                        particles.push(particle);
                    }
                } else if (pattern === 'power') {
                    const numParticles = 300;
                    const maxRadius = minDimension * 0.125; // Dense center cluster
                    const maxTendrilLength = minDimension * 0.3; // Tendrils reach to edge
                    
                    // Create two sets of particles: core and tendrils
                    const coreCount = 0;  // Core particles
                    const tendrilCount = 1000;  // Increased from 100 to 1000
                    
                    // First create core particles
                    for (let i = 0; i < coreCount; i++) {
                        const angle = Math.random() * Math.PI * 2;
                        const radius = Math.pow(Math.random(), 3) * maxRadius;
                        const x = centerX + radius * Math.cos(angle);
                        const y = centerY + radius * Math.sin(angle);
                        
                        const particle = new Particle(
                            x, y, angle, radius,
                            centerX, centerY,
                            null,
                            settings
                        );

                        // Core particles stay in the sphere
                        particle.update = function() {
                            // Core particles move very slowly
                            this.angle += this.orbitSpeed * 0.2; // Reduced speed for core particles
                            const radius = this.orbitRadius + Math.sin(this.angle * 2) * 5;
                            
                            this.x = this.centerX + radius * Math.cos(this.angle);
                            this.y = this.centerY + radius * Math.sin(this.angle);

                            // Standard pulsation logic
                            if (this.pulsationState === 'fadeIn') {
                                this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                                if (this.opacity >= 1) {
                                    this.opacity = 1;
                                    this.pulsationState = 'fadeOut';
                                }
                            } else {
                                this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                                this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                                if (this.opacity <= 0) {
                                    this.opacity = 0;
                                    this.pulsationState = 'fadeIn';
                                    this.radius = 0;
                                }
                            }
                        };

                        // For core particles, modify their draw method
                        particle.draw = function() {
                            // Only draw if particle is visible
                            if (this.opacity <= 0) return;
                            
                            // Calculate darkness based on distance from center
                            const dx = this.x - this.centerX;
                            const dy = this.y - this.centerY;
                            const distanceFromCenter = Math.sqrt(dx * dx + dy * dy);
                            const maxDistance = maxRadius * 2;
                            const brightness = 0.3 + (distanceFromCenter / maxDistance) * 0.7; // 30% to 100% brightness
                            
                            ctx.beginPath();
                            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                            ctx.fillStyle = `rgba(${brightness * 255}, ${brightness * 255}, ${brightness * 255}, ${this.opacity})`;
                            ctx.fill();
                        };

                        particles.push(particle);
                    }

                    // Create tendril clusters
                    const numTendrils = 8; // Number of main tendrils
                    const particlesPerTendril = Math.floor(tendrilCount / numTendrils); // Now ~125 particles per tendril
                    
                    for (let t = 0; t < numTendrils; t++) {
                        // Base angle for this tendril
                        const baseAngle = (t / numTendrils) * Math.PI * 2;
                        
                        // Create cluster of particles for this tendril
                        for (let i = 0; i < particlesPerTendril; i++) {
                            // Calculate position along the tendril (0 to 1)
                            const lengthPercent = i / particlesPerTendril;
                            
                            // Wider spread near the middle of the tendril
                            const spreadFactor = Math.sin(lengthPercent * Math.PI) * 40; // Max spread of 40 pixels
                            
                            // Random spread from the center line of the tendril
                            const angleVariation = (Math.random() - 0.5) * 0.3; // ±0.15 radians variation
                            const angle = baseAngle + angleVariation;
                            const radius = maxRadius; // Start at the surface of the core sphere
                            const x = centerX + radius * Math.cos(angle);
                            const y = centerY + radius * Math.sin(angle);
                            
                            const particle = new Particle(
                                x, y, angle, radius,
                                centerX, centerY,
                                null,
                                {
                                    ...settings,
                                    orbitSpeed: 0.001 + Math.random() * 0.002,
                                    lineOpacity: 0.4,
                                    lineWidth: 0.6
                                }
                            );

                            // Store the initial position and cluster info
                            particle.baseX = x;
                            particle.baseY = y;
                            particle.baseAngle = angle;
                            particle.tendrilGroup = t;
                            particle.lengthPercent = lengthPercent; // Store position along tendril

                            // Tendril particles extend outward
                            particle.update = function() {
                                const time = Date.now() * 0.0005; // Base time factor
                                
                                // Base wave motion
                                const waveOffset = this.baseAngle * 10;
                                const waveSpeed = time;
                                
                                // Primary tendril extension
                                const tendrilExtension = (0.2 + Math.sin(time + waveOffset)) * maxTendrilLength;
                                
                                // Calculate spread based on position along tendril
                                const spread = Math.sin(this.lengthPercent * Math.PI) * spreadFactor;
                                
                                // Speed increases along the length of the tendril
                                const speedMultiplier = this.lengthPercent; // 0 near center, 1 at tip
                                
                                // Secondary wave motion with speed variation
                                const lateralWave = Math.sin(waveSpeed * (0.5 + speedMultiplier) + this.lengthPercent * 5) * spread;
                                const verticalWave = Math.cos(waveSpeed * (0.2 + speedMultiplier * 0.3) + this.lengthPercent * 4) * spread * 0.5;
                                
                                // Calculate base position along tendril
                                const currentRadius = this.patternRadius + tendrilExtension * (1 - this.lengthPercent * 0.2);
                                
                                // Add wave motions to create organic movement
                                this.x = this.centerX + 
                                    currentRadius * Math.cos(this.baseAngle) +
                                    lateralWave * Math.cos(this.baseAngle + Math.PI/2);
                                this.y = this.centerY + 
                                    currentRadius * Math.sin(this.baseAngle) +
                                    lateralWave * Math.sin(this.baseAngle + Math.PI/2) +
                                    verticalWave;

                                // Standard pulsation logic
                                if (this.pulsationState === 'fadeIn') {
                                    this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                    this.radius = Math.min(this.maxRadius, this.radius + this.fadeInSpeed * 2);
                                    if (this.opacity >= 1) {
                                        this.opacity = 1;
                                        this.pulsationState = 'fadeOut';
                                    }
                                } else {
                                    this.opacity = Math.max(0, this.opacity - this.fadeOutSpeed);
                                    this.radius = Math.max(0, this.radius - this.fadeOutSpeed * 0.5);
                                    if (this.opacity <= 0) {
                                        this.opacity = 0;
                                        this.pulsationState = 'fadeIn';
                                        this.radius = 0;
                                    }
                                }
                            };

                            // Override connect method to create more connections within the tendril
                            particle.connect = function(otherParticle) {
                                if (this.tendrilGroup === otherParticle.tendrilGroup) {
                                    const lengthDiff = Math.abs(this.lengthPercent - otherParticle.lengthPercent);
                                    const xDiff = this.x - otherParticle.x;
                                    const yDiff = this.y - otherParticle.y;
                                    const distance = Math.sqrt(xDiff * xDiff + yDiff * yDiff);
                                    
                                    // Connect if either the particles are close in the tendril sequence
                                    // OR if they're physically close to each other
                                    if (lengthDiff < 0.2 || distance < 30) { // Increased from 0.1 to 0.2, added distance check
                                        this.connectedParticles.push(otherParticle);
                                    }
                                }
                            };

                            // For tendril particles, also modify their draw and drawConnections methods
                            particle.draw = function() {
                                // Only draw if particle is visible
                                if (this.opacity <= 0) return;
                                
                                // Calculate darkness based on distance from center
                                const dx = this.x - this.centerX;
                                const dy = this.y - this.centerY;
                                const distanceFromCenter = Math.sqrt(dx * dx + dy * dy);
                                const maxDistance = maxTendrilLength;
                                const brightness = 0.3 + (distanceFromCenter / maxDistance) * 0.7; // 30% to 100% brightness
                                
                                ctx.beginPath();
                                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                                ctx.fillStyle = `rgba(${brightness * 255}, ${brightness * 255}, ${brightness * 255}, ${this.opacity})`;
                                ctx.fill();
                            };

                            particle.drawConnections = function() {
                                // Skip if no connections or particle is invisible
                                if (!this.connectedParticles.length || this.opacity <= 0) return;

                                this.connectedParticles.forEach(otherParticle => {
                                    // Skip if other particle is invisible
                                    if (otherParticle.opacity <= 0) return;

                                    const distance = Math.hypot(this.x - otherParticle.x, this.y - otherParticle.y);
                                    const maxDistance = 200;

                                    if (distance < maxDistance) {
                                        // Calculate average distance from center for this connection
                                        const avgX = (this.x + otherParticle.x) / 2 - this.centerX;
                                        const avgY = (this.y + otherParticle.y) / 2 - this.centerY;
                                        const avgDistanceFromCenter = Math.sqrt(avgX * avgX + avgY * avgY);
                                        const maxDistance = maxTendrilLength;
                                        const brightness = 0.2 + (avgDistanceFromCenter / maxDistance) * 0.8; // 20% to 100% brightness
                                        
                                        // Calculate opacity based on distance between particles
                                        const opacity = (1 - distance / maxDistance) * this.lineOpacity;
                                        
                                        // Only draw if the line would be visible
                                        if (opacity > 0.01) {
                                            ctx.beginPath();
                                            ctx.moveTo(this.x, this.y);
                                            ctx.lineTo(otherParticle.x, otherParticle.y);
                                            ctx.strokeStyle = `rgba(${brightness * 50}, ${brightness * 50}, ${brightness * 50}, ${opacity})`;
                                            ctx.lineWidth = this.lineWidth;
                                            ctx.stroke();
                                        }
                                    }
                                });
                            };

                            particles.push(particle);
                        }
                    }
                }
            }

            // Main animation loop
            function animate() {
                // Only request next frame if tab is visible
                if (!isTabVisible) {
                    return;
                }

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Update spatial grid
                const spatialGrid = new SpatialGrid(canvas.clientWidth, canvas.clientHeight, 30);
                particles.forEach(p => {
                    p.clearConnections();
                    spatialGrid.insert(p);
                });

                // Batch all updates first
                particles.forEach(particle => particle.update());

                // Then batch all connections
                const connectionDistance = 20;
                particles.forEach(particle => {
                    const nearbyParticles = spatialGrid.getNearbyParticles(particle, connectionDistance);
                    nearbyParticles.forEach(other => {
                        if (other !== particle) {
                            const distance = Math.hypot(particle.x - other.x, particle.y - other.y);
                            if (distance < connectionDistance) {
                                particle.connect(other);
                            }
                        }
                    });
                });

                // Finally batch all rendering
                ctx.save();
                // Draw all connections first
                particles.forEach(particle => particle.drawConnections());
                // Then draw all particles
                particles.forEach(particle => particle.draw());
                ctx.restore();

                // Only request next frame if tab is still visible
                if (isTabVisible) {
                    activeAnimations.set(canvas.id, requestAnimationFrame(animate));
                }
            }

            // Get pattern from canvas class
            const pattern = Array.from(canvas.classList).find(className =>
                ['asteroids', 'donut', 'fibonacci', 'gravitydispersion', 'power', 'wave'].includes(className)
            ) || 'fibonacci'; // Changed default to fibonacci

            // Pattern-specific settings
            const patternSettings = {
                asteroids: {
                    orbitSpeed: 0.005,
                    lineOpacity: 0.2,
                    orbitVariation: 0.004,
                    rotationSpeed: 0.003,
                },
                donut: {
                    orbitSpeed: 0.002,
                    orbitVariation: 0.0005,
                    lineOpacity: 0.5
                },
                fibonacci: {
                    lineOpacity: 0.5,
                    fadeInSpeed: 0.009,
                    fadeInVariation: 0.01,
                    fadeOutSpeed: 0.01,
                    fadeOutVariation: 0.06,
                    orbitVariation: 0.001,
                    movement_style: "gravity",
                    cycleLength: 3000,
                    maxVelocity: 0.4,
                    randomization: 0.9,
                },
                gravitydispersion: {
                    orbitSpeed: 0.5,
                    lineOpacity: 0.55,
                    fadeInSpeed: 0.02,
                    fadeOutSpeed: 0.01,
                    rotationSpeed: 0.002,
                    lineWidth: 0.6,
                    movement_style: "gravity",
                    cycleLength: 3000,
                    maxVelocity: 0.5,
                    randomization: 0.9,
                },
                power: {
                    lineOpacity: 0.4,
                    fadeInSpeed: 0.015,
                    fadeOutSpeed: 0.008,
                    orbitSpeed: 0.001,
                    lineWidth: 0.6
                },
                wave: {
                    orbitSpeed: 0.5,
                    lineOpacity: 0.25,
                    fadeInSpeed: 0.02,
                    fadeOutSpeed: 0.01,
                    rotationSpeed: 0.002,
                    lineWidth: 0.5
                },
            };

            // Start animation with pattern from class
            createParticles(pattern, patternSettings[pattern] || {});
            animate();
        }
    }

    // Observe all canvases
    canvases.forEach(canvas => {
        observer.observe(canvas);
    });

    // Update visibility change listener
    document.addEventListener('visibilitychange', () => {
        isTabVisible = !document.hidden;
        
        if (document.hidden) {
            // Pause animations when tab is hidden
            canvases.forEach(canvas => {
                if (activeAnimations.has(canvas.id)) {
                    cancelAnimationFrame(activeAnimations.get(canvas.id));
                    activeAnimations.delete(canvas.id);
                }
            });
        } else {
            // Resume animations by reinitializing the current canvas
            if (currentCanvas) {
                initializeCanvas(currentCanvas);
            }
        }
    });
});
