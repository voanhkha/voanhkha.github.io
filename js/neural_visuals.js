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
            const isHelixCanvas = canvas.classList.contains('helix');

            // Make canvas full screen
            // function resizeCanvas() {
            //     const pixelRatio = window.devicePixelRatio || 1;
            //     //const width = canvas.clientWidth;
            //     //const height = canvas.clientHeight;
            //
            //     // const width = window.innerWidth;
            //     // const height = window.innerHeight;
            //
            //     let width, height;
            //
            //     if (pattern === 'helix') {
            //         // Use container height instead of full window
            //         const rect = canvas.getBoundingClientRect();
            //         width = rect.width;
            //         height = rect.height;
            //     } else {
            //         width = window.innerWidth;
            //         height = window.innerHeight;
            //     }
            //
            //
            //     // Set actual size in memory (scaled to account for extra pixel density)
            //     canvas.width = width * pixelRatio;
            //     canvas.height = height * pixelRatio;
            //
            //     // Style size (CSS pixels)
            //     canvas.style.width = width + 'px';
            //     canvas.style.height = height + 'px';
            //
            //     // Scale all drawing operations by the dpr
            //     // ctx.scale(pixelRatio, pixelRatio);
            //
            //     // ctx.setTransform(1, 0, 0, 1, 0, 0);
            //     ctx.scale(pixelRatio, pixelRatio);
            //
            // }

            function resizeCanvas() {
                const pixelRatio = window.devicePixelRatio || 1;

                let width, height;

                if (isHelixCanvas) {
                    // Use the actual on-page size (prevents huge blank space in page flow)
                    const rect = canvas.getBoundingClientRect();
                    // width = rect.width || canvas.clientWidth || window.innerWidth;
                    // height = rect.height || canvas.clientHeight || 300; // safe fallback
                    width = window.innerWidth;
                    height = window.innerHeight;

                } else {
                    // Keep your original fullscreen behavior for other animations
                    width = window.innerWidth;
                    height = window.innerHeight;
                }

                canvas.width = Math.max(1, Math.floor(width * pixelRatio));
                canvas.height = Math.max(1, Math.floor(height * pixelRatio));

                canvas.style.width = width + 'px';
                canvas.style.height = height + 'px';

                // Prevent cumulative scaling on resize
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.scale(pixelRatio, pixelRatio);
            }


            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);

            // Particle class: Manages individual particle behavior, appearance, and connections
            class Particle {
                constructor(x, y, angle, patternRadius, centerX, centerY, color, settings) {
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
                    this.angle = (this.angle + this.orbitSpeed) % this.TWO_PI;

                    // Calculate position
                    const angleCalc = this.angle;
                    this.x = this.centerX + this.orbitRadius * Math.cos(angleCalc);
                    this.y = this.centerY + this.orbitRadius * Math.sin(angleCalc);

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

                        const distance = Math.hypot(this.x - otherParticle.x, this.y - otherParticle.y);
                        const maxDistance = 250;

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

                } else if (pattern === 'blackhole') {
    // Black Hole Accretion Disk
    // - particles orbit and spiral inward
    // - speed increases near center
    // - particles "fall in" and respawn at outer disk

    const numParticles = 550;                   // 700–1400 depending on perf
    const minDim = Math.min(canvas.clientWidth, canvas.clientHeight);

    const eventHorizon = minDim * 0.01;         // "black hole" radius
    const innerDisk = minDim * 0.10;            // inner edge of visible disk
    const outerDisk = minDim * 0.46;            // outer edge of disk
    const diskThickness = minDim * 0.10;        // vertical thickness (tilt illusion)

    const baseSpin = 0.05;                      // rad/sec baseline (outer)
    const spinBoost = 0.5;                      // extra spin near center
    const baseInfall = 10;                      // px/sec inward (outer)
    const infallBoost = 28;                     // extra infall near center
    const noiseAmp = 0.9;                       // turbulence strength

    // Slight ellipse + tilt for "disk" look
    const ellipse = 0.95;                       // >1 stretches x
    const tilt = 0.85;                          // 0..1 compress y

    // Helper to respawn a particle in the outer disk
    function respawn(p) {
        // Pick radius biased to outer disk (more particles out there)
        const u = Math.random();
        const r = innerDisk + (outerDisk - innerDisk) * Math.sqrt(u); // sqrt bias outward

        const a = Math.random() * Math.PI * 2;
        p.r = r;
        p.a = a;

        // small vertical offset to give thickness
        p.z = (Math.random() - 0.5) * diskThickness;

        // keep a little state for turbulence
        p.seed = Math.random() * Math.PI * 2;

        // initialize position
        const x = centerX + (r * Math.cos(a)) * ellipse;
        const y = centerY + (r * Math.sin(a)) * tilt + p.z * 0.15;
        p.x = x;
        p.y = y;

        // calm velocities at spawn to avoid snapping
        p.vx = 0;
        p.vy = 0;

        // make them a bit larger/brighter than default
        p.maxRadius = 1.2 + Math.random() * 1.6;
    }

    for (let i = 0; i < numParticles; i++) {
        const p = new Particle(centerX, centerY, 0, 0, centerX, centerY, null, {
            ...settings,
            lineOpacity: settings.lineOpacity ?? 0.22,
            lineWidth: settings.lineWidth ?? 0.55,
            fadeInSpeed: settings.fadeInSpeed ?? 0.02,
            fadeOutSpeed: settings.fadeOutSpeed ?? 0.01,
        });

        // motion state
        p.r = 0;
        p.a = 0;
        p.z = 0;
        p.vx = 0;
        p.vy = 0;
        p.seed = 0;

        respawn(p);

        p.update = function() {
            const t = Date.now() * 0.001;

            // Normalize radius: 0 near innerDisk, 1 near outerDisk
            const rn = Math.max(0, Math.min(1, (this.r - innerDisk) / (outerDisk - innerDisk)));
            const nearCenter = 1 - rn;

            // Spin faster near center
            const spin = baseSpin + spinBoost * nearCenter;
            // Infall faster near center
            const infall = baseInfall + infallBoost * nearCenter;

            // Orbit direction (all same looks clean; flip some if you want chaos)
            this.a += (spin * (1 / 60));

            // Spiral inward
            this.r -= (infall * (1 / 60));

            // Turbulence (subtle)
            const turb = noiseAmp * (0.6 + 0.6 * nearCenter);
            const wobX = Math.sin(t * 1.3 + this.seed + this.r * 0.02) * turb;
            const wobY = Math.cos(t * 1.1 + this.seed + this.r * 0.018) * turb;

            // Disk projection: ellipse + tilt + thickness
            const tx = centerX + (this.r * Math.cos(this.a)) * ellipse + wobX;
            const ty = centerY + (this.r * Math.sin(this.a)) * tilt + this.z * 0.15 + wobY;

            // Smooth to target to avoid jitter
            this.vx = this.vx * 0.90 + (tx - this.x) * 0.10;
            this.vy = this.vy * 0.90 + (ty - this.y) * 0.10;

            this.x += this.vx;
            this.y += this.vy;

            // If it crosses event horizon, respawn outside
            if (this.r < eventHorizon) {
                respawn(this);
            }

            // Standard pulsation
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

        // Make particles slightly brighter near center (accretion glow illusion)
        p.draw = function() {
            if (this.opacity <= 0) return;

            const rn = Math.max(0, Math.min(1, (this.r - innerDisk) / (outerDisk - innerDisk)));
            const nearCenter = 1 - rn;

            const boost = 1.25 + 0.9 * nearCenter;        // brighter near center
            const alpha = Math.min(1, this.opacity * boost);

            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.fill();
        };

        particles.push(p);
    }

    // Optional: draw the event horizon as a dark circle by using a background pass.
    // If you want it, tell me—your current loop clears the whole canvas each frame,
    // so we’d add a lightweight "drawBackground" hook for this pattern.


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
                    const numParticles = 800;
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


                    } else if (pattern === 'helix') {
                        // DNA-like double helix with simple 3D-ish perspective projection
                        const strandCount = 2;
                        const pointsPerStrand = 480;                 // total particles = ~520
                        const totalParticles = strandCount * pointsPerStrand;

                        const spanY = minDimension * 0.95;           // vertical span of the helix
                        const baseTopY = centerY - spanY / 2;

                        const helixRadius = minDimension * 0.8;     // radius of the helix
                        const twistStep = (Math.PI * 2) / 22;        // controls twist density
                        const scrollSpeed = 25;                      // px/sec along y (visual flow)
                        const rotateSpeed = 0.3;                     // rad/sec for twisting

                        const perspective = 520;                     // bigger = flatter projection
                        const zScale = 1.0;                          // depth scale
                        const yDepthWobble = 0.06;                   // subtle vertical wobble from depth

                        // Create particles on two strands (phase offset by PI)
                        for (let strand = 0; strand < strandCount; strand++) {
                            const phaseOffset = strand * Math.PI;

                            for (let i = 0; i < pointsPerStrand; i++) {
                                // We seed x/y, but we'll override update() so it doesn't matter much
                                const seedAngle = i * twistStep + phaseOffset;
                                const seedX = centerX + helixRadius * Math.cos(seedAngle);
                                const seedY = baseTopY + (i / pointsPerStrand) * spanY;

                                const p = new Particle(
                                    seedX,
                                    seedY,
                                    seedAngle,
                                    helixRadius,
                                    centerX,
                                    centerY,
                                    null,
                                    {
                                        ...settings,
                                        // keep your existing vibe
                                        fadeInSpeed: (settings.fadeInSpeed ?? 0.012),
                                        fadeOutSpeed: (settings.fadeOutSpeed ?? 0.008),
                                        lineOpacity: (settings.lineOpacity ?? 0.35),
                                        lineWidth: (settings.lineWidth ?? 0.55),
                                    }
                                );

                                // Make rain particles bigger
                                p.maxRadius = 1.6 + Math.random() * 1.2;  // was ~0.6–1.4 in default

                                // Strand indexing
                                p.strand = strand;
                                p.strandPhase = phaseOffset;
                                p.idx = i;

                                // Per-particle speed variation (some faster, some slower)
                                p.scrollMul = 0.65 + Math.random() * 0.9;   // ~0.65..1.55
                                p.rotateMul = 0.70 + Math.random() * 0.9;   // ~0.70..1.60

                                // Optional: make "streaks" of speed along the strand (looks nicer than pure random)
                                p.scrollMul *= 0.85 + 0.3 * Math.sin(i * 0.15 + strand * 2.1);
                                p.rotateMul *= 0.85 + 0.3 * Math.cos(i * 0.12 + strand * 1.7);

                                // Store depth to use in draw / connections
                                p.z = 0;
                                p.depthScale = 1;

                                // Override update: helix motion + depth projection + your pulsation
                        p.update = function() {
                            const t = Date.now() * 0.001;

                            // --- Horizontal travel (wrap in X) ---
                            const spanX = minDimension * 0.95;
                            const baseLeftX = centerX - spanX / 2;

                            const xFloat = (i * (spanX / pointsPerStrand) + t * scrollSpeed * this.scrollMul) % spanX;
                            const x = baseLeftX + xFloat;

                            // --- Twist over time ---
                            const theta = i * twistStep + this.strandPhase + t * rotateSpeed * this.rotateMul;

                            // --- Hourglass radius now depends on X (pinch at center) ---
                            const xNormalized = ((x - centerX) / (spanX / 2)); // [-1, 1]
                            const pinchStrength = 0.8; // tune 0..0.85
                            const hourglassFactor = 1 - pinchStrength * Math.exp(-4 * xNormalized * xNormalized);
                            const dynamicRadius = helixRadius * hourglassFactor;

                            // --- 3D helix around the X-axis (circle in Y-Z plane) ---
                            const y3 = dynamicRadius * Math.cos(theta);
                            const z3 = dynamicRadius * Math.sin(theta) * zScale;

                            // Perspective projection
                            const depth = (perspective / (perspective + z3 + helixRadius));
                            this.depthScale = depth;
                            this.z = z3;

                            // Project to screen
                            this.x = x;
                            this.y = centerY + y3 * depth + (z3 * yDepthWobble) * depth;

                            // --- Standard pulsation (same as your style) ---
                            if (this.pulsationState === 'fadeIn') {
                                this.opacity = Math.min(1, this.opacity + this.fadeInSpeed);
                                const targetMax = this.maxRadius * (0.7 + 0.8 * depth);
                                this.radius = Math.min(targetMax, this.radius + this.fadeInSpeed * 2);

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

                                // Optional: depth-aware draw (still "white", just subtly stronger when closer)
                                p.draw = function() {
                                    if (this.opacity <= 0) return;

                                    // 0.8..1.15-ish boost based on depth (keep it subtle)
                                    const boost = 0.8 + 0.6 * this.depthScale;
                                    const a = Math.min(1, this.opacity * boost);

                                    ctx.beginPath();
                                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                                    ctx.fillStyle = `rgba(255, 255, 255, ${a})`;
                                    ctx.fill();
                                };

                                // // Depth-aware connections (still dark grey like your default)
                                // p.drawConnections = function() {
                                //     if (!this.connectedParticles.length || this.opacity <= 0) return;
                                //
                                //     this.connectedParticles.forEach(other => {
                                //         if (other.opacity <= 0) return;
                                //
                                //         const dx = this.x - other.x;
                                //         const dy = this.y - other.y;
                                //         const dist = Math.hypot(dx, dy);
                                //
                                //         const maxDist = 42; // tuned so neighbors connect nicely
                                //         if (dist < maxDist) {
                                //             const base = (1 - dist / maxDist) * (this.lineOpacity ?? 0.35);
                                //
                                //             // Slight depth weight for nicer “front strand” feel
                                //             const depthW = 0.75 + 0.5 * ((this.depthScale + other.depthScale) * 0.5);
                                //             const opacity = base * depthW;
                                //
                                //             if (opacity > 0.01) {
                                //                 ctx.beginPath();
                                //                 ctx.moveTo(this.x, this.y);
                                //                 ctx.lineTo(other.x, other.y);
                                //                 ctx.strokeStyle = `rgba(50, 50, 50, ${opacity})`;
                                //                 ctx.lineWidth = this.lineWidth ?? 0.55;
                                //                 ctx.stroke();
                                //             }
                                //         }
                                //     });
                                // };

                                particles.push(p);
                            }
                        }

                        // Make connections “structured”:
                        // neighbors along each strand + occasional cross-bridge between strands
                        //
                        // Note: your main loop will clearConnections() then spatial-grid connect.
                        // We can keep that AND add deterministic links by overriding connect() to allow
                        // only “good” helix links (prevents random clutter).
                        const allowCrossEvery = 4;

                        // particles.forEach(p => {
                        //     p.connect = function(other) {
                        //         // same strand: connect to near indices only
                        //         if (this.strand === other.strand) {
                        //             const d = Math.abs(this.idx - other.idx);
                        //             if (d === 1 || d === 2) this.connectedParticles.push(other);
                        //             return;
                        //         }
                        //
                        //         // different strands: connect "rungs" occasionally (like DNA base pairs)
                        //         if (Math.abs(this.idx - other.idx) <= 1 && (this.idx % allowCrossEvery === 0)) {
                        //             this.connectedParticles.push(other);
                        //         }
                        //     };
                        // });



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

} else if (pattern === 'brainnet') {
    // Neural Constellation Brain (Organic Network Growth)
    // - particles live inside a "brain" silhouette (two overlapping ellipses)
    // - they drift with gentle noise + slight attraction to wandering anchors
    // - your global proximity connection logic draws the dim network

    const numParticles = 500;
    const minDim = Math.min(canvas.clientWidth, canvas.clientHeight);

    // Brain silhouette parameters (two lobes)
    const brainW = minDim * 0.65;
    const brainH = minDim * 0.65;
    const lobeOffset = brainW * 0.18;

    // Ellipse radii
    const rx = brainW * 0.42;
    const ry = brainH * 0.50;

    // Helpers
    function insideEllipse(x, y, cx, cy, rx, ry) {
        const dx = (x - cx) / rx;
        const dy = (y - cy) / ry;
        return (dx * dx + dy * dy) <= 1;
    }

    function insideBrain(x, y) {
        // union of two ellipses
        const left = insideEllipse(x, y, centerX - lobeOffset, centerY, rx, ry);
        const right = insideEllipse(x, y, centerX + lobeOffset, centerY, rx, ry);
        // add small “stem” hint by allowing a tiny lower ellipse
        const stem = insideEllipse(x, y, centerX, centerY + ry * 0.75, rx * 0.35, ry * 0.35);
        return left || right || stem;
    }

    function randomPointInBrain() {
        // rejection sampling (fast enough for ~1k points)
        for (let tries = 0; tries < 5000; tries++) {
            const x = centerX + (Math.random() - 0.5) * brainW;
            const y = centerY + (Math.random() - 0.5) * brainH * 1.2;
            if (insideBrain(x, y)) return { x, y };
        }
        return { x: centerX, y: centerY };
    }

    // A few “growth anchors” that wander; particles are gently pulled to nearest anchor
    const anchors = [];
    const numAnchors = 8;
    for (let a = 0; a < numAnchors; a++) {
        const pt = randomPointInBrain();
        anchors.push({
            x: pt.x, y: pt.y,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            seed: Math.random() * Math.PI * 2,
        });
    }

    function updateAnchors() {
        const t = Date.now() * 0.0001;
        for (const an of anchors) {
            // drift with gentle sinusoidal bias
            const ax = Math.sin(t * 0.35 + an.seed) * 0.12;
            const ay = Math.cos(t * 0.30 + an.seed) * 0.12;

            // const ax = Math.sin(t * 0.35 + an.seed) * 0.22;
            // const ay = Math.cos(t * 0.30 + an.seed) * 0.22;

            an.vx = an.vx * 0.96 + ax;
            an.vy = an.vy * 0.96 + ay;

            an.x += an.vx;
            an.y += an.vy;

            // keep anchor inside brain (reflect)
            if (!insideBrain(an.x, an.y)) {
                an.vx *= -0.2;
                an.vy *= -0.2;
                // nudge back inward
                an.x = Math.max(centerX - brainW * 0.45, Math.min(centerX + brainW * 0.45, an.x));
                an.y = Math.max(centerY - brainH * 0.55, Math.min(centerY + brainH * 0.70, an.y));
            }
        }
    }

    for (let i = 0; i < numParticles; i++) {
        const pt = randomPointInBrain();

        const p = new Particle(pt.x, pt.y, 0, 0, centerX, centerY, null, {
            ...settings,
            lineOpacity: settings.lineOpacity ?? 0.28,
            lineWidth: settings.lineWidth ?? 0.55,
            fadeInSpeed: settings.fadeInSpeed ?? 0.018,
            fadeOutSpeed: settings.fadeOutSpeed ?? 0.010
        });

        // Make these particles a bit more visible
        p.maxRadius = 1.1 + Math.random() * 1.2;

        // Local motion state
        p.vx = 0;
        p.vy = 0;
        p.seed = Math.random() * Math.PI * 2;
        p.anchorBias = Math.random(); // slightly different pull per particle

        p.update = function() {
            const t = Date.now() * 0.001;
            updateAnchors();

            // Find nearest anchor (small count => cheap)
            let best = anchors[0];
            let bestD = Infinity;
            for (let k = 0; k < anchors.length; k++) {
                const dx = anchors[k].x - this.x;
                const dy = anchors[k].y - this.y;
                const d2 = dx * dx + dy * dy;
                if (d2 < bestD) { bestD = d2; best = anchors[k]; }
            }

            // “organic” noise drift
            const n1 = Math.sin(t * 0.9 + this.seed + this.x * 0.01) * 0.35;
            const n2 = Math.cos(t * 0.8 + this.seed + this.y * 0.01) * 0.35;

            // Gentle attraction to nearest anchor
            const dx = best.x - this.x;
            const dy = best.y - this.y;

            // const pull = 0.002 + 0.006 * this.anchorBias;
            const pull = 0.001 + 0.005 * this.anchorBias;
            const ax = dx * pull + n1;
            const ay = dy * pull + n2;

            this.vx = this.vx * 0.92 + ax;
            this.vy = this.vy * 0.92 + ay;

            this.x += this.vx;
            this.y += this.vy;

            // Keep inside brain: if outside, bounce back
            // if (!insideBrain(this.x, this.y)) {
            //     this.x -= this.vx * 2.2;
            //     this.y -= this.vy * 2.2;
            //     this.vx *= -0.65;
            //     this.vy *= -0.65;
            // }

            if (!insideBrain(this.x, this.y)) {
              this.x -= this.vx * 1.2;
              this.y -= this.vy * 1.2;
              this.vx *= -0.35;
              this.vy *= -0.35;
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

        particles.push(p);
    }

    } else if (pattern === 'magfield') {
    // Magnetic Field Lines
    // - particles advect along a dipole-like vector field between two poles
    // - global proximity connections give the “field line web” look

    const numParticles = 300;
    const minDim = Math.min(canvas.clientWidth, canvas.clientHeight);

    // Pole positions

// Use current CSS-pixel canvas size for true visual center
    const centerShiftX = -200; // negative = move left, positive = move right
    const cx = canvas.clientWidth * 0.5 + centerShiftX;
    // const cx = canvas.clientWidth * 0.5;
    const cy = canvas.clientHeight * 0.5;
    const poleSep = minDim * 0.28;
    const p1 = { x: cx - poleSep, y: cy };
    const p2 = { x: cx + poleSep, y: cy };

    // const poleSep = minDim * 0.28;
    // const p1 = { x: centerX - poleSep, y: centerY };
    // const p2 = { x: centerX + poleSep, y: centerY };

    // Field tuning
    const baseSpeed = 0.25;     // advection speed
    const swirl = 0.35;         // adds curl-like behavior around poles
    const soft = 200;            // softening to avoid singularity
    const targetR = minDim * 0.56;      // desired big orbit radius
    const band = minDim * 0.24;         // thickness of the allowed band
    const keepStrength = 1.3;           // how strongly to keep the orbit wide (0.3..1.4)
    const centerKickR = minDim * 0.30;  // if inside this, push outward
    const centerKick = 1.2;             // kick strength

    function fieldAt(x, y) {
        // “electric-like” contributions from two opposite poles
        // (not physically perfect magnetostatics, but looks great)
        const dx1 = x - p1.x, dy1 = y - p1.y;
        const dx2 = x - p2.x, dy2 = y - p2.y;

        const r1 = Math.sqrt(dx1 * dx1 + dy1 * dy1 + soft);
        const r2 = Math.sqrt(dx2 * dx2 + dy2 * dy2 + soft);

        // opposite signs create “flow from one pole to the other”
        const s1 = 1 / (r1 * r1);
        const s2 = 1 / (r2 * r2);

        // radial components
        let fx = dx1 * s1 - dx2 * s2;
        let fy = dy1 * s1 - dy2 * s2;

        // add swirl around each pole (perpendicular)
        fx += (-dy1 * s1 + dy2 * s2) * swirl;
        fy += ( dx1 * s1 - dx2 * s2) * swirl;

        // normalize
        const m = Math.hypot(fx, fy) + 1e-6;
        return { fx: fx / m, fy: fy / m };
    }

    for (let i = 0; i < numParticles; i++) {
        const x0 = Math.random() * canvas.clientWidth;
        const y0 = Math.random() * canvas.clientHeight;

        const p = new Particle(x0, y0, 0, 0, centerX, centerY, null, {
            ...settings,
            lineOpacity: settings.lineOpacity ?? 0.24,
            lineWidth: settings.lineWidth ?? 0.50,
            fadeInSpeed: settings.fadeInSpeed ?? 0.018,
            fadeOutSpeed: settings.fadeOutSpeed ?? 0.010
        });

        p.maxRadius = 1.0 + Math.random() * 1.2;
        p.vx = 0;
        p.vy = 0;
        p.speedMul = 0.7 + Math.random() * 1.0;
        p.seed = Math.random() * Math.PI * 2;

        p.update = function() {
            const t = Date.now() * 0.001;

            // Sample vector field
            const f = fieldAt(this.x, this.y);

            // Add subtle time-varying wobble (keeps it alive)
            const wob = 0.25;
            const wx = Math.sin(t * 0.9 + this.seed) * wob;
            const wy = Math.cos(t * 0.8 + this.seed) * wob;

            // --- Keep particles circling at a large radius (avoid collapsing to a dot) ---
            const cx = (p1.x + p2.x) * 0.5;
            const cy = (p1.y + p2.y) * 0.5;

            const dxC = this.x - cx;
            const dyC = this.y - cy;
            const r = Math.hypot(dxC, dyC) + 1e-6;

            // outward unit vector from center
            const ux = dxC / r;
            const uy = dyC / r;

            // spring toward target radius (positive => push outward, negative => pull inward)
            const err = (targetR - r) / band;                 // roughly -1..+1 inside band
            const spring = Math.max(-1, Math.min(1, err)) * keepStrength;

            // “kick out” if too close to center (prevents collapse)
            const kick = (r < centerKickR) ? (centerKick * (1 - r / centerKickR)) : 0;

            // radial correction force
            const radX = (spring + kick) * ux;
            const radY = (spring + kick) * uy;

            const sp = baseSpeed * this.speedMul;
            // Smooth advection
            this.vx = this.vx * 0.90 + (f.fx * sp + wx) * 0.8;
            this.vy = this.vy * 0.90 + (f.fy * sp + wy) * 0.8;

            // const sp = baseSpeed * this.speedMul;
            // Mix field direction + radial keeper (radX/radY)
            // this.vx = this.vx * 0.90 + ((f.fx + radX) * sp + wx) * 0.8;
            // this.vy = this.vy * 0.90 + ((f.fy + radY) * sp + wy) * 0.8;

            this.x += this.vx;
            this.y += this.vy;

            // Wrap
            if (this.x < 0) this.x += canvas.clientWidth;
            if (this.x > canvas.clientWidth) this.x -= canvas.clientWidth;
            if (this.y < 0) this.y += canvas.clientHeight;
            if (this.y > canvas.clientHeight) this.y -= canvas.clientHeight;

            // Standard pulsation
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

        particles.push(p);
    }

    } else if (pattern === 'flock') {
    // Minimal Boids Flocking
    // - classic separation/alignment/cohesion
    // - global proximity connections create a subtle “swarm web”

    const numParticles = 520; // boids are heavier; keep lower than other patterns
    const minDim = Math.min(canvas.clientWidth, canvas.clientHeight);

    // Boids parameters
    // const perception = 55;
    // const separationDist = 20;
    // const maxSpeed = 1.35;
    // const maxForce = 0.04;
    // const wSep = 1.20;
    // const wAli = 0.85;
    // const wCoh = 0.75;

    const perception = 95;        // see neighbors from further away (larger flock structure)
    const separationDist = 54;    // keep personal space larger => bigger cluster
    const maxSpeed = 1.55;        // a bit more roaming
    const maxForce = 0.045;       // slightly more responsiveness
    const wSep = 1.90;            // stronger push apart => fat flock
    const wAli = 0.95;            // keep it coherent
    const wCoh = 0.55;            // reduce “suck into a dot”

    for (let i = 0; i < numParticles; i++) {
        const x0 = Math.random() * canvas.clientWidth;
        const y0 = Math.random() * canvas.clientHeight;

        const p = new Particle(x0, y0, 0, 0, centerX, centerY, null, {
            ...settings,
            lineOpacity: settings.lineOpacity ?? 0.20,
            lineWidth: settings.lineWidth ?? 0.50,
            fadeInSpeed: settings.fadeInSpeed ?? 0.016,
            fadeOutSpeed: settings.fadeOutSpeed ?? 0.010
        });

        p.maxRadius = 1.2 + Math.random() * 1.2;

        // velocity
        p.vx = (Math.random() - 0.5) * 2;
        p.vy = (Math.random() - 0.5) * 2;

        p.update = function() {
            // Use your SpatialGrid to find neighbors (fast)
            // We'll create a local grid each frame in animate(), so here we’ll just use
            // the globally available `spatialGrid` by referencing it if you expose it.
            // But your code doesn't expose it. So: we do a cheap local neighbor sample
            // by scanning a limited number of random boids (keeps it self-contained).

            // If you want the fully correct/faster version using your SpatialGrid directly,
            // tell me and I’ll patch animate() in a tiny safe way.
            const sampleN = 26; // small random sample for performance
            let sepX = 0, sepY = 0, sepCount = 0;
            let aliX = 0, aliY = 0, aliCount = 0;
            let cohX = 0, cohY = 0, cohCount = 0;

            for (let s = 0; s < sampleN; s++) {
                const other = particles[(Math.random() * particles.length) | 0];
                if (other === this) continue;

                const dx = other.x - this.x;
                const dy = other.y - this.y;

                // wrap-aware distance (optional-ish; helps on edges)
                const wx = dx - Math.sign(dx) * Math.max(0, Math.abs(dx) - canvas.clientWidth / 2);
                const wy = dy - Math.sign(dy) * Math.max(0, Math.abs(dy) - canvas.clientHeight / 2);

                const d = Math.hypot(wx, wy);
                if (d < 1e-6) continue;

                if (d < separationDist) {
                    // separation: steer away
                    sepX += (-wx / d) / d;
                    sepY += (-wy / d) / d;
                    sepCount++;
                }

                if (d < perception) {
                    // alignment: match velocity
                    aliX += other.vx;
                    aliY += other.vy;
                    aliCount++;

                    // cohesion: go toward neighbors’ center
                    cohX += other.x;
                    cohY += other.y;
                    cohCount++;
                }
            }

            // Steering helpers
            function limit(x, y, maxVal) {
                const m = Math.hypot(x, y);
                if (m > maxVal) {
                    const k = maxVal / (m + 1e-6);
                    return { x: x * k, y: y * k };
                }
                return { x, y };
            }

            let ax = 0, ay = 0;

            // Separation
            if (sepCount > 0) {
                sepX /= sepCount; sepY /= sepCount;
                const sep = limit(sepX, sepY, maxForce);
                ax += sep.x * wSep;
                ay += sep.y * wSep;
            }

            // Alignment
            if (aliCount > 0) {
                aliX /= aliCount; aliY /= aliCount;
                // desired velocity in direction of average
                const ali = limit(aliX - this.vx, aliY - this.vy, maxForce);
                ax += ali.x * wAli;
                ay += ali.y * wAli;
            }

            // Cohesion
            if (cohCount > 0) {
                cohX /= cohCount; cohY /= cohCount;
                const toCX = cohX - this.x;
                const toCY = cohY - this.y;
                const coh = limit(toCX - this.vx, toCY - this.vy, maxForce);
                ax += coh.x * wCoh;
                ay += coh.y * wCoh;
            }

            // Update velocity + limit speed
            this.vx += ax;
            this.vy += ay;
            const v = limit(this.vx, this.vy, maxSpeed);
            this.vx = v.x; this.vy = v.y;

            // Move
            this.x += this.vx;
            this.y += this.vy;

            // Wrap
            if (this.x < 0) this.x += canvas.clientWidth;
            if (this.x > canvas.clientWidth) this.x -= canvas.clientWidth;
            if (this.y < 0) this.y += canvas.clientHeight;
            if (this.y > canvas.clientHeight) this.y -= canvas.clientHeight;

            // Standard pulsation
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

        particles.push(p);
    }


                } else if (pattern === 'rainwaves') {
            // Particle Rain + Gravity Waves
            // - particles fall down
            // - multiple moving wave fields push them sideways and slightly up/down
            // - connections are handled by your global spatial-grid logic (no override needed)

            const numParticles = 1000;                 // raise/lower for performance
            const margin = 20;

            // Wave field parameters (tuned for subtle, classy motion)
            const wave1 = { amp: 28, kx: 0.010, ky: 0.006, speed: 0.85 };
            const wave2 = { amp: 20, kx: 0.016, ky: -0.004, speed: 1.30 };
            const wave3 = { amp: 16,  kx: -0.008, ky: 0.012, speed: 0.55 };

            // Rain parameters
            const baseFallSpeed = 22;                 // px/sec baseline
            const fallSpeedJitter = 26;               // extra per-particle
            const driftDamping = 0.92;                // smooth velocity

            for (let i = 0; i < numParticles; i++) {
                const x0 = Math.random() * canvas.clientWidth;
                const y0 = Math.random() * canvas.clientHeight;

                const p = new Particle(
                    x0, y0,
                    0, 0,
                    x0, y0,
                    null,
                    {
                        ...settings,
                        lineOpacity: settings.lineOpacity ?? 0.18,
                        lineWidth: settings.lineWidth ?? 0.45,
                        fadeInSpeed: settings.fadeInSpeed ?? 0.001,
                        fadeOutSpeed: settings.fadeOutSpeed ?? 0.001
                    }
                );

                // Per-particle motion state
                p.baseX = x0;
                p.baseY = y0;

                // Give each particle its own fall speed + slight sideways bias
                p.fallSpeed = baseFallSpeed + Math.random() * fallSpeedJitter;
                p.sideBias = (Math.random() - 0.5) * 0.5;   // tiny consistent drift
                p.seed = Math.random() * Math.PI * 2;

                // Velocity (for smoothness)
                p.vx = 0;
                p.vy = 0;

                p.maxRadius = 1.6 + Math.random() * 1.2;  // was ~0.6–1.4 in default

                p.update = function() {
                    const t = Date.now() * 0.001;

                    // --- continuous falling ---
                    this.baseY += this.fallSpeed * (1 / 60); // frame-rate-ish step (stable enough)
                    // Wrap to top when off bottom
                    if (this.baseY > canvas.clientHeight + margin) {
                        this.baseY = -margin;
                        this.baseX = Math.random() * canvas.clientWidth;
                    }


                    if (this.baseY > canvas.clientHeight + margin) {
                      const spawnBand = canvas.clientHeight * 0.25; // <-- bigger band (25% of height)
                      this.baseY = -margin - Math.random() * spawnBand;
                      this.baseX = Math.random() * canvas.clientWidth;

                      // prevent a “teleport velocity spike”
                      this.x = this.baseX;
                      this.y = this.baseY;
                      this.vx = 0;
                      this.vy = 0;
                    }

                    // --- gravity waves (flow field) ---
                    // Combine several traveling wave fields for richer motion
                    const w1 = Math.sin(this.baseX * wave1.kx + this.baseY * wave1.ky + t * wave1.speed + this.seed) * wave1.amp;
                    const w2 = Math.sin(this.baseX * wave2.kx + this.baseY * wave2.ky + t * wave2.speed + this.seed * 1.7) * wave2.amp;
                    const w3 = Math.cos(this.baseX * wave3.kx + this.baseY * wave3.ky + t * wave3.speed + this.seed * 2.3) * wave3.amp;

                    // Horizontal flow + a subtle vertical wobble (like pockets of air)
                    const flowX = (w1 + w2 + w3);
                    const flowY = (0.25 * w2 - 0.15 * w3);

                    // Target position from flow
                    const targetX = this.baseX + flowX;
                    const targetY = this.baseY + flowY;

                    // Smoothly move toward target (avoids jitter)
                    this.vx = this.vx * driftDamping + (targetX - this.x) * 0.08 + this.sideBias;
                    this.vy = this.vy * driftDamping + (targetY - this.y) * 0.08;

                    this.x += this.vx;
                    this.y += this.vy;

                    // Wrap horizontally too (keep density consistent)
                    if (this.x < -margin) this.x = canvas.clientWidth + margin;
                    if (this.x > canvas.clientWidth + margin) this.x = -margin;

                    // --- your standard pulsation logic ---
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

                p.draw = function() {
                    if (this.opacity <= 0) return;

                    // Boost brightness slightly
                    const brightnessBoost = 2.0; // try 1.2–1.8
                    const alpha = Math.min(1, this.opacity * brightnessBoost);
                    // ctx.shadowBlur = 6;
                    // ctx.shadowColor = "rgba(255,255,255,0.6)";
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
                    ctx.fill();
                    // ctx.shadowBlur = 0;
                };

                particles.push(p);
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
                const connectionDistance = 35;
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

            // // Get pattern from canvas class
            // const pattern = Array.from(canvas.classList).find(className =>
            //     ['asteroids', 'donut', 'fibonacci', 'gravitydispersion', 'power', 'wave'].includes(className)
            // ) || 'fibonacci'; // Changed default to fibonacci

            const pattern = Array.from(canvas.classList).find(className =>
                ['asteroids', 'donut', 'fibonacci', 'gravitydispersion', 'power', 'wave',
                    'helix', 'rainwaves', 'blackhole', 'brainnet', 'magfield', 'flock'].includes(className)
            ) || 'fibonacci';

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
                helix: {
                    orbitSpeed: 0.002,        // used as base speed hint (we override update anyway)
                    lineOpacity: 0.35,
                    fadeInSpeed: 0.012,
                    fadeOutSpeed: 0.008,
                    lineWidth: 0.55
                },
                rainwaves: {
                      lineOpacity: 0.6,
                      fadeInSpeed: 0.0012,
                      fadeOutSpeed: 0.008,
                      lineWidth: 0.55,
                    },
                blackhole: {
                      lineOpacity: 0.6,
                      fadeInSpeed: 0.02,
                      fadeOutSpeed: 0.01,
                      lineWidth: 0.55,
                    },
                brainnet: {
                  lineOpacity: 0.28,
                  fadeInSpeed: 0.018,
                  fadeOutSpeed: 0.010,
                  lineWidth: 0.55,
                },
                magfield: {
                  lineOpacity: 0.24,
                  fadeInSpeed: 0.018,
                  fadeOutSpeed: 0.010,
                  lineWidth: 0.50,
                },
                flock: {
                  lineOpacity: 0.20,
                  fadeInSpeed: 0.016,
                  fadeOutSpeed: 0.010,
                  lineWidth: 0.50,
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

                    // const width = windowr.innerWidth;
                // const height = window.innerHeight;
