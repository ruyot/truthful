import { useEffect, useRef } from "react";
import './Particles.css';

interface ParticlesProps {
  particleColors?: string[];
  particleCount?: number;
  particleSpread?: number;
  speed?: number;
  particleBaseSize?: number;
  moveParticlesOnHover?: boolean;
  particleHoverFactor?: number;
  alphaParticles?: boolean;
  sizeRandomness?: number;
  disableRotation?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

const defaultColors = ["#8B5CF6", "#A78BFA", "#C4B5FD", "#EC4899", "#3B82F6", "#10B981"];

const hexToRgb = (hex: string) => {
  hex = hex.replace(/^#/, "");
  if (hex.length === 3) {
    hex = hex.split("").map((c) => c + c).join("");
  }
  const int = parseInt(hex, 16);
  const r = ((int >> 16) & 255) / 255;
  const g = ((int >> 8) & 255) / 255;
  const b = (int & 255) / 255;
  return [r, g, b];
};

const Particles: React.FC<ParticlesProps> = ({
  particleColors = defaultColors,
  particleCount = 100,
  particleSpread = 10,
  speed = 0.1,
  particleBaseSize = 80,
  moveParticlesOnHover = true,
  particleHoverFactor = 1,
  alphaParticles = true,
  sizeRandomness = 1,
  disableRotation = false,
  className = "",
  style = {}
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameId = useRef<number>();
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    
    if (!container || !canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // Create particles
    const particles: any[] = [];
    
    for (let i = 0; i < particleCount; i++) {
      const color = particleColors[Math.floor(Math.random() * particleColors.length)];
      const size = particleBaseSize * (0.5 + Math.random() * sizeRandomness);
      
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: size * 0.01,
        color: color,
        speedX: (Math.random() - 0.5) * speed * 0.2, // Reduced horizontal speed
        speedY: Math.random() * speed + speed * 0.5, // Increased and always positive for falling effect
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.01 * speed,
        opacity: Math.random() * 0.5 + 0.3 // Random opacity for depth effect
      });
    }
    
    // Handle mouse movement
    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouseRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
    };
    
    if (moveParticlesOnHover) {
      container.addEventListener('mousemove', handleMouseMove);
    }
    
    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      particles.forEach(particle => {
        // Update position - particles fall down slowly
        particle.x += particle.speedX;
        particle.y += particle.speedY;
        
        // Wrap around edges
        if (particle.x < -50) particle.x = canvas.width + 50;
        if (particle.x > canvas.width + 50) particle.x = -50;
        if (particle.y > canvas.height + 50) particle.y = -50; // Reset to top when falling below
        
        // Mouse interaction
        if (moveParticlesOnHover) {
          const dx = mouseRef.current.x - particle.x;
          const dy = mouseRef.current.y - particle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 100) {
            const angle = Math.atan2(dy, dx);
            const force = (100 - distance) / 100 * particleHoverFactor;
            
            particle.x -= Math.cos(angle) * force;
            particle.y -= Math.sin(angle) * force;
          }
        }
        
        // Update rotation
        if (!disableRotation) {
          particle.rotation += particle.rotationSpeed;
        }
        
        // Draw particle
        ctx.save();
        ctx.translate(particle.x, particle.y);
        ctx.rotate(particle.rotation);
        
        const rgbValues = hexToRgb(particle.color);
        const alpha = alphaParticles ? particle.opacity : 1;
        
        // Draw a more interesting particle shape - soft gradient circle
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, particle.size * particleSpread);
        gradient.addColorStop(0, `rgba(${rgbValues[0] * 255}, ${rgbValues[1] * 255}, ${rgbValues[2] * 255}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${rgbValues[0] * 255}, ${rgbValues[1] * 255}, ${rgbValues[2] * 255}, 0)`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, particle.size * particleSpread, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
      });
      
      animationFrameId.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (moveParticlesOnHover) {
        container.removeEventListener('mousemove', handleMouseMove);
      }
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [
    particleColors,
    particleCount,
    particleSpread,
    speed,
    particleBaseSize,
    moveParticlesOnHover,
    particleHoverFactor,
    alphaParticles,
    sizeRandomness,
    disableRotation
  ]);

  return (
    <div
      ref={containerRef}
      className={`particles-container ${className}`}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        overflow: "hidden",
        zIndex: -10,
        ...style
      }}
    >
      <canvas ref={canvasRef} className="particles-canvas" />
    </div>
  );
};

export default Particles;