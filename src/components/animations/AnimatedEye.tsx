import React, { useEffect, useRef } from 'react';

interface AnimatedEyeProps {
  size?: number;
  color?: string;
  blinkInterval?: number;
  className?: string;
}

const AnimatedEye: React.FC<AnimatedEyeProps> = ({
  size = 32,
  color = '#8B5CF6',
  blinkInterval = 5000,
  className = ''
}) => {
  const eyeRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    const eye = eyeRef.current;
    if (!eye) return;
    
    // Set up blinking animation
    const blink = () => {
      const eyelid = eye.querySelector('.eyelid') as SVGElement;
      if (!eyelid) return;
      
      // Blink animation
      eyelid.style.transform = 'scaleY(1)';
      
      // Open eye after a short delay
      setTimeout(() => {
        eyelid.style.transform = 'scaleY(0)';
      }, 150);
    };
    
    // Random blink interval
    const getRandomBlinkInterval = () => {
      return blinkInterval + Math.random() * 2000;
    };
    
    // Initial blink
    const initialTimeout = setTimeout(blink, 1000);
    
    // Set up recurring blinks
    let blinkTimeout: number;
    
    const scheduleNextBlink = () => {
      blinkTimeout = window.setTimeout(() => {
        blink();
        scheduleNextBlink();
      }, getRandomBlinkInterval());
    };
    
    scheduleNextBlink();
    
    // Track mouse movement for eye following
    const handleMouseMove = (e: MouseEvent) => {
      const pupil = eye.querySelector('.pupil') as SVGElement;
      const iris = eye.querySelector('.iris') as SVGElement;
      if (!pupil || !iris) return;
      
      // Get eye position
      const eyeRect = eye.getBoundingClientRect();
      const eyeCenterX = eyeRect.left + eyeRect.width / 2;
      const eyeCenterY = eyeRect.top + eyeRect.height / 2;
      
      // Calculate angle between mouse and eye center
      const angle = Math.atan2(e.clientY - eyeCenterY, e.clientX - eyeCenterX);
      
      // Calculate movement distance (limited to keep pupil inside eye)
      const distance = Math.min(size / 10, Math.hypot(e.clientX - eyeCenterX, e.clientY - eyeCenterY) / 10);
      
      // Calculate new position
      const moveX = Math.cos(angle) * distance;
      const moveY = Math.sin(angle) * distance;
      
      // Apply movement
      pupil.style.transform = `translate(${moveX}px, ${moveY}px)`;
      iris.style.transform = `translate(${moveX * 0.7}px, ${moveY * 0.7}px)`;
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      clearTimeout(initialTimeout);
      clearTimeout(blinkTimeout);
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [size, blinkInterval]);
  
  return (
    <svg 
      ref={eyeRef}
      width={size} 
      height={size} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round" 
      strokeLinejoin="round"
      className={`animated-eye ${className}`}
    >
      {/* Eye outline */}
      <circle cx="12" cy="12" r="10" fill="white" stroke={color} />
      
      {/* Iris */}
      <circle className="iris" cx="12" cy="12" r="4" fill={color} fillOpacity="0.3" />
      
      {/* Pupil */}
      <circle className="pupil" cx="12" cy="12" r="2" fill={color} />
      
      {/* Eyelid - now using a circle clip path for natural blinking */}
      <circle 
        className="eyelid" 
        cx="12" 
        cy="12" 
        r="10" 
        fill="white" 
        style={{
          transformOrigin: 'center',
          transform: 'scaleY(0)',
          transition: 'transform 0.15s ease-in-out'
        }}
      />
    </svg>
  );
};

export default AnimatedEye;