import React from 'react';

interface AnimatedGradientTextProps {
  children: React.ReactNode;
  className?: string;
  gradient?: string;
  duration?: number;
}

const AnimatedGradientText: React.FC<AnimatedGradientTextProps> = ({
  children,
  className = '',
  gradient = 'from-purple-600 via-pink-600 to-blue-600',
  duration = 3
}) => {
  return (
    <span 
      className={`bg-gradient-to-r ${gradient} bg-clip-text text-transparent animate-gradient ${className}`}
      style={{ 
        backgroundSize: '300% 300%',
        animation: `gradient-animation ${duration}s ease infinite alternate`
      }}
    >
      {children}
    </span>
  );
};

export default AnimatedGradientText;