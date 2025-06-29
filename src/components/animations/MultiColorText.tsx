import React from 'react';
import { motion } from 'framer-motion';

interface MultiColorTextProps {
  text: string;
  className?: string;
  colors?: string[];
  animationDuration?: number;
  animationDelay?: number;
  staggerChildren?: number;
}

const MultiColorText: React.FC<MultiColorTextProps> = ({
  text,
  className = '',
  colors = ['#8B5CF6', '#EC4899', '#3B82F6', '#EF4444', '#10B981'],
  animationDuration = 3,
  animationDelay = 0,
  staggerChildren = 0.1
}) => {
  const words = text.split(' ');
  
  const container = {
    hidden: { opacity: 0 },
    visible: (i = 1) => ({
      opacity: 1,
      transition: {
        staggerChildren,
        delayChildren: animationDelay * i
      }
    })
  };
  
  const child = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring" as const,
        damping: 12,
        stiffness: 100
      }
    }
  };

  return (
    <motion.div
      className={`inline-flex flex-wrap justify-center ${className}`}
      variants={container}
      initial="hidden"
      animate="visible"
    >
      {words.map((word, index) => (
        <motion.span
          key={index}
          variants={child}
          className="inline-block mr-2 last:mr-0"
          style={{
            background: `linear-gradient(to right, ${colors[index % colors.length]}, ${colors[(index + 1) % colors.length]})`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundSize: '200% 200%',
            animation: `gradient-shift ${animationDuration}s ease infinite alternate ${index * 0.2}s`
          }}
        >
          {word}
        </motion.span>
      ))}
    </motion.div>
  );
};

export default MultiColorText;