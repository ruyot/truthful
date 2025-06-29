import React from 'react'

interface AnimatedListProps {
  items: React.ReactNode[]
  className?: string
  staggerDelay?: number
}

export const AnimatedList: React.FC<AnimatedListProps> = ({ 
  items, 
  className = '', 
  staggerDelay = 100 
}) => {
  return (
    <div className={`space-y-4 ${className}`}>
      {items.map((item, index) => (
        <div
          key={index}
          className="opacity-0 translate-y-4 animate-fade-in-up"
          style={{
            animationDelay: `${index * staggerDelay}ms`,
            animationFillMode: 'forwards'
          }}
        >
          {item}
        </div>
      ))}
    </div>
  )
}