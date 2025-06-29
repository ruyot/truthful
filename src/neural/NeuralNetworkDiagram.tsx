import React, { useState, useCallback } from 'react'
import { Brain, Zap, Eye, Target } from 'lucide-react'

interface NodeData {
  id: string
  label: string
  description: string
  type: 'input' | 'hidden' | 'output'
  x: number
  y: number
}

interface ConnectionData {
  from: string
  to: string
  weight: number
}

export const NeuralNetworkDiagram: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  const nodes: NodeData[] = [
    // Input layer
    { id: 'input1', label: 'Frame 1', description: 'First video frame input', type: 'input', x: 50, y: 100 },
    { id: 'input2', label: 'Frame 2', description: 'Second video frame input', type: 'input', x: 50, y: 200 },
    { id: 'input3', label: 'Frame 3', description: 'Third video frame input', type: 'input', x: 50, y: 300 },
    
    // Hidden layers
    { id: 'hidden1', label: 'CNN', description: 'Convolutional Neural Network feature extraction', type: 'hidden', x: 200, y: 150 },
    { id: 'hidden2', label: 'Attention', description: 'Multi-frame attention mechanism', type: 'hidden', x: 200, y: 250 },
    { id: 'hidden3', label: 'Fusion', description: 'Feature fusion and aggregation', type: 'hidden', x: 350, y: 200 },
    
    // Output layer
    { id: 'output1', label: 'AI Score', description: 'AI generation likelihood (0-100%)', type: 'output', x: 500, y: 200 }
  ]

  const connections: ConnectionData[] = [
    { from: 'input1', to: 'hidden1', weight: 0.8 },
    { from: 'input2', to: 'hidden1', weight: 0.9 },
    { from: 'input3', to: 'hidden1', weight: 0.7 },
    { from: 'input1', to: 'hidden2', weight: 0.6 },
    { from: 'input2', to: 'hidden2', weight: 0.8 },
    { from: 'input3', to: 'hidden2', weight: 0.9 },
    { from: 'hidden1', to: 'hidden3', weight: 0.9 },
    { from: 'hidden2', to: 'hidden3', weight: 0.8 },
    { from: 'hidden3', to: 'output1', weight: 0.95 }
  ]

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'input': return <Eye size={16} />
      case 'hidden': return <Brain size={16} />
      case 'output': return <Target size={16} />
      default: return <Zap size={16} />
    }
  }

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'input': return 'bg-blue-500'
      case 'hidden': return 'bg-purple-500'
      case 'output': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  const handleNodeClick = useCallback((node: NodeData) => {
    setSelectedNode(node)
  }, [])

  return (
    <div className="relative w-full h-96 bg-gradient-to-br from-black to-gray-900 rounded-2xl p-6 overflow-hidden">
      {/* Background grid */}
      <div className="absolute inset-0 opacity-10">
        <svg width="100%" height="100%">
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="white" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      {/* Neural network visualization */}
      <svg className="absolute inset-0 w-full h-full">
        {/* Connections */}
        {connections.map((conn, index) => {
          const fromNode = nodes.find(n => n.id === conn.from)
          const toNode = nodes.find(n => n.id === conn.to)
          if (!fromNode || !toNode) return null

          return (
            <line
              key={index}
              x1={fromNode.x + 20}
              y1={fromNode.y + 20}
              x2={toNode.x + 20}
              y2={toNode.y + 20}
              stroke={`rgba(139, 92, 246, ${conn.weight})`}
              strokeWidth={conn.weight * 3}
              className="animate-pulse"
              style={{ animationDelay: `${index * 0.2}s` }}
            />
          )
        })}

        {/* Animated data flow */}
        {connections.map((conn, index) => {
          const fromNode = nodes.find(n => n.id === conn.from)
          const toNode = nodes.find(n => n.id === conn.to)
          if (!fromNode || !toNode) return null

          return (
            <circle
              key={`flow-${index}`}
              r="3"
              fill="#8b5cf6"
              className="animate-pulse"
              style={{ animationDelay: `${index * 0.3}s` }}
            >
              <animateMotion
                dur="2s"
                repeatCount="indefinite"
                begin={`${index * 0.3}s`}
              >
                <mpath>
                  <path d={`M ${fromNode.x + 20} ${fromNode.y + 20} L ${toNode.x + 20} ${toNode.y + 20}`} />
                </mpath>
              </animateMotion>
            </circle>
          )
        })}
      </svg>

      {/* Nodes */}
      {nodes.map((node) => (
        <div
          key={node.id}
          className={`
            absolute w-10 h-10 rounded-full flex items-center justify-center text-white cursor-pointer
            transition-all duration-300 hover:scale-110 hover:shadow-lg
            ${getNodeColor(node.type)}
            ${hoveredNode === node.id ? 'scale-110 shadow-lg' : ''}
            ${selectedNode?.id === node.id ? 'ring-4 ring-white ring-opacity-50' : ''}
          `}
          style={{ left: node.x, top: node.y }}
          onClick={() => handleNodeClick(node)}
          onMouseEnter={() => setHoveredNode(node.id)}
          onMouseLeave={() => setHoveredNode(null)}
        >
          {getNodeIcon(node.type)}
        </div>
      ))}

      {/* Node labels */}
      {nodes.map((node) => (
        <div
          key={`label-${node.id}`}
          className="absolute text-xs text-white font-medium pointer-events-none"
          style={{ left: node.x - 10, top: node.y + 45 }}
        >
          {node.label}
        </div>
      ))}

      {/* Tooltip */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-80 text-white p-4 rounded-lg">
          <h4 className="font-semibold text-lg mb-2">{selectedNode.label}</h4>
          <p className="text-sm text-gray-300">{selectedNode.description}</p>
          <button
            onClick={() => setSelectedNode(null)}
            className="mt-2 text-xs text-purple-400 hover:text-purple-300"
          >
            Close
          </button>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-black bg-opacity-60 text-white p-3 rounded-lg text-xs">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span>Input Layer</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span>Hidden Layer</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Output Layer</span>
          </div>
        </div>
      </div>
    </div>
  )
}