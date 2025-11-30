"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Maximize, 
  Minimize, 
  RotateCcw, 
  Trash2, 
  Plus, 
  Search, 
  Info, 
  X, 
  Settings2,
  Share2,
  MousePointer2,
  Move,
  ZoomIn
} from "lucide-react";
import { cn } from "@/lib/utils";

// --- Types ---

interface WordNode {
  id: string;
  text: string;
  embedding: number[];
  x: number;
  y: number;
  color: string;
  connections: string[];
}

interface Connection {
  from: string;
  to: string;
  similarity: number;
}

interface ViewState {
  x: number;
  y: number;
  scale: number;
}

// --- Constants ---

const COLORS = [
  "#C084FC", // Purple
  "#22D3EE", // Cyan
  "#34D399", // Emerald
  "#FBBF24", // Amber
  "#F87171", // Red
  "#F472B6", // Pink
  "#818CF8", // Indigo
  "#A3E635", // Lime
  "#FB923C", // Orange
  "#2DD4BF", // Teal
  "#60A5FA", // Blue
  "#A78BFA", // Violet
];

const THEME = {
  gridColor: "rgba(255, 255, 255, 0.03)",
  nodeBaseSize: 24,
  nodeHoverScale: 1.2,
};

// --- Components ---

export default function EmbeddingPage() {
  // -- State --
  const [inputText, setInputText] = useState("");
  const [words, setWords] = useState<WordNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [viewState, setViewState] = useState<ViewState>({ x: 0, y: 0, scale: 1 });
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<string | null>(null);
  
  // UI State
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [error, setError] = useState("");
  const [showHelp, setShowHelp] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Interaction State
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const pipelineRef = useRef<any>(null);

  // -- Initialization --

  useEffect(() => {
    if (containerRef.current) {
      const { clientWidth, clientHeight } = containerRef.current;
      setViewState({
        x: clientWidth / 2,
        y: clientHeight / 2,
        scale: 1,
      });
    }
  }, []);

  // -- Logic --

  const getEmbedding = async (text: string): Promise<number[]> => {
    const { pipeline } = await import("@huggingface/transformers");

    if (!pipelineRef.current) {
      setInitializing(true);
      pipelineRef.current = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        { dtype: "q8", device: "wasm" }
      );
      setInitializing(false);
    }

    const output = await pipelineRef.current(text, {
      pooling: "mean",
      normalize: true,
    });

    return Array.from(output.data);
  };

  const cosineSimilarity = (a: number[], b: number[]) => {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  };

  const generatePosition = useCallback((embedding: number[], existingWords: WordNode[]) => {
    if (existingWords.length === 0) return { x: 0, y: 0 };

    const similarities = existingWords.map((word) => ({
      word,
      similarity: cosineSimilarity(embedding, word.embedding),
    })).sort((a, b) => b.similarity - a.similarity);

    const mostSimilar = similarities[0];
    
    // Create a semantic map where distance roughly correlates to dissimilarity
    if (mostSimilar.similarity > 0.15) {
      const normalizedSim = (mostSimilar.similarity - 0.15) / 0.85;
      // Closer similarity = smaller distance. 
      // We use a non-linear scale to separate clusters better
      const distance = 400 * (1 - Math.pow(normalizedSim, 1.5)) + 60; 
      
      // Add randomness to angle to prevent lines
      const angle = Math.random() * Math.PI * 2;
      
      return {
        x: mostSimilar.word.x + Math.cos(angle) * distance,
        y: mostSimilar.word.y + Math.sin(angle) * distance,
      };
    }

    // If not similar to anything, push it far away in a random direction
    const spread = Math.max(500, Math.sqrt(existingWords.length) * 300);
    const angle = Math.random() * Math.PI * 2;
    return {
      x: Math.cos(angle) * spread,
      y: Math.sin(angle) * spread,
    };
  }, []);

  const addWord = async (text: string) => {
    if (!text.trim()) return;
    if (words.some(w => w.text.toLowerCase() === text.toLowerCase())) {
      setInputText(""); 
      return;
    }

    setLoading(true);
    setError("");

    try {
      const embedding = await getEmbedding(text);
      const position = generatePosition(embedding, words);
      
      // Assign a color based on cluster/position or just cycle for now
      // Ideally we'd do K-means but simple cycling works for visual distinction
      const color = COLORS[words.length % COLORS.length];

      const newWord: WordNode = {
        id: Date.now().toString(),
        text,
        embedding,
        x: position.x,
        y: position.y,
        color,
        connections: [],
      };

      const newConnections: Connection[] = [];
      
      words.forEach(existing => {
        const sim = cosineSimilarity(embedding, existing.embedding);
        if (sim > 0.25) { // Connection threshold
          newConnections.push({ from: existing.id, to: newWord.id, similarity: sim });
          existing.connections.push(newWord.id);
          newWord.connections.push(existing.id);
        }
      });

      setWords(prev => [...prev, newWord]);
      setConnections(prev => [...prev, ...newConnections]);
      setInputText("");
      
      if (words.length === 0) {
         // Wait for render then center
         setTimeout(() => resetView(), 50);
      }
      
    } catch (err) {
      console.error(err);
      setError("Failed to generate embedding. Check console.");
    } finally {
      setLoading(false);
    }
  };

  // -- Interaction Handlers --

  const handleMouseDown = (e: React.MouseEvent) => {
    // Ignore clicks on UI elements (bubbled up)
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('input')) return;
    
    setIsDragging(true);
    setDragStart({ x: e.clientX - viewState.x, y: e.clientY - viewState.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setViewState(prev => ({
        ...prev,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      }));
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(4, viewState.scale * delta));
    
    // Zoom towards mouse pointer could be implemented here, 
    // but center zoom is easier for now
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    // Simple center zoom correction
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    setViewState(prev => ({
        x: centerX - (centerX - prev.x) * (newScale / prev.scale),
        y: centerY - (centerY - prev.y) * (newScale / prev.scale),
        scale: newScale
    }));
  };

  const resetView = () => {
    if (!containerRef.current) return;
    
    const { clientWidth, clientHeight } = containerRef.current;
    
    if (words.length === 0) {
      setViewState({ x: clientWidth / 2, y: clientHeight / 2, scale: 1 });
      return;
    }

    // Calculate bounding box of all words
    const xs = words.map(w => w.x);
    const ys = words.map(w => w.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    setViewState({
      x: clientWidth / 2 - centerX,
      y: clientHeight / 2 - centerY,
      scale: 0.8, // Start slightly zoomed out to see context
    });
  };

  const clearAll = () => {
    setWords([]);
    setConnections([]);
    setHoveredWord(null);
    resetView();
  };

  // -- Rendering Helpers --

  // Filter connections for performance and clarity
  const visibleConnections = React.useMemo(() => {
    if (hoveredWord) {
      return connections.filter(c => c.from === hoveredWord || c.to === hoveredWord);
    }
    // Show only stronger connections by default to reduce noise
    return connections.filter(c => c.similarity > 0.4); 
  }, [connections, hoveredWord]);

  return (
    <div className="relative w-full h-screen bg-black text-foreground overflow-hidden font-sans">
      
      {/* --- Canvas Layer --- */}
      <div 
        ref={containerRef}
        className="absolute inset-0 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onWheel={handleWheel}
      >
        <svg width="100%" height="100%" className="w-full h-full block touch-none">
          <defs>
            <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse"
              patternTransform={`scale(${viewState.scale}) translate(${viewState.x/viewState.scale} ${viewState.y/viewState.scale})`}>
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke={THEME.gridColor} strokeWidth="1" />
            </pattern>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <rect width="100%" height="100%" fill="url(#grid)" />

          <g transform={`translate(${viewState.x}, ${viewState.y}) scale(${viewState.scale})`}>
            
            {/* Connections */}
            <AnimatePresence>
              {visibleConnections.map((conn) => {
                 const fromNode = words.find(w => w.id === conn.from);
                 const toNode = words.find(w => w.id === conn.to);
                 if (!fromNode || !toNode) return null;
                 
                 const isHovered = hoveredConnection === `${conn.from}-${conn.to}` || 
                                   hoveredWord === conn.from || 
                                   hoveredWord === conn.to;

                 return (
                   <motion.line
                     key={`${conn.from}-${conn.to}`}
                     initial={{ opacity: 0 }}
                     animate={{ 
                       opacity: isHovered ? 1 : Math.max(0.1, conn.similarity - 0.2),
                       strokeWidth: isHovered ? 2 : Math.max(0.5, conn.similarity * 2)
                     }}
                     exit={{ opacity: 0 }}
                     x1={fromNode.x} y1={fromNode.y}
                     x2={toNode.x} y2={toNode.y}
                     stroke={isHovered ? "#fff" : fromNode.color}
                     strokeLinecap="round"
                   />
                 );
              })}
            </AnimatePresence>

            {/* Nodes */}
            <AnimatePresence>
              {words.map((word) => {
                const isHovered = hoveredWord === word.id;
                const isConnected = hoveredWord && word.connections.includes(hoveredWord);
                
                // Dim others if something is hovered, but not if it's connected
                const isDimmed = hoveredWord && !isHovered && !isConnected;
                
                return (
                  <motion.g
                    key={word.id}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: isDimmed ? 0.3 : 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredWord(word.id)}
                    onMouseLeave={() => setHoveredWord(null)}
                  >
                    {/* Interaction Area (invisible but larger) */}
                    <circle cx={word.x} cy={word.y} r={THEME.nodeBaseSize * 1.5} fill="transparent" />
                    
                    {/* Glow (only when hovered) */}
                    {isHovered && (
                      <circle 
                        cx={word.x} cy={word.y} 
                        r={THEME.nodeBaseSize * 1.2} 
                        fill={word.color} 
                        fillOpacity="0.3" 
                        filter="url(#glow)"
                      />
                    )}

                    {/* Core Node */}
                    <circle 
                      cx={word.x} cy={word.y} 
                      r={isHovered ? THEME.nodeBaseSize * 0.8 : THEME.nodeBaseSize * 0.6} 
                      fill="#09090b"
                      stroke={word.color}
                      strokeWidth={isHovered ? 3 : 2}
                    />
                    
                    {/* Text Label */}
                    <text
                      x={word.x} 
                      y={word.y + THEME.nodeBaseSize + 12}
                      textAnchor="middle"
                      fill={isHovered ? "#fff" : "#a1a1aa"}
                      fontSize={isHovered ? 14 : 12}
                      fontWeight={isHovered ? 600 : 400}
                      className="pointer-events-none select-none font-mono"
                      style={{ textShadow: "0 2px 4px rgba(0,0,0,0.8)" }}
                    >
                      {word.text}
                    </text>
                  </motion.g>
                );
              })}
            </AnimatePresence>

          </g>
        </svg>
      </div>

      {/* --- HUD Layer --- */}
      
      {/* Top Bar */}
      <header className="absolute top-0 left-0 right-0 p-4 flex justify-between items-start pointer-events-none">
        <div className="pointer-events-auto">
           <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-zinc-500 bg-clip-text text-transparent tracking-tight">
             Embedding Universe
           </h1>
           <div className="flex items-center gap-2 text-xs text-zinc-500 mt-1">
             <span className={cn("w-2 h-2 rounded-full", initializing ? "bg-amber-500 animate-pulse" : "bg-emerald-500")} />
             {initializing ? "Initializing Neural Engine..." : "Neural Engine Ready"}
           </div>
        </div>
        
        <div className="flex gap-2 pointer-events-auto">
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white"
          >
            {sidebarOpen ? <Minimize size={18} /> : <Maximize size={18} />}
          </button>
          <button 
            onClick={() => setShowHelp(!showHelp)}
            className={cn(
              "p-2 rounded-lg border transition-colors",
              showHelp ? "bg-white text-black border-white" : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:text-white"
            )}
          >
            <Info size={18} />
          </button>
        </div>
      </header>

      {/* Sidebar / Controls */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="absolute top-24 left-4 w-80 glass-panel rounded-xl flex flex-col overflow-hidden shadow-2xl pointer-events-auto"
          >
            {/* Input Section */}
            <div className="p-4 border-b border-zinc-800/50 space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" size={16} />
                <input
                  autoFocus
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addWord(inputText)}
                  placeholder="Add a concept..."
                  className="w-full bg-zinc-900/50 border border-zinc-700/50 rounded-lg py-2.5 pl-9 pr-3 text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
                />
              </div>
              <button
                onClick={() => addWord(inputText)}
                disabled={loading || !inputText.trim()}
                className="w-full py-2 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <span className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                ) : (
                  <>
                    <Plus size={16} /> Add Node
                  </>
                )}
              </button>
            </div>

            {/* Stats / Info */}
            <div className="p-4 space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-zinc-900/30 p-3 rounded-lg border border-zinc-800/50">
                   <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Nodes</div>
                   <div className="text-xl font-mono text-white">{words.length}</div>
                </div>
                <div className="bg-zinc-900/30 p-3 rounded-lg border border-zinc-800/50">
                   <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Links</div>
                   <div className="text-xl font-mono text-white">{connections.length}</div>
                </div>
              </div>

              {hoveredWord && (
                 <div className="bg-primary/10 border border-primary/20 p-3 rounded-lg">
                   <div className="text-primary text-xs uppercase tracking-wider mb-1">Selected</div>
                   <div className="font-medium text-white flex items-center gap-2">
                     <span className="w-2 h-2 rounded-full bg-primary" />
                     {words.find(w => w.id === hoveredWord)?.text}
                   </div>
                 </div>
              )}
            </div>

            {/* Actions */}
            <div className="p-4 border-t border-zinc-800/50 flex gap-2">
              <button 
                onClick={resetView}
                className="flex-1 py-2 px-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-xs text-zinc-300 transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw size={14} /> Reset View
              </button>
              <button 
                onClick={clearAll}
                className="flex-1 py-2 px-3 bg-red-950/30 hover:bg-red-900/50 border border-red-900/30 rounded-lg text-xs text-red-400 transition-colors flex items-center justify-center gap-2"
              >
                <Trash2 size={14} /> Clear
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Controls (Bottom Right) */}
      <div className="absolute bottom-6 right-6 flex flex-col gap-2 pointer-events-auto">
        <button onClick={() => setViewState(v => ({...v, scale: v.scale * 1.2}))} className="p-3 bg-zinc-900 border border-zinc-800 rounded-full text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors shadow-lg">
          <ZoomIn size={20} />
        </button>
      </div>

      {/* Help Modal Overlay */}
      <AnimatePresence>
        {showHelp && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 pointer-events-auto"
            onClick={() => setShowHelp(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-zinc-950 border border-zinc-800 rounded-2xl max-w-lg w-full p-8 shadow-2xl relative overflow-hidden"
            >
              {/* Decorative gradient */}
              <div className="absolute -top-24 -right-24 w-48 h-48 bg-primary/20 blur-3xl rounded-full pointer-events-none" />
              
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-2xl font-bold text-white">Welcome to the Universe</h2>
                <button onClick={() => setShowHelp(false)} className="text-zinc-500 hover:text-white">
                  <X size={24} />
                </button>
              </div>
              
              <div className="space-y-6 text-zinc-400 leading-relaxed">
                <p>
                  This is a semantic visualization engine. It uses an AI model running <strong className="text-white">entirely in your browser</strong> to understand the meaning of words.
                </p>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-zinc-900/50 rounded-xl border border-zinc-800">
                    <MousePointer2 className="text-primary mb-2" size={20} />
                    <h3 className="text-white font-medium mb-1">Interactive</h3>
                    <p className="text-xs">Hover nodes to see connections. Drag canvas to explore.</p>
                  </div>
                  <div className="p-4 bg-zinc-900/50 rounded-xl border border-zinc-800">
                    <Move className="text-emerald-400 mb-2" size={20} />
                    <h3 className="text-white font-medium mb-1">Smart Layout</h3>
                    <p className="text-xs">Similar concepts naturally cluster together in space.</p>
                  </div>
                </div>

                <div className="text-sm border-t border-zinc-800 pt-6">
                  <p>Try adding words like <span className="text-white bg-zinc-800 px-1.5 py-0.5 rounded">King</span>, <span className="text-white bg-zinc-800 px-1.5 py-0.5 rounded">Queen</span>, and <span className="text-white bg-zinc-800 px-1.5 py-0.5 rounded">Throne</span> to see how the AI groups them.</p>
                </div>

                <button 
                  onClick={() => setShowHelp(false)}
                  className="w-full py-3 bg-white text-black rounded-xl font-bold hover:bg-zinc-200 transition-colors"
                >
                  Start Exploring
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}