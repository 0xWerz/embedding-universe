"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import * as d3 from "d3";
import { 
  Maximize, 
  Minimize, 
  RotateCcw, 
  Trash2, 
  Plus, 
  Search, 
  Info, 
  X, 
  ZoomIn,
  Activity,
  Cpu,
  Box
} from "lucide-react";
import { cn } from "@/lib/utils";

// --- Types ---

interface WordNode extends d3.SimulationNodeDatum {
  id: string;
  text: string;
  embedding: number[];
  color: string;
  x?: number;
  y?: number;
}

interface Connection extends d3.SimulationLinkDatum<WordNode> {
  source: string | WordNode;
  target: string | WordNode;
  similarity: number;
  id: string; // source-target
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
};

const SIMULATION_CONFIG = {
  chargeStrength: -300,
  linkDistanceBase: 100,
  linkStrengthBase: 0.5,
  collideRadius: 35,
  alphaDecay: 0.02, // Slower decay = longer movement
};

// --- Components ---

export default function EmbeddingPage() {
  // -- State --
  const [inputText, setInputText] = useState("");
  const [viewState, setViewState] = useState<ViewState>({ x: 0, y: 0, scale: 1 });
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<string | null>(null);
  const [is3DMode, setIs3DMode] = useState(false);
  
  // Render State (Sync with D3)
  const [nodes, setNodes] = useState<WordNode[]>([]);
  const [links, setLinks] = useState<Connection[]>([]);

  // UI State
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [error, setError] = useState("");
  const [showHelp, setShowHelp] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [physicsActive, setPhysicsActive] = useState(false);

  // Interaction State
  const [isDraggingCanvas, setIsDraggingCanvas] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const pipelineRef = useRef<any>(null);
  
  // D3 Simulation Refs
  const simulationRef = useRef<d3.Simulation<WordNode, Connection> | null>(null);
  // We keep mutable refs for D3 to operate on, then sync to state for React render
  const nodesRef = useRef<WordNode[]>([]);
  const linksRef = useRef<Connection[]>([]);

  // -- Initialization --

  useEffect(() => {
    if (containerRef.current) {
      const { clientWidth, clientHeight } = containerRef.current;
      setViewState({
        x: clientWidth / 2,
        y: clientHeight / 2,
        scale: 1,
      });

      // Initialize Simulation
      simulationRef.current = d3.forceSimulation<WordNode, Connection>()
        .force("charge", d3.forceManyBody().strength(SIMULATION_CONFIG.chargeStrength))
        .force("collide", d3.forceCollide().radius(SIMULATION_CONFIG.collideRadius))
        .force("center", d3.forceCenter(0, 0).strength(0.05)) // Soft centering
        .force("link", d3.forceLink<WordNode, Connection>()
          .id(d => d.id)
          .distance(link => SIMULATION_CONFIG.linkDistanceBase * (1.5 - link.similarity)) // Higher sim = shorter distance
          .strength(link => SIMULATION_CONFIG.linkStrengthBase * link.similarity) // Higher sim = stronger pull
        )
        .alphaDecay(SIMULATION_CONFIG.alphaDecay)
        .on("tick", () => {
          // Optimization: throttle state updates if needed, but for < 100 nodes, 60fps React render is usually fine
          // We spread to trigger re-render
          setNodes([...nodesRef.current]);
          setLinks([...linksRef.current]);
        })
        .on("end", () => setPhysicsActive(false));
    }

    return () => {
      simulationRef.current?.stop();
    };
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
    // Normalized vectors, so dot product is cosine similarity
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  };

  const addWord = async (text: string) => {
    if (!text.trim()) return;
    if (nodesRef.current.some(w => w.text.toLowerCase() === text.toLowerCase())) {
      setInputText(""); 
      return;
    }

    setLoading(true);
    setError("");
    setPhysicsActive(true);

    try {
      const embedding = await getEmbedding(text);
      const color = COLORS[nodesRef.current.length % COLORS.length];

      // Initial position: Random nearby or center
      const initialX = (Math.random() - 0.5) * 50;
      const initialY = (Math.random() - 0.5) * 50;

      const newWord: WordNode = {
        id: Date.now().toString(),
        text,
        embedding,
        color,
        x: initialX,
        y: initialY,
      };

      // --- KNN & Cluster Topology Strategy ---
      // 1. Calculate ALL similarities first
      const candidates = nodesRef.current.map(existing => ({
        id: existing.id,
        similarity: cosineSimilarity(embedding, existing.embedding)
      }));

      // 2. Sort by similarity (strongest first)
      candidates.sort((a, b) => b.similarity - a.similarity);

      // 3. Determine connections:
      //    - Always connect to Top 3 (K=3) nearest neighbors to ensure graph connectivity
      //    - Also connect to any other node with Similarity > 0.45 (Cluster density)
      //    - Cap at max 8 connections per new node to prevent "super-hubs" from cluttering
      const connectionsToMake = candidates.filter((c, index) => {
        const isTopK = index < 3;
        const isStrong = c.similarity > 0.45;
        return (isTopK || isStrong) && index < 8;
      });

      const newLinks: Connection[] = connectionsToMake.map(c => ({
        source: c.id,
        target: newWord.id,
        similarity: c.similarity,
        id: `${c.id}-${newWord.id}`
      }));

      // Update Mutable Refs
      nodesRef.current.push(newWord);
      linksRef.current.push(...newLinks);

      // Update Simulation
      if (simulationRef.current) {
        simulationRef.current.nodes(nodesRef.current);
        
        // Cast is necessary because d3 modifies link structure
        const forceLink = simulationRef.current.force("link") as d3.ForceLink<WordNode, Connection>;
        forceLink.links(linksRef.current);
        
        // Re-heat simulation to let new node settle
        simulationRef.current.alpha(1).restart();
      }

      setInputText("");
      
    } catch (err) {
      console.error(err);
      setError("Failed to generate embedding.");
    } finally {
      setLoading(false);
    }
  };

  // -- Interaction Handlers --

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('input')) return;
    
    setIsDraggingCanvas(true);
    setDragStart({ x: e.clientX - viewState.x, y: e.clientY - viewState.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingCanvas) {
      setViewState(prev => ({
        ...prev,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      }));
    }
  };

  // D3 Node Dragging Handlers
  const onNodeDragStart = (e: React.MouseEvent, node: WordNode) => {
    e.stopPropagation(); // Prevent canvas drag
    if (!simulationRef.current) return;
    simulationRef.current.alphaTarget(0.3).restart();
    node.fx = node.x;
    node.fy = node.y;
    setPhysicsActive(true);
  };

  const onNodeDrag = (e: React.MouseEvent, node: WordNode) => {
    e.stopPropagation();
    if (isDraggingCanvas) return; // Safety check

    // We need to transform mouse coordinates back to simulation space
    // Mouse (Screen) -> View State (Pan/Zoom) -> Simulation Space
    const mouseX = e.clientX;
    const mouseY = e.clientY;

    node.fx = (mouseX - viewState.x) / viewState.scale;
    node.fy = (mouseY - viewState.y) / viewState.scale;
  };

  const onNodeDragEnd = (e: React.MouseEvent, node: WordNode) => {
    e.stopPropagation();
    if (!simulationRef.current) return;
    simulationRef.current.alphaTarget(0);
    node.fx = null;
    node.fy = null;
  };

  const handleWheel = (e: React.WheelEvent) => {
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(4, viewState.scale * delta));
    
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

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
    setViewState({ x: clientWidth / 2, y: clientHeight / 2, scale: 0.8 });
  };

  const clearAll = () => {
    nodesRef.current = [];
    linksRef.current = [];
    if (simulationRef.current) {
      simulationRef.current.nodes([]);
      (simulationRef.current.force("link") as d3.ForceLink<WordNode, Connection>).links([]);
      simulationRef.current.restart();
    }
    setNodes([]);
    setLinks([]);
    setHoveredWord(null);
    resetView();
  };

  // -- Rendering Helpers --

  const visibleConnections = React.useMemo(() => {
    // Render prioritization
    if (hoveredWord) {
      return links.filter(c => 
        (typeof c.source === 'object' ? c.source.id : c.source) === hoveredWord || 
        (typeof c.target === 'object' ? c.target.id : c.target) === hoveredWord
      );
    }
    // Since we now use KNN topology, every link is meaningful. Show them all.
    return links; 
  }, [links, hoveredWord]);

  return (
    <div className="relative w-full h-screen bg-black text-foreground overflow-hidden font-sans select-none">
      
      {/* --- Canvas Layer --- */}
      <div 
        ref={containerRef}
        className="absolute inset-0 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setIsDraggingCanvas(false)}
        onMouseLeave={() => setIsDraggingCanvas(false)}
        onWheel={handleWheel}
      >
        <svg 
          ref={svgRef}
          width="100%" 
          height="100%" 
          className="w-full h-full block touch-none"
          style={{ 
            perspective: "1000px",
            overflow: "visible"
          }}
        >
          <defs>
            <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse"
              patternTransform={`scale(${viewState.scale}) translate(${viewState.x/viewState.scale} ${viewState.y/viewState.scale})`}>
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke={THEME.gridColor} strokeWidth="1" />
            </pattern>
            <radialGradient id="nodeGradient">
              <stop offset="0%" stopColor="#fff" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#000" stopOpacity="0" />
            </radialGradient>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <rect width="100%" height="100%" fill="url(#grid)" />

          <motion.g 
            initial={false}
            animate={{
              transform: is3DMode 
                ? `translate(${viewState.x}px, ${viewState.y}px) scale(${viewState.scale}) rotateX(45deg) rotateZ(-10deg)`
                : `translate(${viewState.x}px, ${viewState.y}px) scale(${viewState.scale})`
            }}
            transition={{ type: "spring", duration: 0.8, bounce: 0.2 }}
            style={{ transformStyle: "preserve-3d" }}
          >
            
            {/* Connections */}
            <AnimatePresence>
              {visibleConnections.map((conn) => {
                 // D3 replaces string ids with object references after simulation starts
                 const sourceNode = conn.source as WordNode;
                 const targetNode = conn.target as WordNode;
                 
                 if (typeof sourceNode.x !== 'number' || typeof targetNode.x !== 'number') return null;
                 
                 const isHovered = hoveredConnection === conn.id || 
                                   hoveredWord === sourceNode.id || 
                                   hoveredWord === targetNode.id;

                 return (
                   <motion.line
                     key={conn.id}
                     initial={{ opacity: 0 }}
                     animate={{ 
                       opacity: isHovered ? 0.8 : Math.max(0.1, conn.similarity - 0.2),
                       strokeWidth: isHovered ? 2 : Math.max(0.5, conn.similarity * 2)
                     }}
                     x1={sourceNode.x} y1={sourceNode.y}
                     x2={targetNode.x} y2={targetNode.y}
                     stroke={isHovered ? "#fff" : sourceNode.color}
                     strokeLinecap="round"
                   />
                 );
              })}
            </AnimatePresence>

            {/* Nodes */}
            {nodes.map((word) => {
              if (typeof word.x !== 'number' || typeof word.y !== 'number') return null;

              const isHovered = hoveredWord === word.id;
              // Check connections for highlight
              const isConnected = hoveredWord && links.some(l => 
                ((l.source as WordNode).id === word.id && (l.target as WordNode).id === hoveredWord) ||
                ((l.target as WordNode).id === word.id && (l.source as WordNode).id === hoveredWord)
              );
              
              const isDimmed = hoveredWord && !isHovered && !isConnected;
              
              return (
                <motion.g
                  key={word.id}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ 
                    scale: 1, 
                    opacity: isDimmed ? 0.2 : 1,
                    x: word.x, 
                    y: word.y
                  }}
                  transition={{ duration: 0 }} 
                  className="cursor-grab active:cursor-grabbing"
                  onMouseEnter={() => setHoveredWord(word.id)}
                  onMouseLeave={() => setHoveredWord(null)}
                  onMouseDown={(e) => onNodeDragStart(e, word)}
                  onMouseMove={(e) => {
                    if (word.fx !== undefined && word.fx !== null) onNodeDrag(e, word);
                  }}
                  onMouseUp={(e) => onNodeDragEnd(e, word)}
                >
                  {/* Interaction Area */}
                  <circle r={THEME.nodeBaseSize * 1.5} fill="transparent" />
                  
                  {/* Glow */}
                  {isHovered && (
                    <circle 
                      r={THEME.nodeBaseSize * 1.2} 
                      fill={word.color} 
                      fillOpacity="0.3" 
                      filter="url(#glow)"
                    />
                  )}

                  {/* Core Node */}
                  <circle 
                    r={isHovered ? THEME.nodeBaseSize * 0.7 : THEME.nodeBaseSize * 0.5} 
                    fill="#09090b"
                    stroke={word.color}
                    strokeWidth={isHovered ? 3 : 2}
                  />
                  
                  {/* Text Label */}
                  <g transform={is3DMode ? "rotateZ(10deg) rotateX(-45deg) translate(0, -10)" : ""}>
                    <text
                      y={is3DMode ? 0 : THEME.nodeBaseSize + 10}
                      textAnchor="middle"
                      fill={isHovered ? "#fff" : "#a1a1aa"}
                      fontSize={isHovered ? 14 : 12}
                      fontWeight={isHovered ? 600 : 400}
                      className="pointer-events-none select-none font-mono"
                      style={{ textShadow: "0 2px 4px rgba(0,0,0,0.8)" }}
                    >
                      {word.text}
                    </text>
                  </g>
                </motion.g>
              );
            })}
          </motion.g>
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
             {initializing ? (
               <>
                <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                <span>Initializing Neural Engine...</span>
               </>
             ) : (
               <>
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>Model: all-MiniLM-L6-v2 (384d)</span>
               </>
             )}
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
                   <div className="text-xl font-mono text-white">{nodes.length}</div>
                </div>
                <div className="bg-zinc-900/30 p-3 rounded-lg border border-zinc-800/50">
                   <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Links</div>
                   <div className="text-xl font-mono text-white">{links.length}</div>
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs text-zinc-500 bg-zinc-900/20 p-2 rounded border border-zinc-800/50">
                <Activity size={14} className={physicsActive ? "text-emerald-500" : "text-zinc-600"} />
                Physics Engine: {physicsActive ? "Active" : "Stable"}
              </div>

              {hoveredWord && (
                 <div className="bg-primary/10 border border-primary/20 p-3 rounded-lg">
                   <div className="text-primary text-xs uppercase tracking-wider mb-1">Selected</div>
                   <div className="font-medium text-white flex items-center gap-2">
                     <span className="w-2 h-2 rounded-full bg-primary" />
                     {nodes.find(w => w.id === hoveredWord)?.text}
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
        <button 
          onClick={() => setIs3DMode(!is3DMode)} 
          className={cn(
            "p-3 border rounded-full transition-all shadow-lg",
            is3DMode 
              ? "bg-primary text-white border-primary" 
              : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white hover:bg-zinc-800"
          )}
          title="Toggle 3D Mode"
        >
          <Box size={20} />
        </button>
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
                <h2 className="text-2xl font-bold text-white">High-Dimensional Projection</h2>
                <button onClick={() => setShowHelp(false)} className="text-zinc-500 hover:text-white">
                  <X size={24} />
                </button>
              </div>
              
              <div className="space-y-6 text-zinc-400 leading-relaxed">
                <p>
                  You are exploring a <strong className="text-white">384-dimensional semantic space</strong> projected onto your 2D screen.
                </p>
                
                <div className="space-y-3">
                  <div className="flex gap-3">
                    <div className="p-2 bg-zinc-900 rounded-lg h-fit"><Cpu size={18} className="text-indigo-400" /></div>
                    <div>
                      <h3 className="text-white font-medium text-sm">Force-Directed Physics</h3>
                      <p className="text-xs mt-1">Nodes push and pull each other based on semantic similarity. The final layout reveals hidden clusters and relationships in the data.</p>
                    </div>
                  </div>
                  
                  <div className="flex gap-3">
                    <div className="p-2 bg-zinc-900 rounded-lg h-fit"><Activity size={18} className="text-emerald-400" /></div>
                    <div>
                      <h3 className="text-white font-medium text-sm">Live Simulation</h3>
                      <p className="text-xs mt-1">The universe is alive. Adding a new concept shifts the gravity of the entire system as it finds its place among existing ideas.</p>
                    </div>
                  </div>
                </div>

                <div className="text-sm border-t border-zinc-800 pt-6">
                  <p>Try adding opposing concepts like <span className="text-white font-mono">Love</span> vs <span className="text-white font-mono">Hate</span>, or related groups like <span className="text-white font-mono">Cat, Dog, Pet</span>.</p>
                </div>

                <button 
                  onClick={() => setShowHelp(false)}
                  className="w-full py-3 bg-white text-black rounded-xl font-bold hover:bg-zinc-200 transition-colors"
                >
                  Enter the Simulation
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}