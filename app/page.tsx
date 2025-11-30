"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
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
  Box,
  Activity,
  Cpu,
  Globe
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
  // 3D Properties
  z: number; 
}

interface Connection extends d3.SimulationLinkDatum<WordNode> {
  source: string | WordNode;
  target: string | WordNode;
  similarity: number;
  id: string;
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
  nodeBaseSize: 20,
};

const SIMULATION_CONFIG = {
  chargeStrength: -200,
  linkDistanceBase: 120,
  linkStrengthBase: 0.4,
  collideRadius: 25,
  alphaDecay: 0.02,
};

// --- Math Helpers ---

const PROJECT_FL = 1200; // Increased focal length for flatter, less distorted 3D view

function project(
  x: number, 
  y: number, 
  z: number, 
  angleX: number, 
  angleY: number, 
  width: number, 
  height: number,
  zoom: number,
  panX: number,
  panY: number,
  is3D: boolean
) {
  const cx = width / 2;
  const cy = height / 2;

  if (!is3D) {
    // 2D Projection: Simple Pan/Zoom relative to center + offset
    return {
      x: cx + x * zoom + panX,
      y: cy + y * zoom + panY,
      scale: zoom,
      opacity: 1,
      zIndex: 0
    };
  }

  // 3D Projection (Volumetric)
  // 1. Rotate Y (Orbit horizontal)
  const cosY = Math.cos(angleY);
  const sinY = Math.sin(angleY);
  const x1 = x * cosY - z * sinY;
  const z1 = z * cosY + x * sinY;

  // 2. Rotate X (Tilt vertical)
  const cosX = Math.cos(angleX);
  const sinX = Math.sin(angleX);
  const y2 = y * cosX - z1 * sinX;
  const z2 = z1 * cosX + y * sinX;

  // 3. Perspective Projection
  // We ignore panX/panY in 3D mode to ensure rotation is always around the universe center
  const depth = PROJECT_FL - z2; 
  // Prevent division by zero or negative depth behind camera
  const safeDepth = Math.max(10, depth);
  const scale = (PROJECT_FL * zoom) / safeDepth;
  
  return {
    x: cx + x1 * scale,
    y: cy + y2 * scale,
    scale: scale,
    opacity: Math.min(1, Math.max(0.2, scale)), // Depth cueing (fog)
    zIndex: z2 
  };
}

// --- Components ---

export default function EmbeddingPage() {
  // -- State --
  const [inputText, setInputText] = useState("");
  const [is3D, setIs3D] = useState(false);
  
  // Simulation State
  const [nodes, setNodes] = useState<WordNode[]>([]);
  const [links, setLinks] = useState<Connection[]>([]);
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);

  // View State
  const [camera, setCamera] = useState({ 
    panX: 0, panY: 0, // 2D Pan
    zoom: 1, 
    angleX: 0, angleY: 0 // 3D Orbit
  });
  
  // UI State
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showHelp, setShowHelp] = useState(true);

  // Interaction
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const pipelineRef = useRef<any>(null);
  const simulationRef = useRef<d3.Simulation<WordNode, Connection> | null>(null);
  const nodesRef = useRef<WordNode[]>([]);
  const linksRef = useRef<Connection[]>([]);

  // -- Initialization --

  useEffect(() => {
    simulationRef.current = d3.forceSimulation<WordNode, Connection>()
      .force("charge", d3.forceManyBody().strength(SIMULATION_CONFIG.chargeStrength))
      .force("collide", d3.forceCollide().radius(SIMULATION_CONFIG.collideRadius))
      .force("center", d3.forceCenter(0, 0).strength(0.05))
      .force("link", d3.forceLink<WordNode, Connection>()
        .id(d => d.id)
        .distance(link => SIMULATION_CONFIG.linkDistanceBase * (1.5 - link.similarity))
        .strength(link => SIMULATION_CONFIG.linkStrengthBase * link.similarity)
      )
      .alphaDecay(SIMULATION_CONFIG.alphaDecay)
      .on("tick", () => {
        // Sync refs to state for render
        setNodes([...nodesRef.current]);
        setLinks([...linksRef.current]);
      });

    return () => simulationRef.current?.stop();
  }, []);

  // -- Logic --

  const getEmbedding = async (text: string): Promise<number[]> => {
    const { pipeline } = await import("@huggingface/transformers");
    if (!pipelineRef.current) {
      setInitializing(true);
      pipelineRef.current = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8", device: "wasm" });
      setInitializing(false);
    }
    const output = await pipelineRef.current(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  };

  const cosineSimilarity = (a: number[], b: number[]) => {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  };

  const addWord = async (text: string) => {
    if (!text.trim()) return;
    if (nodesRef.current.some(w => w.text.toLowerCase() === text.toLowerCase())) {
      setInputText(""); return;
    }

    setLoading(true);

    try {
      const embedding = await getEmbedding(text);
      const color = COLORS[nodesRef.current.length % COLORS.length];
      
      // 3D Depth Assignment (Random Volume)
      const z = (Math.random() - 0.5) * 600; 

      const newWord: WordNode = {
        id: Date.now().toString(),
        text,
        embedding,
        color,
        x: 0, y: 0, z,
      };

      // KNN Topology
      const candidates = nodesRef.current.map(existing => ({
        id: existing.id,
        similarity: cosineSimilarity(embedding, existing.embedding)
      }));
      candidates.sort((a, b) => b.similarity - a.similarity);

      const connectionsToMake = candidates.filter((c, index) => 
        (index < 3 || c.similarity > 0.45) && index < 6
      );

      const newLinks: Connection[] = connectionsToMake.map(c => ({
        source: c.id,
        target: newWord.id,
        similarity: c.similarity,
        id: `${c.id}-${newWord.id}`
      }));

      nodesRef.current.push(newWord);
      linksRef.current.push(...newLinks);
      
      if (simulationRef.current) {
        simulationRef.current.nodes(nodesRef.current);
        (simulationRef.current.force("link") as d3.ForceLink<WordNode, Connection>).links(linksRef.current);
        simulationRef.current.alpha(1).restart();
      }

      setInputText("");
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // -- Interaction --

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setCamera(prev => ({ ...prev, zoom: Math.max(0.2, Math.min(5, prev.zoom * delta)) }));
    };
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('input')) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;

    if (is3D) {
      // Orbit
      setCamera(prev => ({
        ...prev,
        angleY: prev.angleY + deltaX * 0.005,
        angleX: Math.max(-Math.PI/2, Math.min(Math.PI/2, prev.angleX - deltaY * 0.005))
      }));
    } else {
      // Pan
      setCamera(prev => ({
        ...prev,
        panX: prev.panX + deltaX,
        panY: prev.panY + deltaY
      }));
    }
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  // -- Rendering --

  const { width, height } = containerRef.current?.getBoundingClientRect() || { width: 1000, height: 800 };

  // Compute projected positions for all nodes
  // This is fast enough to do in render for < 200 nodes
  const renderedNodes = useMemo(() => {
    return nodes.map(node => {
      if (typeof node.x !== 'number' || typeof node.y !== 'number') return null;
      const proj = project(
        node.x, node.y, node.z,
        is3D ? camera.angleX : 0,
        is3D ? camera.angleY : 0,
        width, height,
        camera.zoom,
        is3D ? 0 : camera.panX, // No pan in 3D, just orbit
        is3D ? 0 : camera.panY,
        is3D
      );
      return { ...node, proj };
    }).filter(Boolean) as (WordNode & { proj: any })[];
  }, [nodes, camera, is3D, width, height]);

  // Sort for depth in 3D mode (Painter's algorithm)
  const sortedRenderedNodes = useMemo(() => {
    if (!is3D) return renderedNodes;
    return [...renderedNodes].sort((a, b) => a.proj.zIndex - b.proj.zIndex);
  }, [renderedNodes, is3D]);

  const renderedLinks = useMemo(() => {
    return links.map(link => {
      const source = sortedRenderedNodes.find(n => n.id === (link.source as WordNode).id || n.id === link.source);
      const target = sortedRenderedNodes.find(n => n.id === (link.target as WordNode).id || n.id === link.target);
      if (!source || !target) return null;
      return { ...link, source, target };
    }).filter(Boolean) as any[];
  }, [links, sortedRenderedNodes]);

  const handleReset = () => {
    setCamera({ panX: 0, panY: 0, zoom: 1, angleX: 0, angleY: 0 });
  };

  const clearAll = () => {
    nodesRef.current = [];
    linksRef.current = [];
    if (simulationRef.current) {
      simulationRef.current.nodes([]);
      simulationRef.current.restart();
    }
    setNodes([]);
    setLinks([]);
    handleReset();
  };

  return (
    <div className="relative w-full h-screen bg-black text-foreground overflow-hidden font-sans select-none">
      
      {/* --- Canvas --- */}
      <div 
        ref={containerRef}
        className={cn("absolute inset-0", isDragging ? "cursor-grabbing" : "cursor-grab")}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
      >
        <svg width="100%" height="100%" className="block">
          <defs>
             <radialGradient id="node-glow">
               <stop offset="0%" stopColor="white" stopOpacity="0.8" />
               <stop offset="100%" stopColor="transparent" stopOpacity="0" />
             </radialGradient>
          </defs>

          {/* Grid removed per user request */}

          {/* Links */}
          {renderedLinks.map((link, i) => {
            const isHovered = hoveredWord === link.source.id || hoveredWord === link.target.id;
            // Don't show non-hovered links in 3D if there are too many, to reduce clutter? No, show them.
            const opacity = Math.max(0.1, link.similarity - 0.2) * (is3D ? 0.6 : 1);
            
            return (
              <motion.line
                key={link.id}
                animate={{
                  x1: link.source.proj.x, y1: link.source.proj.y,
                  x2: link.target.proj.x, y2: link.target.proj.y,
                }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }}
                stroke={isHovered ? "#fff" : link.source.color}
                strokeWidth={isHovered ? 2 : 1}
                strokeOpacity={isHovered ? 0.8 : opacity}
              />
            );
          })}

          {/* Nodes */}
          {sortedRenderedNodes.map(node => {
            const isHovered = hoveredWord === node.id;
            const baseSize = THEME.nodeBaseSize * node.proj.scale;
            
            return (
              <motion.g
                key={node.id}
                animate={{
                  x: node.proj.x,
                  y: node.proj.y,
                }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }} // Smooths the 2D->3D transition
                onMouseEnter={() => setHoveredWord(node.id)}
                onMouseLeave={() => setHoveredWord(null)}
                className="cursor-pointer"
              >
                {/* Glow */}
                {isHovered && (
                   <circle r={baseSize * 1.5} fill="url(#node-glow)" />
                )}
                
                {/* Core */}
                <motion.circle 
                  animate={{ r: isHovered ? baseSize * 0.6 : baseSize * 0.4 }}
                  fill="#000"
                  stroke={node.color}
                  strokeWidth={2 * node.proj.scale}
                />

                {/* Label */}
                <motion.text
                  animate={{ y: baseSize + 10 * node.proj.scale }}
                  textAnchor="middle"
                  fill={isHovered ? "#fff" : "#888"}
                  fontSize={(isHovered ? 14 : 10) * node.proj.scale}
                  fontWeight={isHovered ? 600 : 400}
                  style={{ textShadow: "0 2px 4px rgba(0,0,0,1)" }}
                >
                  {node.text}
                </motion.text>
              </motion.g>
            );
          })}
        </svg>
      </div>

      {/* --- HUD --- */}
      
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-start pointer-events-none">
        <div className="pointer-events-auto">
           <h1 className="text-2xl font-bold text-white tracking-tight">
             Embedding Universe
           </h1>
           <div className="flex items-center gap-2 text-xs text-zinc-500 mt-1">
             <span className={cn("w-2 h-2 rounded-full", initializing ? "bg-amber-500 animate-pulse" : "bg-emerald-500")} />
             {initializing ? "Initializing..." : "Ready"}
             <span className="border-l border-zinc-800 pl-2 ml-2">{is3D ? "Volumetric Mode" : "2D Graph Mode"}</span>
           </div>
        </div>
        
        <div className="flex gap-2 pointer-events-auto">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 bg-zinc-900 border border-zinc-800 rounded-lg text-zinc-400 hover:text-white">
             {sidebarOpen ? <Minimize size={18}/> : <Maximize size={18}/>}
          </button>
          <button onClick={() => setShowHelp(!showHelp)} className="p-2 bg-zinc-900 border border-zinc-800 rounded-lg text-zinc-400 hover:text-white">
             <Info size={18}/>
          </button>
        </div>
      </div>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            className="absolute top-24 left-4 w-80 glass-panel rounded-xl flex flex-col overflow-hidden shadow-2xl pointer-events-auto bg-zinc-950/80 backdrop-blur-md border border-zinc-800"
          >
            <div className="p-4 border-b border-zinc-800/50 space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" size={16} />
                <input
                  autoFocus
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addWord(inputText)}
                  placeholder="Add concept..."
                  className="w-full bg-zinc-900/50 border border-zinc-700/50 rounded-lg py-2.5 pl-9 pr-3 text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all"
                />
              </div>
              <button
                onClick={() => addWord(inputText)}
                disabled={loading || !inputText.trim()}
                className="w-full py-2 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
              >
                {loading ? <span className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" /> : <><Plus size={16} /> Add Node</>}
              </button>
            </div>

            <div className="p-4 space-y-2 text-sm">
               <div className="flex justify-between text-zinc-500 text-xs uppercase">
                  <span>Nodes: {nodes.length}</span>
                  <span>Links: {links.length}</span>
               </div>
               {hoveredWord && (
                 <div className="p-2 bg-purple-900/20 border border-purple-500/30 rounded text-purple-200 text-xs">
                   Selected: {nodes.find(n => n.id === hoveredWord)?.text}
                 </div>
               )}
            </div>
            
            <div className="p-4 border-t border-zinc-800/50 flex gap-2">
              <button onClick={handleReset} className="flex-1 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-xs text-zinc-300 transition-colors flex items-center justify-center gap-2"><RotateCcw size={14}/> Reset</button>
              <button onClick={clearAll} className="flex-1 py-2 bg-red-950/30 hover:bg-red-900/50 border border-red-900/30 rounded-lg text-xs text-red-400 transition-colors flex items-center justify-center gap-2"><Trash2 size={14}/> Clear</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* View Controls */}
      <div className="absolute bottom-6 right-6 flex flex-col gap-2 pointer-events-auto">
        <button 
          onClick={() => { setIs3D(!is3D); handleReset(); }} 
          className={cn(
            "p-3 border rounded-full transition-all shadow-lg",
            is3D 
              ? "bg-purple-600 text-white border-purple-500" 
              : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-white hover:bg-zinc-800"
          )}
          title={is3D ? "Switch to 2D" : "Switch to 3D"}
        >
          {is3D ? <Globe size={20} /> : <Box size={20} />}
        </button>
        <button onClick={() => setCamera(c => ({...c, zoom: c.zoom * 1.2}))} className="p-3 bg-zinc-900 border border-zinc-800 rounded-full text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors shadow-lg">
          <ZoomIn size={20} />
        </button>
      </div>

      {/* Help Modal */}
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
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              className="bg-zinc-950 border border-zinc-800 rounded-2xl max-w-md w-full p-8 shadow-2xl text-center space-y-6"
              onClick={e => e.stopPropagation()}
            >
               <div className="w-16 h-16 mx-auto bg-gradient-to-br from-purple-500 to-cyan-500 rounded-full flex items-center justify-center">
                 <Activity className="text-white" size={32} />
               </div>
               <h2 className="text-2xl font-bold text-white">Embedding Universe</h2>
               <p className="text-zinc-400">
                 Visualizing language in N-dimensional space.
                 <br/><br/>
                 <span className="text-white">2D Mode:</span> Pan/Zoom flat graph.
                 <br/>
                 <span className="text-white">3D Mode:</span> Orbit volumetric cloud.
               </p>
               <button onClick={() => setShowHelp(false)} className="px-6 py-2 bg-white text-black rounded-lg font-medium hover:bg-zinc-200">
                 Start Exploring
               </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
