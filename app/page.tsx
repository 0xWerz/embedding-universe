"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
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
  Box,
  Orbit,
  Play,
  Pause
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
  z: number; // True 3D depth
  
  // Render Cache (Computed each frame)
  screenX?: number;
  screenY?: number;
  scale?: number;
  depth?: number;
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

const SIMULATION_CONFIG = {
  chargeStrength: -250,
  linkDistanceBase: 120,
  linkStrengthBase: 0.4,
  collideRadius: 30,
  alphaDecay: 0.01, 
};

// --- 3D Math Helper ---
// Simple perspective projection
const PROJECT_FL = 800; // Focal Length

const rotate3D = (x: number, y: number, z: number, angleX: number, angleY: number) => {
  // Rotate around Y axis (Horizontal orbit)
  const cosY = Math.cos(angleY);
  const sinY = Math.sin(angleY);
  const x1 = x * cosY - z * sinY;
  const z1 = z * cosY + x * sinY;
  
  // Rotate around X axis (Vertical tilt)
  const cosX = Math.cos(angleX);
  const sinX = Math.sin(angleX);
  const y2 = y * cosX - z1 * sinX;
  const z2 = z1 * cosX + y * sinX;

  return { x: x1, y: y2, z: z2 };
};

// --- Components ---

export default function EmbeddingPage() {
  // -- State --
  const [inputText, setInputText] = useState("");
  
  // Interaction State
  const [camera, setCamera] = useState({ angleX: -0.2, angleY: 0, zoom: 1 });
  const [autoRotate, setAutoRotate] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  // Simulation State
  const [nodes, setNodes] = useState<WordNode[]>([]);
  const [links, setLinks] = useState<Connection[]>([]);
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);

  // UI State
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [error, setError] = useState("");
  const [showHelp, setShowHelp] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const pipelineRef = useRef<any>(null);
  const simulationRef = useRef<d3.Simulation<WordNode, Connection> | null>(null);
  
  // Mutable Data (for high-freq loop)
  const nodesRef = useRef<WordNode[]>([]);
  const linksRef = useRef<Connection[]>([]);
  const requestRef = useRef<number>();

  // -- Initialization --

  useEffect(() => {
    // Initialize Simulation
    simulationRef.current = d3.forceSimulation<WordNode, Connection>()
      .force("charge", d3.forceManyBody().strength(SIMULATION_CONFIG.chargeStrength))
      .force("collide", d3.forceCollide().radius(SIMULATION_CONFIG.collideRadius))
      .force("center", d3.forceCenter(0, 0).strength(0.02))
      .force("link", d3.forceLink<WordNode, Connection>()
        .id(d => d.id)
        .distance(link => SIMULATION_CONFIG.linkDistanceBase * (1.5 - link.similarity)) 
        .strength(link => SIMULATION_CONFIG.linkStrengthBase * link.similarity)
      )
      .alphaDecay(SIMULATION_CONFIG.alphaDecay)
      .stop(); // We step manually in the render loop

    // Start Render Loop
    const animate = () => {
      // 1. Tick Simulation
      if (simulationRef.current) {
        simulationRef.current.tick();
      }

      // 2. Auto Rotate
      if (autoRotate && !isDragging) {
        setCamera(prev => ({ ...prev, angleY: prev.angleY + 0.002 }));
      }

      // 3. Project 3D -> 2D
      const currentNodes = nodesRef.current;
      const { width = 1000, height = 800 } = containerRef.current?.getBoundingClientRect() || {};
      const cx = width / 2;
      const cy = height / 2;

      // Update cached screen coordinates for all nodes
      currentNodes.forEach(node => {
        if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
        
        // Rotate
        // We use node.x/y from D3 as world X/Y, and node.z as world Z
        const rot = rotate3D(node.x, node.y, node.z, camera.angleX, camera.angleY);
        
        // Project
        const scale = (PROJECT_FL * camera.zoom) / (PROJECT_FL * camera.zoom - rot.z + 1000); // Offset to prevent div/0
        const screenX = cx + rot.x * scale;
        const screenY = cy + rot.y * scale;

        node.screenX = screenX;
        node.screenY = screenY;
        node.scale = scale;
        node.depth = rot.z; // For sorting
      });

      // Trigger React Render (throttled or full speed? React 18 handles this well usually)
      // We create a new array reference to trigger render
      setNodes([...currentNodes]); 
      
      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      simulationRef.current?.stop();
    };
  }, [autoRotate, camera.angleX, camera.angleY, camera.zoom, isDragging]);

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
    setError("");

    try {
      const embedding = await getEmbedding(text);
      const color = COLORS[nodesRef.current.length % COLORS.length];

      // 3D Positioning Strategy:
      // X/Y determined by D3 (semantic). 
      // Z is random to create "Volume".
      const z = (Math.random() - 0.5) * 600; // +/- 300 depth

      const newWord: WordNode = {
        id: Date.now().toString(),
        text,
        embedding,
        color,
        x: 0, y: 0, // Starts at center, physics pushes it out
        z,
      };

      // KNN Topology
      const candidates = nodesRef.current.map(existing => ({
        id: existing.id,
        similarity: cosineSimilarity(embedding, existing.embedding)
      }));
      candidates.sort((a, b) => b.similarity - a.similarity);

      // Top 3 + Strong matches
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
      setLinks([...linksRef.current]);

      if (simulationRef.current) {
        simulationRef.current.nodes(nodesRef.current);
        (simulationRef.current.force("link") as d3.ForceLink<WordNode, Connection>).links(linksRef.current);
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

  // -- Interaction --

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setCamera(prev => ({ ...prev, zoom: Math.max(0.5, Math.min(3, prev.zoom * delta)) }));
    };

    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", onWheel);
    };
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('input')) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const deltaX = e.clientX - dragStart.x;
      const deltaY = e.clientY - dragStart.y;
      
      setCamera(prev => ({
        ...prev,
        angleY: prev.angleY + deltaX * 0.005,
        angleX: Math.max(-Math.PI/2, Math.min(Math.PI/2, prev.angleX - deltaY * 0.005))
      }));
      
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  // Removed handleWheel from here as it is now a native effect

  // -- Rendering --

  // Sort for Painter's Algorithm (draw furthest first)
  const sortedNodes = useMemo(() => {
    return [...nodes].sort((a, b) => (a.depth || 0) - (b.depth || 0));
  }, [nodes]);

  const visibleLinks = useMemo(() => {
    return links.map(link => {
      const source = nodes.find(n => n.id === (link.source as WordNode).id || n.id === link.source);
      const target = nodes.find(n => n.id === (link.target as WordNode).id || n.id === link.target);
      return { ...link, source, target };
    }).filter(l => l.source && l.target);
  }, [links, nodes]);

  return (
    <div className="relative w-full h-screen bg-black text-foreground overflow-hidden font-sans select-none">
      
      {/* --- 3D Viewport --- */}
      <div 
        ref={containerRef}
        className="absolute inset-0 cursor-move active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
      >
        <svg width="100%" height="100%" className="block">
          <defs>
             <radialGradient id="star-glow">
               <stop offset="0%" stopColor="white" stopOpacity="1" />
               <stop offset="100%" stopColor="transparent" stopOpacity="0" />
             </radialGradient>
          </defs>
          
          {/* Background Stars (Static or Parallax? Static for now) */}
          <rect width="100%" height="100%" fill="#050505" />
          
          {/* Links (Lines) */}
          {visibleLinks.map((link, i) => {
            const s = link.source as WordNode;
            const t = link.target as WordNode;
            
            if (!s.screenX || !t.screenX) return null;

            // Average depth for sorting (simplified: just draw lines behind nodes usually)
            const opacity = Math.max(0.1, link.similarity - 0.2) * (hoveredWord ? 0.2 : 0.6);
            const isHovered = hoveredWord === s.id || hoveredWord === t.id;

            return (
              <line 
                key={i}
                x1={s.screenX} y1={s.screenY}
                x2={t.screenX} y2={t.screenY}
                stroke={isHovered ? "#fff" : s.color}
                strokeWidth={isHovered ? 2 : 1}
                strokeOpacity={isHovered ? 0.8 : opacity}
              />
            );
          })}

          {/* Nodes (Sorted by Depth) */}
          {sortedNodes.map(node => {
            if (!node.screenX) return null;
            
            const scale = node.scale || 1;
            const isHovered = hoveredWord === node.id;
            
            // Scale down based on distance
            const size = 20 * scale; 
            const fontSize = (isHovered ? 14 : 10) * scale;

            return (
              <g 
                key={node.id} 
                transform={`translate(${node.screenX}, ${node.screenY})`}
                onMouseEnter={() => { setHoveredWord(node.id); setAutoRotate(false); }}
                onMouseLeave={() => { setHoveredWord(null); setAutoRotate(true); }}
                className="cursor-pointer"
              >
                {/* Glow */}
                {isHovered && <circle r={size * 2} fill="url(#star-glow)" opacity="0.5" />}
                
                {/* Core */}
                <circle 
                  r={size / 2} 
                  fill="#000" 
                  stroke={node.color} 
                  strokeWidth={2 * scale}
                />
                
                {/* Label */}
                <text
                  y={size + 5 * scale}
                  textAnchor="middle"
                  fill={isHovered ? "#fff" : "#888"}
                  fontSize={fontSize}
                  fontWeight={isHovered ? "bold" : "normal"}
                  style={{ textShadow: "0 1px 4px rgba(0,0,0,1)" }}
                >
                  {node.text}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* --- HUD --- */}
      <header className="absolute top-0 left-0 right-0 p-4 flex justify-between items-start pointer-events-none">
        <div className="pointer-events-auto backdrop-blur-md bg-black/30 p-2 rounded-lg border border-white/10">
           <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
             <Orbit size={20} className="text-purple-500" />
             Embedding Universe <span className="text-xs font-normal text-zinc-500 px-2 border-l border-zinc-700">3D Projection Engine</span>
           </h1>
        </div>
        
        <div className="flex gap-2 pointer-events-auto">
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg bg-zinc-900/80 border border-zinc-800 hover:bg-zinc-800 text-zinc-400 hover:text-white"
          >
            {sidebarOpen ? <Minimize size={18} /> : <Maximize size={18} />}
          </button>
          <button onClick={() => setShowHelp(!showHelp)} className="p-2 rounded-lg bg-zinc-900/80 border border-zinc-800 text-zinc-400 hover:text-white">
            <Info size={18} />
          </button>
        </div>
      </header>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            className="absolute top-20 left-4 w-80 bg-zinc-950/80 backdrop-blur-xl border border-zinc-800 rounded-xl shadow-2xl pointer-events-auto overflow-hidden"
          >
            <div className="p-4 space-y-3">
              <div className="relative">
                <input
                  autoFocus
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addWord(inputText)}
                  placeholder="Add concept..."
                  className="w-full bg-black/50 border border-zinc-700 rounded-lg py-2 px-3 text-white focus:outline-none focus:border-purple-500 transition-colors"
                />
                {loading && <div className="absolute right-3 top-2.5 w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />}
              </div>
              <div className="text-xs text-zinc-500 flex justify-between px-1">
                <span>{nodes.length} Nodes</span>
                <span>{links.length} Connections</span>
              </div>
            </div>
            
            <div className="border-t border-zinc-800 p-2 flex gap-2">
               <button onClick={() => { setNodes([]); setLinks([]); }} className="flex-1 py-1.5 bg-red-900/20 text-red-400 text-xs rounded hover:bg-red-900/40">
                 Clear Universe
               </button>
               <button 
                 onClick={() => setAutoRotate(!autoRotate)} 
                 className={cn("flex-1 py-1.5 text-xs rounded flex items-center justify-center gap-2", autoRotate ? "bg-purple-900/30 text-purple-300" : "bg-zinc-800 text-zinc-400")}
               >
                 {autoRotate ? <Pause size={12}/> : <Play size={12}/>} Rotate
               </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Help Overlay */}
      <AnimatePresence>
        {showHelp && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 pointer-events-auto"
            onClick={() => setShowHelp(false)}
          >
            <div className="max-w-md text-center space-y-6">
              <div className="w-20 h-20 mx-auto bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center shadow-[0_0_40px_rgba(168,85,247,0.4)]">
                <Orbit size={40} className="text-white" />
              </div>
              <h2 className="text-3xl font-bold text-white">Volumetric Semantic Space</h2>
              <p className="text-zinc-400 leading-relaxed">
                Welcome to the 3D Embedding Universe.
                <br/><br/>
                • <strong className="text-white">Orbit</strong> by dragging the background.
                <br/>
                • <strong className="text-white">Zoom</strong> with your mouse wheel.
                <br/>
                • <strong className="text-white">Hover</strong> nodes to pause rotation.
              </p>
              <button className="px-8 py-3 bg-white text-black rounded-full font-bold hover:bg-zinc-200 transition-colors">
                Enter Void
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
